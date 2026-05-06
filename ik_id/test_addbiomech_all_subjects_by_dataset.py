#!/usr/bin/env python3
"""
Evaluate a trained IK->ID checkpoint on all available subjects per source dataset
slice (Jinwoo ranges + JSON-discovered slices under --data-dir), and report
overall RMSE/R^2 for each source dataset.

Run from ``os_kinetics/``::

    python -m ik_id.test_addbiomech_all_subjects_by_dataset
    python ik_id/test_addbiomech_all_subjects_by_dataset.py \
        --checkpoint runs/0427_ik_id_all_addbiomech/best_model.pt \
        --data-dir /media/metamobility3/Samsung_T5/Processed/Addbiomech_final
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import KineticsTCNDataset
from training_utils import set_global_seed

from ik_id.test import load_model, load_run_config
from ik_id.inter_dataset_eval import _accumulate_loader_metrics, _finalize_metrics
from ik_id.test_addbiomech_repr_subjects import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
    _assert_paths_only_under_root,
    _subject_sort_key,
    build_h5_eval_kwargs,
    discover_dataset_slices_from_dir,
    list_h5_subjects_under_root,
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "dataset"


def _resolve_dataset_subjects(
    data_root: Path,
    explicit_subjects: Optional[List[str]],
) -> tuple[Dict[str, List[str]], Dict[str, Any]]:
    available = set(list_h5_subjects_under_root(data_root))
    if not available:
        raise FileNotFoundError(f"No S*.h5 files found under {data_root}")

    if explicit_subjects:
        requested: Set[str] = set()
        for sid in explicit_subjects:
            sid_u = sid.strip().upper()
            if sid_u and sid_u in available:
                requested.add(sid_u)
            elif sid_u:
                raise FileNotFoundError(f"Requested subject {sid_u!r} not found under {data_root}")
        available = requested
        if not available:
            raise RuntimeError("No valid subjects left after --subjects filtering.")

    slices, slice_debug = discover_dataset_slices_from_dir(data_root)

    preferred = ("Camargo", "Scherpereel", "Molinaro_Scherpereel")
    ordered_labels: List[str] = [lab for lab in preferred if lab in slices]
    ordered_labels.extend(sorted(lab for lab in slices if lab not in preferred))

    dataset_subjects: Dict[str, List[str]] = {}
    skipped: List[str] = []
    for lab in ordered_labels:
        ids = sorted(slices[lab] & available, key=_subject_sort_key)
        if not ids:
            skipped.append(lab)
            continue
        dataset_subjects[lab] = ids

    if skipped:
        slice_debug["slices_skipped_no_available_subjects"] = skipped

    covered = set()
    for ids in dataset_subjects.values():
        covered.update(ids)
    leftover = sorted(available - covered, key=_subject_sort_key)
    if leftover:
        slice_debug["h5_subjects_not_in_any_slice"] = leftover

    if not dataset_subjects:
        raise RuntimeError("No dataset slices with available subjects were found.")

    return dataset_subjects, slice_debug


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval checkpoint on all available subjects for each dataset slice."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to best_model.pt (default: 0427_ik_id_all_addbiomech).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Folder containing S*.h5 and optional source-dataset JSON manifests.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="",
        help="Optional comma-separated subject IDs; if set, restricts eval to these IDs.",
    )
    parser.add_argument("--output-dir", type=str, default="results/addbiomech_dataset_eval")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--subjects-per-chunk",
        type=int,
        default=1,
        help="Max number of subjects loaded at once for each dataset slice.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=False,
        help="Enable DataLoader pin_memory (default: off for stability).",
    )
    parser.add_argument("--walking-only", action="store_true", default=True)
    parser.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    parser.add_argument("--levelground-only", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rollout",
        action="store_true",
        default=False,
        help="Force stride-2 IK/ID decimation (same as testV2 --rollout).",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    data_root = Path(args.data_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"--data-dir is not a directory: {data_root}")

    explicit_subjects: Optional[List[str]] = None
    if args.subjects.strip():
        explicit_subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    dataset_subjects, slice_debug = _resolve_dataset_subjects(data_root, explicit_subjects)
    for ids in dataset_subjects.values():
        _assert_paths_only_under_root(data_root, ids)

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = (_ROOT / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        ckpt_try = (_ROOT / ckpt_path).resolve()
        if ckpt_try.is_file():
            ckpt_path = ckpt_try
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Data root (H5 only): {data_root}")
    print(f"Checkpoint:          {ckpt_path}")
    print("Dataset slices and #subjects:")
    if args.subjects_per_chunk < 1:
        raise ValueError("--subjects-per-chunk must be >= 1")

    for dataset_name, ids in dataset_subjects.items():
        print(f"  {dataset_name:<32} {len(ids):3d}")
    for msg in slice_debug.get("json_warnings", []) or []:
        print(f"  [json warning] {msg}")
    for ds in slice_debug.get("slices_skipped_no_available_subjects", []) or []:
        print(f"  [skip] {ds}: no available subjects")

    (
        model,
        stats,
        dof_names,
        window_size,
        input_indices,
        moment_indices,
        input_mode,
        output_mode,
        laterality_ckpt,
        unilateral_paired_ckpt,
    ) = load_model(str(ckpt_path), args.device)
    run_cfg = load_run_config(str(ckpt_path))

    common_kw = dict(
        data_root=data_root,
        model=model,
        stats=stats,
        window_size=window_size,
        input_indices=input_indices,
        moment_indices=moment_indices,
        input_mode=input_mode,
        output_mode=output_mode,
        laterality_ckpt=laterality_ckpt,
        unilateral_paired_ckpt=unilateral_paired_ckpt,
        run_cfg=run_cfg,
        walking_only=args.walking_only,
        levelground_only_cli=args.levelground_only,
        rollout_force=args.rollout,
    )

    summary: Dict[str, Any] = {
        "data_dir": str(data_root),
        "checkpoint": str(ckpt_path),
        "slice_discovery": slice_debug,
        "datasets": {},
    }

    for dataset_name, ids in dataset_subjects.items():
        ds_out = out_root / _slugify(dataset_name)
        ds_out.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}\nDataset {dataset_name} ({len(ids)} subjects)\n{'=' * 70}")

        sum_sq_ch: Optional[np.ndarray] = None
        sum_abs_ch: Optional[np.ndarray] = None
        sum_t_ch: Optional[np.ndarray] = None
        sum_t2_ch: Optional[np.ndarray] = None
        n_elem_ch = 0
        sum_sq_all = 0.0
        sum_abs_all = 0.0
        sum_t_all = 0.0
        sum_t2_all = 0.0
        n_all = 0
        smooth_abs_sum = 0.0
        smooth_n = 0
        n_scatter = 0
        scatter_gt_chunks: List[np.ndarray] = []
        scatter_pred_chunks: List[np.ndarray] = []

        n_windows = 0
        n_subjects_no_trials = 0
        n_subjects_with_trials = 0

        for idx in range(0, len(ids), args.subjects_per_chunk):
            chunk_ids = ids[idx : idx + args.subjects_per_chunk]
            test_ds_kwargs = build_h5_eval_kwargs(**common_kw, subject_ids=chunk_ids)
            try:
                test_ds = KineticsTCNDataset(**test_ds_kwargs)
            except ValueError as e:
                msg = str(e)
                if "No valid trials found" not in msg:
                    raise
                n_subjects_no_trials += len(chunk_ids)
                print(f"  [chunk skip] {chunk_ids[0]}..{chunk_ids[-1]}: no valid trials")
                continue

            chunk_windows = len(test_ds)
            n_windows += chunk_windows
            if chunk_windows == 0:
                n_subjects_no_trials += len(chunk_ids)
                print(f"  [chunk skip] {chunk_ids[0]}..{chunk_ids[-1]}: zero windows")
                continue
            n_subjects_with_trials += len(chunk_ids)

            loader = DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(args.pin_memory and args.device == "cuda"),
            )
            (
                sum_sq_ch,
                sum_abs_ch,
                sum_t_ch,
                sum_t2_ch,
                n_elem_ch,
                sum_sq_all,
                sum_abs_all,
                sum_t_all,
                sum_t2_all,
                n_all,
                smooth_abs_sum,
                smooth_n,
                n_scatter,
            ) = _accumulate_loader_metrics(
                model,
                loader,
                args.device,
                sum_sq_ch=sum_sq_ch,
                sum_abs_ch=sum_abs_ch,
                sum_t_ch=sum_t_ch,
                sum_t2_ch=sum_t2_ch,
                n_elem_ch=n_elem_ch,
                sum_sq_all=sum_sq_all,
                sum_abs_all=sum_abs_all,
                sum_t_all=sum_t_all,
                sum_t2_all=sum_t2_all,
                n_all=n_all,
                smooth_abs_sum=smooth_abs_sum,
                smooth_n=smooth_n,
                scatter_gt_chunks=scatter_gt_chunks,
                scatter_pred_chunks=scatter_pred_chunks,
                n_scatter=n_scatter,
                scatter_max_points=0,
            )
            # Reduce long-run allocator pressure between many dataset chunks.
            del loader
            del test_ds
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

        print(f"  Windows: {n_windows:,}")
        if n_windows == 0 or sum_sq_ch is None or sum_abs_ch is None or sum_t_ch is None or sum_t2_ch is None:
            print("  [skip] No windows after filters.")
            summary["datasets"][dataset_name] = {
                "n_subjects": len(ids),
                "subject_ids": ids,
                "n_subjects_with_trials": 0,
                "n_subjects_no_trials": len(ids),
                "n_windows": 0,
                "skipped": "no_windows",
            }
            continue

        metrics = _finalize_metrics(
            dof_names=list(dof_names),
            sum_sq_ch=sum_sq_ch,
            sum_abs_ch=sum_abs_ch,
            sum_t_ch=sum_t_ch,
            sum_t2_ch=sum_t2_ch,
            n_elem_ch=n_elem_ch,
            sum_sq_all=sum_sq_all,
            sum_abs_all=sum_abs_all,
            sum_t_all=sum_t_all,
            sum_t2_all=sum_t2_all,
            n_all=n_all,
        )
        smoothness = float(smooth_abs_sum / max(smooth_n, 1))
        overall = metrics.get("overall", {})
        result = {
            "n_subjects": len(ids),
            "subject_ids": ids,
            "n_subjects_with_trials": n_subjects_with_trials,
            "n_subjects_no_trials": n_subjects_no_trials,
            "n_windows": n_windows,
            "overall_rmse": overall.get("rmse"),
            "overall_mae": overall.get("mae"),
            "overall_r2": overall.get("r2"),
            "smoothness_score": smoothness,
        }
        summary["datasets"][dataset_name] = result

        print(f"  Overall RMSE: {result['overall_rmse']:.6f}")
        print(f"  Overall MAE:  {result['overall_mae']:.6f}")
        print(f"  Overall R^2:  {result['overall_r2']:.6f}")

        with open(ds_out / "metrics.json", "w") as f:
            json.dump(
                {
                    "dataset_slice": dataset_name,
                    "n_subjects": len(ids),
                    "subject_ids": ids,
                    "n_subjects_with_trials": n_subjects_with_trials,
                    "n_subjects_no_trials": n_subjects_no_trials,
                    "n_windows": n_windows,
                    "metrics": metrics,
                    "smoothness_score": smoothness,
                },
                f,
                indent=2,
            )

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote per-dataset metrics to {out_root}")


if __name__ == "__main__":
    main()
