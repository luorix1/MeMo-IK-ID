#!/usr/bin/env python3
"""
Extension of ``test_addbiomech_all_subjects_by_dataset.py``: same checkpoint,
data root, and per-source-dataset subject slices, but **metrics are split by
H5 task condition** (the condition group name under each ``S*.h5``).

For every source-dataset slice (Camargo / JSON cohorts / …), this script pools
all eval windows, attributes each window to its trial’s **condition** string,
and reports RMSE / R² **per condition**. It then flags conditions that are
**substantially worse** than that dataset’s pooled baseline: higher RMSE
(``--worse-rmse-ratio``), lower R² (``--worse-r2-gap``), or both — any
trigger counts if ``--min-windows-condition`` is met.

Run from ``os_kinetics/``::

    python -m ik_id.test_addbiomech_inaccuracy_by_dataset_condition
    python ik_id/test_addbiomech_inaccuracy_by_dataset_condition.py \\
        --checkpoint runs/0427_ik_id_all_addbiomech/best_model.pt \\
        --data-dir /media/metamobility3/Samsung_T5/Processed/Addbiomech_final

Outputs (under ``--output-dir``): ``summary.json`` plus one
``<dataset_slug>/condition_breakdown.json`` per slice.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import KineticsTCNDataset, classify_loc_condition_family
from training_utils import set_global_seed

from ik_id.test import load_model, load_run_config
from ik_id.test_addbiomech_all_subjects_by_dataset import _resolve_dataset_subjects, _slugify
from ik_id.test_addbiomech_repr_subjects import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
    _assert_paths_only_under_root,
    build_h5_eval_kwargs,
)


def _new_cond_bucket() -> Dict[str, float]:
    return {
        "sum_sq_all": 0.0,
        "sum_abs_all": 0.0,
        "sum_t_all": 0.0,
        "sum_t2_all": 0.0,
        "n_all": 0.0,
        "n_windows": 0.0,
    }


def _merge_cond_stats(
    dst: DefaultDict[str, Dict[str, float]], src: Dict[str, Dict[str, float]]
) -> None:
    for cond, b in src.items():
        d = dst[cond]
        for k in ("sum_sq_all", "sum_abs_all", "sum_t_all", "sum_t2_all", "n_all", "n_windows"):
            d[k] += float(b[k])


def _overall_from_bucket(b: Dict[str, float]) -> Dict[str, Optional[float]]:
    n = int(b["n_all"])
    if n <= 0:
        return {"rmse": None, "mae": None, "r2": None, "n_scalar_elements": 0, "n_windows": 0}
    sum_sq = float(b["sum_sq_all"])
    mse = sum_sq / n
    mean_t = float(b["sum_t_all"] / n)
    ss_tot = float(b["sum_t2_all"] - b["sum_t_all"] * mean_t)
    r2 = float(1.0 - sum_sq / (ss_tot + 1e-12))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(b["sum_abs_all"] / n),
        "r2": r2,
        "n_scalar_elements": n,
        "n_windows": int(b["n_windows"]),
    }


@torch.no_grad()
def _accumulate_metrics_by_h5_condition(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: KineticsTCNDataset,
    device: str,
    out: DefaultDict[str, Dict[str, float]],
) -> None:
    """Fill ``out`` keyed by H5 condition name (second component of ``h5_trial_refs``)."""
    if not getattr(dataset, "use_h5", False):
        raise RuntimeError("This script expects H5 eval (use_h5=True).")
    global_idx = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        pred = model(x)
        pb = pred.detach().cpu().numpy().astype(np.float64)
        tb = y.numpy().astype(np.float64)
        bsz = pb.shape[0]
        for b in range(bsz):
            widx = global_idx + b
            t_idx, _start, _side = dataset.windows[widx]
            ref = dataset.h5_trial_refs[t_idx]
            cond = ref[1]
            diff = pb[b] - tb[b]
            bucket = out[cond]
            bucket["sum_sq_all"] += float(np.sum(diff**2))
            bucket["sum_abs_all"] += float(np.sum(np.abs(diff)))
            bucket["sum_t_all"] += float(np.sum(tb[b]))
            bucket["sum_t2_all"] += float(np.sum(tb[b] ** 2))
            bucket["n_all"] += float(tb[b].size)
            bucket["n_windows"] += 1.0
        global_idx += bsz


def _flag_conditions(
    *,
    baseline_rmse: float,
    baseline_r2: float,
    by_condition: Dict[str, Dict[str, Any]],
    worse_rmse_ratio: float,
    worse_r2_gap: float,
    min_windows: int,
) -> List[str]:
    """Return condition names flagged as highly inaccurate for this dataset."""
    flagged: List[str] = []
    br = float(baseline_rmse) if baseline_rmse is not None else 0.0
    b2 = float(baseline_r2) if baseline_r2 is not None else 0.0
    for cond, row in sorted(by_condition.items()):
        m = row.get("metrics") or {}
        rmse = m.get("rmse")
        r2 = m.get("r2")
        nw = int(row.get("n_windows") or 0)
        if rmse is None or r2 is None or nw < min_windows:
            continue
        bad_rmse = float(rmse) >= br * float(worse_rmse_ratio)
        bad_r2 = float(r2) <= b2 - float(worse_r2_gap)
        if bad_rmse or bad_r2:
            flagged.append(cond)
    return sorted(flagged)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per H5 task-condition inaccuracy within each AddBiomech source-dataset slice."
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/addbiomech_dataset_condition_inaccuracy",
    )
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
    parser.add_argument(
        "--worse-rmse-ratio",
        type=float,
        default=1.15,
        help="Flag if condition RMSE >= this times the dataset pooled RMSE.",
    )
    parser.add_argument(
        "--worse-r2-gap",
        type=float,
        default=0.08,
        help="Flag if condition R² <= dataset pooled R² minus this gap.",
    )
    parser.add_argument(
        "--min-windows-condition",
        type=int,
        default=32,
        help="Minimum windows per condition to consider it for inaccuracy flags.",
    )
    parser.add_argument(
        "--top-k-print",
        type=int,
        default=12,
        help="How many worst conditions to print per dataset (0 disables).",
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

    if args.subjects_per_chunk < 1:
        raise ValueError("--subjects-per-chunk must be >= 1")

    print(f"Data root (H5 only): {data_root}")
    print(f"Checkpoint:          {ckpt_path}")
    print("Dataset slices and #subjects:")
    for dataset_name, ids in dataset_subjects.items():
        print(f"  {dataset_name:<32} {len(ids):3d}")

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
    model.eval()

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
        "flag_params": {
            "worse_rmse_ratio": args.worse_rmse_ratio,
            "worse_r2_gap": args.worse_r2_gap,
            "min_windows_condition": args.min_windows_condition,
        },
        "datasets": {},
    }

    global_rows: List[Dict[str, Any]] = []

    for dataset_name, ids in dataset_subjects.items():
        ds_out = out_root / _slugify(dataset_name)
        ds_out.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}\nDataset {dataset_name} ({len(ids)} subjects)\n{'=' * 70}")

        cond_total: DefaultDict[str, Dict[str, float]] = defaultdict(_new_cond_bucket)
        n_windows_ds = 0
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
            if chunk_windows == 0:
                n_subjects_no_trials += len(chunk_ids)
                print(f"  [chunk skip] {chunk_ids[0]}..{chunk_ids[-1]}: zero windows")
                continue
            n_subjects_with_trials += len(chunk_ids)
            n_windows_ds += chunk_windows

            loader = DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(args.pin_memory and args.device == "cuda"),
            )
            chunk_stats: DefaultDict[str, Dict[str, float]] = defaultdict(_new_cond_bucket)
            _accumulate_metrics_by_h5_condition(model, loader, test_ds, args.device, chunk_stats)
            _merge_cond_stats(cond_total, chunk_stats)

            del loader
            del test_ds
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

        print(f"  Windows (dataset): {n_windows_ds:,}")

        if n_windows_ds == 0 or not cond_total:
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

        pooled = _new_cond_bucket()
        for _c, b in cond_total.items():
            for k in pooled:
                pooled[k] += b[k]
        ds_overall = _overall_from_bucket(pooled)
        baseline_rmse = ds_overall["rmse"] or 0.0
        baseline_r2 = ds_overall["r2"] or 0.0

        by_condition: Dict[str, Dict[str, Any]] = {}
        for cond, b in sorted(cond_total.items(), key=lambda kv: (-kv[1]["n_all"], kv[0])):
            m = _overall_from_bucket(b)
            fam = classify_loc_condition_family(cond)
            row = {
                "condition": cond,
                "loc_family": fam,
                "n_windows": m["n_windows"],
                "n_scalar_elements": m["n_scalar_elements"],
                "metrics": {"rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"]},
            }
            by_condition[cond] = row
            global_rows.append(
                {
                    "dataset": dataset_name,
                    "condition": cond,
                    "loc_family": fam,
                    "n_windows": m["n_windows"],
                    "rmse": m["rmse"],
                    "r2": m["r2"],
                }
            )

        flagged = _flag_conditions(
            baseline_rmse=baseline_rmse,
            baseline_r2=baseline_r2,
            by_condition=by_condition,
            worse_rmse_ratio=args.worse_rmse_ratio,
            worse_r2_gap=args.worse_r2_gap,
            min_windows=args.min_windows_condition,
        )

        # Rank conditions by RMSE (worst first) among those with enough windows
        ranked = sorted(
            (c, by_condition[c]) for c in by_condition if by_condition[c]["n_windows"] >= args.min_windows_condition
        )
        ranked.sort(key=lambda t: (-(t[1]["metrics"]["rmse"] or -1.0), t[0]))

        summary["datasets"][dataset_name] = {
            "n_subjects": len(ids),
            "subject_ids": ids,
            "n_subjects_with_trials": n_subjects_with_trials,
            "n_subjects_no_trials": n_subjects_no_trials,
            "n_windows": n_windows_ds,
            "pooled_overall_rmse": ds_overall["rmse"],
            "pooled_overall_mae": ds_overall["mae"],
            "pooled_overall_r2": ds_overall["r2"],
            "n_distinct_conditions": len(by_condition),
            "highly_inaccurate_conditions": flagged,
            "worst_conditions_by_rmse": [c for c, _ in ranked[: args.top_k_print]],
        }

        print(f"  Pooled RMSE: {ds_overall['rmse']:.6f}  R^2: {ds_overall['r2']:.6f}")
        print(f"  Distinct H5 conditions: {len(by_condition)}")
        if flagged:
            print(f"  Flagged ({len(flagged)}): {', '.join(flagged[:20])}{' …' if len(flagged) > 20 else ''}")
        else:
            print("  Flagged: (none under current thresholds)")
        if args.top_k_print > 0 and ranked:
            print(f"  Top {min(args.top_k_print, len(ranked))} worst by RMSE:")
            for c, row in ranked[: args.top_k_print]:
                mm = row["metrics"]
                print(
                    f"    {c[:56]:<56}  RMSE={mm['rmse']:.5f}  R2={mm['r2']:.4f}  n_win={row['n_windows']}"
                )

        with open(ds_out / "condition_breakdown.json", "w") as f:
            json.dump(
                {
                    "dataset_slice": dataset_name,
                    "pooled": {
                        "rmse": ds_overall["rmse"],
                        "mae": ds_overall["mae"],
                        "r2": ds_overall["r2"],
                        "n_windows": n_windows_ds,
                    },
                    "by_condition": by_condition,
                    "highly_inaccurate_conditions": flagged,
                },
                f,
                indent=2,
            )

    global_rows.sort(key=lambda r: (-(r["rmse"] or -1.0), r["dataset"], r["condition"]))
    summary["worst_conditions_global"] = global_rows[: max(50, args.top_k_print * 5)]

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote per-dataset condition breakdown under {out_root}")


if __name__ == "__main__":
    main()
