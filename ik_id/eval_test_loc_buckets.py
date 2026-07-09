#!/usr/bin/env python3
"""
Evaluate a checkpoint on held-out test subjects, reporting metrics per locomotion bucket
(LG / RA / RD) for each subject.

Uses ``dataset.classify_loc_bucket`` (same buckets as training oversampling / RA-RD filters)
and the checkpoint's ``subject_split.json`` for test-subject selection.

Run from ``os_kinetics/``::

    python -m ik_id.eval_test_loc_buckets \\
        --checkpoint runs/0706_ik_id_knee_causal_in_zero_out/best_model.pt \\
        --data-dir /media/metamobility3/Samsung_T52/Processed/AddBiomechanics_final/

Outputs under ``--output-dir``:
  - ``summary.json`` — full breakdown + trial inventory + RMSE/R² pivots
  - ``metrics_subject_loc_bucket.csv`` — one row per subject × bucket (RMSE, MAE, R²)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from dataset import KineticsTCNDataset, classify_loc_bucket, load_loc_ascent_descent_mapping
from training_utils import set_global_seed

from ik_id.inter_dataset_eval import _accumulate_loader_metrics, _finalize_metrics
from ik_id.run_subject_task_eval import _apply_lpf_settings_from_run_cfg, _eval_window_subset
from ik_id.test import load_model, load_run_config, load_subject_split
from ik_id.test_addbiomech_repr_subjects import (
    DEFAULT_DATA_DIR,
    _assert_paths_only_under_root,
    build_h5_eval_kwargs,
)

DEFAULT_CHECKPOINT = _ROOT / "runs" / "0706_ik_id_knee_causal_in_zero_out" / "best_model.pt"
DEFAULT_BUCKETS: Tuple[str, ...] = ("LG", "RA", "RD")


def _parse_csv_list(value: str, *, upper: bool = False) -> List[str]:
    out = [x.strip() for x in value.split(",") if x.strip()]
    return [x.upper() for x in out] if upper else out


def _resolve_checkpoint(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_file():
        p = (_ROOT / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path_str}")
    return p


def _resolve_data_dir(cli: Optional[str], run_cfg: Optional[Dict[str, Any]]) -> Path:
    if cli:
        root = Path(cli).expanduser().resolve()
    elif run_cfg and run_cfg.get("train_dir"):
        root = Path(str(run_cfg["train_dir"])).expanduser().resolve()
    else:
        root = DEFAULT_DATA_DIR.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--data-dir is not a directory: {root}")
    return root


def _resolve_loc_map(run_cfg: Optional[Dict[str, Any]]) -> Optional[Dict[Tuple[str, str, str], str]]:
    if run_cfg is None:
        return None
    rel = run_cfg.get("loc_ascent_descent_map")
    if not rel:
        return None
    candidates = [Path(str(rel)), _ROOT / str(rel)]
    for p in candidates:
        if p.is_file():
            return load_loc_ascent_descent_mapping(str(p))
    print(f"[warn] loc_ascent_descent_map not found: {rel!r}")
    return None


def _subjects_from_split(
    split: Optional[Dict[str, Any]],
    *,
    eval_split: str,
    cli_subjects: Optional[List[str]],
) -> List[str]:
    if cli_subjects:
        return sorted({s.upper() for s in cli_subjects})
    if split is None:
        raise ValueError("No subject_split.json and no --subjects provided.")
    key = "test_subjects" if eval_split == "test" else "val_subjects"
    subjects = split.get(key) or []
    if not subjects:
        raise ValueError(f"subject_split.json has no {key!r}.")
    return sorted({str(s).upper() for s in subjects})


def _build_eval_dataset_kwargs(
    *,
    data_root: Path,
    subject_id: str,
    model: torch.nn.Module,
    stats: Any,
    window_size: int,
    input_indices: Any,
    moment_indices: Any,
    input_mode: str,
    output_mode: str,
    laterality_ckpt: str,
    unilateral_paired_ckpt: Optional[bool],
    run_cfg: Optional[Dict[str, Any]],
    walking_only: bool,
    rollout_force: bool,
    loc_map_path: Optional[str],
) -> Dict[str, Any]:
    kwargs = build_h5_eval_kwargs(
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
        subject_ids=[subject_id],
        walking_only=walking_only,
        levelground_only_cli=False,
        rollout_force=rollout_force,
    )
    _apply_lpf_settings_from_run_cfg(kwargs, run_cfg)
    # Eval all locomotion buckets on test subjects (do not mirror training filters).
    kwargs["levelground_only"] = False
    kwargs["ra_rd_only"] = False
    kwargs["exclude_stair_tasks"] = False
    if loc_map_path:
        kwargs["loc_ascent_descent_map"] = loc_map_path
    return kwargs


def _index_windows_by_bucket(
    ds: KineticsTCNDataset,
    *,
    subject_id: str,
    loc_map: Optional[Dict[Tuple[str, str, str], str]],
    target_buckets: Set[str],
) -> Tuple[Dict[str, List[int]], Dict[str, int], List[Dict[str, Any]]]:
    by_bucket: Dict[str, List[int]] = {b: [] for b in sorted(target_buckets)}
    other_counts: DefaultDict[str, int] = defaultdict(int)
    trial_rows: List[Dict[str, Any]] = []
    seen_trials: Set[Tuple[int, Optional[str]]] = set()

    for widx, (trial_idx, _start, side) in enumerate(ds.windows):
        sid, cond, trial, h5_path = ds.h5_trial_refs[trial_idx]
        bucket = classify_loc_bucket(sid, cond, trial, loc_map)
        side_key = side if side in {"r", "l"} else None
        trial_key = (int(trial_idx), side_key)
        if trial_key not in seen_trials:
            seen_trials.add(trial_key)
            trial_rows.append(
                {
                    "subject_id": sid,
                    "condition": cond,
                    "trial": trial,
                    "h5_path": h5_path,
                    "loc_bucket": bucket,
                    "trial_idx": int(trial_idx),
                    "side": side_key,
                }
            )
        if bucket in target_buckets:
            by_bucket[bucket].append(widx)
        else:
            other_counts[bucket] += 1

    return by_bucket, dict(other_counts), trial_rows


def _metrics_row(
    subject: str,
    bucket: str,
    metrics: Dict[str, Any],
    *,
    dof_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    overall = metrics.get("overall", {})
    row: Dict[str, Any] = {
        "subject": subject,
        "loc_bucket": bucket,
        "rmse_nmpkg": overall.get("rmse"),
        "mae_nmpkg": overall.get("mae"),
        "r_squared": overall.get("r2"),
        "n_windows": metrics.get("n_windows"),
        "smoothness_score": metrics.get("smoothness_score"),
    }
    per_ch = metrics.get("per_channel") or []
    if dof_names and len(per_ch) == len(dof_names):
        for ch in per_ch:
            name = str(ch.get("name", "dof"))
            row[f"r_squared__{name}"] = ch.get("r2")
            row[f"rmse_nmpkg__{name}"] = ch.get("rmse")
    return row


def _format_metric(value: Any, *, width: int = 10, precision: int = 4) -> str:
    if value is None:
        return f"{'—':>{width}}"
    return f"{float(value):{width}.{precision}f}"


def _print_table(rows: Sequence[Dict[str, Any]], *, title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("(no rows)")
        return
    hdr = f"{'subject':<12} {'bucket':<6} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'n_win':>8}"
    print(hdr)
    for r in rows:
        print(
            f"{r['subject']:<12} {r['loc_bucket']:<6} "
            f"{_format_metric(r['rmse_nmpkg'])} {_format_metric(r['mae_nmpkg'])} "
            f"{_format_metric(r['r_squared'], precision=3)} "
            f"{int(r['n_windows']):8d}"
        )


def _pivot_metric(
    rows: Sequence[Dict[str, Any]],
    *,
    subjects: Sequence[str],
    buckets: Sequence[str],
    metric_key: str,
) -> Dict[str, Dict[str, Optional[float]]]:
    lookup = {(r["subject"], r["loc_bucket"]): r.get(metric_key) for r in rows}
    return {
        sid: {b: lookup.get((sid, b)) for b in buckets}
        for sid in subjects
    }


def _print_pivot(
    pivot: Dict[str, Dict[str, Optional[float]]],
    *,
    title: str,
    precision: int = 4,
) -> None:
    if not pivot:
        return
    buckets = sorted({b for row in pivot.values() for b in row})
    print(f"\n{title}")
    print("-" * len(title))
    hdr = f"{'subject':<12}" + "".join(f"{b:>12}" for b in buckets)
    print(hdr)
    for sid in sorted(pivot):
        vals = pivot[sid]
        line = f"{sid:<12}" + "".join(
            _format_metric(vals.get(b), width=12, precision=precision) for b in buckets
        )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-subject LG/RA/RD metrics on held-out test subjects."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to best_model.pt",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="H5 root (default: train_dir from config.json next to checkpoint)",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Which subject_split.json list to use when --subjects is omitted",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="",
        help="Comma-separated subject IDs (overrides subject_split.json)",
    )
    parser.add_argument(
        "--buckets",
        type=str,
        default=",".join(DEFAULT_BUCKETS),
        help="Locomotion buckets to report (default: LG,RA,RD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/0706_test_loc_buckets",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--walking-only", action="store_true", default=True)
    parser.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    parser.add_argument(
        "--rollout",
        action="store_true",
        default=False,
        help="Force stride-2 decimation (default: follow config.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    target_buckets = {b.strip().upper() for b in args.buckets.split(",") if b.strip()}
    if not target_buckets:
        raise ValueError("No buckets requested.")

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    run_cfg = load_run_config(str(ckpt_path))
    data_root = _resolve_data_dir(args.data_dir or None, run_cfg)
    split = load_subject_split(str(ckpt_path))
    cli_subjects = _parse_csv_list(args.subjects, upper=True) if args.subjects.strip() else None
    subjects = _subjects_from_split(split, eval_split=args.eval_split, cli_subjects=cli_subjects)
    _assert_paths_only_under_root(data_root, subjects)

    loc_map = _resolve_loc_map(run_cfg)
    loc_map_path = None
    if run_cfg and run_cfg.get("loc_ascent_descent_map"):
        for p in (Path(str(run_cfg["loc_ascent_descent_map"])), _ROOT / str(run_cfg["loc_ascent_descent_map"])):
            if p.is_file():
                loc_map_path = str(p)
                break

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = (_ROOT / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

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
        rollout_force=args.rollout,
        loc_map_path=loc_map_path,
    )

    per_subject_bucket: Dict[str, Dict[str, Any]] = {}
    subject_datasets: Dict[str, KineticsTCNDataset] = {}
    subject_windows_by_bucket: Dict[str, Dict[str, List[int]]] = {}
    csv_rows: List[Dict[str, Any]] = []
    trial_inventory: List[Dict[str, Any]] = []
    subject_meta: Dict[str, Any] = {}

    print(f"Checkpoint: {ckpt_path}")
    print(f"Data dir:   {data_root}")
    print(f"Subjects:   {subjects}")
    print(f"Buckets:    {sorted(target_buckets)}")

    for sid in subjects:
        ds_kwargs = _build_eval_dataset_kwargs(**common_kw, subject_id=sid)
        ds = KineticsTCNDataset(**ds_kwargs)
        by_bucket, other_counts, trials = _index_windows_by_bucket(
            ds,
            subject_id=sid,
            loc_map=loc_map,
            target_buckets=target_buckets,
        )
        subject_datasets[sid] = ds
        subject_windows_by_bucket[sid] = by_bucket
        trial_inventory.extend(trials)
        subject_meta[sid] = {
            "n_total_windows": len(ds),
            "windows_per_bucket": {b: len(v) for b, v in by_bucket.items()},
            "windows_other_buckets": other_counts,
            "n_trials_loaded": len({(t["trial_idx"], t["side"]) for t in trials}),
        }

        per_subject_bucket[sid] = {}
        for bucket in sorted(target_buckets):
            win_idx = by_bucket.get(bucket, [])
            if not win_idx:
                continue
            metrics = _eval_window_subset(
                model,
                ds,
                win_idx,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dof_names=dof_names,
            )
            per_subject_bucket[sid][bucket] = metrics
            csv_rows.append(_metrics_row(sid, bucket, metrics, dof_names=dof_names))

    # Pooled across subjects per bucket.
    pooled_by_bucket: Dict[str, Any] = {}
    for bucket in sorted(target_buckets):
        sum_sq_ch = None
        sum_abs_ch = None
        sum_t_ch = None
        sum_t2_ch = None
        n_elem_ch = 0
        sum_sq_all = 0.0
        sum_abs_all = 0.0
        sum_t_all = 0.0
        sum_t2_all = 0.0
        n_all = 0
        smooth_abs_sum = 0.0
        smooth_n = 0
        n_scatter = 0
        n_windows = 0
        subjects_hit: Set[str] = set()

        for sid in subjects:
            win_idx = subject_windows_by_bucket.get(sid, {}).get(bucket, [])
            if not win_idx:
                continue
            subjects_hit.add(sid)
            ds = subject_datasets[sid]
            loader = DataLoader(
                Subset(ds, win_idx),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(args.device == "cuda"),
            )
            scatter_gt_chunks: List[Any] = []
            scatter_pred_chunks: List[Any] = []
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
            n_windows += len(win_idx)

        if sum_sq_ch is None:
            continue
        metrics = _finalize_metrics(
            dof_names=dof_names,
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
        metrics["smoothness_score"] = float(smooth_abs_sum / max(smooth_n, 1))
        metrics["n_windows"] = int(n_windows)
        metrics["n_subjects"] = len(subjects_hit)
        metrics["subjects"] = sorted(subjects_hit)
        pooled_by_bucket[bucket] = metrics
        csv_rows.append(_metrics_row("POOLED", bucket, metrics, dof_names=dof_names))

    subject_rows = [r for r in csv_rows if r["subject"] != "POOLED"]
    pooled_rows = [r for r in csv_rows if r["subject"] == "POOLED"]
    pivot_rmse = _pivot_metric(
        subject_rows, subjects=subjects, buckets=sorted(target_buckets), metric_key="rmse_nmpkg"
    )
    pivot_r_squared = _pivot_metric(
        subject_rows, subjects=subjects, buckets=sorted(target_buckets), metric_key="r_squared"
    )

    summary: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_root),
        "eval_split": args.eval_split,
        "subjects": subjects,
        "buckets_requested": sorted(target_buckets),
        "loc_map_path": loc_map_path,
        "subject_meta": subject_meta,
        "pivot_rmse_nmpkg": pivot_rmse,
        "pivot_r_squared": pivot_r_squared,
        "results_subject_bucket": {
            sid: {b: v for b, v in buckets.items()}
            for sid, buckets in per_subject_bucket.items()
        },
        "results_pooled_by_bucket": pooled_by_bucket,
        "trial_inventory": trial_inventory,
    }

    csv_path = out_root / "metrics_subject_loc_bucket.csv"
    base_fields = [
        "subject",
        "loc_bucket",
        "rmse_nmpkg",
        "mae_nmpkg",
        "r_squared",
        "n_windows",
        "smoothness_score",
    ]
    extra_fields = sorted(
        k for row in csv_rows for k in row if k not in base_fields
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    json_path = out_root / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _print_table(subject_rows, title="Per-subject × loc-bucket metrics")
    _print_table(pooled_rows, title="Pooled across subjects")
    _print_pivot(pivot_rmse, title="RMSE (N·m/kg) — subject × bucket")
    _print_pivot(pivot_r_squared, title="R² — subject × bucket", precision=3)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
