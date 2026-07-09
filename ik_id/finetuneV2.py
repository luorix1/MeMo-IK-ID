#!/usr/bin/env python3
"""
Fine-tune an IK→ID checkpoint on a **balanced** LG / RA / RD training subset.

Loads the original run's ``config.json`` and ``subject_split.json`` (next to the
checkpoint), builds the **train** split with all locomotion buckets available
(``ra_rd_only=False``), counts sliding windows per bucket, then subsamples so
each requested bucket contributes the same count as the smallest bucket
(random sampling without replacement for larger buckets).

Validation uses the same subject split and preprocessing as the source run
(optional ``--val-ra-rd-only`` to mirror original ramp-only validation).

Run from ``os_kinetics/``::

    python -m ik_id.finetuneV2 \\
        --checkpoint runs/0706_ik_id_knee_causal_in_zero_out/best_model.pt \\
        --output-dir runs/0706_knee_finetune_balanced_lg_ra_rd \\
        --epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset import (
    KineticsTCNDataset,
    _subject_id_excluded_temp_broken_h5,
    classify_loc_bucket,
    extract_subject_id,
    find_trial_dirs,
    load_loc_ascent_descent_mapping,
)
from model import GaussianDiffusion1D
from training_utils import MomentLoss, evaluate, set_global_seed, train_one_epoch

from ik_id.test import load_model, load_run_config, load_subject_split
from ik_id.trainV2 import (
    _save_checkpoint,
    _write_run_metadata_json,
    plot_curves,
    plot_sample_prediction,
    train_one_epoch_diffusion,
)

DEFAULT_BUCKETS: Tuple[str, ...] = ("LG", "RA", "RD")


def _resolve_checkpoint(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_file():
        p = (_ROOT / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path_str}")
    return p


def _parse_bucket_list(value: str) -> Tuple[str, ...]:
    buckets = tuple(x.strip().upper() for x in value.split(",") if x.strip())
    if not buckets:
        raise ValueError("At least one balance bucket is required.")
    return buckets


def _resolve_loc_map(run_cfg: Optional[Dict[str, Any]]) -> Optional[Dict[Tuple[str, str, str], str]]:
    if run_cfg is None:
        return None
    rel = run_cfg.get("loc_ascent_descent_map")
    if not rel:
        return None
    for p in (Path(str(rel)), _ROOT / str(rel)):
        if p.is_file():
            return load_loc_ascent_descent_mapping(str(p))
    print(f"[warn] loc_ascent_descent_map not found: {rel!r}")
    return None


def _trial_loc_bucket(
    ds: KineticsTCNDataset,
    trial_idx: int,
    loc_map: Optional[Dict[Tuple[str, str, str], str]],
) -> str:
    if ds.use_h5:
        sid, cond, trial, _ = ds.h5_trial_refs[trial_idx]
    else:
        td = ds.trial_dirs[trial_idx]
        sid = extract_subject_id(td)
        cond = td.parent.name
        trial = td.name
    return classify_loc_bucket(sid, cond, trial, loc_map)


def _index_windows_by_bucket(
    ds: KineticsTCNDataset,
    *,
    target_buckets: Sequence[str],
    loc_map: Optional[Dict[Tuple[str, str, str], str]],
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    bucket_set = set(target_buckets)
    by_bucket: Dict[str, List[int]] = {b: [] for b in target_buckets}
    other_counts: Dict[str, int] = {}

    for widx, (trial_idx, _start, _side) in enumerate(ds.windows):
        bucket = _trial_loc_bucket(ds, trial_idx, loc_map)
        if bucket in bucket_set:
            by_bucket[bucket].append(widx)
        else:
            other_counts[bucket] = other_counts.get(bucket, 0) + 1

    return by_bucket, other_counts


def _subsample_balanced_indices(
    by_bucket: Dict[str, List[int]],
    *,
    seed: int,
) -> Tuple[List[int], Dict[str, Any]]:
    present = {k: v for k, v in by_bucket.items() if len(v) > 0}
    missing = [k for k, v in by_bucket.items() if len(v) == 0]
    if missing:
        raise ValueError(
            f"Cannot balance buckets: no windows for {missing}. "
            f"Counts before subsample: {{{', '.join(f'{k}:{len(v)}' for k, v in by_bucket.items())}}}"
        )
    if len(present) < 1:
        raise ValueError("Need at least one non-empty bucket to balance.")

    target = min(len(v) for v in present.values())
    rng = np.random.default_rng(int(seed))
    chosen: List[int] = []
    counts_after: Dict[str, int] = {}
    for bucket, indices in by_bucket.items():
        if not indices:
            counts_after[bucket] = 0
            continue
        n = len(indices)
        if n == target:
            pick = indices
        else:
            pick_idx = rng.choice(n, size=target, replace=False)
            pick = [indices[i] for i in pick_idx.tolist()]
        chosen.extend(pick)
        counts_after[bucket] = len(pick)

    rng.shuffle(chosen)
    meta = {
        "balance_mode": "undersample",
        "target_per_bucket": int(target),
        "counts_before": {k: len(v) for k, v in by_bucket.items()},
        "counts_after": counts_after,
        "n_train_windows_balanced": len(chosen),
        "seed": int(seed),
    }
    return chosen, meta


def _oversample_balanced_indices(
    by_bucket: Dict[str, List[int]],
    *,
    seed: int,
) -> Tuple[List[int], Dict[str, Any]]:
    present = {k: v for k, v in by_bucket.items() if len(v) > 0}
    missing = [k for k, v in by_bucket.items() if len(v) == 0]
    if missing:
        raise ValueError(
            f"Cannot balance buckets: no windows for {missing}. "
            f"Counts before oversample: {{{', '.join(f'{k}:{len(v)}' for k, v in by_bucket.items())}}}"
        )
    if len(present) < 1:
        raise ValueError("Need at least one non-empty bucket to balance.")

    target = max(len(v) for v in present.values())
    rng = np.random.default_rng(int(seed))
    chosen: List[int] = []
    counts_after: Dict[str, int] = {}
    for bucket, indices in by_bucket.items():
        if not indices:
            counts_after[bucket] = 0
            continue
        n = len(indices)
        if n == target:
            pick = indices
        else:
            pick_idx = rng.choice(n, size=target, replace=True)
            pick = [indices[i] for i in pick_idx.tolist()]
        chosen.extend(pick)
        counts_after[bucket] = len(pick)

    rng.shuffle(chosen)
    meta = {
        "balance_mode": "oversample",
        "target_per_bucket": int(target),
        "counts_before": {k: len(v) for k, v in by_bucket.items()},
        "counts_after": counts_after,
        "n_train_windows_balanced": len(chosen),
        "seed": int(seed),
    }
    return chosen, meta


def _balance_window_indices(
    by_bucket: Dict[str, List[int]],
    *,
    mode: str,
    seed: int,
) -> Tuple[List[int], Dict[str, Any]]:
    if mode == "undersample":
        return _subsample_balanced_indices(by_bucket, seed=seed)
    if mode == "oversample":
        return _oversample_balanced_indices(by_bucket, seed=seed)
    raise ValueError(f"balance_mode must be 'undersample' or 'oversample', got {mode!r}")


def _evaluate_val_per_bucket(
    model: torch.nn.Module,
    val_ds: KineticsTCNDataset,
    device: str,
    loc_map: Optional[Dict[Tuple[str, str, str], str]],
    buckets: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Per-bucket val MSE/R² on the validation loader windows."""
    by_bucket: Dict[str, List[int]] = {b: [] for b in buckets}
    for widx, (trial_idx, _start, _side) in enumerate(val_ds.windows):
        bucket = _trial_loc_bucket(val_ds, trial_idx, loc_map)
        if bucket in by_bucket:
            by_bucket[bucket].append(widx)

    mse_crit = nn.MSELoss()
    out: Dict[str, Dict[str, float]] = {}
    for bucket, indices in by_bucket.items():
        if not indices:
            continue
        loader = DataLoader(
            Subset(val_ds, indices),
            batch_size=512,
            shuffle=False,
            num_workers=0,
        )
        val_loss, _per_rmse, r2_global, _per_r2 = evaluate(
            model, loader, mse_crit, device
        )
        out[bucket] = {
            "n_windows": float(len(indices)),
            "val_mse": float(val_loss),
            "val_r2": float(r2_global) if np.isfinite(r2_global) else float("nan"),
        }
    return out


def _detect_layout(train_root: Path) -> Tuple[bool, List[str], Optional[Dict[str, List[Path]]]]:
    h5_files = sorted(p for p in train_root.glob("S*.h5") if p.is_file())
    if h5_files:
        subjects = sorted(
            p.stem.upper()
            for p in h5_files
            if not _subject_id_excluded_temp_broken_h5(p.stem.upper())
        )
        return True, subjects, None

    all_trials = find_trial_dirs(str(train_root))
    subject_to_trials: Dict[str, List[Path]] = {}
    for td in all_trials:
        sid = extract_subject_id(td)
        subject_to_trials.setdefault(sid, []).append(td)
    return False, sorted(subject_to_trials.keys()), subject_to_trials


def _files_for_subjects(
    subjects: List[str],
    *,
    is_h5: bool,
    subject_to_trials: Optional[Dict[str, List[Path]]],
) -> List[Any]:
    if is_h5:
        return sorted(subjects)
    assert subject_to_trials is not None
    return [td for s in subjects for td in subject_to_trials.get(s, [])]


def _denoise_kwargs_from_run_cfg(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    rollout_step = int(run_cfg.get("rollout_decimate_step", 1))
    if rollout_step == 1 and bool(run_cfg.get("rollout")):
        rollout_step = 2
    rollout_step = max(1, rollout_step)
    return dict(
        apply_lowpass_filter=True,
        input_lowpass_mode=str(run_cfg.get("input_lowpass_mode", "zero_phase")),
        apply_moment_lowpass_filter=(str(run_cfg.get("output_lowpass_mode", "zero_phase")) != "none"),
        moment_lowpass_mode=(
            "zero_phase"
            if str(run_cfg.get("output_lowpass_mode", "zero_phase")) == "none"
            else str(run_cfg.get("output_lowpass_mode", "zero_phase"))
        ),
        lowpass_cutoff_hz=float(run_cfg.get("lowpass_cutoff_hz", 6.0)),
        lowpass_order=int(run_cfg.get("lowpass_order", 4)),
        rollout_decimate_step=rollout_step,
        apply_velocity_lowpass_filter=bool(run_cfg.get("velocity_lowpass_filter", True)),
        velocity_lowpass_cutoff_hz=run_cfg.get("velocity_lowpass_cutoff_hz"),
        velocity_lowpass_order=run_cfg.get("velocity_lowpass_order"),
        velocity_lowpass_mode=str(run_cfg.get("input_lowpass_mode", "zero_phase")),
    )


def _make_dataset(
    *,
    train_dir: str,
    files_or_subjects: List[Any],
    is_h5: bool,
    run_cfg: Dict[str, Any],
    stats: Any,
    window_size: int,
    stride: int,
    ra_rd_only: bool,
    loc_map_path: Optional[str],
    pair_kw: Dict[str, Any],
) -> KineticsTCNDataset:
    ds_filter_kw: Dict[str, Any] = dict(
        walking_only=bool(run_cfg.get("walking_only", True)),
        levelground_only=False,
        exclude_stair_tasks=bool(run_cfg.get("exclude_stair_tasks", False)),
        ra_rd_only=bool(ra_rd_only),
    )
    if loc_map_path:
        ds_filter_kw["loc_ascent_descent_map"] = loc_map_path

    common = dict(
        window_size=window_size,
        stride=stride,
        normalize=False,
        stats=stats,
        input_mode=str(run_cfg.get("input_mode", "lower_limb")),
        output_mode=str(run_cfg.get("output_mode", "lower_limb")),
        laterality=str(run_cfg.get("laterality", "unilateral")),
        **_denoise_kwargs_from_run_cfg(run_cfg),
        **ds_filter_kw,
        **pair_kw,
    )
    if is_h5:
        return KineticsTCNDataset(
            data_dir=train_dir,
            h5_dir=train_dir,
            use_h5=True,
            subject_ids=files_or_subjects,
            **common,
        )
    return KineticsTCNDataset(
        data_dir=train_dir,
        b3d_files=files_or_subjects,
        **common,
    )


def _namespace_from_run_cfg(
    run_cfg: Dict[str, Any],
    *,
    finetune_args: argparse.Namespace,
    rollout_decimate_step: int,
) -> SimpleNamespace:
    merged = dict(run_cfg)
    merged.update(
        output_dir=finetune_args.output_dir,
        epochs=finetune_args.epochs,
        batch_size=finetune_args.batch_size,
        lr=finetune_args.lr,
        weight_decay=finetune_args.weight_decay,
        grad_clip=finetune_args.grad_clip,
        lr_scheduler=finetune_args.lr_scheduler,
        early_stopping_patience=finetune_args.early_stopping_patience,
        num_workers=finetune_args.num_workers,
        seed=finetune_args.seed,
        device=finetune_args.device,
        rollout_decimate_step=rollout_decimate_step,
        finetune_from_checkpoint=str(finetune_args.checkpoint),
        finetune_balance_buckets=list(finetune_args.balance_buckets),
        finetune_balance_seed=int(finetune_args.balance_seed),
        finetune_balance_mode=str(finetune_args.balance_mode),
    )
    return SimpleNamespace(**merged)


def _resolve_optional_float(cli_value: Optional[float], run_cfg: Dict[str, Any], key: str) -> float:
    return float(cli_value if cli_value is not None else run_cfg.get(key, 0.0))


def _resolve_optional_bool(cli_value: Optional[bool], run_cfg: Dict[str, Any], key: str) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    return bool(run_cfg.get(key, False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a checkpoint on a balanced LG/RA/RD window subset (trainV2-compatible)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Source checkpoint (.pt); config.json and subject_split.json are read from the same run dir.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="H5 / trial root (default: train_dir from source config.json).",
    )
    parser.add_argument("--output-dir", type=str, default="runs/finetune_v2_balanced")
    parser.add_argument(
        "--balance-buckets",
        type=str,
        default=",".join(DEFAULT_BUCKETS),
        help="Comma-separated locomotion buckets to balance (default: LG,RA,RD).",
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=None,
        help="RNG seed for subsampling (default: --seed).",
    )
    parser.add_argument(
        "--balance-mode",
        type=str,
        default="undersample",
        choices=["undersample", "oversample"],
        help="undersample: cap each bucket to the smallest count; "
             "oversample: repeat windows so each bucket matches the largest count.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=None,
        choices=["cosine", "none"],
        help="LR schedule (default: cosine).",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-ra-rd-only",
        action="store_true",
        default=False,
        help="Validate with ra_rd_only=True (mirror original ramp-only training). "
             "Default: validate on all walking buckets.",
    )
    parser.add_argument(
        "--angle-jitter-std",
        type=float,
        default=None,
        help="Override angle jitter (rad) on position channels during training.",
    )
    parser.add_argument(
        "--correlated-vel-noise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recompute velocity from jittered angles (cascade noise model).",
    )
    parser.add_argument(
        "--angle-input-offset-augment-deg",
        type=float,
        default=None,
        help="Override per-sample angle bias augment (deg).",
    )
    parser.add_argument(
        "--smoothness-lambda",
        type=float,
        default=None,
        help="Override temporal smoothness penalty on predictions.",
    )
    parser.add_argument(
        "--input-noise-std",
        type=float,
        default=None,
        help="Override Gaussian noise std added to all input channels.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="os-kinetics-tcn")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    args.checkpoint = str(_resolve_checkpoint(args.checkpoint))
    args.balance_buckets = _parse_bucket_list(args.balance_buckets)
    args.balance_seed = (
        int(args.balance_seed) if args.balance_seed is not None else int(args.seed)
    )

    run_cfg = load_run_config(args.checkpoint)
    if run_cfg is None:
        raise FileNotFoundError(
            f"No config.json next to checkpoint {args.checkpoint}. "
            "Fine-tune requires the original trainV2 run metadata."
        )
    split = load_subject_split(args.checkpoint)
    if split is None:
        raise FileNotFoundError(
            f"No subject_split.json next to checkpoint {args.checkpoint}."
        )

    train_subjects = sorted(str(s).upper() for s in split.get("train_subjects") or [])
    val_subjects = sorted(str(s).upper() for s in split.get("val_subjects") or [])
    test_subjects = sorted(str(s).upper() for s in split.get("test_subjects") or [])
    if not train_subjects:
        raise ValueError("subject_split.json has no train_subjects.")

    train_dir = args.train_dir or str(run_cfg.get("train_dir", ""))
    if not train_dir:
        raise ValueError("No --train-dir and no train_dir in source config.json.")
    train_root = Path(train_dir).expanduser().resolve()
    if not train_root.is_dir():
        raise FileNotFoundError(f"train-dir is not a directory: {train_root}")

    if args.batch_size is None:
        args.batch_size = int(run_cfg.get("batch_size", 256))
    if args.lr is None:
        args.lr = float(run_cfg.get("lr", 1e-5))
    if args.weight_decay is None:
        args.weight_decay = float(run_cfg.get("weight_decay", 1e-4))
    if args.grad_clip is None:
        args.grad_clip = float(run_cfg.get("grad_clip", 1.0))
    if args.lr_scheduler is None:
        args.lr_scheduler = str(run_cfg.get("lr_scheduler", "cosine"))
    if args.num_workers is None:
        args.num_workers = int(run_cfg.get("num_workers", 4))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    model, stats, _dof_names, window_size, *_rest = load_model(
        args.checkpoint, args.device
    )
    model.train()

    loc_map = _resolve_loc_map(run_cfg)
    loc_map_path = run_cfg.get("loc_ascent_descent_map")

    is_h5, all_subjects, subject_to_trials = _detect_layout(train_root)
    train_files = _files_for_subjects(train_subjects, is_h5=is_h5, subject_to_trials=subject_to_trials)
    val_files = _files_for_subjects(val_subjects, is_h5=is_h5, subject_to_trials=subject_to_trials)
    test_files_info = (
        test_subjects
        if is_h5
        else [str(p) for s in test_subjects for p in (subject_to_trials or {}).get(s, [])]
    )

    pair_kw: Dict[str, Any] = {}
    if bool(run_cfg.get("legacy_unilateral_full_window", False)):
        pair_kw["unilateral_paired_side_windows"] = False

    train_stride = int(run_cfg.get("stride", 1))
    denoise_kw = _denoise_kwargs_from_run_cfg(run_cfg)
    rollout_decimate_step = int(denoise_kw["rollout_decimate_step"])

    print("=" * 70)
    print("FINE-TUNE V2 — balanced loc buckets")
    print("=" * 70)
    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Train dir:         {train_root}")
    print(f"Output dir:        {out_dir.resolve()}")
    print(f"Balance buckets:   {args.balance_buckets}")
    print(f"Balance mode:      {args.balance_mode}")
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects   ({len(val_subjects)}): {val_subjects}")

    print("\n" + "=" * 70)
    print("LOADING FULL TRAIN SPLIT (all loc buckets)")
    print("=" * 70)
    train_ds_full = _make_dataset(
        train_dir=str(train_root),
        files_or_subjects=train_files,
        is_h5=is_h5,
        run_cfg=run_cfg,
        stats=stats,
        window_size=window_size,
        stride=train_stride,
        ra_rd_only=False,
        loc_map_path=str(loc_map_path) if loc_map_path else None,
        pair_kw=pair_kw,
    )

    by_bucket, other_counts = _index_windows_by_bucket(
        train_ds_full,
        target_buckets=args.balance_buckets,
        loc_map=loc_map,
    )
    print("\nWindow counts per bucket (before balance):")
    for b in args.balance_buckets:
        print(f"  {b}: {len(by_bucket[b]):6d}")
    if other_counts:
        print(f"  other buckets (excluded from train): {other_counts}")

    balanced_indices, balance_meta = _balance_window_indices(
        by_bucket, mode=args.balance_mode, seed=args.balance_seed
    )
    print(
        f"\nBalanced {args.balance_mode}: {balance_meta['target_per_bucket']} windows per bucket "
        f"→ {balance_meta['n_train_windows_balanced']} total (seed={args.balance_seed})"
    )
    for b in args.balance_buckets:
        print(f"  {b}: {balance_meta['counts_before'][b]} → {balance_meta['counts_after'][b]}")

    train_ds = Subset(train_ds_full, balanced_indices)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    val_loader = None
    val_ds = None
    if val_files:
        print("\n" + "=" * 70)
        print("LOADING VALIDATION DATA")
        print("=" * 70)
        val_ds = _make_dataset(
            train_dir=str(train_root),
            files_or_subjects=val_files,
            is_h5=is_h5,
            run_cfg=run_cfg,
            stats=stats,
            window_size=window_size,
            stride=1,
            ra_rd_only=bool(args.val_ra_rd_only),
            loc_map_path=str(loc_map_path) if loc_map_path else None,
            pair_kw=pair_kw,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )
        print(f"  Val windows: {len(val_ds)}  (ra_rd_only={args.val_ra_rd_only})")

    args_ns = _namespace_from_run_cfg(
        run_cfg,
        finetune_args=args,
        rollout_decimate_step=rollout_decimate_step,
    )
    _write_run_metadata_json(
        out_dir,
        args_ns,
        subjects=all_subjects,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        train_files=train_files,
        val_files=val_files,
        test_files_info=test_files_info,
        unilateral_paired_side_windows=train_ds_full.unilateral_paired,
    )
    smoothness_lambda = _resolve_optional_float(
        args.smoothness_lambda, run_cfg, "smoothness_lambda"
    )
    angle_jitter_std = _resolve_optional_float(args.angle_jitter_std, run_cfg, "angle_jitter_std")
    correlated_vel_noise = _resolve_optional_bool(
        args.correlated_vel_noise, run_cfg, "correlated_vel_noise"
    )
    angle_offset_deg = _resolve_optional_float(
        args.angle_input_offset_augment_deg, run_cfg, "angle_input_offset_augment_deg"
    )
    input_noise_std = _resolve_optional_float(args.input_noise_std, run_cfg, "input_noise_std")

    finetune_meta = {
        "finetune_from_checkpoint": args.checkpoint,
        "balance_buckets": list(args.balance_buckets),
        "balance_seed": int(args.balance_seed),
        "balance_mode": str(args.balance_mode),
        **balance_meta,
        "val_ra_rd_only": bool(args.val_ra_rd_only),
        "angle_jitter_std": angle_jitter_std,
        "correlated_vel_noise": correlated_vel_noise,
        "angle_input_offset_augment_deg": angle_offset_deg,
        "smoothness_lambda": smoothness_lambda,
        "input_noise_std": input_noise_std,
    }
    cfg_path = out_dir / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_out = json.load(f)
    cfg_out["finetune"] = finetune_meta
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    loss_type = str(run_cfg.get("loss_type", "mse"))
    criterion = MomentLoss(
        loss_type=loss_type,
        huber_delta=float(run_cfg.get("huber_delta", 0.5)),
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

    sample_rate_hz = 200.0 / float(rollout_decimate_step)
    n_position_channels = train_ds_full.n_input_channels // 2

    print(f"\nModel: in={train_ds_full.n_input_channels} out={train_ds_full.n_output_channels}")
    print(f"  DOFs: {train_ds_full.output_dof_names}")
    print(f"  Optimizer: AdamW lr={args.lr}  scheduler={args.lr_scheduler}")
    print(f"  Loss: {loss_type}  epochs={args.epochs}  batch={args.batch_size}")
    print(
        f"  Augment: angle_jitter={angle_jitter_std}  correlated_vel={correlated_vel_noise}  "
        f"angle_offset_deg={angle_offset_deg}  input_noise={input_noise_std}  "
        f"smoothness_lambda={smoothness_lambda}"
    )

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    last_epoch_idx = -1
    is_diffusion = isinstance(model, GaussianDiffusion1D)
    t0 = time.time()

    print(f"\n{'=' * 70}")
    print("TRAINING")
    print(f"{'=' * 70}")

    for epoch in range(args.epochs):
        last_epoch_idx = epoch
        ep_start = time.time()
        if is_diffusion:
            train_loss = train_one_epoch_diffusion(
                model,
                train_loader,
                optimizer,
                args.device,
                grad_clip=args.grad_clip,
                input_noise_std=input_noise_std,
            )
        else:
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                args.device,
                epoch,
                grad_clip=args.grad_clip,
                input_noise_std=input_noise_std,
                angle_jitter_std=angle_jitter_std,
                n_position_channels=n_position_channels,
                angle_input_offset_augment_deg=angle_offset_deg,
                correlated_vel_noise=correlated_vel_noise,
                sample_rate_hz=sample_rate_hz,
                smoothness_lambda=smoothness_lambda,
            )
        train_losses.append(train_loss)
        log_parts = [f"Epoch {epoch + 1:3d}/{args.epochs}  train_loss={train_loss:.6f}"]

        val_loss: Optional[float] = None
        if val_loader is not None:
            mse_crit = nn.MSELoss()
            val_loss, _per_rmse, r2_global, _per_r2 = evaluate(
                model, val_loader, mse_crit, args.device
            )
            val_losses.append(val_loss)
            log_parts.append(f"val_mse={val_loss:.6f}")
            r2_str = f"{r2_global:.4f}" if np.isfinite(r2_global) else "nan"
            log_parts.append(f"val_R2={r2_str}")
            if val_ds is not None:
                per_bucket = _evaluate_val_per_bucket(
                    model,
                    val_ds,
                    args.device,
                    loc_map,
                    args.balance_buckets,
                )
                bucket_bits = []
                for b in args.balance_buckets:
                    if b not in per_bucket:
                        continue
                    m = per_bucket[b]
                    r2b = m["val_r2"]
                    r2b_str = f"{r2b:.3f}" if np.isfinite(r2b) else "nan"
                    bucket_bits.append(f"{b}:mse={m['val_mse']:.4f},R2={r2b_str}")
                if bucket_bits:
                    log_parts.append("val_bucket[" + " | ".join(bucket_bits) + "]")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                _save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    train_loss,
                    val_loss,
                    train_ds_full,
                    args_ns,
                    out_dir / "best_model.pt",
                    list(train_ds_full.output_dof_names),
                )
                log_parts.append("*best*")
            else:
                epochs_no_improve += 1
            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {epochs_no_improve} epochs)."
                )
                if scheduler is not None:
                    scheduler.step()
                break

        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        log_parts.append(f"lr={lr_now:.2e}  time={time.time() - ep_start:.1f}s")
        print("  ".join(log_parts))

        if (epoch + 1) % args.save_freq == 0:
            _save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                train_ds_full,
                args_ns,
                out_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                list(train_ds_full.output_dof_names),
            )

    _save_checkpoint(
        model,
        optimizer,
        max(last_epoch_idx, 0),
        train_losses[-1] if train_losses else float("nan"),
        val_losses[-1] if val_losses else None,
        train_ds_full,
        args_ns,
        out_dir / "final_model.pt",
        list(train_ds_full.output_dof_names),
    )

    plot_curves(train_losses, val_losses, out_dir / "training_curves.png")
    plot_ds = val_ds if val_ds is not None else train_ds_full
    plot_sample_prediction(
        model,
        plot_ds,
        args.device,
        out_dir / "sample_prediction.png",
        list(train_ds_full.output_dof_names),
        sample_rate_hz=sample_rate_hz,
    )

    with open(out_dir / "finetune_balance.json", "w", encoding="utf-8") as f:
        json.dump(finetune_meta, f, indent=2)

    print(f"\n{'=' * 70}")
    print("FINE-TUNE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time: {(time.time() - t0) / 60:.1f} min")
    if val_losses:
        print(f"  Best val MSE: {best_val_loss:.6f}")
    print(f"  Output: {out_dir.resolve()}")
    print(f"  Balance summary: {out_dir / 'finetune_balance.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
