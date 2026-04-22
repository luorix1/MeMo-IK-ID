#!/usr/bin/env python3
"""
Shared evaluation logic for IMU → sagittal angle / moment checkpoints (see ``imu_sagittal/test_imu_sagittal_*.py``).

Mirrors ``ik_id.test`` for ``ik_id.train``: load checkpoint, build H5 test dataset with training normalization,
run streaming inference, save metrics.json / eval_subjects.json and plots. When ``use_wandb`` is set in the run
``config.json`` next to the checkpoint, resumes the local training run under ``<repo>/wandb/`` (same layout as
training invoked from ``os_kinetics/``) and uploads eval plots.

Unilateral IMU pipeline: **24** channels (pelvis + one-side thigh/shank/foot); **3** sagittal targets for that leg;
same IK flip as ``ik_id.train`` ``--laterality unilateral``. Checkpoints store paired R/L IMU column layouts; training
mixes right- and left-chain windows when ``sides="both"``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from dataset import (
    IK_DOF_NAMES,
    MOMENT_NAMES,
    SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES,
    SAGITTAL_INPUT_INDICES,
)
from ik_id.test import (
    _find_matching_local_wandb_run,
    load_run_config,
    load_subject_split,
    resolve_dataset_stride,
    run_inference_streaming,
)
from imu_sagittal.imu_sagittal_dataset import ImuSagittalH5Dataset
from model import TCN

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _imu_schema_pairs_from_ckpt_dict(ckpt: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], bool]:
    """Returns (right_schema, left_schema, legacy_single_schema_duplicated)."""
    if "imu_schema_right" in ckpt and "imu_schema_left" in ckpt:
        sr = [(str(d["segment"]), str(d["column"])) for d in ckpt["imu_schema_right"]]
        sl = [(str(d["segment"]), str(d["column"])) for d in ckpt["imu_schema_left"]]
        return sr, sl, False
    raw = ckpt["imu_schema"]
    sr = [(str(d["segment"]), str(d["column"])) for d in raw]
    return sr, list(sr), True


def _output_names_paired_from_ckpt(ckpt: Dict[str, Any], target: str) -> Tuple[List[str], List[str]]:
    onr = ckpt.get("output_names_right")
    if onr is None:
        onr = ckpt.get("output_names")
    if onr is None:
        raise ValueError("Checkpoint missing output_names / output_names_right.")
    onr = [str(x) for x in onr]
    onl = ckpt.get("output_names_left")
    if onl is None:
        if target == "angle":
            onl = [IK_DOF_NAMES[i] for i in SAGITTAL_INPUT_INDICES[3:]]
        else:
            onl = [MOMENT_NAMES[i] for i in SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES[3:]]
    else:
        onl = [str(x) for x in onl]
    return onr, onl


def load_imu_checkpoint(
    ckpt_path: str,
    device: str,
) -> Tuple[
    torch.nn.Module,
    Dict[str, Any],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    str,
    List[str],
    List[str],
    int,
    int,
    Dict[str, np.ndarray],
]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["model_config"]
    model = TCN(
        n_input_channels=cfg["n_input_channels"],
        n_output_channels=cfg["n_output_channels"],
        hidden_channels=cfg["hidden_channels"],
        n_blocks=cfg["n_blocks"],
        kernel_size=cfg["kernel_size"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    stats = ckpt.get("normalization")
    if stats is not None:
        stats = dict(stats)
        for k, v in list(stats.items()):
            if isinstance(v, torch.Tensor):
                stats[k] = v.detach().cpu().numpy()

    imu_schema_right, imu_schema_left, legacy_dup = _imu_schema_pairs_from_ckpt_dict(ckpt)
    if legacy_dup:
        print(
            "  [warn] Checkpoint has only legacy ``imu_schema``; using it for both R/L chains. "
            "Retrain with current ``imu_sagittal/train_imu_sagittal.py`` for correct paired layouts."
        )
    target = str(ckpt["target"])
    output_names_right, output_names_left = _output_names_paired_from_ckpt(ckpt, target)
    window_size = int(ckpt.get("window_size", 200))
    stride = int(ckpt.get("stride", 1))
    if stats is None or "imu_mean" not in stats or "imu_std" not in stats:
        raise ValueError("Checkpoint missing IMU normalization (imu_mean / imu_std).")
    return (
        model,
        ckpt,
        imu_schema_right,
        imu_schema_left,
        target,
        output_names_right,
        output_names_left,
        window_size,
        stride,
        stats,
    )


def plot_imu_per_channel_rmse(metrics: Dict[str, Any], out_path: Path, y_label: str) -> None:
    if not HAS_MPL:
        return
    names = [m["name"] for m in metrics["per_channel"]]
    rmses = [m["rmse"] for m in metrics["per_channel"]]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 5))
    colors = []
    for n in names:
        nl = n.lower()
        if "hip" in nl:
            colors.append("#E74C3C")
        elif "knee" in nl:
            colors.append("#3498DB")
        elif "ankle" in nl:
            colors.append("#2ECC71")
        else:
            colors.append("#95A5A6")
    ax.bar(range(len(names)), rmses, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(y_label)
    ax.set_title("Per-DOF RMSE (test set)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_imu_per_channel_r2(metrics: Dict[str, Any], out_path: Path) -> None:
    if not HAS_MPL:
        return
    names = [m["name"] for m in metrics["per_channel"]]
    r2s = [m["r2"] for m in metrics["per_channel"]]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 5))
    colors = ["#2ECC71" if r > 0.8 else "#F39C12" if r > 0.5 else "#E74C3C" for r in r2s]
    ax.bar(range(len(names)), r2s, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Per-DOF R² (test set)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_imu_time_series(
    pred: np.ndarray,
    true: np.ndarray,
    dof_names: List[str],
    out_path: Path,
    *,
    sample_idx: int,
    y_label: str,
    title: str,
    n_dofs: Optional[int] = None,
    sample_rate_hz: float = 200.0,
) -> None:
    if not HAS_MPL:
        return
    p = pred[sample_idx]
    t_arr = true[sample_idx]
    n_lim = n_dofs if n_dofs is not None else p.shape[0]
    n_plot = min(n_lim, p.shape[0])
    time_axis = np.arange(p.shape[1]) / sample_rate_hz
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2.8 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for i in range(n_plot):
        ax = axes[i]
        ax.plot(time_axis, t_arr[i], label="Ground truth", linewidth=1.5, color="#2E86AB")
        ax.plot(time_axis, p[i], label="Predicted", linewidth=1.5, linestyle="--", color="#A23B72")
        name = dof_names[i] if i < len(dof_names) else f"ch_{i}"
        ax.set_ylabel(f"{name}\n({y_label})", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{title} — sample {sample_idx}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_imu_scatter(
    scatter_pred: np.ndarray,
    scatter_gt: np.ndarray,
    out_path: Path,
    *,
    x_label: str,
    y_label: str,
    title: str,
    overall_r2: Optional[float] = None,
    max_points: int = 50_000,
) -> None:
    if not HAS_MPL:
        return
    gt_flat = scatter_gt.astype(np.float64)
    pred_flat = scatter_pred.astype(np.float64)
    if overall_r2 is not None:
        r2 = float(overall_r2)
    else:
        ss_res = np.sum((gt_flat - pred_flat) ** 2)
        ss_tot = np.sum((gt_flat - np.mean(gt_flat)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    if gt_flat.shape[0] > max_points:
        idx = np.random.choice(gt_flat.shape[0], size=max_points, replace=False)
        gt_plot = gt_flat[idx]
        pred_plot = pred_flat[idx]
    else:
        gt_plot = gt_flat
        pred_plot = pred_flat
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_plot, pred_plot, s=5, alpha=0.3, color="#2E86AB", edgecolors="none")
    vmin = min(gt_plot.min(), pred_plot.min())
    vmax = max(gt_plot.max(), pred_plot.max())
    ax.plot([vmin, vmax], [vmin, vmax], color="#A23B72", linestyle="--", linewidth=2, label="y = x")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}\nR² = {r2:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_main(expected_target: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Evaluate IMU → sagittal {expected_target} checkpoint (H5 test directory)"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt / final_model.pt")
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Folder with S###.h5 (same layout as training)",
    )
    parser.add_argument(
        "--meta-root",
        type=str,
        default=None,
        help="Directory with dataset_metadata.json (default: --test-dir)",
    )
    parser.add_argument("--output-dir", type=str, default="results/imu_sagittal_eval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 recommended for HDF5 + many windows).",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Max subject H5 files after split filter")
    parser.add_argument("--walking-only", action="store_true", default=True)
    parser.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    parser.add_argument("--levelground-only", action="store_true", default=False)
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Use subject_split.json test or val cohort next to checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-plot-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride (default: from checkpoint / config.json, else window size)",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        default=False,
        help="Keep every 2nd sample after alignment (~200 Hz -> ~100 Hz). "
             "Overrides config/checkpoint rate settings.",
    )
    parser.add_argument(
        "--eval-side",
        type=str,
        default="right",
        choices=["right", "left", "both"],
        help="Which IMU chain and joint labels to score (right, left, or both separately).",
    )
    args = parser.parse_args()

    if expected_target not in ("angle", "moment"):
        raise ValueError("expected_target must be 'angle' or 'moment'")

    set_global_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_root = args.meta_root or args.test_dir

    print(f"Loading checkpoint: {args.checkpoint}")
    (
        model,
        _ckpt,
        imu_schema_right,
        imu_schema_left,
        target,
        output_names_right,
        output_names_left,
        window_size,
        stride_ckpt,
        stats,
    ) = load_imu_checkpoint(args.checkpoint, args.device)
    if target != expected_target:
        raise ValueError(
            f"This script evaluates {expected_target} models, but checkpoint target is {target!r}."
        )
    print(
        f"  target={target}  window_size={window_size}  imu_ch/R={len(imu_schema_right)} "
        f"imu_ch/L={len(imu_schema_left)}"
    )
    print(f"  outputs (R): {output_names_right}")
    print(f"  outputs (L): {output_names_left}")

    run_cfg = load_run_config(args.checkpoint)

    wandb_run = None
    if HAS_WANDB and run_cfg is not None and run_cfg.get("use_wandb", False):
        if "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = "offline"

        wandb_project = run_cfg.get("wandb_project", None) or "os-kinetics-imu-sagittal"
        wandb_entity = run_cfg.get("wandb_entity", None)

        wanted_output_dir = run_cfg.get("output_dir", None)
        wandb_root = _ROOT / "wandb"
        matching = _find_matching_local_wandb_run(wandb_root, str(wanted_output_dir))
        if matching is not None:
            run_id, _run_dir = matching
            try:
                wandb_run = wandb.init(
                    project=wandb_project, entity=wandb_entity, id=run_id, resume="allow"
                )
            except Exception as e:
                print(f"  [wandb] Failed to resume run id={run_id}: {e}")
                wandb_run = None
        else:
            print(
                "  [wandb] Could not find matching local wandb run by training --output-dir; "
                "skipping wandb plot upload (runs live under <repo>/wandb/ when training from os_kinetics/)."
            )

    eval_stride = resolve_dataset_stride(
        stride_from_ckpt=stride_ckpt,
        run_cfg=run_cfg,
        window_size=window_size,
        override=args.stride,
    )
    print(f"  eval stride: {eval_stride}")

    rollout_step = 1
    if args.rollout:
        rollout_step = 2
    elif run_cfg is not None:
        rollout_step = int(run_cfg.get("rollout_decimate_step", 1))
    elif _ckpt.get("rollout_decimate_step") is not None:
        rollout_step = int(_ckpt["rollout_decimate_step"])
    rollout_step = max(1, rollout_step)

    if rollout_step > 1:
        plot_sr = 200.0 / float(rollout_step)
        print(f"  rollout decimation: stride={rollout_step} (~{plot_sr:.0f} Hz)")
    else:
        plot_sr = 200.0

    _lat_cfg = str(run_cfg.get("laterality")) if run_cfg and run_cfg.get("laterality") else "unilateral"
    if _lat_cfg != "unilateral":
        print(f"  [warn] config/checkpoint laterality={_lat_cfg!r}; eval assumes unilateral IMU training only.")

    apply_lowpass_filter = True
    lowpass_cutoff_hz = 4.0
    lowpass_order = 4
    if run_cfg is not None:
        if "lowpass_cutoff_hz" in run_cfg:
            lowpass_cutoff_hz = float(run_cfg["lowpass_cutoff_hz"])
        if "lowpass_order" in run_cfg:
            lowpass_order = int(run_cfg["lowpass_order"])
        if "lowpass_cutoff_hz" in run_cfg or "lowpass_order" in run_cfg:
            print(
                f"  denoise (from config): zero-phase LPF on "
                f"({lowpass_cutoff_hz} Hz, order {lowpass_order})"
            )

    _levelground_only = args.levelground_only
    if run_cfg is not None and "levelground_only" in run_cfg:
        _levelground_only = bool(run_cfg["levelground_only"])

    _walking_only = args.walking_only
    if run_cfg is not None and "walking_only" in run_cfg:
        _walking_only = bool(run_cfg["walking_only"])

    print(f"\n{'='*70}\nRESOLVING TEST SUBJECTS\n{'='*70}")
    test_root = Path(args.test_dir)
    h5_subject_files = sorted([p for p in test_root.glob("S*.h5") if p.is_file()])
    if not h5_subject_files:
        raise ValueError(f"No S*.h5 files under {args.test_dir}")
    split = load_subject_split(args.checkpoint)
    eval_subject_ids: Optional[List[str]] = None
    mode = "independent"

    if split is not None:
        print(f"  Found subject_split.json next to checkpoint.")
        print(f"  Train: {split.get('train_subjects')}")
        print(f"  Val:   {split.get('val_subjects')}")
        print(f"  Test:  {split.get('test_subjects')}")
        subjects_in_dir = sorted([p.stem.upper() for p in h5_subject_files])
        train_subjects = set(split.get("train_subjects", []))
        val_subjects = set(split.get("val_subjects", []))
        test_subjects = set(split.get("test_subjects", []))
        all_split = train_subjects | val_subjects | test_subjects

        if args.eval_split == "test":
            keep = test_subjects if test_subjects else val_subjects
            label = "test"
        else:
            keep = val_subjects
            label = "val"

        overlap = set(subjects_in_dir) & all_split
        if not overlap:
            print(f"  No overlap with split in test-dir — using all {len(subjects_in_dir)} H5 subject(s).")
            eval_subject_ids = subjects_in_dir
            mode = "independent"
        else:
            eval_subject_ids = sorted(set(subjects_in_dir) & keep)
            mode = label
            print(f"  Keeping ({len(eval_subject_ids)}): {eval_subject_ids}")
    else:
        print("  No subject_split.json — using all H5 subjects under test-dir.")
        eval_subject_ids = sorted([p.stem.upper() for p in h5_subject_files])
        mode = "independent"

    if args.max_files is not None and eval_subject_ids is not None:
        eval_subject_ids = eval_subject_ids[: args.max_files]

    if not eval_subject_ids:
        raise ValueError("No subjects left for evaluation after filtering.")

    if target == "angle":
        rmse_ylabel = "RMSE (rad)"
        ts_ylabel = "rad"
        scatter_x = "Ground truth angle (rad)"
        scatter_y = "Predicted angle (rad)"
        scatter_title = "GT vs predicted angles (all DOFs × time)"
        ts_title = "Sagittal angle prediction"
    else:
        rmse_ylabel = "RMSE (N·m/kg)"
        ts_ylabel = "N·m/kg"
        scatter_x = "Ground truth moment (N·m/kg)"
        scatter_y = "Predicted moment (N·m/kg)"
        scatter_title = "GT vs predicted moments (all DOFs × time)"
        ts_title = "Sagittal moment prediction"

    side_runs: List[Tuple[str, str]]
    if args.eval_side == "both":
        side_runs = [("right", "_right"), ("left", "_left")]
    else:
        side_runs = [(args.eval_side, "")]

    wandb_plot_paths: Dict[str, str] = {}

    for side_kw, file_tag in side_runs:
        output_names = output_names_right if side_kw == "right" else output_names_left
        print(f"\nLoading test dataset ({len(eval_subject_ids)} subject(s), side={side_kw})…")
        test_ds = ImuSagittalH5Dataset(
            h5_dir=str(test_root),
            meta_root_dir=meta_root,
            subject_ids=eval_subject_ids,
            imu_schema_right=imu_schema_right,
            imu_schema_left=imu_schema_left,
            sides=cast(Literal["right", "left"], side_kw),
            target=target,
            window_size=window_size,
            stride=eval_stride,
            walking_only=_walking_only,
            levelground_only=_levelground_only,
            normalize=True,
            stats=stats,
            apply_lowpass_filter=apply_lowpass_filter,
            lowpass_cutoff_hz=lowpass_cutoff_hz,
            lowpass_order=lowpass_order,
            target_sample_rate_hz=None,
            rollout_decimate_step=rollout_step,
            preload_trials=False,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )

        print("Running inference (streaming)…")
        metrics, pred_plot, true_plot, scatter_gt, scatter_pred = run_inference_streaming(
            model,
            test_loader,
            args.device,
            output_names,
            n_plot_samples=args.n_plot_samples,
            scatter_max_points=50_000,
        )
        print(f"  Windows: {len(test_ds):,}")

        label_side = f" ({side_kw})" if args.eval_side == "both" else ""
        print(f"\n{'='*70}\nRESULTS ({target}){label_side}\n{'='*70}")
        print(f"  Overall MSE:  {metrics['overall']['mse']:.6f}")
        print(f"  Overall RMSE: {metrics['overall']['rmse']:.6f}")
        print(f"  Overall MAE:  {metrics['overall']['mae']:.6f}")
        if "r2" in metrics["overall"]:
            print(f"  Overall R²:   {metrics['overall']['r2']:.6f}")
        print(f"\n  {'DOF':<25s}  {'RMSE':>8s}  {'MAE':>8s}  {'R²':>8s}")
        print(f"  {'-'*55}")
        for ch in metrics["per_channel"]:
            print(f"  {ch['name']:<25s}  {ch['rmse']:8.4f}  {ch['mae']:8.4f}  {ch['r2']:8.4f}")
        print(f"{'='*70}")

        metrics_name = f"metrics{file_tag}.json" if file_tag else "metrics.json"
        with open(out_dir / metrics_name, "w") as f:
            json.dump(metrics, f, indent=2)

        plot_imu_per_channel_rmse(metrics, out_dir / f"per_dof_rmse{file_tag}.png", rmse_ylabel)
        plot_imu_per_channel_r2(metrics, out_dir / f"per_dof_r2{file_tag}.png")
        plot_imu_scatter(
            scatter_pred,
            scatter_gt,
            out_dir / f"scatter_gt_vs_pred{file_tag}.png",
            x_label=scatter_x,
            y_label=scatter_y,
            title=scatter_title,
            overall_r2=metrics["overall"].get("r2"),
        )
        for s in range(min(args.n_plot_samples, len(pred_plot))):
            plot_imu_time_series(
                pred_plot,
                true_plot,
                output_names,
                out_dir / f"timeseries_sample_{s}{file_tag}.png",
                sample_idx=s,
                y_label=ts_ylabel,
                title=ts_title,
                sample_rate_hz=plot_sr,
            )

        prefix = f"{side_kw}/" if file_tag else ""
        wandb_plot_paths[f"{prefix}per_dof_rmse"] = str(out_dir / f"per_dof_rmse{file_tag}.png")
        wandb_plot_paths[f"{prefix}per_dof_r2"] = str(out_dir / f"per_dof_r2{file_tag}.png")
        wandb_plot_paths[f"{prefix}scatter_gt_vs_pred"] = str(out_dir / f"scatter_gt_vs_pred{file_tag}.png")

    with open(out_dir / "eval_subjects.json", "w") as f:
        json.dump(
            {
                "test_dir": args.test_dir,
                "eval_split": args.eval_split,
                "eval_side": args.eval_side,
                "mode": mode,
                "subjects_evaluated": eval_subject_ids,
                "target": target,
                "checkpoint": str(Path(args.checkpoint).resolve()),
            },
            f,
            indent=2,
        )

    if wandb_run is not None:
        try:
            to_log: Dict[str, Any] = {}
            for key, path_str in wandb_plot_paths.items():
                p = Path(path_str)
                if p.exists():
                    to_log[f"plots/{key}"] = wandb.Image(str(p))
            if to_log:
                wandb.log(to_log)
            for side_kw, file_tag in side_runs:
                prefix = f"{side_kw}/" if file_tag else ""
                for s in range(args.n_plot_samples):
                    img_path = out_dir / f"timeseries_sample_{s}{file_tag}.png"
                    if img_path.exists():
                        wandb.log(
                            {f"plots/{prefix}timeseries_sample_{s}": wandb.Image(str(img_path))}
                        )
        except Exception as e:
            print(f"  [wandb] Failed to log images: {e}")
        finally:
            wandb.finish()

    print(f"\nResults saved to {out_dir}")
