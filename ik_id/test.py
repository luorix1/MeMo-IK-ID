#!/usr/bin/env python3
"""
Test / evaluation script for a trained TCN moment-prediction model (IK+vel → ID).

Run from ``os_kinetics/``::

    python -m ik_id.test ...
    python ik_id/test.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

# Reduces multiprocessing FD use when num_workers>0; safe to call once at import.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from dataset import (
    KineticsTCNDataset,
    DOF_NAMES,
    normalize_laterality,
    # Unused but kept for backward compatibility in older runs/plots.
    find_trial_dirs,
    extract_subject_id,
)
from model import TCN
from training_utils import set_global_seed

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


def resolve_unilateral_paired_for_eval(
    *,
    laterality: str,
    ckpt_flag: Optional[bool],
    n_in_model: int,
    input_indices: Optional[List[int]],
) -> bool:
    """Match ``KineticsTCNDataset`` layout to the saved TCN width (paired ipsilateral vs full R+L window)."""
    if ckpt_flag is not None:
        return bool(ckpt_flag)
    if normalize_laterality(laterality) != "unilateral":
        return False
    if input_indices is None:
        return False
    return int(n_in_model) == len(input_indices)


def load_model(
    ckpt_path: str, device: str = "cpu",
) -> Tuple[Any, Any, Any, int, Any, Any, str, str, Optional[int], str, Optional[bool]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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

    stats          = ckpt.get("normalization", None)
    dof_names      = ckpt.get("dof_names", DOF_NAMES)
    window_size    = ckpt.get("window_size", 200)
    input_indices  = ckpt.get("input_indices", None)
    moment_indices = ckpt.get("moment_indices", None)
    input_mode     = ckpt.get("input_mode", "full")
    output_mode    = ckpt.get("output_mode", "all")
    train_stride   = ckpt.get("stride", None)
    if train_stride is not None:
        train_stride = int(train_stride)
    laterality = str(ckpt.get("laterality", "bilateral"))
    uni_paired = ckpt.get("unilateral_paired_side_windows", None)
    if uni_paired is not None:
        uni_paired = bool(uni_paired)
    return (
        model,
        stats,
        dof_names,
        window_size,
        input_indices,
        moment_indices,
        input_mode,
        output_mode,
        train_stride,
        laterality,
        uni_paired,
    )


def resolve_dataset_stride(
    *,
    stride_from_ckpt: Optional[int],
    run_cfg: Optional[Dict[str, Any]],
    window_size: int,
    override: Optional[int],
) -> int:
    """Training --stride: checkpoint, then config.json, else window_size (legacy eval fallback)."""
    if override is not None:
        return int(override)
    if stride_from_ckpt is not None:
        return int(stride_from_ckpt)
    if run_cfg is not None and run_cfg.get("stride") is not None:
        return int(run_cfg["stride"])
    return int(window_size)


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: Any, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run full inference, return stacked predictions and ground truth.

    Warning: materializes the entire eval set in RAM. For large N (e.g. stride=1 on
    full MeMo), use ``run_inference_streaming`` instead to avoid OOM.
    """
    all_pred, all_true = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        pred = model(x)
        all_pred.append(pred.cpu())
        all_true.append(y)
    return torch.cat(all_pred, 0), torch.cat(all_true, 0)


@torch.no_grad()
def run_inference_streaming(
    model: torch.nn.Module,
    loader: Any,
    device: str,
    dof_names: Any,
    *,
    n_plot_samples: int = 3,
    scatter_max_points: int = 50_000,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One pass over ``loader``: O(1) extra RAM for metrics (no full pred/true tensors).

    Returns:
        metrics dict (same structure as ``compute_metrics``),
        pred_plot, true_plot: (min(n_plot_samples, N), C, W) for time-series plots,
        scatter_gt, scatter_pred: 1D arrays of length <= scatter_max_points for scatter plot.
    """
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

    plot_pred_chunks: List[Any] = []
    plot_true_chunks: List[Any] = []
    n_plot_collected = 0

    scatter_gt_chunks: List[Any] = []
    scatter_pred_chunks: List[Any] = []
    n_scatter = 0

    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        pred = model(x)
        pb = pred.detach().cpu().numpy().astype(np.float64)
        tb = y.numpy().astype(np.float64)
        n_ch = pb.shape[1]

        if sum_sq_ch is None:
            sum_sq_ch = np.zeros(n_ch, dtype=np.float64)
            sum_abs_ch = np.zeros(n_ch, dtype=np.float64)
            sum_t_ch = np.zeros(n_ch, dtype=np.float64)
            sum_t2_ch = np.zeros(n_ch, dtype=np.float64)

        diff = pb - tb
        sum_sq_ch += np.sum(diff ** 2, axis=(0, 2))
        sum_abs_ch += np.sum(np.abs(diff), axis=(0, 2))
        sum_t_ch += np.sum(tb, axis=(0, 2))
        sum_t2_ch += np.sum(tb ** 2, axis=(0, 2))
        n_elem_ch += pb.shape[0] * pb.shape[2]

        sum_sq_all += float(np.sum(diff ** 2))
        sum_abs_all += float(np.sum(np.abs(diff)))
        sum_t_all += float(np.sum(tb))
        sum_t2_all += float(np.sum(tb ** 2))
        n_all += tb.size

        if n_plot_collected < n_plot_samples:
            need = n_plot_samples - n_plot_collected
            take = min(pb.shape[0], need)
            plot_pred_chunks.append(pb[:take].astype(np.float32))
            plot_true_chunks.append(tb[:take].astype(np.float32))
            n_plot_collected += take

        if n_scatter < scatter_max_points:
            need = scatter_max_points - n_scatter
            flat_g = tb.reshape(-1)
            flat_p = pb.reshape(-1)
            take = min(flat_g.size, need)
            scatter_gt_chunks.append(flat_g[:take].astype(np.float32))
            scatter_pred_chunks.append(flat_p[:take].astype(np.float32))
            n_scatter += take

    assert sum_sq_ch is not None and sum_abs_ch is not None
    per_ch: List[Dict[str, Any]] = []
    for c in range(n_ch):
        mse = float(sum_sq_ch[c] / n_elem_ch)
        rmse = float(np.sqrt(mse))
        mae = float(sum_abs_ch[c] / n_elem_ch)
        ss_res = float(sum_sq_ch[c])
        mean_t = sum_t_ch[c] / n_elem_ch
        ss_tot = float(sum_t2_ch[c] - sum_t_ch[c] * mean_t)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        name = dof_names[c] if c < len(dof_names) else f"dof_{c}"
        per_ch.append({"name": name, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2})

    overall_mse = float(sum_sq_all / max(n_all, 1))
    overall_rmse = float(np.sqrt(overall_mse))
    overall_mae = float(sum_abs_all / max(n_all, 1))
    mean_all = sum_t_all / max(n_all, 1)
    ss_tot_all = float(sum_t2_all - sum_t_all * mean_all)
    overall_r2 = float(1.0 - sum_sq_all / (ss_tot_all + 1e-12))

    metrics: Dict[str, Any] = {
        "per_channel": per_ch,
        "overall": {
            "mse": overall_mse,
            "rmse": overall_rmse,
            "mae": overall_mae,
            "r2": overall_r2,
        },
    }

    pred_plot = np.concatenate(plot_pred_chunks, axis=0) if plot_pred_chunks else np.zeros((0, n_ch, 0), dtype=np.float32)
    true_plot = np.concatenate(plot_true_chunks, axis=0) if plot_true_chunks else np.zeros((0, n_ch, 0), dtype=np.float32)
    scatter_gt = np.concatenate(scatter_gt_chunks, axis=0) if scatter_gt_chunks else np.zeros(0, dtype=np.float32)
    scatter_pred = np.concatenate(scatter_pred_chunks, axis=0) if scatter_pred_chunks else np.zeros(0, dtype=np.float32)

    return metrics, pred_plot, true_plot, scatter_gt, scatter_pred


def compute_metrics(pred: np.ndarray, true: np.ndarray, dof_names: Any) -> Dict[str, Any]:
    """
    Compute per-channel and overall metrics.
    pred, true: (N, C, W)
    """
    n_ch = pred.shape[1]
    results = {}

    # Flatten samples and time for overall metrics
    pred_flat = pred.reshape(-1, n_ch)
    true_flat = true.reshape(-1, n_ch)

    per_ch = []
    for c in range(n_ch):
        p, t = pred_flat[:, c], true_flat[:, c]
        mse = float(np.mean((p - t) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(p - t)))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        name = dof_names[c] if c < len(dof_names) else f"dof_{c}"
        per_ch.append({"name": name, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2})

    results["per_channel"] = per_ch

    overall_mse = float(np.mean((pred_flat - true_flat) ** 2))
    results["overall"] = {
        "mse": overall_mse,
        "rmse": float(np.sqrt(overall_mse)),
        "mae": float(np.mean(np.abs(pred_flat - true_flat))),
    }
    return results


def plot_per_channel_rmse(metrics: Dict[str, Any], out_path: Path) -> None:
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
        elif "pelvis" in nl:
            colors.append("#9B59B6")
        else:
            colors.append("#95A5A6")

    ax.bar(range(len(names)), rmses, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (N·m/kg)")
    ax.set_title("Per-DOF RMSE on Test Set")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_channel_r2(metrics: Dict[str, Any], out_path: Path) -> None:
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
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="R²=0.8")
    ax.set_title("Per-DOF R² on Test Set")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_time_series(
    pred: np.ndarray,
    true: np.ndarray,
    dof_names: Any,
    out_path: Path,
    sample_idx: int = 0,
    n_dofs: int = 6,
    *,
    sample_rate_hz: float = 200.0,
) -> None:
    """Plot time-series comparison for a single sample."""
    if not HAS_MPL:
        return
    p = pred[sample_idx]  # (C, W)
    t_arr = true[sample_idx]
    n_plot = min(n_dofs, p.shape[0])
    time_axis = np.arange(p.shape[1]) / float(sample_rate_hz)

    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2.8 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for i in range(n_plot):
        ax = axes[i]
        ax.plot(time_axis, t_arr[i], label="Ground Truth", linewidth=1.5, color="#2E86AB")
        ax.plot(time_axis, p[i], label="Predicted", linewidth=1.5, linestyle="--", color="#A23B72")
        name = dof_names[i] if i < len(dof_names) else f"DOF {i}"
        ax.set_ylabel(f"{name}\n(N·m/kg)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Moment Prediction — Sample {sample_idx}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_gt_vs_pred(
    pred: np.ndarray,
    true: np.ndarray,
    dof_names: Any,
    out_path: Path,
    max_points: int = 50000,
    overall_r2: Optional[float] = None,
) -> None:
    """
    Scatter plot of GT (x-axis) vs predictions (y-axis) across all DOFs and time.
    Includes y=x identity line and overall R².

    If ``overall_r2`` is provided (e.g. from streaming eval on huge sets), it is used for
    the title instead of recomputing from full ``pred``/``true`` (which may be subsampled).
    """
    if not HAS_MPL:
        return

    if pred.ndim == 1 and true.ndim == 1:
        gt_flat = true.astype(np.float64)
        pred_flat = pred.astype(np.float64)
    else:
        gt_flat = true.reshape(-1)
        pred_flat = pred.reshape(-1)

    # Optionally subsample for speed/visual clarity
    if gt_flat.shape[0] > max_points:
        idx = np.random.choice(gt_flat.shape[0], size=max_points, replace=False)
        gt_plot = gt_flat[idx]
        pred_plot = pred_flat[idx]
    else:
        gt_plot = gt_flat
        pred_plot = pred_flat

    if overall_r2 is not None:
        r2 = float(overall_r2)
    else:
        ss_res = np.sum((gt_flat - pred_flat) ** 2)
        ss_tot = np.sum((gt_flat - np.mean(gt_flat)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_plot, pred_plot, s=5, alpha=0.3, color="#2E86AB", edgecolors="none")

    # y = x line
    vmin = min(gt_plot.min(), pred_plot.min())
    vmax = max(gt_plot.max(), pred_plot.max())
    ax.plot([vmin, vmax], [vmin, vmax], color="#A23B72", linestyle="--", linewidth=2, label="y = x")

    ax.set_xlabel("Ground Truth Moment (N·m/kg)")
    ax.set_ylabel("Predicted Moment (N·m/kg)")
    ax.set_title(f"GT vs Predicted Moments (All DOFs)\nR² = {r2:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_subject_split(checkpoint_path: str) -> Optional[dict]:
    """
    Look for subject_split.json next to the checkpoint and return it, or None.
    The split file lives in the same directory as the checkpoint.
    """
    split_file = Path(checkpoint_path).parent / "subject_split.json"
    if split_file.exists():
        with open(split_file) as f:
            return json.load(f)
    return None


def load_run_config(checkpoint_path: str) -> Optional[dict]:
    """
    Load `config.json` from the same run directory as the checkpoint.
    Train writes this at `out_dir/config.json`.
    """
    ckpt_dir = Path(checkpoint_path).resolve().parent
    candidate_paths = [ckpt_dir / "config.json", ckpt_dir.parent / "config.json"]
    for p in candidate_paths:
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    return None


def _extract_flag_value(args_list: List[Any], flag: str) -> Optional[str]:
    for i in range(len(args_list) - 1):
        if args_list[i] == flag:
            return str(args_list[i + 1])
    return None


def _find_matching_local_wandb_run(wandb_root: Path, wanted_output_dir: str) -> Optional[Tuple[str, Path]]:
    """
    Find a local wandb run directory under `run_root/wandb/` whose metadata matches `--output-dir`.

    Returns: (run_id, run_dir_path) or None.
    """
    if not wanted_output_dir:
        return None

    if not wandb_root.exists():
        return None

    for meta_path in wandb_root.glob("run-*/files/wandb-metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        args_list = meta.get("args", [])
        out_dir_val = _extract_flag_value(args_list, "--output-dir")
        if out_dir_val == wanted_output_dir:
            # run dir name is like run-20260320_104717-za3s6vm9
            run_dir = meta_path.parents[1]  # .../run-xxx/files/wandb-metadata.json -> .../run-xxx
            run_id = run_dir.name.split("-")[-1]
            return run_id, run_dir
    return None


def resolve_test_files(test_dir: str, split: dict, split_key: str = "test",
                       max_files: Optional[int] = None) -> Tuple[List[Path], str]:
    """
    Return b3d files from test_dir that belong to the requested split subset.

    split_key:
      "test" — use split["test_subjects"] (the final held-out set)
      "val"  — use split["val_subjects"]  (the early-stopping set)

    If the test_dir has no overlap at all with the split (independent directory),
    all files are returned.
    """
    all_files = find_trial_dirs(test_dir)
    subjects_in_dir = {extract_subject_id(f) for f in all_files}

    train_subjects = set(split.get("train_subjects", []))
    val_subjects   = set(split.get("val_subjects",   []))
    test_subjects  = set(split.get("test_subjects",  []))
    all_split_subjects = train_subjects | val_subjects | test_subjects

    # Determine which subjects to keep
    if split_key == "test":
        keep = test_subjects if test_subjects else val_subjects  # backward compat
        label = "test"
    else:
        keep = val_subjects
        label = "val"

    overlap_with_split = subjects_in_dir & all_split_subjects
    overlap_keep       = subjects_in_dir & keep
    overlap_train      = subjects_in_dir & train_subjects

    if not overlap_with_split:
        # Completely independent directory — no leakage possible, use all
        print(f"  No overlap with training split found in {test_dir}.")
        print(f"  Using all {len(all_files)} trial dir(s).")
        filtered = all_files
        mode = "independent"
    else:
        excluded = subjects_in_dir - keep
        filtered = [f for f in all_files if extract_subject_id(f) in keep]
        mode = label
        if overlap_train:
            print(f"  Train subjects detected in test-dir — restricting to {label} subjects.")
        print(f"  Keeping   ({len(overlap_keep):2d}): {sorted(overlap_keep)}")
        if excluded & all_split_subjects:
            print(f"  Excluding ({len(excluded):2d}): {sorted(excluded & all_split_subjects)}")

    if max_files is not None:
        filtered = filtered[:max_files]

    return filtered, mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained TCN model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Directory with test b3d files")
    parser.add_argument("--output-dir", type=str, default="results/tcn_eval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes. Default 0 (load in main process): recommended for HDF5 "
            "and large window counts (e.g. training stride 1) to avoid 'too many open files'. "
            "Increase only if I/O is the bottleneck and ulimit allows it."
        ),
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--walking-only", action="store_true", default=True)
    parser.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    parser.add_argument(
        "--levelground-only",
        action="store_true",
        help=(
            "Match training: only level-included conditions "
            "(see dataset.is_levelground_subset_condition). "
            "Overridden by levelground_only in config.json next to checkpoint when present."
        ),
    )
    parser.add_argument("--eval-split", type=str, default="test",
                        choices=["test", "val"],
                        help="Which split subset to evaluate: "
                             "'test' (final held-out, default) or 'val' (early-stopping set)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-plot-samples", type=int, default=3,
                        help="Number of sample time-series to plot")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible eval/plots")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Sliding-window stride for evaluation windows. "
            "Default: same as training (--stride in config.json or checkpoint), "
            "else window size (legacy)."
        ),
    )
    parser.add_argument(
        "--target-sample-rate-hz",
        type=float,
        default=None,
        help=(
            "Override trial resampling rate (Hz) for KineticsTCNDataset. "
            "Default: use target_sample_rate_hz from config.json next to checkpoint (training value)."
        ),
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    (
        model,
        stats,
        dof_names,
        window_size,
        input_indices,
        moment_indices,
        input_mode,
        output_mode,
        stride_from_ckpt,
        laterality_ckpt,
        unilateral_paired_ckpt,
    ) = load_model(args.checkpoint, args.device)
    print(f"  Input mode:  {input_mode}")
    print(f"  Output mode: {output_mode}")
    print(f"  Output DOFs: {dof_names}")
    print(f"  Window size: {window_size}")

    run_cfg = load_run_config(args.checkpoint)
    eval_stride = resolve_dataset_stride(
        stride_from_ckpt=stride_from_ckpt,
        run_cfg=run_cfg,
        window_size=window_size,
        override=args.stride,
    )
    print(f"  Dataset stride: {eval_stride}")
    if run_cfg is not None and "laterality" in run_cfg:
        print(f"  Run laterality: {run_cfg.get('laterality')}")

    wandb_run = None
    if HAS_WANDB and run_cfg is not None and run_cfg.get("use_wandb", False):
        # Avoid accidental network use.
        if "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = "offline"

        wandb_project = run_cfg.get("wandb_project", None) or "os-kinetics-tcn"
        wandb_entity = run_cfg.get("wandb_entity", None)

        # config.json uses relative `runs/...` paths; wandb metadata stores the raw value passed.
        wanted_output_dir = run_cfg.get("output_dir", None)
        wandb_root = Path(__file__).resolve().parent / "wandb"
        matching = _find_matching_local_wandb_run(wandb_root, str(wanted_output_dir))
        if matching is not None:
            run_id, _run_dir = matching
            try:
                wandb_run = wandb.init(project=wandb_project, entity=wandb_entity, id=run_id, resume="allow")
            except Exception as e:
                print(f"  [wandb] Failed to resume run id={run_id}: {e}")
                wandb_run = None
        else:
            print("  [wandb] Could not find matching local wandb run by --output-dir; skipping wandb plots.")

    # Adjust stats arrays to numpy
    if stats is not None:
        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                stats[k] = v.numpy()

    # ---- Resolve which files to evaluate on ----
    print(f"\n{'='*70}")
    print("RESOLVING TEST FILES")
    print(f"{'='*70}")
    test_root = Path(args.test_dir)
    h5_subject_files = sorted([p for p in test_root.glob("S*.h5") if p.is_file()])
    is_h5_only_layout = len(h5_subject_files) > 0
    eval_subject_ids: Optional[List[str]] = None

    split = load_subject_split(args.checkpoint)
    if split is not None:
        print(f"  Found subject_split.json next to checkpoint.")
        print(f"  Train subjects: {split.get('train_subjects')}")
        print(f"  Val subjects:   {split.get('val_subjects')}")
        print(f"  Test subjects:  {split.get('test_subjects', '(not recorded)')}")
        print(f"  Evaluating:     --eval-split={args.eval_split}")
        if is_h5_only_layout:
            subjects_in_dir = sorted([p.stem.upper() for p in h5_subject_files])
            train_subjects = set(split.get("train_subjects", []))
            val_subjects = set(split.get("val_subjects", []))
            test_subjects = set(split.get("test_subjects", []))
            all_split_subjects = train_subjects | val_subjects | test_subjects

            if args.eval_split == "test":
                keep = test_subjects if test_subjects else val_subjects
                label = "test"
            else:
                keep = val_subjects
                label = "val"

            overlap_with_split = set(subjects_in_dir) & all_split_subjects
            if not overlap_with_split:
                print(f"  No overlap with training split found in {args.test_dir}.")
                print(f"  Using all {len(subjects_in_dir)} subject h5 file(s).")
                eval_subject_ids = subjects_in_dir
                mode = "independent"
            else:
                eval_subject_ids = sorted(list(set(subjects_in_dir) & keep))
                mode = label
                print(f"  Keeping subjects ({len(eval_subject_ids):2d}): {eval_subject_ids}")

            if args.max_files is not None:
                eval_subject_ids = eval_subject_ids[:args.max_files]
            test_files = []
        else:
            test_files, mode = resolve_test_files(
                args.test_dir, split,
                split_key=args.eval_split,
                max_files=args.max_files,
            )
    else:
        print(f"  No subject_split.json found next to checkpoint.")
        print(f"  Using all files in {args.test_dir} (no leakage protection).")
        if is_h5_only_layout:
            eval_subject_ids = sorted([p.stem.upper() for p in h5_subject_files])
            if args.max_files is not None:
                eval_subject_ids = eval_subject_ids[:args.max_files]
            test_files = []
        else:
            test_files = find_trial_dirs(args.test_dir)
            if args.max_files is not None:
                test_files = test_files[:args.max_files]
        mode = "independent"

    if is_h5_only_layout:
        subjects_used = sorted(eval_subject_ids or [])
        print(f"\n  Evaluating on {len(subjects_used)} subject h5 file(s): {subjects_used}")
    else:
        subjects_used = sorted({extract_subject_id(f) for f in test_files})
        print(f"\n  Evaluating on {len(test_files)} file(s) from subjects: {subjects_used}")

    if is_h5_only_layout and len(subjects_used) == 0:
        raise ValueError("No test subjects found after applying subject split filter.")
    if (not is_h5_only_layout) and len(test_files) == 0:
        raise ValueError("No test files found after applying subject split filter.")

    # Load test data
    print(f"\nLoading test data...")
    if is_h5_only_layout:
        test_ds_kwargs = dict(
            data_dir=args.test_dir,
            h5_dir=args.test_dir,
            use_h5=True,
            subject_ids=eval_subject_ids,
            window_size=window_size,
            stride=eval_stride,
            walking_only=args.walking_only,
            normalize=False,
            stats=stats,
        )
    else:
        test_ds_kwargs = dict(
            data_dir=args.test_dir,
            b3d_files=test_files,
            window_size=window_size,
            stride=eval_stride,
            walking_only=args.walking_only,
            normalize=False,
            stats=stats,
        )

    _levelground_only = args.levelground_only
    if run_cfg is not None and "levelground_only" in run_cfg:
        _levelground_only = bool(run_cfg["levelground_only"])
    test_ds_kwargs["levelground_only"] = _levelground_only

    # Match training: input/output modes and laterality from config.json when available.
    if run_cfg is not None:
        _lat = str(run_cfg.get("laterality", laterality_ckpt))
        test_ds_kwargs.update(
            input_mode=run_cfg.get("input_mode", input_mode),
            output_mode=run_cfg.get("output_mode", output_mode),
            laterality=_lat,
        )
        _cfg_paired = run_cfg.get("unilateral_paired_side_windows", None)
        if _cfg_paired is not None:
            _cfg_paired = bool(_cfg_paired)
        _paired_flag = _cfg_paired if _cfg_paired is not None else unilateral_paired_ckpt
        test_ds_kwargs["unilateral_paired_side_windows"] = resolve_unilateral_paired_for_eval(
            laterality=_lat,
            ckpt_flag=_paired_flag,
            n_in_model=model.n_input_channels,
            input_indices=input_indices,
        )
    else:
        test_ds_kwargs.update(
            input_indices=input_indices,
            moment_indices=moment_indices,
            laterality=laterality_ckpt,
            unilateral_paired_side_windows=resolve_unilateral_paired_for_eval(
                laterality=laterality_ckpt,
                ckpt_flag=unilateral_paired_ckpt,
                n_in_model=model.n_input_channels,
                input_indices=input_indices,
            ),
        )

    # Match training-time denoising when saved in config.json (ik_id.train writes vars(args)).
    if run_cfg is not None and any(
        k in run_cfg
        for k in (
            "no_lowpass",
            "lowpass_cutoff_hz",
            "lowpass_order",
            "median_kernel_samples",
        )
    ):
        test_ds_kwargs.update(
            apply_lowpass_filter=not bool(run_cfg.get("no_lowpass", False)),
            lowpass_cutoff_hz=float(run_cfg.get("lowpass_cutoff_hz", 4.0)),
            lowpass_order=int(run_cfg.get("lowpass_order", 4)),
            median_kernel_samples=int(run_cfg.get("median_kernel_samples", 0)),
        )
        print(
            f"  Dataset denoise (from config.json): zero-phase LPF={test_ds_kwargs['apply_lowpass_filter']} "
            f"({test_ds_kwargs['lowpass_cutoff_hz']} Hz), "
            f"median_k={test_ds_kwargs['median_kernel_samples']}"
        )

    eval_target_sr: Optional[float] = None
    if run_cfg is not None and run_cfg.get("target_sample_rate_hz") is not None:
        eval_target_sr = float(run_cfg["target_sample_rate_hz"])
    if args.target_sample_rate_hz is not None:
        eval_target_sr = float(args.target_sample_rate_hz)
    if eval_target_sr is not None:
        test_ds_kwargs["target_sample_rate_hz"] = eval_target_sr
        print(f"  Dataset resampling: target_sample_rate_hz={eval_target_sr}")

    test_ds = KineticsTCNDataset(**test_ds_kwargs)
    print(f"  unilateral_paired_side_windows: {test_ds.unilateral_paired}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # Inference (streaming: train only keeps one batch; test used to concat all windows → OOM on huge N)
    print("Running inference (streaming metrics; full pred tensor not stored)...")
    metrics, pred_plot, true_plot, scatter_gt, scatter_pred = run_inference_streaming(
        model,
        test_loader,
        args.device,
        dof_names,
        n_plot_samples=args.n_plot_samples,
        scatter_max_points=50_000,
    )
    print(f"  Windows evaluated: {len(test_ds):,}  |  plot / scatter use small subsamples only")

    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
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

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "eval_subjects.json", "w") as f:
        json.dump({
            "test_dir": args.test_dir,
            "eval_split": args.eval_split,
            "mode": mode,
            "subjects_evaluated": subjects_used,
            "n_files": len(test_files),
            "train_subjects": split.get("train_subjects") if split else None,
            "val_subjects":   split.get("val_subjects")   if split else None,
            "test_subjects":  split.get("test_subjects")  if split else None,
        }, f, indent=2)

    # Plots
    plot_per_channel_rmse(metrics, out_dir / "per_dof_rmse.png")
    plot_per_channel_r2(metrics, out_dir / "per_dof_r2.png")
    plot_scatter_gt_vs_pred(
        scatter_pred,
        scatter_gt,
        dof_names,
        out_dir / "scatter_gt_vs_pred.png",
        overall_r2=metrics["overall"].get("r2"),
    )

    plot_sample_hz = float(eval_target_sr) if eval_target_sr is not None else 200.0
    for s in range(min(args.n_plot_samples, len(pred_plot))):
        plot_time_series(
            pred_plot,
            true_plot,
            dof_names,
            out_dir / f"timeseries_sample_{s}.png",
            sample_idx=s,
            sample_rate_hz=plot_sample_hz,
        )

    # ---- W&B: log plots to the training run ----
    if wandb_run is not None:
        try:
            to_log: Dict[str, Any] = {}
            p_rmse = out_dir / "per_dof_rmse.png"
            p_r2 = out_dir / "per_dof_r2.png"
            p_scatter = out_dir / "scatter_gt_vs_pred.png"
            if p_rmse.exists():
                to_log["plots/per_dof_rmse"] = wandb.Image(str(p_rmse))
            if p_r2.exists():
                to_log["plots/per_dof_r2"] = wandb.Image(str(p_r2))
            if p_scatter.exists():
                to_log["plots/scatter_gt_vs_pred"] = wandb.Image(str(p_scatter))
            if to_log:
                wandb.log(to_log)
            for s in range(min(args.n_plot_samples, len(pred_plot))):
                img_path = out_dir / f"timeseries_sample_{s}.png"
                if img_path.exists():
                    wandb.log({f"plots/timeseries_sample_{s}": wandb.Image(str(img_path))})
        except Exception as e:
            print(f"  [wandb] Failed to log images: {e}")
        finally:
            wandb.finish()

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
