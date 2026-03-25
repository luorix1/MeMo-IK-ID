#!/usr/bin/env python3
"""
Test / evaluation script for a trained TCN moment-prediction model.

Usage:
    python test.py \
        --checkpoint runs/tcn_run1/best_model.pt \
        --test-dir /media/metamobility3/T7_Shield/test/No_Arm/Camargo2021_Formatted_No_Arm \
        --output-dir results/tcn_eval
"""

import argparse
import json
import random
import os
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    KineticsTCNDataset,
    DOF_NAMES,
    # Unused but kept for backward compatibility in older runs/plots.
    find_trial_dirs,
    extract_subject_id,
)
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
    """Set seeds for Python, NumPy, and Torch (CPU & CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(ckpt_path: str, device: str = "cpu") -> Tuple[Any, Any, Any, int, Any, Any, str, str]:
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
    return model, stats, dof_names, window_size, input_indices, moment_indices, input_mode, output_mode


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: Any, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run full inference, return stacked predictions and ground truth."""
    all_pred, all_true = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        all_pred.append(pred.cpu())
        all_true.append(y)
    return torch.cat(all_pred, 0), torch.cat(all_true, 0)


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
) -> None:
    """Plot time-series comparison for a single sample."""
    if not HAS_MPL:
        return
    p = pred[sample_idx]  # (C, W)
    t_arr = true[sample_idx]
    n_plot = min(n_dofs, p.shape[0])
    time_axis = np.arange(p.shape[1]) / 200.0

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
) -> None:
    """
    Scatter plot of GT (x-axis) vs predictions (y-axis) across all DOFs and time.
    Includes y=x identity line and overall R².
    """
    if not HAS_MPL:
        return

    # Flatten over batch and time: (N, C, W) -> (N*C*W,)
    n_samples, n_ch, n_t = pred.shape
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

    # Compute overall R²
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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--walking-only", action="store_true", default=True)
    parser.add_argument("--no-walking-only", dest="walking_only", action="store_false")
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
    args = parser.parse_args()

    set_global_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, stats, dof_names, window_size, input_indices, moment_indices, input_mode, output_mode = \
        load_model(args.checkpoint, args.device)
    print(f"  Input mode:  {input_mode}")
    print(f"  Output mode: {output_mode}")
    print(f"  Output DOFs: {dof_names}")
    print(f"  Window size: {window_size}")

    run_cfg = load_run_config(args.checkpoint)
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
            stride=window_size,  # non-overlapping
            walking_only=args.walking_only,
            normalize=False,
            stats=stats,
        )
    else:
        test_ds_kwargs = dict(
            data_dir=args.test_dir,
            b3d_files=test_files,
            window_size=window_size,
            stride=window_size,  # non-overlapping
            walking_only=args.walking_only,
            normalize=False,
            stats=stats,
        )

    # Match training: input/output modes and laterality from config.json when available.
    if run_cfg is not None:
        test_ds_kwargs.update(
            input_mode=run_cfg.get("input_mode", input_mode),
            output_mode=run_cfg.get("output_mode", output_mode),
            laterality=run_cfg.get("laterality", "bilateral"),
        )
    else:
        test_ds_kwargs.update(
            input_indices=input_indices,
            moment_indices=moment_indices,
        )

    # Match training-time denoising when saved in config.json (train.py writes vars(args)).
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
            f"  Dataset denoise (from config.json): LPF={test_ds_kwargs['apply_lowpass_filter']} "
            f"({test_ds_kwargs['lowpass_cutoff_hz']} Hz), "
            f"median_k={test_ds_kwargs['median_kernel_samples']}"
        )

    test_ds = KineticsTCNDataset(**test_ds_kwargs)

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # Inference
    print("Running inference...")
    pred, true = run_inference(model, test_loader, args.device)
    pred_np = pred.numpy()
    true_np = true.numpy()
    print(f"  Predictions shape: {pred_np.shape}")

    # Metrics
    metrics = compute_metrics(pred_np, true_np, dof_names)

    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Overall MSE:  {metrics['overall']['mse']:.6f}")
    print(f"  Overall RMSE: {metrics['overall']['rmse']:.6f}")
    print(f"  Overall MAE:  {metrics['overall']['mae']:.6f}")
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
    plot_scatter_gt_vs_pred(pred_np, true_np, dof_names, out_dir / "scatter_gt_vs_pred.png")

    for s in range(min(args.n_plot_samples, len(pred_np))):
        plot_time_series(pred_np, true_np, dof_names,
                         out_dir / f"timeseries_sample_{s}.png", sample_idx=s)

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
            for s in range(min(args.n_plot_samples, len(pred_np))):
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
