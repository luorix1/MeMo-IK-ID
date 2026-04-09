#!/usr/bin/env python3
"""
Training script for TCN-based joint moment prediction.

Key behaviors:
  - Subject-based split: train and validation subjects are disjoint.
  - Optional training-time Gaussian input noise (--input-noise-std); not applied at validation.
  - Optional Weights & Biases logging.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import (
    KineticsTCNDataset,
    extract_subject_id,
    find_trial_dirs,
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


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    grad_clip: float = 1.0,
    input_noise_std: float = 0.0,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if input_noise_std > 0:
            x = x + torch.randn_like(x) * input_noise_std
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, np.ndarray, float, np.ndarray]:
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_pred, all_true = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        running_loss += loss.item()
        n_batches += 1
        all_pred.append(pred.cpu())
        all_true.append(y.cpu())

    avg_loss = running_loss / max(n_batches, 1)

    all_pred = torch.cat(all_pred, dim=0)  # (N, C_out, W)
    all_true = torch.cat(all_true, dim=0)

    # Per-channel RMSE: sqrt(mean over samples and time)
    per_ch_mse = ((all_pred - all_true) ** 2).mean(dim=(0, 2))  # (C_out,)
    per_ch_rmse = per_ch_mse.sqrt().numpy()

    # R²: 1 - SS_res / SS_tot (per channel and global over samples × time)
    ss_res = ((all_pred - all_true) ** 2).sum(dim=(0, 2))  # (C_out,)
    t_mean = all_true.mean(dim=(0, 2), keepdim=True)  # (1, C_out, 1)
    ss_tot = ((all_true - t_mean) ** 2).sum(dim=(0, 2))
    per_ch_r2 = torch.where(
        ss_tot > 0,
        1.0 - ss_res / ss_tot,
        torch.full_like(ss_res, float("nan")),
    )
    per_ch_r2_np = per_ch_r2.numpy()

    flat_p = all_pred.reshape(-1)
    flat_t = all_true.reshape(-1)
    ss_res_g = ((flat_p - flat_t) ** 2).sum()
    ss_tot_g = ((flat_t - flat_t.mean()) ** 2).sum()
    if ss_tot_g.item() > 0:
        r2_global = float((1.0 - ss_res_g / ss_tot_g).item())
    else:
        r2_global = float("nan")

    return avg_loss, per_ch_rmse, r2_global, per_ch_r2_np


def plot_curves(train_losses: List[float], val_losses: List[float], out_path: Path) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train MSE")
    if val_losses:
        ax.plot(val_losses, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_prediction(
    model: torch.nn.Module,
    dataset: Any,
    device: str,
    out_path: Path,
    dof_names: List[str],
    n_dofs_to_plot: int = 6,
) -> None:
    """Plot ground-truth vs predicted moments for the first window."""
    if not HAS_MPL:
        return
    model.eval()
    x, y = dataset[0]
    x_batch = x.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_batch).squeeze(0).cpu().numpy()  # (C_out, W)
    y = y.numpy()

    n_out = y.shape[0]
    n_plot = min(n_dofs_to_plot, n_out)
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]

    t = np.arange(y.shape[1]) / 200.0  # seconds at 200 Hz

    for i in range(n_plot):
        ax = axes[i]
        ax.plot(t, y[i], label="Ground Truth", linewidth=1.5)
        ax.plot(t, pred[i], label="Predicted", linewidth=1.5, linestyle="--")
        name = dof_names[i] if i < len(dof_names) else f"DOF {i}"
        ax.set_ylabel(f"{name}\n(N·m/kg)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Sample Moment Prediction (first window)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCN for joint moment prediction")
    parser.add_argument("--train-dir", type=str, required=True,
                        help="Directory containing trial directories")
    parser.add_argument("--output-dir", type=str, default="runs/tcn_run")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding-window step in samples (same for every condition). Validation uses --window-size.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--input-noise-std",
        type=float,
        default=0.0,
        help=(
            "If > 0, add Gaussian noise to training inputs each batch (N(0, σ²) i.i.d. per element). "
            "Not applied to validation. Units match the model input tensor (e.g. rad and rad/s when "
            "normalize=False). Typical starting range: 1e-4–1e-2 depending on scale."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-val-files", type=int, default=None)
    parser.add_argument("--n-val-subjects", type=int, default=1,
                        help="Number of subjects to hold out for validation (default: 1)")
    parser.add_argument("--n-test-subjects", type=int, default=2,
                        help="Number of subjects to hold out as final test set "
                             "(never used during training or early-stopping; default: 2)")
    parser.add_argument("--val-subjects", nargs="+", default=None,
                        help="Explicit val subject IDs, overrides --n-val-subjects")
    parser.add_argument("--test-subjects", nargs="+", default=None,
                        help="Explicit test subject IDs, overrides --n-test-subjects")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walking-only", action="store_true", default=True)
    parser.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    parser.add_argument(
        "--levelground-only",
        action="store_true",
        help=(
            "Use only level-included conditions: levelground_*, treadmill_normal_walk*, "
            "treadmill_transient*, treadmill_0p*, treadmill_1p*, treadmill_2p*, "
            "treadmill_unspecified_speed*. When set, determines which conditions are kept "
            "(instead of the broad --walking-only filter)."
        ),
    )
    parser.add_argument(
        "--no-lowpass",
        action="store_true",
        help="Disable Butterworth low-pass in the dataset loader (median filter still applies if set).",
    )
    parser.add_argument(
        "--lowpass-cutoff-hz",
        type=float,
        default=4.0,
        help="Zero-phase Butterworth low-pass cutoff (Hz). Try 3–6 for gait; lower = smoother.",
    )
    parser.add_argument("--lowpass-order", type=int, default=4)
    parser.add_argument(
        "--median-kernel-samples",
        type=int,
        default=0,
        help=(
            "Temporal median kernel length (samples, >=3, odd enforced). "
            "Try 5–9 at ~200 Hz to suppress impulse / double-peak noise before LPF."
        ),
    )
    parser.add_argument("--input-mode", type=str, default="lower_limb",
                        choices=["full", "lower_limb", "sagittal"],
                        help=(
                            "Input DOF set: "
                            "full=all 23 DOFs (46 ch), "
                            "lower_limb=hip+knee+ankle R/L (10 DOFs, 20 ch), "
                            "sagittal=hip_flex+knee+ankle R/L (6 DOFs, 12 ch)"
                        ))
    parser.add_argument("--output-mode", type=str, default="sagittal_hip_knee_ankle",
                        choices=["all", "lower_limb", "hip_knee", "sagittal_hip_knee", "sagittal_hip_knee_ankle"],
                        help=(
                            "Output moment set: "
                            "all=all 23 moments, "
                            "lower_limb=hip+knee+ankle R/L (10), "
                            "hip_knee=hip+knee R/L (8), "
                            "sagittal_hip_knee=hip_flex+knee R/L (4), "
                            "sagittal_hip_knee_ankle=hip_flex+knee+ankle R/L (6)"
                        ))
    parser.add_argument(
        "--laterality",
        type=str,
        default="unilateral",
        choices=["bilateral", "unilateral", "both"],
        help=(
            "bilateral (alias: both): use all R/L channels as in the files. "
            "unilateral: same DOFs; negate left hip adduction & rotation (angles, velocities, moments) "
            "so both legs share one sign convention; right side unchanged."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help=(
            "Early stopping patience (epochs) based on validation MSE. "
            "If set to 0, early stopping is disabled."
        ),
    )
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--wandb-project", type=str, default="os-kinetics-tcn")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)

    train_root = Path(args.train_dir)
    h5_subject_files = sorted([p for p in train_root.glob("S*.h5") if p.is_file()])
    # H5-only layout (e.g. MeMo_processed): /.../Processed/MeMo/S###.h5
    is_h5_only_layout = len(h5_subject_files) > 0

    # ---- Subject-based split ----
    if is_h5_only_layout:
        subjects = sorted([p.stem.upper() for p in h5_subject_files])
        subject_to_trials = None
    else:
        # Processed Camargo-style layout: /.../Processed/Camargo/S###/<condition>/trial_XX/...
        all_trials = find_trial_dirs(args.train_dir)
        subject_to_trials = {}  # type: ignore[var-annotated]
        for td in all_trials:
            sid = extract_subject_id(td)
            subject_to_trials.setdefault(sid, []).append(td)
        subjects = sorted(subject_to_trials.keys())

    n_total = len(subjects)
    if n_total < 4:
        raise ValueError(f"Need at least 4 subjects for an 18/1/2 split, found {n_total}.")

    # ---- Determine test subjects (held out completely) ----
    if args.test_subjects:
        test_subjects = sorted([s.upper() for s in args.test_subjects])
    else:
        shuffled = subjects.copy()
        random.shuffle(shuffled)
        test_subjects = sorted(shuffled[:args.n_test_subjects])

    remaining = [s for s in subjects if s not in set(test_subjects)]

    # ---- Determine val subjects (used for early stopping only) ----
    if args.val_subjects:
        val_subjects = sorted([s.upper() for s in args.val_subjects])
    else:
        random.shuffle(remaining)
        val_subjects = sorted(remaining[:args.n_val_subjects])

    train_subjects = sorted([s for s in remaining if s not in set(val_subjects)])

    if len(train_subjects) == 0:
        raise ValueError("Split consumed all subjects — reduce --n-val-subjects or --n-test-subjects.")

    if is_h5_only_layout:
        # For H5-only datasets we filter by subject_ids directly in the dataset.
        train_files = train_subjects  # type: ignore[assignment]
        val_files = val_subjects  # type: ignore[assignment]
        # Recorded for completeness; not used during training.
        test_files_info = test_subjects  # type: ignore[assignment]
    else:
        train_files = [td for s in train_subjects for td in subject_to_trials[s]]
        val_files   = [td for s in val_subjects   for td in subject_to_trials[s]]
        # test files are recorded but never loaded during training
        test_files_info = [str(td) for s in test_subjects for td in subject_to_trials[s]]

    if args.max_train_files is not None:
        train_files = train_files[:args.max_train_files]
    if args.max_val_files is not None:
        val_files = val_files[:args.max_val_files]

    print("=" * 70)
    print("SUBJECT SPLIT")
    print("=" * 70)
    print(f"All subjects ({n_total}): {subjects}")
    print(f"Train  ({len(train_subjects):2d}): {train_subjects}")
    print(f"Val    ({len(val_subjects):2d}): {val_subjects}")
    print(f"Test   ({len(test_subjects):2d}): {test_subjects}  ← never used during training")
    print(f"Train files: {len(train_files)}  |  Val files: {len(val_files)}")

    # ---- Data ----
    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)
    ds_denoise_kw = dict(
        apply_lowpass_filter=not args.no_lowpass,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        lowpass_order=args.lowpass_order,
        median_kernel_samples=args.median_kernel_samples,
    )
    print(
        f"  Dataset denoise: LPF={ds_denoise_kw['apply_lowpass_filter']} "
        f"({ds_denoise_kw['lowpass_cutoff_hz']} Hz, order {ds_denoise_kw['lowpass_order']}), "
        f"median_k={args.median_kernel_samples}"
    )
    if args.levelground_only:
        print("  Condition filter: --levelground-only (level-included tasks only; see dataset.py)")
    elif args.walking_only:
        print("  Condition filter: walking-like trials only (--walking-only)")
    if is_h5_only_layout:
        train_ds = KineticsTCNDataset(
            data_dir=args.train_dir,
            h5_dir=args.train_dir,
            use_h5=True,
            subject_ids=train_subjects,
            window_size=args.window_size,
            stride=args.stride,
            walking_only=args.walking_only,
            levelground_only=args.levelground_only,
            normalize=False,
            input_mode=args.input_mode,
            output_mode=args.output_mode,
            laterality=args.laterality,
            max_files=args.max_train_files,
            **ds_denoise_kw,
        )
    else:
        train_ds = KineticsTCNDataset(
            data_dir=args.train_dir,
            b3d_files=train_files,
            window_size=args.window_size,
            stride=args.stride,
            walking_only=args.walking_only,
            levelground_only=args.levelground_only,
            normalize=False,
            input_mode=args.input_mode,
            output_mode=args.output_mode,
            laterality=args.laterality,
            **ds_denoise_kw,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    val_loader = None
    val_ds = None
    if len(val_files) > 0:
        print("\n" + "=" * 70)
        print("LOADING VALIDATION DATA")
        print("=" * 70)
        if is_h5_only_layout:
            val_ds = KineticsTCNDataset(
                data_dir=args.train_dir,
                h5_dir=args.train_dir,
                use_h5=True,
                subject_ids=val_subjects,
                window_size=args.window_size,
                stride=args.window_size,  # non-overlapping for eval
                walking_only=args.walking_only,
                levelground_only=args.levelground_only,
                normalize=False,
                stats=train_ds.get_stats(),
                input_mode=args.input_mode,
                output_mode=args.output_mode,
                laterality=args.laterality,
                max_files=args.max_val_files,
                **ds_denoise_kw,
            )
        else:
            val_ds = KineticsTCNDataset(
                data_dir=args.train_dir,
                b3d_files=val_files,
                window_size=args.window_size,
                stride=args.window_size,  # non-overlapping for eval
                walking_only=args.walking_only,
                levelground_only=args.levelground_only,
                normalize=False,
                stats=train_ds.get_stats(),
                input_mode=args.input_mode,
                output_mode=args.output_mode,
                laterality=args.laterality,
                **ds_denoise_kw,
            )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        )

    # ---- Model ----
    n_in = train_ds.n_input_channels
    n_out = train_ds.n_output_channels
    model = TCN(
        n_input_channels=n_in,
        n_output_channels=n_out,
        hidden_channels=args.hidden_channels,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: TCN  |  params: {n_params:,}  |  in={n_in}  out={n_out}")
    print(f"  hidden={args.hidden_channels}  blocks={args.n_blocks}  "
          f"kernel={args.kernel_size}  dropout={args.dropout}")
    print(f"  device={args.device}")
    print(f"  input_mode={args.input_mode}  ({n_in} channels)")
    print(f"  output_mode={args.output_mode}  ({n_out} moments)")
    print(f"  Input DOFs:  {train_ds.input_dof_names}")
    print(f"  Output DOFs: {train_ds.output_dof_names}")

    out_dof_names = train_ds.output_dof_names

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    wandb_run = None
    if args.use_wandb:
        if not HAS_WANDB:
            print("wandb is not installed. Install with: pip install wandb")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
            )
            wandb.config.update({
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
                "n_train_files": len(train_files),
                "n_val_files": len(val_files),
            }, allow_val_change=True)

    # ---- Training loop ----
    train_losses, val_losses = [], []
    val_r2_globals: List[float] = []
    best_val_loss = float("inf")
    best_val_r2 = float("nan")
    epochs_no_improve = 0
    should_stop = False
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"TRAINING  |  epochs={args.epochs}  batch={args.batch_size}  "
          f"lr={args.lr}  window={args.window_size}")
    if args.input_noise_std > 0:
        print(f"  Input noise (train only): Gaussian std={args.input_noise_std}")
    print(f"{'='*70}")

    for epoch in range(args.epochs):
        ep_start = time.time()
        should_stop = False
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch,
            grad_clip=args.grad_clip,
            input_noise_std=args.input_noise_std,
        )
        train_losses.append(train_loss)

        log_parts = [f"Epoch {epoch+1:3d}/{args.epochs}  train_mse={train_loss:.6f}"]

        if val_loader is not None:
            val_loss, per_ch_rmse, r2_global, per_ch_r2 = evaluate(
                model, val_loader, criterion, args.device)
            val_losses.append(val_loss)
            val_r2_globals.append(r2_global)
            log_parts.append(f"val_mse={val_loss:.6f}")
            r2_str = f"{r2_global:.4f}" if np.isfinite(r2_global) else "nan"
            log_parts.append(f"val_R2={r2_str}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_r2 = r2_global
                epochs_no_improve = 0
                _save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                                 train_ds, args, out_dir / "best_model.pt",
                                 out_dof_names)
                log_parts.append("*best*")
            else:
                epochs_no_improve += 1

            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1}: "
                    f"val_mse did not improve for {epochs_no_improve} epochs "
                    f"(patience={args.early_stopping_patience})."
                )
                should_stop = True

        scheduler.step()
        ep_time = time.time() - ep_start
        lr_now = optimizer.param_groups[0]["lr"]
        log_parts.append(f"lr={lr_now:.2e}  time={ep_time:.1f}s")
        print("  ".join(log_parts))

        if wandb_run is not None:
            log_dict = {
                "epoch": epoch + 1,
                "train/mse": train_loss,
                "train/lr": lr_now,
                "train/epoch_time_sec": ep_time,
            }
            if val_loader is not None:
                log_dict["val/mse"] = val_loss
                if np.isfinite(r2_global):
                    log_dict["val/r2"] = float(r2_global)
                for i, rmse in enumerate(per_ch_rmse):
                    if i < len(out_dof_names):
                        log_dict[f"val/rmse/{out_dof_names[i]}"] = float(rmse)
                for i, r2c in enumerate(per_ch_r2):
                    if i < len(out_dof_names) and np.isfinite(r2c):
                        log_dict[f"val/r2/{out_dof_names[i]}"] = float(r2c)
            wandb.log(log_dict)

        if (epoch + 1) % args.save_freq == 0:
            _save_checkpoint(model, optimizer, epoch, train_loss,
                             val_losses[-1] if val_losses else None,
                             train_ds, args,
                             out_dir / f"checkpoint_epoch_{epoch+1}.pt",
                             out_dof_names)

        if should_stop:
            break

    total_time = time.time() - t0

    # ---- Save final model ----
    _save_checkpoint(model, optimizer, args.epochs - 1, train_losses[-1],
                     val_losses[-1] if val_losses else None,
                     train_ds, args, out_dir / "final_model.pt", out_dof_names)

    # ---- Plots ----
    plot_curves(train_losses, val_losses, out_dir / "training_curves.png")

    plot_ds = val_ds if val_ds is not None else train_ds
    plot_sample_prediction(model, plot_ds, args.device,
                           out_dir / "sample_prediction.png", out_dof_names)

    if wandb_run is not None:
        if (out_dir / "training_curves.png").exists():
            wandb.log({"plots/training_curves": wandb.Image(str(out_dir / "training_curves.png"))})
        if (out_dir / "sample_prediction.png").exists():
            wandb.log({"plots/sample_prediction": wandb.Image(str(out_dir / "sample_prediction.png"))})

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Final train MSE: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"  Final val MSE:   {val_losses[-1]:.6f}")
        print(f"  Best val MSE:    {best_val_loss:.6f}")
        if val_r2_globals:
            fr2 = val_r2_globals[-1]
            br2 = best_val_r2
            fr2_s = f"{fr2:.4f}" if np.isfinite(fr2) else "nan"
            br2_s = f"{br2:.4f}" if np.isfinite(br2) else "nan"
            print(f"  Final val R²:    {fr2_s}")
            print(f"  Best val R²:     {br2_s}  (at best val MSE checkpoint)")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    # Save run config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    with open(out_dir / "subject_split.json", "w") as f:
        json.dump({
            "all_subjects": subjects,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
            "n_train_files": len(train_files),
            "n_val_files": len(val_files),
            "n_test_files": len(test_files_info),
            "test_files": test_files_info,
        }, f, indent=2)

    if wandb_run is not None:
        wandb.finish()


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    dataset: KineticsTCNDataset,
    args: Any,
    path: Path,
    dof_names: List[str],
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_config": {
            "n_input_channels": model.n_input_channels,
            "n_output_channels": model.n_output_channels,
            "hidden_channels": args.hidden_channels,
            "n_blocks": args.n_blocks,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
        },
        "normalization": dataset.get_stats(),
        "dof_names": dof_names,
        "window_size": args.window_size,
        "stride": args.stride,
        "input_mode": args.input_mode,
        "output_mode": args.output_mode,
        "input_indices": dataset.input_indices,
        "moment_indices": dataset.moment_indices,
    }, path)


if __name__ == "__main__":
    main()
