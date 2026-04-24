#!/usr/bin/env python3
"""
Training script V2 for TCN-based joint moment prediction (IK+vel → ID).

Changes vs train.py
-------------------
1. **Correlated angle+velocity noise** (``--correlated-vel-noise``)
   When ``--angle-jitter-std > 0``, velocity channels are *recomputed* from the
   noisy angle channels via finite difference rather than left as pre-loaded
   (LPF-smoothed) values.  This reproduces the cascade-inference noise model:
   the upstream IMU estimator delivers slightly noisy joint angles, and the
   ik_id model receives velocities that are the finite-difference of those noisy
   angles.  Without this flag the angle and velocity noise are uncorrelated,
   which never occurs in deployment.

2. **Robust / weighted loss** (``--loss-type``, ``--huber-delta``, ``--dof-loss-weights``)
   Supports Huber (smooth-L1) loss in addition to MSE.  Huber applies L2 to
   small residuals and L1 to large ones, reducing the influence of brief
   high-moment events (stair/ramp transitions) that may overwhelm gradient
   updates.  Optional per-DOF weights (``--dof-loss-weights``) allow
   upweighting under-performing joints such as hip flexion.

3. **Temporal smoothness regularisation** (``--smoothness-lambda``)
   Adds ``λ · mean(||Δŷ||²)`` to the training loss.  Real joint moments are
   smooth (≤4 Hz content); penalising high-frequency prediction jitter
   encourages the model to produce physically plausible outputs even when
   cascade-noise pollutes the velocity input.

4. **Half-rate via decimation** (``--rollout``)
   Takes every 2nd IK/ID sample after alignment (~200 Hz → ~100 Hz) with no
   interpolation.

Run from ``os_kinetics/``::

    python -m ik_id.trainV2 ...
    python ik_id/trainV2.py ...
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
from model import TCN, TransformerMoment, GaussianDiffusion1D
from training_utils import MomentLoss, evaluate, set_global_seed, train_one_epoch

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


# ---------------------------------------------------------------------------
# Plotting helpers (unchanged from train.py)
# ---------------------------------------------------------------------------

def plot_curves(train_losses: List[float], val_losses: List[float], out_path: Path) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train loss")
    if val_losses:
        ax.plot(val_losses, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
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
    *,
    sample_rate_hz: float = 200.0,
) -> None:
    if not HAS_MPL:
        return
    model.eval()
    x, y = dataset[0]
    x_batch = x.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_batch).squeeze(0).cpu().numpy()
    y = y.numpy()

    n_out = y.shape[0]
    n_plot = min(n_dofs_to_plot, n_out)
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    t = np.arange(y.shape[1]) / float(sample_rate_hz)

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


# ---------------------------------------------------------------------------
# Diffusion training loop
# ---------------------------------------------------------------------------

def train_one_epoch_diffusion(
    model: "GaussianDiffusion1D",
    loader: Any,
    optimizer: optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
    input_noise_std: float = 0.0,
) -> float:
    """
    Training epoch for GaussianDiffusion1D.

    Unlike the standard epoch which calls ``model(x)`` and ``criterion(pred, y)``,
    diffusion training randomly samples a timestep and calls ``model.p_losses(y, x)``.
    Input noise augmentation (iid Gaussian on x) is still supported.
    Smoothness regularisation and angle-jitter are skipped because the
    diffusion process already acts as a form of noise regularisation.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if input_noise_std > 0:
            x = x + torch.randn_like(x) * input_noise_std
        loss = model.p_losses(y, x)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

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
    # Keep checkpoint metadata self-contained so eval scripts can reconstruct
    # preprocessing and indexing behavior without relying on CLI defaults.
    model_type = getattr(args, "model_type", "tcn")
    if model_type == "transformer":
        arch_cfg = {
            "model_type": "transformer",
            "n_input_channels": model.n_input_channels,
            "n_output_channels": model.n_output_channels,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
        }
    elif model_type == "diffusion":
        arch_cfg = {
            "model_type": "diffusion",
            "n_input_channels": model.n_input_channels,
            "n_output_channels": model.n_output_channels,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "n_timesteps": args.n_diffusion_timesteps,
            "schedule": args.diffusion_schedule,
            "predict_epsilon": bool(args.diffusion_predict_epsilon),
            "n_inference_steps": args.n_inference_steps,
        }
    else:
        arch_cfg = {
            "model_type": "tcn",
            "n_input_channels": model.n_input_channels,
            "n_output_channels": model.n_output_channels,
            "hidden_channels": args.hidden_channels,
            "n_blocks": args.n_blocks,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
        }
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_config": arch_cfg,
        "normalization": dataset.get_stats(),
        "dof_names": dof_names,
        "window_size": args.window_size,
        "stride": args.stride,
        "input_mode": args.input_mode,
        "output_mode": args.output_mode,
        "input_indices": dataset.input_indices,
        "moment_indices": dataset.moment_indices,
        "laterality": args.laterality,
        "unilateral_paired_side_windows": dataset.unilateral_paired,
        "rollout_decimate_step": int(getattr(args, "rollout_decimate_step", 1)),
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TCN for joint moment prediction (V2: correlated noise, Huber, smoothness reg)"
    )

    # ---- Data / split ----
    parser.add_argument("--train-dir", type=str, required=True,
                        help="Directory containing trial directories (Camargo) or S*.h5 files (MeMo)")
    parser.add_argument("--output-dir", type=str, default="runs/tcn_run_v2")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=1,
                        help="Sliding-window step (samples). Validation always uses stride=1.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-val-files", type=int, default=None)
    parser.add_argument("--n-val-subjects", type=int, default=1)
    parser.add_argument("--n-test-subjects", type=int, default=2)
    parser.add_argument("--val-subjects", nargs="+", default=None)
    parser.add_argument("--test-subjects", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walking-only", action="store_true", default=True)
    parser.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    parser.add_argument(
        "--levelground-only", action="store_true",
        help="Use only level-included conditions (levelground, treadmill_normal_walk, etc.)"
    )

    # ---- Preprocessing / LPF ----
    parser.add_argument("--lowpass-cutoff-hz", type=float, default=4.0)
    parser.add_argument("--lowpass-order", type=int, default=4)
    parser.add_argument(
        "--rollout",
        action="store_true",
        default=False,
        help="After IK/ID alignment, keep every 2nd sample (~200 Hz → ~100 Hz). No interpolation.",
    )
    parser.add_argument("--velocity-lowpass-filter", action="store_true", default=True,
                        help="Apply zero-phase LPF to computed angular velocity. Default: on.")
    parser.add_argument("--no-velocity-lowpass-filter", dest="velocity_lowpass_filter",
                        action="store_false")
    parser.add_argument("--velocity-lowpass-cutoff-hz", type=float, default=None,
                        help="Cutoff for velocity LPF. Default: same as --lowpass-cutoff-hz.")
    parser.add_argument("--velocity-lowpass-order", type=int, default=None,
                        help="Order for velocity LPF. Default: same as --lowpass-order.")

    # ---- Model ----
    parser.add_argument(
        "--model-type", type=str, default="tcn", choices=["tcn", "transformer", "diffusion"],
        help="Model architecture: 'tcn' (causal, streaming-friendly) or 'transformer' "
             "(bidirectional, offline, higher accuracy ceiling). Default: tcn.",
    )
    # ---- TCN-specific ----
    parser.add_argument("--hidden-channels", type=int, default=80)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    # ---- Transformer-specific ----
    parser.add_argument(
        "--d-model", type=int, default=256,
        help="[transformer] Internal embedding dimension. Must be divisible by --n-heads. Default: 256.",
    )
    parser.add_argument(
        "--n-heads", type=int, default=8,
        help="[transformer] Number of self-attention heads. Default: 8.",
    )
    parser.add_argument(
        "--n-layers", type=int, default=6,
        help="[transformer] Number of stacked encoder blocks. Default: 6.",
    )
    parser.add_argument(
        "--d-ff", type=int, default=1024,
        help="[transformer/diffusion] Feed-forward hidden dimension inside each block. Default: 1024.",
    )
    # ---- Diffusion-specific ----
    parser.add_argument(
        "--n-diffusion-timesteps", type=int, default=1000,
        help="[diffusion] Total DDPM timesteps T. Default: 1000.",
    )
    parser.add_argument(
        "--diffusion-schedule", type=str, default="cosine", choices=["linear", "cosine"],
        help="[diffusion] Beta schedule. Default: cosine.",
    )
    parser.add_argument(
        "--diffusion-predict-epsilon", action="store_true", default=True,
        help="[diffusion] Predict noise ε (default). Use --no-diffusion-predict-epsilon to predict x0.",
    )
    parser.add_argument("--no-diffusion-predict-epsilon", dest="diffusion_predict_epsilon", action="store_false")
    parser.add_argument(
        "--n-inference-steps", type=int, default=50,
        help="[diffusion] DDIM steps at inference. Default: 50.",
    )
    parser.add_argument("--input-mode", type=str, default="lower_limb",
                        choices=["full", "lower_limb", "sagittal", "sagittal_hip_flexion",
                                 "sagittal_knee", "sagittal_ankle"])
    parser.add_argument("--output-mode", type=str, default="sagittal_hip_knee_ankle",
                        choices=["all", "lower_limb", "hip_knee", "sagittal_hip_knee",
                                 "sagittal_hip_knee_ankle", "sagittal_hip_flexion",
                                 "sagittal_knee", "sagittal_ankle"])
    parser.add_argument("--laterality", type=str, default="unilateral",
                        choices=["bilateral", "unilateral", "both"])
    parser.add_argument("--legacy-unilateral-full-window", action="store_true", default=False)

    # ---- Optimiser ----
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)

    # ---- Augmentation ----
    parser.add_argument(
        "--input-noise-std", type=float, default=0.0,
        help="Gaussian noise std added to all input channels each training batch (N(0,σ²)). "
             "Not applied at validation. Typical: 1e-4–1e-2.",
    )
    parser.add_argument(
        "--angle-jitter-std", type=float, default=0.0,
        help="Gaussian noise std added to joint angle (position) channels only. "
             "When --correlated-vel-noise is set, velocities are recomputed from the noisy "
             "angles (finite diff), reproducing cascade noise structure. Try 1e-3–5e-2 rad.",
    )
    parser.add_argument(
        "--correlated-vel-noise", action="store_true", default=False,
        help="[V2] When set and --angle-jitter-std > 0, recompute velocity channels from "
             "noisy angle channels via finite difference.  Mimics cascade deployment: "
             "IMU estimator delivers slightly noisy angles, so velocity = finite_diff(noisy_angle). "
             "Without this flag the two noise sources are independent, which never occurs in practice.",
    )

    # ---- Loss ----
    parser.add_argument(
        "--loss-type", type=str, default="mse", choices=["mse", "huber"],
        help="[V2] Primary loss: 'mse' (L2, default) or 'huber' (smooth-L1). "
             "Huber reduces the influence of large residuals at transition events "
             "(stair/ramp transitions, foot strike).",
    )
    parser.add_argument(
        "--huber-delta", type=float, default=0.5,
        help="[V2] Huber loss delta (N·m/kg). Residuals below delta use L2; above, L1. "
             "Typical sagittal moment RMSE is 0.1–0.2 N·m/kg; delta=0.5 means the vast "
             "majority of errors use the L2 branch while large outliers are down-weighted.",
    )
    parser.add_argument(
        "--dof-loss-weights", nargs="+", type=float, default=None,
        help="[V2] Per-output-DOF loss weights (space-separated, length must match n_out). "
             "Default: uniform (all 1.0). The DOF order is printed at startup. "
             "Example for sagittal_hip_knee_ankle, unilateral (6 DOFs): "
             "--dof-loss-weights 1.5 1.0 1.2 1.5 1.0 1.2  (upweight hip and ankle).",
    )
    parser.add_argument(
        "--smoothness-lambda", type=float, default=0.0,
        help="[V2] Weight λ for temporal smoothness regularisation: adds "
             "λ·mean(||Δŷ||²) to the training loss where Δŷ is the first temporal "
             "difference of the predicted moment sequence.  Real joint moments are "
             "inherently smooth (≤4 Hz); this penalty discourages high-frequency "
             "prediction jitter caused by noisy cascade inputs. Try 1e-3–1e-2.",
    )

    # ---- Misc ----
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--wandb-project", type=str, default="os-kinetics-tcn")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    # V2 rollout mode is a fixed "keep every 2nd sample" policy.
    # We convert the flag to an explicit numeric step once and thread that
    # through dataset/config usage to avoid ad-hoc branching later.
    args.rollout_decimate_step = 2 if args.rollout else 1

    # Validate joint-specific sagittal mode pairing.
    _joint_sagittal_modes = frozenset({"sagittal_hip_flexion", "sagittal_knee", "sagittal_ankle"})
    if args.input_mode in _joint_sagittal_modes or args.output_mode in _joint_sagittal_modes:
        if args.input_mode != args.output_mode:
            raise ValueError(
                "For sagittal_hip_flexion / sagittal_knee / sagittal_ankle, "
                f"--input-mode and --output-mode must match "
                f"(got {args.input_mode!r} vs {args.output_mode!r})."
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    # ---- Subject-based split ----
    # Support both layouts:
    #   1) H5-only subject files (S001.h5, ...)
    #   2) trial-directory trees (.../S001/condition/trial_xx)
    train_root = Path(args.train_dir)
    h5_subject_files = sorted([p for p in train_root.glob("S*.h5") if p.is_file()])
    is_h5_only_layout = len(h5_subject_files) > 0

    if is_h5_only_layout:
        subjects = sorted([p.stem.upper() for p in h5_subject_files])
        subject_to_trials = None
    else:
        all_trials = find_trial_dirs(args.train_dir)
        subject_to_trials: Dict[str, Any] = {}
        for td in all_trials:
            sid = extract_subject_id(td)
            subject_to_trials.setdefault(sid, []).append(td)
        subjects = sorted(subject_to_trials.keys())

    n_total = len(subjects)
    if n_total < 4:
        raise ValueError(f"Need at least 4 subjects, found {n_total}.")

    if args.test_subjects:
        test_subjects = sorted([s.upper() for s in args.test_subjects])
    else:
        # Reproducible random split due to set_global_seed().
        shuffled = subjects.copy()
        random.shuffle(shuffled)
        test_subjects = sorted(shuffled[:args.n_test_subjects])

    remaining = [s for s in subjects if s not in set(test_subjects)]

    if args.val_subjects:
        val_subjects = sorted([s.upper() for s in args.val_subjects])
    else:
        random.shuffle(remaining)
        val_subjects = sorted(remaining[:args.n_val_subjects])

    train_subjects = sorted([s for s in remaining if s not in set(val_subjects)])

    if len(train_subjects) == 0:
        raise ValueError("Split consumed all subjects.")

    if is_h5_only_layout:
        train_files = train_subjects
        val_files = val_subjects
        test_files_info = test_subjects
    else:
        train_files = [td for s in train_subjects for td in subject_to_trials[s]]
        val_files   = [td for s in val_subjects   for td in subject_to_trials[s]]
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
        apply_lowpass_filter=True,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        lowpass_order=args.lowpass_order,
        rollout_decimate_step=args.rollout_decimate_step,
        apply_velocity_lowpass_filter=args.velocity_lowpass_filter,
        velocity_lowpass_cutoff_hz=args.velocity_lowpass_cutoff_hz,
        velocity_lowpass_order=args.velocity_lowpass_order,
    )
    _vel_cut = args.velocity_lowpass_cutoff_hz or args.lowpass_cutoff_hz
    _vel_ord = args.velocity_lowpass_order or args.lowpass_order
    print(f"  Dataset denoise: zero-phase LPF ({args.lowpass_cutoff_hz} Hz, order {args.lowpass_order})")
    print(f"  Velocity LPF: {'on' if args.velocity_lowpass_filter else 'off'}"
          + (f" ({_vel_cut} Hz, order {_vel_ord})" if args.velocity_lowpass_filter else ""))
    if args.rollout_decimate_step > 1:
        print(f"  Rollout decimation: stride={args.rollout_decimate_step} (native ~200 Hz → ~{200.0/args.rollout_decimate_step:.0f} Hz)")

    _pair_kw: Dict[str, Any] = {}
    if args.legacy_unilateral_full_window:
        _pair_kw["unilateral_paired_side_windows"] = False

    def _make_ds(files_or_subjects, stride, max_files=None):
        # Keep train/val dataset construction in one place so shared kwargs
        # cannot drift between branches.
        if is_h5_only_layout:
            return KineticsTCNDataset(
                data_dir=args.train_dir,
                h5_dir=args.train_dir,
                use_h5=True,
                subject_ids=files_or_subjects,
                window_size=args.window_size,
                stride=stride,
                walking_only=args.walking_only,
                levelground_only=args.levelground_only,
                normalize=False,
                input_mode=args.input_mode,
                output_mode=args.output_mode,
                laterality=args.laterality,
                max_files=max_files,
                **_pair_kw,
                **ds_denoise_kw,
            )
        return KineticsTCNDataset(
            data_dir=args.train_dir,
            b3d_files=files_or_subjects,
            window_size=args.window_size,
            stride=stride,
            walking_only=args.walking_only,
            levelground_only=args.levelground_only,
            normalize=False,
            input_mode=args.input_mode,
            output_mode=args.output_mode,
            laterality=args.laterality,
            **_pair_kw,
            **ds_denoise_kw,
        )

    train_ds = _make_ds(train_files, args.stride, args.max_train_files)
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
                data_dir=args.train_dir, h5_dir=args.train_dir, use_h5=True,
                subject_ids=val_subjects, window_size=args.window_size, stride=1,
                walking_only=args.walking_only, levelground_only=args.levelground_only,
                normalize=False, stats=train_ds.get_stats(),
                input_mode=args.input_mode, output_mode=args.output_mode,
                laterality=args.laterality, max_files=args.max_val_files,
                **_pair_kw, **ds_denoise_kw,
            )
        else:
            val_ds = KineticsTCNDataset(
                data_dir=args.train_dir, b3d_files=val_files,
                window_size=args.window_size, stride=1,
                walking_only=args.walking_only, levelground_only=args.levelground_only,
                normalize=False, stats=train_ds.get_stats(),
                input_mode=args.input_mode, output_mode=args.output_mode,
                laterality=args.laterality, **_pair_kw, **ds_denoise_kw,
            )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        )

    # ---- Model ----
    n_in = train_ds.n_input_channels
    n_out = train_ds.n_output_channels

    if args.model_type == "transformer":
        if args.d_model % args.n_heads != 0:
            raise ValueError(
                f"--d-model ({args.d_model}) must be divisible by --n-heads ({args.n_heads})."
            )
        model = TransformerMoment(
            n_input_channels=n_in,
            n_output_channels=n_out,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ).to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: TransformerMoment  |  params: {n_params:,}  |  in={n_in}  out={n_out}")
        print(f"  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}  "
              f"d_ff={args.d_ff}  dropout={args.dropout}")
        print("  [bidirectional — not suitable for real-time streaming]")
    elif args.model_type == "diffusion":
        if args.d_model % args.n_heads != 0:
            raise ValueError(
                f"--d-model ({args.d_model}) must be divisible by --n-heads ({args.n_heads})."
            )
        model = GaussianDiffusion1D(
            n_input_channels=n_in,
            n_output_channels=n_out,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            n_timesteps=args.n_diffusion_timesteps,
            schedule=args.diffusion_schedule,
            predict_epsilon=args.diffusion_predict_epsilon,
            n_inference_steps=args.n_inference_steps,
        ).to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: GaussianDiffusion1D  |  params: {n_params:,}  |  in={n_in}  out={n_out}")
        print(f"  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}  "
              f"d_ff={args.d_ff}  dropout={args.dropout}")
        print(f"  timesteps={args.n_diffusion_timesteps}  schedule={args.diffusion_schedule}  "
              f"predict_epsilon={args.diffusion_predict_epsilon}  ddim_steps={args.n_inference_steps}")
        print("  [offline — DDIM inference is not real-time suitable]")
    else:
        model = TCN(
            n_input_channels=n_in, n_output_channels=n_out,
            hidden_channels=args.hidden_channels, n_blocks=args.n_blocks,
            kernel_size=args.kernel_size, dropout=args.dropout,
        ).to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        rf = model.receptive_field
        print(f"\nModel: TCN  |  params: {n_params:,}  |  in={n_in}  out={n_out}")
        print(f"  hidden={args.hidden_channels}  blocks={args.n_blocks}  "
              f"kernel={args.kernel_size}  dropout={args.dropout}")
        print(f"  receptive_field={rf} samples  (window={args.window_size})")
        if rf > args.window_size:
            print(f"  WARNING: RF ({rf}) > window_size ({args.window_size}). "
                  f"Deep blocks see only zero-padding during training.")

    print(f"  device={args.device}")
    print(f"  input_mode={args.input_mode}  ({n_in} channels)")
    print(f"  output_mode={args.output_mode}  ({n_out} moments)")
    print(f"  Input DOFs:  {train_ds.input_dof_names}")
    print(f"  Output DOFs: {train_ds.output_dof_names}")
    print(f"  unilateral_paired_side_windows: {train_ds.unilateral_paired}")

    out_dof_names = train_ds.output_dof_names

    # ---- Loss ----
    dof_weights_tensor: Optional[torch.Tensor] = None
    if args.dof_loss_weights is not None:
        if len(args.dof_loss_weights) != n_out:
            raise ValueError(
                f"--dof-loss-weights has {len(args.dof_loss_weights)} values but "
                f"n_out={n_out} ({out_dof_names}). Must match."
            )
        dof_weights_tensor = torch.tensor(args.dof_loss_weights, dtype=torch.float32)

    criterion = MomentLoss(
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        dof_weights=dof_weights_tensor,
    )

    print(f"\n  Loss: {args.loss_type.upper()}"
          + (f"  delta={args.huber_delta}" if args.loss_type == "huber" else ""))
    if dof_weights_tensor is not None:
        w_str = "  ".join(f"{n}:{w:.2f}" for n, w in zip(out_dof_names, args.dof_loss_weights))
        print(f"  DOF weights: {w_str}")
    if args.smoothness_lambda > 0:
        print(f"  Smoothness reg: λ={args.smoothness_lambda}")
    if args.angle_jitter_std > 0:
        flag = "(correlated vel)" if args.correlated_vel_noise else "(angle-only, vel unchanged)"
        print(f"  Angle jitter: std={args.angle_jitter_std} rad  {flag}")

    # ---- Optimiser / scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Effective sample rate for correlated-vel-noise finite differencing / plots (~native 200 Hz).
    if args.rollout_decimate_step > 1:
        _sample_rate_hz = 200.0 / float(args.rollout_decimate_step)
    else:
        _sample_rate_hz = 200.0

    # ---- W&B ----
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
                "n_train_windows": len(train_ds),
                "n_val_windows": len(val_ds) if val_ds is not None else 0,
            }, allow_val_change=True)

    # ---- Training loop ----
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_r2_globals: List[float] = []
    best_val_loss = float("inf")
    best_val_r2 = float("nan")
    epochs_no_improve = 0
    last_epoch_idx = -1
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"TRAINING  |  epochs={args.epochs}  batch={args.batch_size}  "
          f"lr={args.lr}  window={args.window_size}")
    print(f"{'='*70}")

    _is_diffusion = isinstance(model, GaussianDiffusion1D)

    for epoch in range(args.epochs):
        last_epoch_idx = epoch
        ep_start = time.time()
        if _is_diffusion:
            train_loss = train_one_epoch_diffusion(
                model, train_loader, optimizer, args.device,
                grad_clip=args.grad_clip,
                input_noise_std=args.input_noise_std,
            )
        else:
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, args.device, epoch,
                grad_clip=args.grad_clip,
                input_noise_std=args.input_noise_std,
                angle_jitter_std=args.angle_jitter_std,
                n_position_channels=train_ds.n_input_channels // 2,
                correlated_vel_noise=args.correlated_vel_noise,
                sample_rate_hz=_sample_rate_hz,
                smoothness_lambda=args.smoothness_lambda,
            )
        train_losses.append(train_loss)

        log_parts = [f"Epoch {epoch+1:3d}/{args.epochs}  train_loss={train_loss:.6f}"]

        if val_loader is not None:
            # Validation remains MSE for apples-to-apples model selection even if
            # training loss uses Huber or extra regularization terms.
            _mse_crit = nn.MSELoss()
            val_loss, per_ch_rmse, r2_global, per_ch_r2 = evaluate(
                model, val_loader, _mse_crit, args.device)
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
                                 train_ds, args, out_dir / "best_model.pt", out_dof_names)
                log_parts.append("*best*")
            else:
                epochs_no_improve += 1

            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no improvement for {epochs_no_improve} epochs).")
                last_epoch_idx = epoch
                scheduler.step()
                break

        scheduler.step()
        ep_time = time.time() - ep_start
        lr_now = optimizer.param_groups[0]["lr"]
        log_parts.append(f"lr={lr_now:.2e}  time={ep_time:.1f}s")
        print("  ".join(log_parts))

        if wandb_run is not None:
            log_dict: Dict[str, Any] = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
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
            _save_checkpoint(
                model, optimizer, epoch, train_loss,
                val_losses[-1] if val_losses else None,
                train_ds, args,
                out_dir / f"checkpoint_epoch_{epoch+1}.pt",
                out_dof_names,
            )

    total_time = time.time() - t0

    # ---- Final checkpoint ----
    _save_checkpoint(
        model, optimizer, max(last_epoch_idx, 0), train_losses[-1],
        val_losses[-1] if val_losses else None,
        train_ds, args, out_dir / "final_model.pt", out_dof_names,
    )

    # ---- Plots ----
    plot_curves(train_losses, val_losses, out_dir / "training_curves.png")
    plot_ds = val_ds if val_ds is not None else train_ds
    _plot_hz = _sample_rate_hz
    plot_sample_prediction(model, plot_ds, args.device, out_dir / "sample_prediction.png",
                           out_dof_names, sample_rate_hz=_plot_hz)

    if wandb_run is not None:
        for img_name in ("training_curves.png", "sample_prediction.png"):
            img_path = out_dir / img_name
            if img_path.exists():
                wandb.log({f"plots/{img_path.stem}": wandb.Image(str(img_path))})

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"  Final val MSE:    {val_losses[-1]:.6f}")
        print(f"  Best val MSE:     {best_val_loss:.6f}")
        if val_r2_globals:
            fr2_s = f"{val_r2_globals[-1]:.4f}" if np.isfinite(val_r2_globals[-1]) else "nan"
            br2_s = f"{best_val_r2:.4f}" if np.isfinite(best_val_r2) else "nan"
            print(f"  Final val R²:     {fr2_s}")
            print(f"  Best val R²:      {br2_s}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    # ---- Config / split JSON ----
    _cfg_out = dict(vars(args))
    _cfg_out["unilateral_paired_side_windows"] = train_ds.unilateral_paired
    with open(out_dir / "config.json", "w") as f:
        json.dump(_cfg_out, f, indent=2)

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


if __name__ == "__main__":
    main()
