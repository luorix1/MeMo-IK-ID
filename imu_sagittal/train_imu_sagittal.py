#!/usr/bin/env python3
"""
Train a TCN: IMU (interpolated to IK time) → sagittal lower-limb joint angles or moments.

Uses subjects **S035–S056** by default. Entry points: ``imu_sagittal/train_imu_sagittal_angles.py`` /
``imu_sagittal/train_imu_sagittal_moments.py`` (or ``python -m imu_sagittal.train_imu_sagittal``).

Run from ``os_kinetics/``::

    python -m imu_sagittal.train_imu_sagittal ...

**Unilateral IMU → one-leg sagittal** (aligned with ``ik_id.train`` / ``--laterality unilateral``):

- **IMU (per sample):** **24** channels = **pelvis + thigh + shank + foot** on **one** side; training mixes right- and
  left-chain windows (``sides="both"``).
- **Targets:** **3** DOFs — hip flexion, knee, ankle on the **same** side — after the same left hip adduction/rotation
  IK/ID flip.

**Compared to** ``ik_id.train`` **/** ``KineticsTCNDataset``: train window **stride** is fixed at **1**. **Validation**
stride defaults to **``--window-size``** (non-overlapping). IMU channels are z-scored; IK pipeline uses raw rad/rad/s.
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

from imu_sagittal.imu_sagittal_dataset import (
    IMU_UNILATERAL_N_CHANNELS,
    ImuSagittalH5Dataset,
    discover_imu_schemas_paired_first_trial,
    imu_paired_chain_orders,
    imu_unilateral_24_segment_order,
    molinaro_subject_ids,
)
from model import TCN
from training_utils import evaluate, set_global_seed, train_one_epoch

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


def _plot_sample(
    model: torch.nn.Module,
    dataset: ImuSagittalH5Dataset,
    device: str,
    out_path: Path,
    y_label: str,
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
    _side = dataset.windows[0][2]
    names = dataset.output_names_for_side(_side)
    n_plot = min(6, y.shape[0])
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    t = np.arange(y.shape[1]) / float(sample_rate_hz)
    for i in range(n_plot):
        ax = axes[i]
        ax.plot(t, y[i], label="Target", linewidth=1.5)
        ax.plot(t, pred[i], label="Pred", linewidth=1.5, linestyle="--")
        ax.set_ylabel(f"{names[i]}\n({y_label})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("IMU → sagittal (first window)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_curves(train_losses: List[float], val_losses: List[float], out_path: Path) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train MSE")
    if val_losses:
        ax.plot(val_losses, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("IMU sagittal training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    args: argparse.Namespace,
    dataset: ImuSagittalH5Dataset,
    imu_schema_right: List[Tuple[str, str]],
    imu_schema_left: List[Tuple[str, str]],
) -> None:
    torch.save(
        {
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
            "imu_schema_right": [{"segment": s, "column": c} for s, c in imu_schema_right],
            "imu_schema_left": [{"segment": s, "column": c} for s, c in imu_schema_left],
            "imu_schema": [{"segment": s, "column": c} for s, c in imu_schema_right],
            "target": dataset.target,
            "output_names_right": dataset.output_names_right,
            "output_names_left": dataset.output_names_left,
            "output_names": dataset.output_names_right,
            "window_size": args.window_size,
            "stride": 1,
            "laterality": "unilateral",
            "imu_segment_order": list(imu_unilateral_24_segment_order()),
            "imu_chain_right": list(imu_paired_chain_orders()[0]),
            "imu_chain_left": list(imu_paired_chain_orders()[1]),
            "target_sample_rate_hz": getattr(args, "target_sample_rate_hz", None),
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train IMU → sagittal angle or moment TCN")
    p.add_argument("--h5-dir", type=str, required=True, help="Folder with S###.h5 (e.g. Processed/Jinwoo)")
    p.add_argument(
        "--meta-root",
        type=str,
        default=None,
        help="Directory containing dataset_metadata.json (defaults to --h5-dir)",
    )
    p.add_argument("--output-dir", type=str, default="runs/imu_sagittal")
    p.add_argument(
        "--target",
        type=str,
        choices=["angle", "moment"],
        required=True,
        help="angle: IK rad; moment: ID N·m/kg",
    )
    p.add_argument("--window-size", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=7)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--input-noise-std", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-val-subjects", type=int, default=1)
    p.add_argument("--n-test-subjects", type=int, default=2)
    p.add_argument("--val-subjects", nargs="+", default=None)
    p.add_argument("--test-subjects", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--walking-only", action="store_true", default=True)
    p.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    p.add_argument("--levelground-only", action="store_true", default=False)
    p.add_argument(
        "--no-lowpass",
        action="store_true",
        help=(
            "Disable zero-phase Butterworth on IK/ID targets in the loader (SciPy sosfiltfilt; "
            "median filter still applies if set)."
        ),
    )
    p.add_argument(
        "--lowpass-cutoff-hz",
        type=float,
        default=4.0,
        help="Zero-phase Butterworth cutoff (Hz) on IK positions and ID moments. Try 3–6 for gait.",
    )
    p.add_argument(
        "--lowpass-order",
        type=int,
        default=4,
        help="Butterworth order for zero-phase (forward-backward) IK/ID low-pass in the loader.",
    )
    p.add_argument("--median-kernel-samples", type=int, default=0)
    p.add_argument(
        "--target-sample-rate-hz",
        type=float,
        default=None,
        help=(
            "Resample IK/ID (and IMU, interpolated to IK time) to this uniform Hz before denoise/velocities. "
            "Default: native ~200 Hz. Use a smaller --window-size for the same time span at lower Hz."
        ),
    )
    p.add_argument(
        "--val-stride",
        type=int,
        default=None,
        help=(
            "Sliding-window stride for the validation set only. "
            "Default: same as ik_id.train — ``--window-size`` (non-overlapping val windows). "
            "Use 1 to match dense train windows (not recommended for early stopping)."
        ),
    )
    p.add_argument("--early-stopping-patience", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-train-trials", type=int, default=None)
    p.add_argument("--max-val-trials", type=int, default=None)
    p.add_argument("--use-wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", type=str, default="os-kinetics-imu-sagittal")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    args = p.parse_args()

    meta_root = args.meta_root or args.h5_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)

    subjects = molinaro_subject_ids()
    if len(subjects) < args.n_val_subjects + args.n_test_subjects + 1:
        raise ValueError("Not enough subjects in S035–S056 for this split.")

    if args.test_subjects:
        test_subjects = [s.upper() for s in args.test_subjects]
    else:
        shuffled = subjects.copy()
        random.shuffle(shuffled)
        test_subjects = sorted(shuffled[: args.n_test_subjects])

    remaining = [s for s in subjects if s not in set(test_subjects)]

    if args.val_subjects:
        val_subjects = [s.upper() for s in args.val_subjects]
    else:
        random.shuffle(remaining)
        val_subjects = sorted(remaining[: args.n_val_subjects])

    train_subjects = sorted([s for s in remaining if s not in set(val_subjects)])
    if not train_subjects:
        raise ValueError("Empty train split; reduce val/test holdout.")

    print("=" * 70)
    print("SUBJECT SPLIT (S035–S056)")
    print("=" * 70)
    print(f"Train ({len(train_subjects)}): {train_subjects}")
    print(f"Val   ({len(val_subjects)}): {val_subjects}")
    print(f"Test  ({len(test_subjects)}): {test_subjects} (held out; not loaded here)")

    imu_schema_right, imu_schema_left = discover_imu_schemas_paired_first_trial(
        args.h5_dir,
        subjects,
        walking_only=args.walking_only,
        levelground_only=args.levelground_only,
    )
    chain_r, chain_l = imu_paired_chain_orders()
    print(
        f"IMU schemas: R {len(imu_schema_right)} ch (segments={list(chain_r)}), "
        f"L {len(imu_schema_left)} ch (segments={list(chain_l)}); expect {IMU_UNILATERAL_N_CHANNELS} each."
    )

    val_stride = args.window_size if args.val_stride is None else args.val_stride

    ds_kw_train: Dict[str, Any] = dict(
        h5_dir=args.h5_dir,
        meta_root_dir=meta_root,
        imu_schema_right=imu_schema_right,
        imu_schema_left=imu_schema_left,
        sides="both",
        target=args.target,
        window_size=args.window_size,
        stride=1,
        walking_only=args.walking_only,
        levelground_only=args.levelground_only,
        normalize=True,
        apply_lowpass_filter=not args.no_lowpass,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        lowpass_order=args.lowpass_order,
        median_kernel_samples=args.median_kernel_samples,
        target_sample_rate_hz=args.target_sample_rate_hz,
        preload_trials=False,
    )
    ds_kw_val = {**ds_kw_train, "stride": val_stride}

    print("=" * 70)
    print("LOADING TRAIN")
    print("=" * 70)
    if args.target_sample_rate_hz is not None:
        print(f"  target_sample_rate_hz={args.target_sample_rate_hz} (IMU+IK+ID resampled before denoise)")
    train_ds = ImuSagittalH5Dataset(
        subject_ids=train_subjects,
        stats=None,
        max_trials=args.max_train_trials,
        **ds_kw_train,
    )

    print("=" * 70)
    print("LOADING VAL")
    print("=" * 70)
    print(f"  Val stride: {val_stride} (train stride: 1)")
    val_ds = ImuSagittalH5Dataset(
        subject_ids=val_subjects,
        stats=train_ds.get_stats(),
        max_trials=args.max_val_trials,
        **ds_kw_val,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    model = TCN(
        n_input_channels=train_ds.n_input_channels,
        n_output_channels=train_ds.n_output_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(args.device)

    print(
        f"\nModel: TCN | params={sum(p.numel() for p in model.parameters()):,} | "
        f"in={train_ds.n_input_channels} | out={train_ds.n_output_channels} | target={args.target}"
    )

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    wandb_run = None
    if args.use_wandb and HAS_WANDB:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    y_label = "rad" if args.target == "angle" else "N·m/kg"
    best_val = float("inf")
    epochs_no_improve = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            args.device,
            epoch,
            grad_clip=args.grad_clip,
            input_noise_std=args.input_noise_std,
        )
        train_losses.append(tr_loss)
        val_loss, _, _, _ = evaluate(model, val_loader, criterion, args.device)
        val_losses.append(val_loss)
        scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_no_improve = 0
            _save_checkpoint(
                out_dir / "best_model.pt",
                model,
                optimizer,
                epoch,
                tr_loss,
                val_loss,
                args,
                train_ds,
                imu_schema_right,
                imu_schema_left,
            )
            tag = " *best*"
        else:
            epochs_no_improve += 1
            tag = ""

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  train_mse={tr_loss:.6f}  val_mse={val_loss:.6f}{tag}  "
            f"time={time.time()-t0:.1f}s"
        )

        if wandb_run is not None:
            wandb.log({"epoch": epoch + 1, "train/mse": tr_loss, "val/mse": val_loss})

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    _save_checkpoint(
        out_dir / "final_model.pt",
        model,
        optimizer,
        args.epochs - 1,
        train_losses[-1],
        val_losses[-1] if val_losses else None,
        args,
        train_ds,
        imu_schema_right,
        imu_schema_left,
    )

    _plot_curves(train_losses, val_losses, out_dir / "training_curves.png")
    _phz = float(args.target_sample_rate_hz) if args.target_sample_rate_hz is not None else 200.0
    _plot_sample(
        model,
        val_ds,
        args.device,
        out_dir / "sample_prediction.png",
        y_label=y_label,
        sample_rate_hz=_phz,
    )

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(out_dir / "subject_split.json", "w") as f:
        json.dump(
            {
                "cohort": "S035_S056",
                "laterality": "unilateral",
                "imu_segment_order": list(imu_unilateral_24_segment_order()),
                "imu_chain_right": list(chain_r),
                "imu_chain_left": list(chain_l),
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
                "imu_schema_right": [{"segment": s, "column": c} for s, c in imu_schema_right],
                "imu_schema_left": [{"segment": s, "column": c} for s, c in imu_schema_left],
                "imu_schema": [{"segment": s, "column": c} for s, c in imu_schema_right],
            },
            f,
            indent=2,
        )

    if wandb_run is not None:
        wandb.finish()

    print(f"Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
