"""
Compare Camargo kinetics dataset loading:
  - `KineticsTCNDataset` (text `.mot/.sto` from Processed/Camargo)
  - `KineticsTCNH5Dataset` (arrays from Processed/Camargo_h5)

The script:
  1) Picks a specific window from the processed dataset (default: window index 0)
  2) Finds the corresponding window in the H5 dataset (by subject/condition/trial folder + start frame)
  3) Verifies that `__getitem__` outputs (`x`, `y`) are numerically identical
  4) Plots overlayed knee kinematics (positions) and knee moments for that window
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

# In case this is run headless.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly via:
#   python os_kinetics/compare_camargo_processed_vs_h5_trial_plot.py ...
# without needing PYTHONPATH configuration.
import sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from os_kinetics.dataset import KineticsTCNDataset
from os_kinetics.benchmark_dataset_loading_camargo_h5 import KineticsTCNH5Dataset


def _trial_key_from_processed(ds: KineticsTCNDataset, t_idx: int, start: int) -> Tuple[str, str, str, int]:
    td = ds.trial_dirs[t_idx]
    # Processed directory depth: .../<subject>/<condition>/<trial_XX>
    subject_id = td.parent.parent.name
    condition_name = td.parent.name
    trial_folder = td.name
    return (subject_id.upper(), condition_name, trial_folder, int(start))


def _trial_key_from_h5(ds: KineticsTCNH5Dataset, t_idx: int, start: int) -> Tuple[str, str, str, int]:
    tr = ds._trial_refs[t_idx]
    return (tr.subject_id.upper(), tr.condition_name, tr.trial_name, int(start))


def _find_matching_h5_window(
    ds_processed: KineticsTCNDataset,
    ds_h5: KineticsTCNH5Dataset,
    processed_window_idx: int,
) -> Tuple[int, Tuple[str, str, str, int], Tuple[int, int]]:
    p_t_idx, p_start = ds_processed.windows[processed_window_idx]
    key = _trial_key_from_processed(ds_processed, p_t_idx, p_start)

    for h_idx, (h_t_idx, h_start) in enumerate(ds_h5.windows):
        if h_start != p_start:
            continue
        if _trial_key_from_h5(ds_h5, h_t_idx, h_start) == key:
            return h_idx, key, (p_t_idx, p_start)

    raise RuntimeError(f"Could not find matching H5 window for key={key}")


def _get_channel_indices(names: List[str], targets: List[str]) -> List[int]:
    idxs: List[int] = []
    for t in targets:
        if t in names:
            idxs.append(names.index(t))
    return idxs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Camargo processed vs Camargo_h5 dataset loading")
    parser.add_argument("--processed_root", default="/media/metamobility3/Samsung_T51/Processed/Camargo")
    parser.add_argument("--h5_root", default="/media/metamobility3/Samsung_T51/Processed/Camargo_h5")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--max_trials", type=int, default=5, help="Max trials to scan during dataset init")
    parser.add_argument(
        "--walking_only",
        action="store_true",
        help="Use walking-only filtering (mirrors default in dataset.py). Default is True.",
    )
    parser.add_argument(
        "--no_walking_only",
        action="store_true",
        help="Disable walking-only filtering (overrides --walking_only).",
    )
    parser.add_argument("--normalize", action="store_true", help="Apply normalization (default off for equality checks)")
    parser.add_argument("--processed_window_idx", type=int, default=0, help="Which processed-window index to compare")
    parser.add_argument("--out_dir", default="/home/metamobility3/Jinwoo/os_kinetics/runs", help="Output directory for plots")
    args = parser.parse_args()

    # Defaults: dataset.py uses walking_only=True; mirror that unless disabled.
    walking_only = True
    if args.no_walking_only:
        walking_only = False
    elif args.walking_only:
        walking_only = True

    # Normalize is opt-in so we can do a direct equality check by default.
    normalize = bool(args.normalize)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use `max_files` to cap scan time; keep normalize disabled for direct equality comparison.
    ds_processed = KineticsTCNDataset(
        data_dir=args.processed_root,
        window_size=args.window_size,
        stride=args.stride,
        walking_only=walking_only,
        normalize=normalize,
        max_files=args.max_trials,
        preload_trials=False,
    )
    ds_h5 = KineticsTCNH5Dataset(
        h5_dir=args.h5_root,
        meta_root_dir=args.processed_root,
        window_size=args.window_size,
        stride=args.stride,
        walking_only=walking_only,
        normalize=normalize,
        max_files=args.max_trials,
        preload_trials=False,
    )

    if len(ds_processed) == 0 or len(ds_h5) == 0:
        raise RuntimeError(f"Empty dataset(s). processed_len={len(ds_processed)} h5_len={len(ds_h5)}")
    if args.processed_window_idx >= len(ds_processed):
        raise RuntimeError(f"processed_window_idx out of range. idx={args.processed_window_idx} len={len(ds_processed)}")

    t0 = time.perf_counter()
    h5_window_idx, key, (p_t_idx, p_start) = _find_matching_h5_window(ds_processed, ds_h5, args.processed_window_idx)
    t1 = time.perf_counter()

    x_p, y_p = ds_processed[args.processed_window_idx]
    x_h, y_h = ds_h5[h5_window_idx]

    x_p_np = x_p.detach().cpu().numpy()
    x_h_np = x_h.detach().cpu().numpy()
    y_p_np = y_p.detach().cpu().numpy()
    y_h_np = y_h.detach().cpu().numpy()

    if x_p_np.shape != x_h_np.shape or y_p_np.shape != y_h_np.shape:
        raise RuntimeError(f"Shape mismatch: x {x_p_np.shape} vs {x_h_np.shape}, y {y_p_np.shape} vs {y_h_np.shape}")

    x_max_abs = float(np.max(np.abs(x_p_np - x_h_np)))
    y_max_abs = float(np.max(np.abs(y_p_np - y_h_np)))
    identical = (x_max_abs == 0.0 and y_max_abs == 0.0) or (x_max_abs < 1e-7 and y_max_abs < 1e-7)

    print("=== Match ===")
    print(f"trial key: {key}")
    print(f"processed window idx: {args.processed_window_idx} -> h5 window idx: {h5_window_idx}")
    print(f"match search time: {t1 - t0:.3f}s")
    print("=== Numerical identity check ===")
    print(f"x max abs diff: {x_max_abs:.8g}")
    print(f"y max abs diff: {y_max_abs:.8g}")
    print(f"identical (tol=1e-7): {identical}")

    # Load raw slices (time + positions/moments) for plotting.
    trial_p = ds_processed._get_trial(p_t_idx)
    trial_h = ds_h5._get_trial(ds_h5.windows[h5_window_idx][0])

    start = p_start
    end = start + args.window_size

    pos_p = trial_p["positions"][start:end].copy()
    vel_p = trial_p["velocities"][start:end].copy()
    mom_p = trial_p["moments"][start:end].copy()

    pos_h = trial_h["positions"][start:end].copy()
    vel_h = trial_h["velocities"][start:end].copy()
    mom_h = trial_h["moments"][start:end].copy()

    # Match channel selection + normalization done in __getitem__.
    if normalize:
        pos_p = (pos_p - ds_processed.pos_mean) / ds_processed.pos_std
        vel_p = (vel_p - ds_processed.vel_mean) / ds_processed.vel_std
        pos_h = (pos_h - ds_h5.pos_mean) / ds_h5.pos_std
        vel_h = (vel_h - ds_h5.vel_mean) / ds_h5.vel_std

    if ds_processed.input_indices is not None:
        pos_p = pos_p[:, ds_processed.input_indices]
        vel_p = vel_p[:, ds_processed.input_indices]
        pos_h = pos_h[:, ds_processed.input_indices]
        vel_h = vel_h[:, ds_processed.input_indices]

    if ds_processed.moment_indices is not None:
        mom_p = mom_p[:, ds_processed.moment_indices]
        mom_h = mom_h[:, ds_processed.moment_indices]

    time_p = trial_p["time"][start:end].copy()
    time_h = trial_h["time"][start:end].copy()

    # Use channel names to pick the signals to plot.
    kin_names = ds_processed.input_dof_names
    mom_names = ds_processed.output_dof_names

    knee_r = "knee_angle_r"
    knee_l = "knee_angle_l"
    pos_r = kin_names.index(knee_r) if knee_r in kin_names else 0
    pos_l = kin_names.index(knee_l) if knee_l in kin_names else (1 if len(kin_names) > 1 else 0)
    mom_r = mom_names.index(knee_r) if knee_r in mom_names else 0
    mom_l = mom_names.index(knee_l) if knee_l in mom_names else (1 if len(mom_names) > 1 else 0)

    # Plot 1: Kinematics (positions only)
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    colors = ["#d62728", "#1f77b4"]  # red, blue
    ax1.plot(time_p, pos_p[:, pos_r], color=colors[0], linestyle="-", label="knee_angle_r processed")
    ax1.plot(time_h, pos_h[:, pos_r], color=colors[0], linestyle="--", label="knee_angle_r h5")
    ax1.plot(time_p, pos_p[:, pos_l], color=colors[1], linestyle="-", label="knee_angle_l processed")
    ax1.plot(time_h, pos_h[:, pos_l], color=colors[1], linestyle="--", label="knee_angle_l h5")
    ax1.set_title("Camargo kinematics overlay (joint angles, rad)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("angle (rad)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)
    kin_path = out_dir / f"camargo_kinematics_overlay_w{args.window_size}_s{args.stride}_idx{args.processed_window_idx}.png"
    fig1.tight_layout()
    fig1.savefig(kin_path, dpi=150)
    plt.close(fig1)

    # Plot 2: Joint moments (subset, N*m/kg)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(time_p, mom_p[:, mom_r], color=colors[0], linestyle="-", label="knee_angle_r processed")
    ax2.plot(time_h, mom_h[:, mom_r], color=colors[0], linestyle="--", label="knee_angle_r h5")
    ax2.plot(time_p, mom_p[:, mom_l], color=colors[1], linestyle="-", label="knee_angle_l processed")
    ax2.plot(time_h, mom_h[:, mom_l], color=colors[1], linestyle="--", label="knee_angle_l h5")
    ax2.set_title("Camargo joint moments overlay (N*m/kg)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("moment (N*m/kg)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)
    mom_path = out_dir / f"camargo_moments_overlay_w{args.window_size}_s{args.stride}_idx{args.processed_window_idx}.png"
    fig2.tight_layout()
    fig2.savefig(mom_path, dpi=150)
    plt.close(fig2)

    print("=== Plots ===")
    print(str(kin_path))
    print(str(mom_path))


if __name__ == "__main__":
    # Allow running directly: ensure repo root is in sys.path if needed.
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()

