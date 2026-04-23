#!/usr/bin/env python3
"""
Visualize joint angles and angular velocities for one H5 trial, aligned with ik_id training loader.

Pipeline order (matches ``KineticsTCNDataset``):
1) load IK (deg->rad) and ID moments from H5 trial
2) optional uniform resampling
3) zero-phase LPF on joint angles (and moments for parity with dataset path)
4) differentiate angles -> angular velocity
5) optional extra zero-phase LPF on the computed velocity for comparison

Default trial is the requested:
  S040 / treadmill_normal_walk_bundle1 / trial_01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise SystemExit("Install plotly: pip install plotly") from e

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import h5py
except ImportError as e:
    raise SystemExit("Install h5py: pip install h5py") from e

from dataset import (
    IK_DOF_NAMES,
    MOMENT_NAMES,
    _compute_velocity,
    _denoise_pos_and_moments,
    _ik_time_and_pos_deg,
    _lowpass_zero_phase,
    _read_h5_opensim_table,
    resample_trial_to_uniform_hz,
)


def _load_single_h5_trial(
    *,
    h5_dir: Path,
    subject_id: str,
    condition: str,
    trial: str,
    target_sample_rate_hz: Optional[float],
    angle_lowpass_cutoff_hz: float,
    angle_lowpass_order: int,
) -> Dict[str, np.ndarray]:
    sid = subject_id.upper()
    h5_path = h5_dir / f"{sid}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing subject H5: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        if condition not in h5f:
            raise KeyError(f"Condition not found in {h5_path.name}: {condition}")
        if trial not in h5f[condition]:
            raise KeyError(f"Trial not found in {h5_path.name}/{condition}: {trial}")
        trial_group = h5f[condition][trial]
        if "ik" not in trial_group or "id" not in trial_group:
            raise KeyError("Trial must contain both 'ik' and 'id' groups.")

        ik_group = trial_group["ik"]
        id_group = trial_group["id"]
        if len(ik_group.keys()) == 0 or len(id_group.keys()) == 0:
            raise ValueError("IK/ID groups are empty.")

        ik_key = sorted(list(ik_group.keys()))[0]
        id_key = sorted(list(id_group.keys()))[0]
        ik_cols, ik_data = _read_h5_opensim_table(ik_group[ik_key])
        id_cols, id_data = _read_h5_opensim_table(id_group[id_key])

    ik_tp = _ik_time_and_pos_deg(ik_cols, ik_data)
    if ik_tp is None:
        raise ValueError("IK table missing time column.")
    time, pos_deg = ik_tp
    pos = np.deg2rad(pos_deg)

    if "time" not in id_cols:
        raise ValueError("ID table missing time column.")
    id_time = id_data[:, id_cols.index("time")]
    n = min(len(time), len(id_time))
    time = time[:n]
    pos = pos[:n]
    id_data = id_data[:n]

    moments = np.full((n, len(MOMENT_NAMES)), np.nan, dtype=np.float64)
    for j, name in enumerate(MOMENT_NAMES):
        col = f"{name}_moment"
        if col in id_cols:
            moments[:, j] = id_data[:, id_cols.index(col)]

    if target_sample_rate_hz is not None and target_sample_rate_hz > 0:
        time, pos, moments = resample_trial_to_uniform_hz(
            time.astype(np.float64),
            pos.astype(np.float64),
            moments.astype(np.float64),
            float(target_sample_rate_hz),
        )

    # Same denoise entry-point as dataset loader: LPF on angles first.
    pos_lpf, _mom_lpf = _denoise_pos_and_moments(
        pos.astype(np.float64),
        moments.astype(np.float64),
        time.astype(np.float64),
        apply_lowpass_filter=True,
        lowpass_cutoff_hz=float(angle_lowpass_cutoff_hz),
        lowpass_order=int(angle_lowpass_order),
    )
    vel_from_lpf_pos = _compute_velocity(pos_lpf, time.astype(np.float64))

    return {
        "time": time.astype(np.float64),
        "angles_rad_lpf": pos_lpf.astype(np.float64),
        "vel_rad_s_from_lpf_pos": vel_from_lpf_pos.astype(np.float64),
    }


def _plot_angles_and_velocities(
    *,
    time: np.ndarray,
    angle: np.ndarray,
    vel_no_lpf: np.ndarray,
    vel_lpf: np.ndarray,
    dof_names: List[str],
    out_path: Path,
    vel_lpf_cutoff_hz: float,
    vel_lpf_order: int,
) -> None:
    t_rel = time - float(time[0])
    n = angle.shape[1]
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(14, max(2.2 * n, 8)), sharex="col")
    if n == 1:
        axes = np.array([axes])  # shape -> (1,2)

    for i in range(n):
        axes[i, 0].plot(t_rel, angle[:, i], color="black", linewidth=1.5)
        axes[i, 0].set_ylabel(f"{dof_names[i]}\n(rad)")
        axes[i, 0].grid(True, alpha=0.25)

        axes[i, 1].plot(t_rel, vel_no_lpf[:, i], color="#808080", linewidth=1.3, label="velocity (no output LPF)")
        axes[i, 1].plot(
            t_rel,
            vel_lpf[:, i],
            color="#d62728",
            linewidth=1.6,
            label=f"velocity + output LPF ({vel_lpf_cutoff_hz} Hz, order {vel_lpf_order})",
        )
        axes[i, 1].set_ylabel("rad/s")
        axes[i, 1].grid(True, alpha=0.25)
        if i == 0:
            axes[i, 1].legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 0].set_title("Joint angle (LPF before differentiation)")
    axes[0, 1].set_title("Angular velocity from differentiated angle")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_angles_and_velocities_plotly(
    *,
    time: np.ndarray,
    angle: np.ndarray,
    vel_no_lpf: np.ndarray,
    vel_lpf: np.ndarray,
    dof_names: List[str],
    out_path: Path,
    vel_lpf_cutoff_hz: float,
    vel_lpf_order: int,
) -> None:
    t_rel = time - float(time[0])
    n = angle.shape[1]
    subplot_titles: List[str] = []
    for i in range(n):
        subplot_titles.append(f"{dof_names[i]} angle (LPF)")
        subplot_titles.append(f"{dof_names[i]} angular velocity")
    fig = make_subplots(
        rows=n,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=subplot_titles,
    )
    for i in range(n):
        row = i + 1
        fig.add_trace(
            go.Scatter(
                x=t_rel,
                y=angle[:, i],
                mode="lines",
                name=f"{dof_names[i]} angle",
                line=dict(color="black", width=1.6),
                legendgroup=f"a{i}",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="rad", row=row, col=1)

        fig.add_trace(
            go.Scatter(
                x=t_rel,
                y=vel_no_lpf[:, i],
                mode="lines",
                name="velocity (no output LPF)",
                line=dict(color="#808080", width=1.4),
                legendgroup="v0",
                showlegend=(i == 0),
            ),
            row=row,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=t_rel,
                y=vel_lpf[:, i],
                mode="lines",
                name=f"velocity + output LPF ({vel_lpf_cutoff_hz} Hz, order {vel_lpf_order})",
                line=dict(color="#d62728", width=1.8),
                legendgroup="v1",
                showlegend=(i == 0),
            ),
            row=row,
            col=2,
        )
        fig.update_yaxes(title_text="rad/s", row=row, col=2)

    fig.update_xaxes(title_text="Time (s)", row=n, col=1)
    fig.update_xaxes(title_text="Time (s)", row=n, col=2)
    fig.update_layout(
        height=max(320 * n, 700),
        width=1400,
        template="plotly_white",
        hovermode="x unified",
        title="Angle LPF -> differentiate -> optional velocity LPF",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visualize angle and angular velocity for one H5 trial (ik_id loader-aligned)."
    )
    p.add_argument("--h5-dir", type=str, required=True, help="Directory containing S*.h5 files.")
    p.add_argument("--subject-id", type=str, default="S040")
    p.add_argument("--condition", type=str, default="treadmill_normal_walk_bundle1")
    p.add_argument("--trial", type=str, default="trial_01")
    p.add_argument(
        "--dofs",
        nargs="+",
        default=["hip_flexion_r", "knee_angle_r", "ankle_angle_r", "hip_flexion_l", "knee_angle_l", "ankle_angle_l"],
        help="IK DOF names to plot (must exist in dataset.IK_DOF_NAMES).",
    )
    p.add_argument(
        "--target-sample-rate-hz",
        type=float,
        default=None,
        help="Optional resample rate before LPF/differentiation (same stage as train loader).",
    )
    p.add_argument("--angle-lowpass-cutoff-hz", type=float, default=4.0)
    p.add_argument("--angle-lowpass-order", type=int, default=4)
    p.add_argument("--vel-output-lowpass-cutoff-hz", type=float, default=4.0)
    p.add_argument("--vel-output-lowpass-order", type=int, default=4)
    p.add_argument(
        "--backend",
        type=str,
        choices=["plotly", "matplotlib"],
        default="plotly",
        help="Plot backend (default: plotly).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/trial_angle_velocity_viz",
    )
    args = p.parse_args()

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bad = [d for d in args.dofs if d not in IK_DOF_NAMES]
    if bad:
        raise ValueError(f"Unknown DOF names: {bad}")
    dof_idx = [IK_DOF_NAMES.index(d) for d in args.dofs]

    trial = _load_single_h5_trial(
        h5_dir=h5_dir,
        subject_id=args.subject_id,
        condition=args.condition,
        trial=args.trial,
        target_sample_rate_hz=args.target_sample_rate_hz,
        angle_lowpass_cutoff_hz=args.angle_lowpass_cutoff_hz,
        angle_lowpass_order=args.angle_lowpass_order,
    )

    t = trial["time"]
    pos_lpf = trial["angles_rad_lpf"][:, dof_idx]
    vel_no_lpf = trial["vel_rad_s_from_lpf_pos"][:, dof_idx]
    vel_lpf = _lowpass_zero_phase(
        vel_no_lpf.astype(np.float64),
        t.astype(np.float64),
        cutoff_hz=float(args.vel_output_lowpass_cutoff_hz),
        order=int(args.vel_output_lowpass_order),
    )

    tag = f"{args.subject_id.upper()}_{args.condition}_{args.trial}"
    if args.backend == "plotly":
        fig_path = out_dir / f"{tag}_angle_velocity_comparison.html"
    else:
        fig_path = out_dir / f"{tag}_angle_velocity_comparison.png"
    npz_path = out_dir / f"{tag}_angle_velocity_comparison.npz"

    if args.backend == "plotly":
        _plot_angles_and_velocities_plotly(
            time=t,
            angle=pos_lpf,
            vel_no_lpf=vel_no_lpf,
            vel_lpf=vel_lpf,
            dof_names=list(args.dofs),
            out_path=fig_path,
            vel_lpf_cutoff_hz=float(args.vel_output_lowpass_cutoff_hz),
            vel_lpf_order=int(args.vel_output_lowpass_order),
        )
    else:
        if not HAS_MPL:
            raise SystemExit("matplotlib backend requested but matplotlib is not installed.")
        _plot_angles_and_velocities(
            time=t,
            angle=pos_lpf,
            vel_no_lpf=vel_no_lpf,
            vel_lpf=vel_lpf,
            dof_names=list(args.dofs),
            out_path=fig_path,
            vel_lpf_cutoff_hz=float(args.vel_output_lowpass_cutoff_hz),
            vel_lpf_order=int(args.vel_output_lowpass_order),
        )

    np.savez(
        npz_path,
        time=t,
        dof_names=np.array(args.dofs, dtype=object),
        angle_rad_lpf=pos_lpf,
        vel_rad_s_no_output_lpf=vel_no_lpf,
        vel_rad_s_with_output_lpf=vel_lpf,
    )

    print(f"Saved figure: {fig_path}")
    print(f"Saved arrays: {npz_path}")
    print("Pipeline: angle LPF -> differentiate -> optional velocity output LPF (comparison plotted).")


if __name__ == "__main__":
    main()

