"""
Plot model I/O traces from a knee-exo .npz log using Plotly.

Default use case:
  python utils/plot_npz_model_io_plotly.py \
      --npz /home/metamobility3/Jinwoo/os_kinetics/0426_gain0p2.npz \
      --side right \
      --command-scale 0.2 \
      --out /home/metamobility3/Jinwoo/os_kinetics/0426_gain0p2_plot.html
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot model inputs/outputs from npz log.")
    p.add_argument("--npz", required=True, help="Path to .npz log file.")
    p.add_argument("--side", choices=["right", "left"], default="right")
    p.add_argument(
        "--command-scale",
        type=float,
        default=0.2,
        help="Runtime gain applied in main loop (used to unnormalize cmd trace).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output HTML path. Defaults to <npz_basename>_model_io_plot.html",
    )
    return p.parse_args()


def _get_first_available(data: np.lib.npyio.NpzFile, keys: list[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in data.files:
            return np.asarray(data[k], dtype=np.float32)
    return None


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)

    n = min(len(np.asarray(data["time"])), *(len(np.asarray(data[k])) for k in data.files if np.asarray(data[k]).ndim == 1))
    t = np.asarray(data["time"][:n], dtype=np.float32)

    sfx = "r" if args.side == "right" else "l"
    cmd_key = "cmd_R" if args.side == "right" else "cmd_L"

    # Prefer logged model-input channels (exact values fed to TRT), fallback to legacy names.
    angle = _get_first_available(data, [f"model_in_knee_angle_raw", f"knee_angle_{sfx}"])
    vel = _get_first_available(data, [f"model_in_knee_vel_raw", f"knee_angle_{sfx}_u_gyr"])
    model_out_nmpkg = _get_first_available(data, ["model_out_nmpkg"])
    cmd_nm = _get_first_available(data, [cmd_key])
    moment_raw_nm = _get_first_available(data, ["moment_raw"])

    if angle is None or vel is None or model_out_nmpkg is None or cmd_nm is None:
        raise KeyError(
            "Missing one of required traces: model input angle/vel, model_out_nmpkg, command trace."
        )

    angle = angle[:n]
    vel = vel[:n]
    model_out_nmpkg = model_out_nmpkg[:n]
    cmd_nm = cmd_nm[:n]
    moment_raw_nm = moment_raw_nm[:n] if moment_raw_nm is not None else None

    # "Unnormalized torque command" from commanded torque / runtime gain.
    if abs(args.command_scale) > 1e-9:
        cmd_unnorm_nm = cmd_nm / float(args.command_scale)
    else:
        cmd_unnorm_nm = np.full_like(cmd_nm, np.nan, dtype=np.float32)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f"Input Channel 0 ({args.side}): Knee Angle (rad)",
            f"Input Channel 1 ({args.side}): Knee Angular Velocity (rad/s)",
            "Model Output: Knee Moment (Nm/kg)",
            f"Torque Commands ({args.side}): Unnormalized Nm",
        ),
    )

    fig.add_trace(go.Scatter(x=t, y=angle, mode="lines", name="knee_angle_raw"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=vel, mode="lines", name="knee_vel_raw"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=model_out_nmpkg, mode="lines", name="model_out_nmpkg"), row=3, col=1)

    fig.add_trace(
        go.Scatter(
            x=t, y=cmd_unnorm_nm, mode="lines", name=f"cmd_unnorm_nm (= {cmd_key}/{args.command_scale:g})"
        ),
        row=4,
        col=1,
    )
    if moment_raw_nm is not None:
        fig.add_trace(
            go.Scatter(x=t, y=moment_raw_nm, mode="lines", name="moment_raw_nm (logged)"),
            row=4,
            col=1,
        )

    fig.update_yaxes(title_text="rad", row=1, col=1)
    fig.update_yaxes(title_text="rad/s", row=2, col=1)
    fig.update_yaxes(title_text="Nm/kg", row=3, col=1)
    fig.update_yaxes(title_text="Nm", row=4, col=1)
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)

    fig.update_layout(
        height=1100,
        width=1400,
        title_text=f"Model Inputs/Outputs from {os.path.basename(args.npz)}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )

    out = args.out
    if out is None:
        base = os.path.splitext(os.path.abspath(args.npz))[0]
        out = f"{base}_model_io_plot.html"
    out = os.path.abspath(out)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[INFO] Wrote Plotly HTML: {out}")


if __name__ == "__main__":
    main()

