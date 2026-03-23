#!/usr/bin/env python3
"""
Run inference on one MeMo H5 trial and plot GT vs prediction.

Creates one figure per output joint moment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise SystemExit("Install plotly: pip install plotly") from e

from dataset import IK_DOF_NAMES, MOMENT_NAMES, _compute_velocity, _read_h5_opensim_table
from model import TCN


def load_checkpoint_model(checkpoint_path: Path, device: str):
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = TCN(
        n_input_channels=cfg["n_input_channels"],
        n_output_channels=cfg["n_output_channels"],
        hidden_channels=cfg["hidden_channels"],
        n_blocks=cfg["n_blocks"],
        kernel_size=cfg["kernel_size"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    input_indices = ckpt.get("input_indices", None)
    moment_indices = ckpt.get("moment_indices", None)
    dof_names = ckpt.get("dof_names", None)
    window_size = int(ckpt.get("window_size", 200))
    return model, input_indices, moment_indices, dof_names, window_size


def load_memo_trial(
    memo_root: Path,
    subject_id: str,
    condition_name: str,
    trial_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h5_path = memo_root / f"{subject_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing subject h5: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        if condition_name not in h5f:
            raise KeyError(f"Condition '{condition_name}' not found in {h5_path.name}")
        cond_group = h5f[condition_name]
        if trial_name not in cond_group:
            raise KeyError(f"Trial '{trial_name}' not found under condition '{condition_name}'")
        trial_group = cond_group[trial_name]

        ik_group = trial_group["ik"]
        id_group = trial_group["id"]
        ik_key = sorted(list(ik_group.keys()))[0]
        id_key = sorted(list(id_group.keys()))[0]
        ik_cols, ik_data = _read_h5_opensim_table(ik_group[ik_key])
        id_cols, id_data = _read_h5_opensim_table(id_group[id_key])

    if "time" not in ik_cols or "time" not in id_cols:
        raise RuntimeError("Both IK and ID tables must have a 'time' column.")

    time = ik_data[:, ik_cols.index("time")]

    # Build full IK matrix (T, 23). Missing columns become NaN.
    pos_deg = np.full((len(time), len(IK_DOF_NAMES)), np.nan, dtype=np.float64)
    for j, name in enumerate(IK_DOF_NAMES):
        if name in ik_cols:
            pos_deg[:, j] = ik_data[:, ik_cols.index(name)]
    pos = np.deg2rad(pos_deg)
    vel = _compute_velocity(pos, time)

    id_time = id_data[:, id_cols.index("time")]
    n = min(len(time), len(id_time))
    time = time[:n]
    pos = pos[:n]
    vel = vel[:n]
    id_data = id_data[:n]

    # Build full ID moments (T, 20). Missing channels become NaN.
    moments = np.full((n, len(MOMENT_NAMES)), np.nan, dtype=np.float64)
    for j, name in enumerate(MOMENT_NAMES):
        col = f"{name}_moment"
        if col in id_cols:
            moments[:, j] = id_data[:, id_cols.index(col)]

    return time.astype(np.float32), pos.astype(np.float32), vel.astype(np.float32), moments.astype(np.float32)


@torch.no_grad()
def infer_full_trial(
    model: torch.nn.Module,
    pos: np.ndarray,
    vel: np.ndarray,
    moments: np.ndarray,
    input_indices: List[int] | None,
    moment_indices: List[int] | None,
    window_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if input_indices is not None:
        pos_in = pos[:, input_indices]
        vel_in = vel[:, input_indices]
    else:
        pos_in = pos
        vel_in = vel

    if moment_indices is not None:
        y_true = moments[:, moment_indices]
    else:
        y_true = moments

    n = pos_in.shape[0]
    c_out = y_true.shape[1]
    if n < window_size:
        raise ValueError(f"Trial too short ({n}) for window_size={window_size}.")

    pred_sum = np.zeros((n, c_out), dtype=np.float64)
    pred_cnt = np.zeros((n, c_out), dtype=np.float64)

    for start in range(0, n - window_size + 1):
        end = start + window_size
        x_w = np.concatenate([pos_in[start:end], vel_in[start:end]], axis=1).T  # (C_in, W)
        x_t = torch.from_numpy(x_w.astype(np.float32)).unsqueeze(0).to(device)
        pred_w = model(x_t).squeeze(0).detach().cpu().numpy().T  # (W, C_out)
        pred_sum[start:end] += pred_w
        pred_cnt[start:end] += 1.0

    pred = pred_sum / np.maximum(pred_cnt, 1.0)
    return pred.astype(np.float32), y_true.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive inference plots on one MeMo trial")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--memo-root", type=str, default="/media/metamobility3/Samsung_T51/Processed/MeMo")
    parser.add_argument("--subject-id", type=str, required=True, help="Example: S056")
    parser.add_argument("--condition", type=str, required=True, help="Example: dynamic_walk_1")
    parser.add_argument("--trial", type=str, default="trial_01")
    parser.add_argument("--output-dir", type=str, default="runs/memo_trial_inference")
    parser.add_argument(
        "--write-combined-html",
        action="store_true",
        help="Also write one combined multi-panel interactive HTML for all outputs.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, input_indices, moment_indices, dof_names, window_size = load_checkpoint_model(
        Path(args.checkpoint), args.device
    )

    time, pos, vel, moments = load_memo_trial(
        Path(args.memo_root), args.subject_id, args.condition, args.trial
    )
    pred, true = infer_full_trial(
        model=model,
        pos=pos,
        vel=vel,
        moments=moments,
        input_indices=input_indices,
        moment_indices=moment_indices,
        window_size=window_size,
        device=args.device,
    )

    if dof_names is None:
        dof_names = [f"dof_{i}" for i in range(pred.shape[1])]

    t_rel = time - time[0]
    saved_html = 0
    for c in range(pred.shape[1]):
        name = dof_names[c] if c < len(dof_names) else f"dof_{c}"
        # GT may include NaNs if a channel is missing in the source .sto.
        gt = true[:, c]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t_rel,
                y=gt,
                mode="lines",
                name="Ground Truth",
                line=dict(width=2),
                connectgaps=False,  # keep NaN gaps visible
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_rel,
                y=pred[:, c],
                mode="lines",
                name="Predicted",
                line=dict(width=2, dash="dash"),
            )
        )
        fig.update_layout(
            title=f"{args.subject_id} {args.condition} {args.trial} — {name}",
            xaxis_title="Time (s)",
            yaxis_title=f"{name} (N·m/kg)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
        out_path = out_dir / f"{args.subject_id}_{args.condition}_{args.trial}_{name}.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
        saved_html += 1

    if args.write_combined_html:
        n = pred.shape[1]
        fig_all = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            subplot_titles=[dof_names[i] if i < len(dof_names) else f"dof_{i}" for i in range(n)],
        )
        for c in range(n):
            name = dof_names[c] if c < len(dof_names) else f"dof_{c}"
            fig_all.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=true[:, c],
                    mode="lines",
                    name=f"{name} GT",
                    legendgroup=name,
                    line=dict(width=1.6),
                    connectgaps=False,
                    showlegend=(c == 0),
                ),
                row=c + 1,
                col=1,
            )
            fig_all.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=pred[:, c],
                    mode="lines",
                    name=f"{name} Pred",
                    legendgroup=name,
                    line=dict(width=1.6, dash="dash"),
                    showlegend=(c == 0),
                ),
                row=c + 1,
                col=1,
            )
            fig_all.update_yaxes(title_text=name, row=c + 1, col=1)
        fig_all.update_layout(
            height=max(320 * n, 700),
            title=f"{args.subject_id} {args.condition} {args.trial} — all outputs",
            template="plotly_white",
            hovermode="x unified",
        )
        fig_all.update_xaxes(title_text="Time (s)", row=n, col=1)
        combined_path = out_dir / f"{args.subject_id}_{args.condition}_{args.trial}_all_outputs.html"
        fig_all.write_html(str(combined_path), include_plotlyjs="cdn", full_html=True)

    # Save a small run manifest for reproducibility.
    with open(out_dir / "inference_manifest.json", "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "memo_root": args.memo_root,
                "subject_id": args.subject_id,
                "condition": args.condition,
                "trial": args.trial,
                "window_size": window_size,
                "n_outputs": pred.shape[1],
                "output_dof_names": dof_names[: pred.shape[1]],
                "plot_format": "html_plotly",
                "write_combined_html": bool(args.write_combined_html),
            },
            f,
            indent=2,
        )

    print(f"Saved {saved_html} interactive per-joint HTML plots to: {out_dir}")


if __name__ == "__main__":
    main()

