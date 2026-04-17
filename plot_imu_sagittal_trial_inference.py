#!/usr/bin/env python3
"""
Run IMU → sagittal angle/moment inference on one H5 trial and plot GT vs prediction (Plotly HTML).

Uses the same trial pipeline as ``ImuSagittalH5Dataset`` / ``_load_trial_imu_sagittal_paired`` (IK time base,
IMU interpolated, optional zero-phase LPF, unilateral left-hip flip on kinematics/kinetics).

The TCN uses **causal** convolutions: default ``--inference-mode causal`` matches ``plot_memo_trial_inference.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise SystemExit("Install plotly: pip install plotly") from e

from dataset import _load_subject_metadata_map
from ik_id.test import load_run_config
from imu_sagittal.imu_sagittal_eval import load_imu_checkpoint
from imu_sagittal.imu_sagittal_leg_dataset import (
    IMU_UNILATERAL_N_CHANNELS,
    TrialRef,
    _load_trial_imu_sagittal_paired,
)


@torch.no_grad()
def infer_imu_full_trial(
    model: torch.nn.Module,
    imu: np.ndarray,
    y_true: np.ndarray,
    imu_mean: np.ndarray,
    imu_std: np.ndarray,
    window_size: int,
    device: str,
    *,
    inference_mode: str = "causal",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window inference: IMU (T, 24) normalized with checkpoint stats → predictions (T, 3).

    ``y_true`` must align frame-wise with ``imu`` (same length).
    """
    if imu.shape[1] != IMU_UNILATERAL_N_CHANNELS:
        raise ValueError(f"Expected IMU width {IMU_UNILATERAL_N_CHANNELS}, got {imu.shape[1]}")
    if y_true.shape[1] != 3:
        raise ValueError(f"Expected 3 target channels, got {y_true.shape[1]}")

    n = imu.shape[0]
    c_out = 3
    if n < window_size:
        raise ValueError(f"Trial too short ({n}) for window_size={window_size}.")

    imu_mean = np.asarray(imu_mean, dtype=np.float64).reshape(1, -1)
    imu_std = np.asarray(imu_std, dtype=np.float64).reshape(1, -1)
    imu_n = (imu.astype(np.float64) - imu_mean) / imu_std

    W = int(window_size)

    def _forward_window(start: int) -> np.ndarray:
        end = start + W
        x_w = imu_n[start:end].T  # (24, W)
        x_t = torch.from_numpy(x_w.astype(np.float32)).unsqueeze(0).to(device)
        pred_w = model(x_t).squeeze(0).detach().cpu().numpy()  # (3, W)
        return pred_w.T  # (W, 3)

    if inference_mode == "overlap_mean":
        pred_sum = np.zeros((n, c_out), dtype=np.float64)
        pred_cnt = np.zeros((n, c_out), dtype=np.float64)
        for start in range(0, n - W + 1):
            pred_w = _forward_window(start)
            pred_sum[start : start + W] += pred_w
            pred_cnt[start : start + W] += 1.0
        pred = pred_sum / np.maximum(pred_cnt, 1.0)
    elif inference_mode == "causal":
        pred = np.zeros((n, c_out), dtype=np.float64)
        pw0 = _forward_window(0)
        for g in range(W - 1):
            pred[g] = pw0[g]
        for start in range(0, n - W + 1):
            pred_w = _forward_window(start)
            pred[start + W - 1] = pred_w[W - 1]
    else:
        raise ValueError(f"Unknown inference_mode: {inference_mode!r} (use 'causal' or 'overlap_mean')")

    return pred.astype(np.float32), y_true.astype(np.float32)


def run_single_trial_inference(
    *,
    model: torch.nn.Module,
    imu_schema_right: List[Tuple[str, str]],
    imu_schema_left: List[Tuple[str, str]],
    target: str,
    output_names_right: List[str],
    output_names_left: List[str],
    window_size: int,
    stats: dict,
    h5_dir: Path,
    meta_root: Path,
    subject_id: str,
    condition: str,
    trial: str,
    out_dir: Path,
    write_combined_html: bool,
    device: str,
    inference_mode: str = "causal",
    checkpoint_path: str | None = None,
    side: str = "both",
    apply_lowpass_filter: bool = True,
    lowpass_cutoff_hz: float = 4.0,
    lowpass_order: int = 4,
    target_sample_rate_hz: Optional[float] = None,
) -> None:
    """Load one trial with IMU+IK+ID, run inference per leg, write Plotly HTML."""
    out_dir.mkdir(parents=True, exist_ok=True)

    sid = subject_id.upper()
    h5_path = h5_dir / f"{sid}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing subject H5: {h5_path}")

    meta_map = _load_subject_metadata_map(str(meta_root))
    ref: TrialRef = (sid, condition, trial, str(h5_path))

    trial_data = _load_trial_imu_sagittal_paired(
        ref,
        meta_map,
        imu_schema_right,
        imu_schema_left,
        target,
        apply_lowpass_filter=apply_lowpass_filter,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        lowpass_order=lowpass_order,
        target_sample_rate_hz=target_sample_rate_hz,
    )
    if trial_data is None:
        raise RuntimeError(
            f"Failed to load trial (need ik+id+imu with paired chains): {sid} / {condition} / {trial}"
        )

    imu_mean = stats["imu_mean"]
    imu_std = stats["imu_std"]
    time = trial_data["time"]
    t_rel = (time - time[0]).astype(np.float64)

    y_unit = "rad" if target == "angle" else "N·m/kg"

    runs: List[Tuple[str, np.ndarray, np.ndarray, List[str]]] = []
    if side in ("both", "right"):
        pr, tr = infer_imu_full_trial(
            model,
            trial_data["imu_right"],
            trial_data["y_right"],
            imu_mean,
            imu_std,
            window_size,
            device,
            inference_mode=inference_mode,
        )
        runs.append(("right", pr, tr, output_names_right))
    if side in ("both", "left"):
        pl, tl = infer_imu_full_trial(
            model,
            trial_data["imu_left"],
            trial_data["y_left"],
            imu_mean,
            imu_std,
            window_size,
            device,
            inference_mode=inference_mode,
        )
        runs.append(("left", pl, tl, output_names_left))

    saved_html = 0
    combined_rows: List[Tuple[str, str, np.ndarray, np.ndarray, List[str]]] = []

    for leg_label, pred, true, dof_names in runs:
        for c in range(pred.shape[1]):
            name = dof_names[c] if c < len(dof_names) else f"dof_{c}"
            gt = true[:, c]
            safe = name.replace("/", "_")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=gt,
                    mode="lines",
                    name="Ground Truth",
                    line=dict(width=2),
                    connectgaps=False,
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
                title=f"{sid} {condition} {trial} — {leg_label} — {name}",
                xaxis_title="Time (s)",
                yaxis_title=f"{name} ({y_unit})",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
            out_path = out_dir / f"{sid}_{condition}_{trial}_{leg_label}_{safe}.html"
            fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
            saved_html += 1
            combined_rows.append((leg_label, name, pred[:, c], gt, dof_names))

    if write_combined_html and combined_rows:
        n = len(combined_rows)
        fig_all = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            subplot_titles=[
                f"{lg} — {combined_rows[i][1]}" for i, (lg, _, _, _, _) in enumerate(combined_rows)
            ],
        )
        for i, (leg_label, name, pred_c, gt_c, _) in enumerate(combined_rows):
            row = i + 1
            fig_all.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=gt_c,
                    mode="lines",
                    name=f"{leg_label} {name} GT",
                    legendgroup=f"{leg_label}_{name}",
                    line=dict(width=1.6),
                    connectgaps=False,
                    showlegend=(i == 0),
                ),
                row=row,
                col=1,
            )
            fig_all.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=pred_c,
                    mode="lines",
                    name=f"{leg_label} {name} Pred",
                    legendgroup=f"{leg_label}_{name}",
                    line=dict(width=1.6, dash="dash"),
                    showlegend=(i == 0),
                ),
                row=row,
                col=1,
            )
            fig_all.update_yaxes(title_text=f"{name} ({y_unit})", row=row, col=1)
        fig_all.update_layout(
            height=max(280 * n, 700),
            title=f"{sid} {condition} {trial} — IMU → sagittal ({target}) all outputs",
            template="plotly_white",
            hovermode="x unified",
        )
        fig_all.update_xaxes(title_text="Time (s)", row=n, col=1)
        combined_path = out_dir / f"{sid}_{condition}_{trial}_all_outputs.html"
        fig_all.write_html(str(combined_path), include_plotlyjs="cdn", full_html=True)

    with open(out_dir / "inference_manifest.json", "w") as f:
        json.dump(
            {
                "checkpoint": checkpoint_path,
                "h5_dir": str(h5_dir),
                "meta_root": str(meta_root),
                "subject_id": sid,
                "condition": condition,
                "trial": trial,
                "target": target,
                "window_size": window_size,
                "side": side,
                "n_outputs_per_leg": 3,
                "imu_channels": IMU_UNILATERAL_N_CHANNELS,
                "output_names_right": output_names_right,
                "output_names_left": output_names_left,
                "plot_format": "html_plotly",
                "write_combined_html": bool(write_combined_html),
                "apply_lowpass_filter": bool(apply_lowpass_filter),
                "lowpass_cutoff_hz": float(lowpass_cutoff_hz),
                "lowpass_order": int(lowpass_order),
                "inference_mode": inference_mode,
            },
            f,
            indent=2,
        )

    print(f"Saved {saved_html} interactive per-channel HTML plots to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive IMU → sagittal inference plots on one H5 trial"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--h5-dir",
        type=str,
        required=True,
        help="Directory with S###.h5 (same as training --h5-dir)",
    )
    parser.add_argument(
        "--meta-root",
        type=str,
        default=None,
        help="dataset_metadata.json directory (defaults to --h5-dir)",
    )
    parser.add_argument("--subject-id", type=str, required=True, help="Example: S056")
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--trial", type=str, default="trial_01")
    parser.add_argument("--output-dir", type=str, default="runs/imu_sagittal_trial_inference")
    parser.add_argument(
        "--side",
        type=str,
        choices=("right", "left", "both"),
        default="both",
        help="Which pelvis+limb IMU chain(s) to run (training uses both legs separately).",
    )
    parser.add_argument(
        "--write-combined-html",
        action="store_true",
        help="Also write one multi-panel HTML (all legs × all DOFs).",
    )
    parser.add_argument(
        "--no-lowpass",
        action="store_true",
        help="Disable Butterworth low-pass on IK/ID (matches train --no-lowpass).",
    )
    parser.add_argument("--lowpass-cutoff-hz", type=float, default=4.0)
    parser.add_argument("--lowpass-order", type=int, default=4)
    parser.add_argument(
        "--target-sample-rate-hz",
        type=float,
        default=None,
        help="Resample trial to this Hz before denoise (default: from config.json / checkpoint, else native ~200).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=("causal", "overlap_mean"),
        default="causal",
    )
    args = parser.parse_args()

    h5_dir = Path(args.h5_dir)
    meta_root = Path(args.meta_root) if args.meta_root else h5_dir

    (
        model,
        _ckpt,
        imu_schema_right,
        imu_schema_left,
        target,
        output_names_right,
        output_names_left,
        window_size,
        _stride,
        stats,
    ) = load_imu_checkpoint(args.checkpoint, args.device)

    run_cfg = load_run_config(args.checkpoint)
    tgt_sr: Optional[float] = None
    if args.target_sample_rate_hz is not None:
        tgt_sr = float(args.target_sample_rate_hz)
    elif run_cfg is not None and run_cfg.get("target_sample_rate_hz") is not None:
        tgt_sr = float(run_cfg["target_sample_rate_hz"])
    elif _ckpt.get("target_sample_rate_hz") is not None:
        tgt_sr = float(_ckpt["target_sample_rate_hz"])
    if tgt_sr is not None:
        print(f"  target_sample_rate_hz={tgt_sr}")

    run_single_trial_inference(
        model=model,
        imu_schema_right=imu_schema_right,
        imu_schema_left=imu_schema_left,
        target=target,
        output_names_right=output_names_right,
        output_names_left=output_names_left,
        window_size=window_size,
        stats=stats,
        h5_dir=h5_dir,
        meta_root=meta_root,
        subject_id=args.subject_id,
        condition=args.condition,
        trial=args.trial,
        out_dir=Path(args.output_dir),
        write_combined_html=bool(args.write_combined_html),
        device=args.device,
        inference_mode=str(args.inference_mode),
        checkpoint_path=str(args.checkpoint),
        side=str(args.side),
        apply_lowpass_filter=not bool(args.no_lowpass),
        lowpass_cutoff_hz=float(args.lowpass_cutoff_hz),
        lowpass_order=int(args.lowpass_order),
        target_sample_rate_hz=tgt_sr,
    )


if __name__ == "__main__":
    main()
