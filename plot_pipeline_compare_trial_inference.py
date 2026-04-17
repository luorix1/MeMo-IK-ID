#!/usr/bin/env python3
"""
Plot **ground-truth** sagittal moments vs **direct** (IMU → moment) vs **cascade**
(IMU → angle → IK TCN → moment) on a **single** H5 trial, plus a **separate** set of
plots for **IMU → joint angle** vs IK ground truth.

Also writes moment plots that add **zero-phase LPF on the cascade moment trajectory** (full trial, same
cutoff/order as IK loading when LPF is enabled), as ``*_pipeline_compare_with_cascade_moment_lpf.html``.

Matches the pipelines in ``compare_pipeline.py`` (same windowing, hybrid 6-DOF sagittal
angles: predicted side from IMU angle model, other side from GT; velocities from the
mixed trajectory; IK moment TCN with training normalization).

Uses **causal** (or ``overlap_mean``) alignment per channel, consistent with
``plot_memo_trial_inference.py`` / ``plot_imu_sagittal_trial_inference.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise SystemExit("Install plotly: pip install plotly") from e

from compare_pipeline import (
    _cascade_pos6_vel6_from_full_ik,
    _ik_moment_tcn_input,
    _lowpass_predicted_angles,
)
from dataset import (
    IK_DOF_NAMES,
    MOMENT_NAMES,
    SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES,
    SAGITTAL_INPUT_INDICES,
    _load_subject_metadata_map,
    _lowpass_zero_phase,
)
from ik_id.test import load_model, load_run_config
from imu_sagittal.imu_sagittal_eval import load_imu_checkpoint
from imu_sagittal.imu_sagittal_leg_dataset import (
    IMU_UNILATERAL_N_CHANNELS,
    TrialRef,
    _load_trial_imu_sagittal_paired,
)
from model import TCN
from plot_imu_sagittal_trial_inference import infer_imu_full_trial

# Moment plots: GT black solid, direct gray dotted, cascade red dotted; cascade+LPF red solid (4th series).
_MOMENT_LINE_GT = dict(color="black", width=2)
_MOMENT_LINE_DIRECT = dict(color="#6e6e6e", width=2, dash="dot")
_MOMENT_LINE_CASCADE = dict(color="red", width=2, dash="dot")
_MOMENT_LINE_CASCADE_LPF = dict(color="red", width=2)
_MOMENT_LINE_GT_SUB = dict(color="black", width=1.8)
_MOMENT_LINE_DIRECT_SUB = dict(color="#6e6e6e", width=1.8, dash="dot")
_MOMENT_LINE_CASCADE_SUB = dict(color="red", width=1.8, dash="dot")
_MOMENT_LINE_CASCADE_LPF_SUB = dict(color="red", width=1.8)


def _ik_stats_as_numpy(ik_stats: dict) -> dict:
    out = dict(ik_stats)
    for k, v in list(out.items()):
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
    return out


def _lowpass_cascade_moment_trajectory(
    pred_c: np.ndarray,
    time_s: np.ndarray,
    *,
    apply: bool,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """
    Zero-phase Butterworth on full-trial cascade moment predictions (T, 3), same helper as IK/moment denoising.
    If ``apply`` is False or ``cutoff_hz`` <= 0, returns ``pred_c`` unchanged.
    """
    if not apply or cutoff_hz <= 0:
        return np.asarray(pred_c, dtype=np.float32)
    t = np.asarray(time_s, dtype=np.float64)
    x = np.asarray(pred_c, dtype=np.float64)
    if x.shape[0] != len(t):
        raise ValueError("pred_c rows must match time length.")
    xf = _lowpass_zero_phase(x, t, cutoff_hz=float(cutoff_hz), order=int(order))
    return xf.astype(np.float32)


@torch.no_grad()
def infer_cascade_full_trial(
    angle_model: torch.nn.Module,
    ik_model: torch.nn.Module,
    imu: np.ndarray,
    positions_full: np.ndarray,
    time_1d: np.ndarray,
    imu_mean: np.ndarray,
    imu_std: np.ndarray,
    ik_stats: dict,
    input_indices: Sequence[int],
    window_size: int,
    device: str,
    eval_side: str,
    *,
    inference_mode: str = "causal",
    cascade_angle_lowpass: bool = True,
    cascade_angle_lowpass_cutoff_hz: float = 4.0,
    cascade_angle_lowpass_order: int = 4,
) -> np.ndarray:
    """
    Cascade IMU → angle → IK moment TCN; return (T, 3) for ``eval_side`` moments.

    ``positions_full`` is (T, 23) rad (unilateral convention), same as ``ImuSagittalH5Dataset`` trials.
    Hybrid velocities use ``dataset._compute_velocity`` via ``compare_pipeline._cascade_pos6_vel6_from_full_ik``.
    Predicted angles can be zero-phase low-pass filtered (same helper as ``compare_pipeline``) before the merge.
    """
    n = imu.shape[0]
    if n < window_size:
        raise ValueError(f"Trial too short ({n}) for window_size={window_size}.")
    if positions_full.shape != (n, 23):
        raise ValueError(f"positions_full must be (T, 23), got {positions_full.shape}")
    if len(time_1d) != n:
        raise ValueError("time_1d length must match IMU length.")
    if imu.shape[1] != IMU_UNILATERAL_N_CHANNELS:
        raise ValueError(f"IMU must have {IMU_UNILATERAL_N_CHANNELS} columns.")

    imu_mean = np.asarray(imu_mean, dtype=np.float64).reshape(1, -1)
    imu_std = np.asarray(imu_std, dtype=np.float64).reshape(1, -1)
    imu_n = (imu.astype(np.float64) - imu_mean) / imu_std

    W = int(window_size)
    dev = torch.device(device)
    es = eval_side.lower()
    if es == "right":
        out_lo, out_hi = 0, 3
    elif es == "left":
        out_lo, out_hi = 3, 6
    else:
        raise ValueError("eval_side must be 'right' or 'left'")

    pos23 = positions_full.astype(np.float32)
    tvec = time_1d.astype(np.float32)

    def forward_cascade_window(start: int) -> np.ndarray:
        end = start + W
        x_imu = torch.from_numpy(imu_n[start:end].T.astype(np.float32)).unsqueeze(0).to(device)
        pred_a = angle_model(x_imu)
        pos23_w = torch.from_numpy(pos23[start:end].T.copy()).unsqueeze(0).to(device)
        time_w = torch.from_numpy(tvec[start:end].copy()).unsqueeze(0).to(device)
        pred_a = _lowpass_predicted_angles(
            pred_a,
            time_w,
            apply=cascade_angle_lowpass,
            cutoff_hz=cascade_angle_lowpass_cutoff_hz,
            order=cascade_angle_lowpass_order,
        )
        pos6, vel6 = _cascade_pos6_vel6_from_full_ik(pred_a, pos23_w, time_w, es, dev)
        x_ik = _ik_moment_tcn_input(pos6, vel6)
        pred_full = ik_model(x_ik).squeeze(0).detach().cpu().numpy()  # (6, W)
        return pred_full[out_lo:out_hi, :].T.astype(np.float32)  # (W, 3)

    c_out = 3
    if inference_mode == "overlap_mean":
        pred_sum = np.zeros((n, c_out), dtype=np.float64)
        pred_cnt = np.zeros((n, c_out), dtype=np.float64)
        for start in range(0, n - W + 1):
            pred_w = forward_cascade_window(start)
            pred_sum[start : start + W] += pred_w
            pred_cnt[start : start + W] += 1.0
        pred = pred_sum / np.maximum(pred_cnt, 1.0)
    elif inference_mode == "causal":
        pred = np.zeros((n, c_out), dtype=np.float64)
        pw0 = forward_cascade_window(0)
        for g in range(W - 1):
            pred[g] = pw0[g]
        for start in range(0, n - W + 1):
            pred_w = forward_cascade_window(start)
            pred[start + W - 1] = pred_w[W - 1]
    else:
        raise ValueError(f"Unknown inference_mode: {inference_mode!r}")

    return pred.astype(np.float32)


def run_single_trial(
    *,
    m_direct: torch.nn.Module,
    m_angle: torch.nn.Module,
    ik_model: torch.nn.Module,
    stats_imu: dict,
    ik_stats: dict,
    input_indices: List[int],
    imu_schema_right: List[Tuple[str, str]],
    imu_schema_left: List[Tuple[str, str]],
    window_size: int,
    h5_dir: Path,
    meta_root: Path,
    subject_id: str,
    condition: str,
    trial: str,
    eval_side: str,
    out_dir: Path,
    write_combined_html: bool,
    device: str,
    inference_mode: str,
    sample_rate_hz: float,
    apply_lowpass_filter: bool,
    lowpass_cutoff_hz: float,
    lowpass_order: int,
    ckpt_paths: dict,
    target_sample_rate_hz: Optional[float] = None,
) -> None:
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
        "moment",
        apply_lowpass_filter=apply_lowpass_filter,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        lowpass_order=lowpass_order,
        target_sample_rate_hz=target_sample_rate_hz,
    )
    if trial_data is None:
        raise RuntimeError(f"Could not load trial {sid} / {condition} / {trial} (need ik+id+imu).")

    if eval_side == "right":
        imu = trial_data["imu_right"]
        y_true = trial_data["y_right"]
    else:
        imu = trial_data["imu_left"]
        y_true = trial_data["y_left"]

    pos6 = trial_data["pos_sagittal_rl"]
    time = trial_data["time"]
    t_rel = (time - time[0]).astype(np.float64)

    pred_d, _ = infer_imu_full_trial(
        m_direct,
        imu,
        y_true,
        stats_imu["imu_mean"],
        stats_imu["imu_std"],
        window_size,
        device,
        inference_mode=inference_mode,
    )

    pred_c = infer_cascade_full_trial(
        m_angle,
        ik_model,
        imu,
        trial_data["positions"],
        trial_data["time"],
        stats_imu["imu_mean"],
        stats_imu["imu_std"],
        ik_stats,
        input_indices,
        window_size,
        device,
        eval_side,
        inference_mode=inference_mode,
        cascade_angle_lowpass=apply_lowpass_filter,
        cascade_angle_lowpass_cutoff_hz=lowpass_cutoff_hz,
        cascade_angle_lowpass_order=lowpass_order,
    )

    pred_c_mom_lpf = _lowpass_cascade_moment_trajectory(
        pred_c,
        time,
        apply=apply_lowpass_filter,
        cutoff_hz=lowpass_cutoff_hz,
        order=lowpass_order,
    )

    if eval_side == "right":
        y_true_ang = pos6[:, :3].copy()
        angle_dof_names = [IK_DOF_NAMES[i] for i in SAGITTAL_INPUT_INDICES[:3]]
    else:
        y_true_ang = pos6[:, 3:6].copy()
        angle_dof_names = [IK_DOF_NAMES[i] for i in SAGITTAL_INPUT_INDICES[3:]]

    pred_ang, _ = infer_imu_full_trial(
        m_angle,
        imu,
        y_true_ang,
        stats_imu["imu_mean"],
        stats_imu["imu_std"],
        window_size,
        device,
        inference_mode=inference_mode,
    )

    if eval_side == "right":
        dof_names = [MOMENT_NAMES[i] for i in SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES[:3]]
    else:
        dof_names = [MOMENT_NAMES[i] for i in SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES[3:]]

    saved_moments = 0
    for c in range(3):
        name = dof_names[c]
        safe = name.replace("/", "_")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t_rel,
                y=y_true[:, c],
                mode="lines",
                name="Ground truth",
                line=dict(_MOMENT_LINE_GT),
                connectgaps=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_rel,
                y=pred_d[:, c],
                mode="lines",
                name="Direct (IMU → moment)",
                line=dict(_MOMENT_LINE_DIRECT),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_rel,
                y=pred_c[:, c],
                mode="lines",
                name="Cascade (IMU → angle → IK → moment)",
                line=dict(_MOMENT_LINE_CASCADE),
            )
        )
        fig.update_layout(
            title=f"{sid} {condition} {trial} — {eval_side} — {name}",
            xaxis_title="Time (s)",
            yaxis_title=f"{name} (N·m/kg)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0.0),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
        out_path = out_dir / f"{sid}_{condition}_{trial}_{eval_side}_{safe}_pipeline_compare.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
        saved_moments += 1

        fig_lpf = go.Figure()
        fig_lpf.add_trace(
            go.Scatter(
                x=t_rel,
                y=y_true[:, c],
                mode="lines",
                name="Ground truth",
                line=dict(_MOMENT_LINE_GT),
                connectgaps=False,
            )
        )
        fig_lpf.add_trace(
            go.Scatter(
                x=t_rel,
                y=pred_d[:, c],
                mode="lines",
                name="Direct (IMU → moment)",
                line=dict(_MOMENT_LINE_DIRECT),
            )
        )
        fig_lpf.add_trace(
            go.Scatter(
                x=t_rel,
                y=pred_c[:, c],
                mode="lines",
                name="Cascade (raw)",
                line=dict(_MOMENT_LINE_CASCADE),
            )
        )
        fig_lpf.add_trace(
            go.Scatter(
                x=t_rel,
                y=pred_c_mom_lpf[:, c],
                mode="lines",
                name="Cascade + moment LPF"
                + (
                    f" ({lowpass_cutoff_hz} Hz)"
                    if apply_lowpass_filter
                    else " (off)"
                ),
                line=dict(_MOMENT_LINE_CASCADE_LPF),
            )
        )
        lpf_title_suffix = (
            f"cascade moment LPF {lowpass_cutoff_hz} Hz"
            if apply_lowpass_filter
            else "cascade moment LPF disabled (same as raw)"
        )
        fig_lpf.update_layout(
            title=f"{sid} {condition} {trial} — {eval_side} — {name} — {lpf_title_suffix}",
            xaxis_title="Time (s)",
            yaxis_title=f"{name} (N·m/kg)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0.0),
        )
        fig_lpf.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
        fig_lpf.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
        out_lpf = (
            out_dir
            / f"{sid}_{condition}_{trial}_{eval_side}_{safe}_pipeline_compare_with_cascade_moment_lpf.html"
        )
        fig_lpf.write_html(str(out_lpf), include_plotlyjs="cdn", full_html=True)
        saved_moments += 1

    saved_angles = 0
    for c in range(3):
        name = angle_dof_names[c]
        safe = name.replace("/", "_")
        fig_a = go.Figure()
        fig_a.add_trace(
            go.Scatter(
                x=t_rel,
                y=y_true_ang[:, c],
                mode="lines",
                name="Ground truth (IK)",
                line=dict(width=2),
                connectgaps=False,
            )
        )
        fig_a.add_trace(
            go.Scatter(
                x=t_rel,
                y=pred_ang[:, c],
                mode="lines",
                name="IMU → angle (estimator)",
                line=dict(width=2, dash="dash"),
            )
        )
        fig_a.update_layout(
            title=f"{sid} {condition} {trial} — {eval_side} — {name} (joint angle)",
            xaxis_title="Time (s)",
            yaxis_title=f"{name} (rad)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0.0),
        )
        fig_a.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
        fig_a.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
        out_a = out_dir / f"{sid}_{condition}_{trial}_{eval_side}_{safe}_imu_angles.html"
        fig_a.write_html(str(out_a), include_plotlyjs="cdn", full_html=True)
        saved_angles += 1

    if write_combined_html:
        fig_all = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=dof_names,
        )
        for c in range(3):
            row = c + 1
            fig_all.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=y_true[:, c],
                    mode="lines",
                    name="GT",
                    line=dict(_MOMENT_LINE_GT_SUB),
                    legendgroup="gt",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_all.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=pred_d[:, c],
                    mode="lines",
                    name="Direct",
                    line=dict(_MOMENT_LINE_DIRECT_SUB),
                    legendgroup="dir",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_all.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=pred_c[:, c],
                    mode="lines",
                    name="Cascade",
                    line=dict(_MOMENT_LINE_CASCADE_SUB),
                    legendgroup="cas",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_all.update_yaxes(title_text=f"{dof_names[c]} (N·m/kg)", row=row, col=1)
        fig_all.update_layout(
            height=900,
            title_text=f"{sid} {condition} {trial} — {eval_side} leg — direct vs cascade vs GT",
            template="plotly_white",
            hovermode="x unified",
        )
        fig_all.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig_all.write_html(str(out_dir / f"{sid}_{condition}_{trial}_{eval_side}_all_pipeline_compare.html"), include_plotlyjs="cdn", full_html=True)

        fig_all_lpf = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=dof_names,
        )
        for c in range(3):
            row = c + 1
            fig_all_lpf.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=y_true[:, c],
                    mode="lines",
                    name="GT",
                    line=dict(_MOMENT_LINE_GT_SUB),
                    legendgroup="gt",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_all_lpf.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=pred_d[:, c],
                    mode="lines",
                    name="Direct",
                    line=dict(_MOMENT_LINE_DIRECT_SUB),
                    legendgroup="dir",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_all_lpf.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=pred_c[:, c],
                    mode="lines",
                    name="Cascade (raw)",
                    line=dict(_MOMENT_LINE_CASCADE_SUB),
                    legendgroup="cas",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_all_lpf.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=pred_c_mom_lpf[:, c],
                    mode="lines",
                    name="Cascade + moment LPF",
                    line=dict(_MOMENT_LINE_CASCADE_LPF_SUB),
                    legendgroup="caslpf",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_all_lpf.update_yaxes(title_text=f"{dof_names[c]} (N·m/kg)", row=row, col=1)
        _lpf_sub = (
            f"cascade moment LPF {lowpass_cutoff_hz} Hz"
            if apply_lowpass_filter
            else "cascade moment LPF off"
        )
        fig_all_lpf.update_layout(
            height=960,
            title_text=f"{sid} {condition} {trial} — {eval_side} leg — incl. {_lpf_sub}",
            template="plotly_white",
            hovermode="x unified",
        )
        fig_all_lpf.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig_all_lpf.write_html(
            str(out_dir / f"{sid}_{condition}_{trial}_{eval_side}_all_pipeline_compare_with_cascade_moment_lpf.html"),
            include_plotlyjs="cdn",
            full_html=True,
        )

        fig_ang = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=angle_dof_names,
        )
        for c in range(3):
            row = c + 1
            fig_ang.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=y_true_ang[:, c],
                    mode="lines",
                    name="GT (IK)",
                    line=dict(width=1.8),
                    legendgroup="gt",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_ang.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=pred_ang[:, c],
                    mode="lines",
                    name="IMU → angle",
                    line=dict(width=1.8, dash="dash"),
                    legendgroup="pred",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_ang.update_yaxes(title_text=f"{angle_dof_names[c]} (rad)", row=row, col=1)
        fig_ang.update_layout(
            height=900,
            title_text=f"{sid} {condition} {trial} — {eval_side} leg — IMU joint angle estimator vs IK",
            template="plotly_white",
            hovermode="x unified",
        )
        fig_ang.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig_ang.write_html(
            str(out_dir / f"{sid}_{condition}_{trial}_{eval_side}_all_imu_angles.html"),
            include_plotlyjs="cdn",
            full_html=True,
        )

    with open(out_dir / "inference_manifest.json", "w") as f:
        json.dump(
            {
                **ckpt_paths,
                "h5_dir": str(h5_dir),
                "meta_root": str(meta_root),
                "subject_id": sid,
                "condition": condition,
                "trial": trial,
                "eval_side": eval_side,
                "window_size": window_size,
                "sample_rate_hz": sample_rate_hz,
                "target_sample_rate_hz": target_sample_rate_hz,
                "inference_mode": inference_mode,
                "apply_lowpass_filter": apply_lowpass_filter,
                "lowpass_cutoff_hz": lowpass_cutoff_hz,
                "lowpass_order": lowpass_order,
                "cascade_predicted_angle_lowpass": {
                    "apply": bool(apply_lowpass_filter),
                    "cutoff_hz": float(lowpass_cutoff_hz),
                    "order": int(lowpass_order),
                },
                "cascade_output_moment_lowpass": {
                    "apply": bool(apply_lowpass_filter),
                    "cutoff_hz": float(lowpass_cutoff_hz),
                    "order": int(lowpass_order),
                    "note": "Full-trial zero-phase LPF on cascade N·m/kg trajectories for extra HTML plots.",
                },
                "n_moment_plots": saved_moments,
                "n_angle_plots": saved_angles,
                "angle_dof_names": angle_dof_names,
                "write_combined_html": write_combined_html,
            },
            f,
            indent=2,
        )

    print(
        f"Saved {saved_moments} moment Plotly HTML file(s) (includes per-DOF + cascade moment LPF variants) "
        f"+ {saved_angles} angle (IMU estimator) → {out_dir}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot GT vs direct IMU→moment vs cascade on one trial (see compare_pipeline.py)"
    )
    p.add_argument("--imu-moment-ckpt", type=str, required=True)
    p.add_argument("--imu-angle-ckpt", type=str, required=True)
    p.add_argument("--ik-moment-ckpt", type=str, required=True)
    p.add_argument("--h5-dir", type=str, required=True)
    p.add_argument("--meta-root", type=str, default=None)
    p.add_argument("--subject-id", type=str, required=True)
    p.add_argument("--condition", type=str, required=True)
    p.add_argument("--trial", type=str, default="trial_01")
    p.add_argument("--output-dir", type=str, default="runs/pipeline_compare_trial_plot")
    p.add_argument("--eval-side", type=str, default="right", choices=["right", "left"])
    p.add_argument(
        "--sample-rate-hz",
        type=float,
        default=200.0,
        help="Nominal Hz for metadata when checkpoint has no stored resampling (cascade uses trial timestamps).",
    )
    p.add_argument(
        "--target-sample-rate-hz",
        type=float,
        default=None,
        help="Override trial resampling Hz (default: IMU moment ckpt / config.json).",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--inference-mode", type=str, choices=("causal", "overlap_mean"), default="causal")
    p.add_argument("--write-combined-html", action="store_true")
    p.add_argument("--no-lowpass", action="store_true")
    p.add_argument("--lowpass-cutoff-hz", type=float, default=4.0)
    p.add_argument("--lowpass-order", type=int, default=4)
    args = p.parse_args()

    h5_dir = Path(args.h5_dir)
    meta_root = Path(args.meta_root) if args.meta_root else h5_dir
    device = args.device

    print("Loading IMU→moment …")
    m_direct, ck_m, schema_mr, schema_ml, tgt_m, _, _, w_imu, _, stats_imu = load_imu_checkpoint(
        args.imu_moment_ckpt, device
    )
    if tgt_m != "moment":
        raise ValueError(f"--imu-moment-ckpt must be target=moment, got {tgt_m!r}")

    print("Loading IMU→angle …")
    m_angle, ck_a, schema_ar, schema_al, tgt_a, _, _, w_ang, _, stats_imu_a = load_imu_checkpoint(
        args.imu_angle_ckpt, device
    )
    if tgt_a != "angle":
        raise ValueError(f"--imu-angle-ckpt must be target=angle, got {tgt_a!r}")

    if schema_mr != schema_ar or schema_ml != schema_al:
        raise ValueError("IMU moment and angle checkpoints have different imu_schema_right/left.")
    if not np.allclose(stats_imu["imu_mean"], stats_imu_a["imu_mean"], rtol=1e-5, atol=1e-8):
        raise ValueError("IMU moment/angle checkpoints have different imu_mean.")
    if not np.allclose(stats_imu["imu_std"], stats_imu_a["imu_std"], rtol=1e-5, atol=1e-8):
        raise ValueError("IMU moment/angle checkpoints have different imu_std.")
    if w_imu != w_ang:
        raise ValueError(f"IMU window_size mismatch: moment={w_imu} angle={w_ang}")

    print("Loading IK→moment …")
    input_mode = "unknown"
    output_mode = "unknown"
    try:
        ik_model, ik_stats, _dof_ik, w_ik, input_indices, _mi, input_mode, output_mode, _, _, _ = load_model(
            args.ik_moment_ckpt, device
        )
    except TypeError:
        ck = torch.load(args.ik_moment_ckpt, map_location=device, weights_only=False)
        cfg = ck["model_config"]
        ik_model = TCN(
            n_input_channels=cfg["n_input_channels"],
            n_output_channels=cfg["n_output_channels"],
            hidden_channels=cfg["hidden_channels"],
            n_blocks=cfg["n_blocks"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
        )
        ik_model.load_state_dict(ck["model_state_dict"])
        ik_model.to(device)
        ik_model.eval()
        ik_stats = ck.get("normalization")
        w_ik = int(ck.get("window_size", w_imu))
        input_indices = ck.get("input_indices")
        input_mode = str(ck.get("input_mode", "unknown"))
        output_mode = str(ck.get("output_mode", "unknown"))

    if ik_stats is None:
        raise ValueError("IK moment checkpoint missing normalization stats.")
    ik_stats = _ik_stats_as_numpy(ik_stats)

    if w_ik != w_imu:
        raise ValueError(f"IK window {w_ik} != IMU window {w_imu}")

    n_in = ik_model.n_input_channels
    n_out = ik_model.n_output_channels
    if n_in != 2 * len(SAGITTAL_INPUT_INDICES) or n_out != len(SAGITTAL_INPUT_INDICES):
        raise ValueError(
            f"Expected sagittal 6-DOF IK model (12 in, 6 out); got n_in={n_in}, n_out={n_out} "
            f"(input_mode={input_mode!r} output_mode={output_mode!r})"
        )
    if input_indices is None:
        input_indices = list(SAGITTAL_INPUT_INDICES)
    else:
        input_indices = [int(i) for i in input_indices]
        if input_indices != list(SAGITTAL_INPUT_INDICES):
            raise ValueError(
                f"IK input_indices {input_indices} != sagittal {list(SAGITTAL_INPUT_INDICES)}."
            )

    run_cfg = load_run_config(args.imu_moment_ckpt)
    apply_lp = not args.no_lowpass
    lp_hz = float(args.lowpass_cutoff_hz)
    lp_ord = int(args.lowpass_order)
    if run_cfg is not None and any(
        k in run_cfg for k in ("no_lowpass", "lowpass_cutoff_hz", "lowpass_order")
    ):
        apply_lp = not bool(run_cfg.get("no_lowpass", False))
        lp_hz = float(run_cfg.get("lowpass_cutoff_hz", lp_hz))
        lp_ord = int(run_cfg.get("lowpass_order", lp_ord))

    imu_tgt_sr: Optional[float] = None
    if args.target_sample_rate_hz is not None:
        imu_tgt_sr = float(args.target_sample_rate_hz)
    elif ck_m.get("target_sample_rate_hz") is not None:
        imu_tgt_sr = float(ck_m["target_sample_rate_hz"])
    elif run_cfg is not None and run_cfg.get("target_sample_rate_hz") is not None:
        imu_tgt_sr = float(run_cfg["target_sample_rate_hz"])
    report_sr = float(imu_tgt_sr) if imu_tgt_sr is not None else float(args.sample_rate_hz)

    ckpt_paths = {
        "imu_moment_checkpoint": str(Path(args.imu_moment_ckpt).resolve()),
        "imu_angle_checkpoint": str(Path(args.imu_angle_ckpt).resolve()),
        "ik_moment_checkpoint": str(Path(args.ik_moment_ckpt).resolve()),
    }

    run_single_trial(
        m_direct=m_direct,
        m_angle=m_angle,
        ik_model=ik_model,
        stats_imu=stats_imu,
        ik_stats=ik_stats,
        input_indices=input_indices,
        imu_schema_right=schema_mr,
        imu_schema_left=schema_ml,
        window_size=w_imu,
        h5_dir=h5_dir,
        meta_root=meta_root,
        subject_id=args.subject_id,
        condition=args.condition,
        trial=args.trial,
        eval_side=args.eval_side,
        out_dir=Path(args.output_dir),
        write_combined_html=bool(args.write_combined_html),
        device=device,
        inference_mode=str(args.inference_mode),
        sample_rate_hz=report_sr,
        apply_lowpass_filter=apply_lp,
        lowpass_cutoff_hz=lp_hz,
        lowpass_order=lp_ord,
        ckpt_paths=ckpt_paths,
        target_sample_rate_hz=imu_tgt_sr,
    )


if __name__ == "__main__":
    main()
