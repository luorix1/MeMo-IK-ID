#!/usr/bin/env python3
"""
Single-trial Plotly plots for **paired ipsilateral IK** checkpoints (6→3 sagittal).

Default ``--inference-mode causal`` uses **single-stream** inference from ``compare_pipelineV3``:
one TCN forward per head on the full trial ``(1, C, T)``, then zero-phase LPF along the full timeaxis. That matches **native** causal context from trial start (not isolated training windows).

``--inference-mode overlap_mean`` keeps the slower sliding-window path (per-window LPF, aligned
with batched ``compare_pipelineV3`` / dataloader evaluation).

Outputs (default):
  - Joint angles: GT (IK) vs IMU→angle.
  - Moments: GT vs direct vs cascade.

Optional ``--write-combined-html`` adds a single 6-row figure (angles + moments).

V1 ``plot_pipeline_compare_trial_inference.py`` targets legacy 12→6 IK TCNs.
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

from compare_pipelineV3 import (
    _cascade_pos6_vel6_from_full_ik,
    _ik_moment_tcn_input,
    _lowpass_predicted_angles,
    _lowpass_window_batch,
    _normalize_ik_tcn_input,
    infer_cascade_moments_full_sequence,
    infer_imu_head_full_sequence,
)
from dataset import (
    IK_DOF_NAMES,
    MOMENT_NAMES,
    SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES,
    SAGITTAL_INPUT_INDICES,
    _load_subject_metadata_map,
)
from imu_sagittal.imu_sagittal_eval import load_imu_checkpoint
from imu_sagittal.imu_sagittal_leg_dataset import (
    IMU_UNILATERAL_N_CHANNELS,
    TrialRef,
    _load_trial_imu_sagittal_paired,
)
from ik_id.test import load_model, load_run_config
from model import TCN

_MOMENT_LINE_GT = dict(color="black", width=2)
_MOMENT_LINE_DIRECT = dict(color="#808080", width=2)
_MOMENT_LINE_CASCADE = dict(color="red", width=2)
_MOMENT_LINE_GT_SUB = dict(color="black", width=1.8)
_MOMENT_LINE_DIRECT_SUB = dict(color="#808080", width=1.8)
_MOMENT_LINE_CASCADE_SUB = dict(color="red", width=1.8)


def _ik_stats_as_numpy(ik_stats: dict) -> dict:
    out = dict(ik_stats)
    for k, v in list(out.items()):
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
    return out


def _side_sagittal_ik_indices(full_indices: Sequence[int], eval_side: str) -> List[int]:
    h = len(full_indices) // 2
    if h * 2 != len(full_indices):
        raise ValueError("IK input_indices must split into equal R/L halves.")
    if eval_side == "right":
        return [int(x) for x in full_indices[:h]]
    if eval_side == "left":
        return [int(x) for x in full_indices[h:]]
    raise ValueError(f"eval_side must be 'right' or 'left', got {eval_side!r}")


@torch.no_grad()
def infer_imu_full_trial_pipeline_v3(
    model: torch.nn.Module,
    imu: np.ndarray,
    y_true: np.ndarray,
    imu_mean: np.ndarray,
    imu_std: np.ndarray,
    time_1d: np.ndarray,
    window_size: int,
    device: str,
    *,
    inference_mode: str = "causal",
    pipeline_lpf_apply: bool = True,
    pipeline_lpf_cutoff_hz: float = 4.0,
    pipeline_lpf_order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window IMU→head inference with per-window LPF on outputs (``compare_pipelineV3`` / ``_lowpass_window_batch``).
    """
    if imu.shape[1] != IMU_UNILATERAL_N_CHANNELS:
        raise ValueError(f"Expected IMU width {IMU_UNILATERAL_N_CHANNELS}, got {imu.shape[1]}")
    if y_true.shape[1] != 3:
        raise ValueError(f"Expected 3 target channels, got {y_true.shape[1]}")
    n = imu.shape[0]
    c_out = 3
    if n < window_size:
        raise ValueError(f"Trial too short ({n}) for window_size={window_size}.")
    if len(time_1d) != n:
        raise ValueError("time_1d must match IMU length.")

    imu_mean = np.asarray(imu_mean, dtype=np.float64).reshape(1, -1)
    imu_std = np.asarray(imu_std, dtype=np.float64).reshape(1, -1)
    imu_n = (imu.astype(np.float64) - imu_mean) / imu_std
    tvec = time_1d.astype(np.float32)
    dev = torch.device(device)
    W = int(window_size)

    def _forward_window(start: int) -> np.ndarray:
        end = start + W
        x_w = imu_n[start:end].T
        x_t = torch.from_numpy(x_w.astype(np.float32)).unsqueeze(0).to(dev)
        time_w = torch.from_numpy(tvec[start:end].copy()).unsqueeze(0).to(dev)
        pred_t = model(x_t)
        pred_t = _lowpass_window_batch(
            pred_t,
            time_w,
            apply=pipeline_lpf_apply,
            cutoff_hz=pipeline_lpf_cutoff_hz,
            order=pipeline_lpf_order,
        )
        pred_w = pred_t.squeeze(0).detach().cpu().numpy()
        return pred_w.T

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
        raise ValueError(f"Unknown inference_mode: {inference_mode!r}")

    return pred.astype(np.float32), y_true.astype(np.float32)


@torch.no_grad()
def infer_cascade_full_trial_v2(
    angle_model: torch.nn.Module,
    ik_model: torch.nn.Module,
    imu: np.ndarray,
    positions_full: np.ndarray,
    time_1d: np.ndarray,
    imu_mean: np.ndarray,
    imu_std: np.ndarray,
    ik_stats: dict,
    side_ik_indices: List[int],
    window_size: int,
    device: str,
    eval_side: str,
    *,
    inference_mode: str = "causal",
    pipeline_lpf_apply: bool = True,
    pipeline_lpf_cutoff_hz: float = 4.0,
    pipeline_lpf_order: int = 4,
    ik_input_normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cascade for **paired** IK TCN — same steps as ``compare_pipelineV3``:
    angle LPF → hybrid IK → vel LPF → IK head → moment LPF (per window).

    Returns ``(moments_T3, imu_angle_T3)`` where ``imu_angle_*`` matches a standalone
    IMU→angle head (LPF per window) so callers need not run the angle model twice.
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
        sl6 = slice(0, 3)
    elif es == "left":
        sl6 = slice(3, 6)
    else:
        raise ValueError("eval_side must be 'right' or 'left'")

    pos23 = positions_full.astype(np.float32)
    tvec = time_1d.astype(np.float32)

    def forward_cascade_window(start: int) -> Tuple[np.ndarray, np.ndarray]:
        end = start + W
        x_imu = torch.from_numpy(imu_n[start:end].T.astype(np.float32)).unsqueeze(0).to(device)
        pred_a = angle_model(x_imu)
        pos23_w = torch.from_numpy(pos23[start:end].T.copy()).unsqueeze(0).to(device)
        time_w = torch.from_numpy(tvec[start:end].copy()).unsqueeze(0).to(device)
        pred_a = _lowpass_predicted_angles(
            pred_a,
            time_w,
            apply=pipeline_lpf_apply,
            cutoff_hz=pipeline_lpf_cutoff_hz,
            order=pipeline_lpf_order,
        )
        pos6, vel6 = _cascade_pos6_vel6_from_full_ik(
            pred_a,
            pos23_w,
            time_w,
            es,
            dev,
            vel_lowpass_apply=pipeline_lpf_apply,
            vel_lowpass_cutoff_hz=pipeline_lpf_cutoff_hz,
            vel_lowpass_order=pipeline_lpf_order,
        )
        pos3 = pos6[:, sl6, :]
        vel3 = vel6[:, sl6, :]
        if ik_input_normalize:
            x_ik = _normalize_ik_tcn_input(pos3, vel3, ik_stats, side_ik_indices, dev)
        else:
            x_ik = _ik_moment_tcn_input(pos3, vel3)
        pred_ik = ik_model(x_ik)
        pred_ik = _lowpass_window_batch(
            pred_ik,
            time_w,
            apply=pipeline_lpf_apply,
            cutoff_hz=pipeline_lpf_cutoff_hz,
            order=pipeline_lpf_order,
        )
        pred_m = pred_ik.squeeze(0).detach().cpu().numpy().T.astype(np.float32)
        pred_ang = pred_a.squeeze(0).detach().cpu().numpy().T.astype(np.float32)
        return pred_m, pred_ang

    c_out = 3
    if inference_mode == "overlap_mean":
        pred_sum = np.zeros((n, c_out), dtype=np.float64)
        pred_cnt = np.zeros((n, c_out), dtype=np.float64)
        ang_sum = np.zeros((n, c_out), dtype=np.float64)
        ang_cnt = np.zeros((n, c_out), dtype=np.float64)
        for start in range(0, n - W + 1):
            pred_w, pred_aw = forward_cascade_window(start)
            pred_sum[start : start + W] += pred_w
            pred_cnt[start : start + W] += 1.0
            ang_sum[start : start + W] += pred_aw
            ang_cnt[start : start + W] += 1.0
        pred = pred_sum / np.maximum(pred_cnt, 1.0)
        pred_ang_out = ang_sum / np.maximum(ang_cnt, 1.0)
    elif inference_mode == "causal":
        pred = np.zeros((n, c_out), dtype=np.float64)
        pred_ang_out = np.zeros((n, c_out), dtype=np.float64)
        pw0m, pw0a = forward_cascade_window(0)
        for g in range(W - 1):
            pred[g] = pw0m[g]
            pred_ang_out[g] = pw0a[g]
        for start in range(0, n - W + 1):
            pwm, pwa = forward_cascade_window(start)
            pred[start + W - 1] = pwm[W - 1]
            pred_ang_out[start + W - 1] = pwa[W - 1]
    else:
        raise ValueError(f"Unknown inference_mode: {inference_mode!r}")

    return pred.astype(np.float32), pred_ang_out.astype(np.float32)


def run_single_trial(
    *,
    m_direct: torch.nn.Module,
    m_angle: torch.nn.Module,
    ik_model: torch.nn.Module,
    stats_imu: dict,
    ik_stats: dict,
    side_ik_indices: List[int],
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
    ik_input_normalize: bool,
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
        trim_nonfinite_imu_suffix=True,
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

    if inference_mode == "causal":
        pred_d = infer_imu_head_full_sequence(
            m_direct,
            imu,
            stats_imu["imu_mean"],
            stats_imu["imu_std"],
            time,
            device,
            pipeline_lpf_apply=apply_lowpass_filter,
            pipeline_lpf_cutoff_hz=lowpass_cutoff_hz,
            pipeline_lpf_order=lowpass_order,
        )
        pred_c, pred_ang = infer_cascade_moments_full_sequence(
            m_angle,
            ik_model,
            imu,
            trial_data["positions"],
            trial_data["time"],
            stats_imu["imu_mean"],
            stats_imu["imu_std"],
            ik_stats,
            side_ik_indices,
            device,
            eval_side,
            pipeline_lpf_apply=apply_lowpass_filter,
            pipeline_lpf_cutoff_hz=lowpass_cutoff_hz,
            pipeline_lpf_order=lowpass_order,
            ik_input_normalize=ik_input_normalize,
        )
    else:
        pred_d, _ = infer_imu_full_trial_pipeline_v3(
            m_direct,
            imu,
            y_true,
            stats_imu["imu_mean"],
            stats_imu["imu_std"],
            time,
            window_size,
            device,
            inference_mode=inference_mode,
            pipeline_lpf_apply=apply_lowpass_filter,
            pipeline_lpf_cutoff_hz=lowpass_cutoff_hz,
            pipeline_lpf_order=lowpass_order,
        )
        pred_c, pred_ang = infer_cascade_full_trial_v2(
            m_angle,
            ik_model,
            imu,
            trial_data["positions"],
            trial_data["time"],
            stats_imu["imu_mean"],
            stats_imu["imu_std"],
            ik_stats,
            side_ik_indices,
            window_size,
            device,
            eval_side,
            inference_mode=inference_mode,
            pipeline_lpf_apply=apply_lowpass_filter,
            pipeline_lpf_cutoff_hz=lowpass_cutoff_hz,
            pipeline_lpf_order=lowpass_order,
            ik_input_normalize=ik_input_normalize,
        )

    if eval_side == "right":
        y_true_ang = pos6[:, :3].copy()
        angle_dof_names = [IK_DOF_NAMES[i] for i in SAGITTAL_INPUT_INDICES[:3]]
    else:
        y_true_ang = pos6[:, 3:6].copy()
        angle_dof_names = [IK_DOF_NAMES[i] for i in SAGITTAL_INPUT_INDICES[3:]]

    if eval_side == "right":
        dof_names = [MOMENT_NAMES[i] for i in SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES[:3]]
    else:
        dof_names = [MOMENT_NAMES[i] for i in SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES[3:]]

    _lpf_base = (
        f"pipeline LPF {lowpass_cutoff_hz} Hz, order {lowpass_order}"
        if apply_lowpass_filter
        else "pipeline LPF off"
    )
    if inference_mode == "causal":
        _lpf_note = f"single-stream causal + trial-axis LPF ({_lpf_base})"
    else:
        _lpf_note = f"sliding-window ({_lpf_base})"
    _ang_pred_label = (
        "IMU → angle (full-sequence causal)"
        if inference_mode == "causal"
        else "IMU → angle (LPF per window, V3)"
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
                line=dict(color="black", width=1.8),
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
                name=_ang_pred_label,
                line=dict(width=2, dash="dash", color="#2ecc71"),
                legendgroup="pred",
                showlegend=(c == 0),
            ),
            row=row,
            col=1,
        )
        fig_ang.update_yaxes(title_text=f"{angle_dof_names[c]} (rad)", row=row, col=1)
    fig_ang.update_layout(
        height=900,
        title_text=f"{sid} {condition} {trial} — {eval_side} — joint angles ({_lpf_note}) — compare_pipelineV3",
        template="plotly_white",
        hovermode="x unified",
    )
    fig_ang.update_xaxes(title_text="Time (s)", row=3, col=1)
    out_ang = out_dir / f"{sid}_{condition}_{trial}_{eval_side}_joint_angles_pipeline_v3.html"
    fig_ang.write_html(str(out_ang), include_plotlyjs="cdn", full_html=True)
    saved_angles = 1

    fig_mom = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=dof_names,
    )
    for c in range(3):
        row = c + 1
        fig_mom.add_trace(
            go.Scatter(
                x=t_rel,
                y=y_true[:, c],
                mode="lines",
                name="GT",
                line=dict(_MOMENT_LINE_GT_SUB),
                connectgaps=False,
                legendgroup="gt",
                showlegend=(c == 0),
            ),
            row=row,
            col=1,
        )
        fig_mom.add_trace(
            go.Scatter(
                x=t_rel,
                y=pred_d[:, c],
                mode="lines",
                name="Direct (LPF)",
                line=dict(_MOMENT_LINE_DIRECT_SUB),
                legendgroup="dir",
                showlegend=(c == 0),
            ),
            row=row,
            col=1,
        )
        fig_mom.add_trace(
            go.Scatter(
                x=t_rel,
                y=pred_c[:, c],
                mode="lines",
                name="Cascade (LPF)",
                line=dict(_MOMENT_LINE_CASCADE_SUB),
                legendgroup="cas",
                showlegend=(c == 0),
            ),
            row=row,
            col=1,
        )
        fig_mom.update_yaxes(title_text=f"{dof_names[c]} (N·m/kg)", row=row, col=1)
    fig_mom.update_layout(
        height=900,
        title_text=f"{sid} {condition} {trial} — {eval_side} — moments ({_lpf_note}) — compare_pipelineV3",
        template="plotly_white",
        hovermode="x unified",
    )
    fig_mom.update_xaxes(title_text="Time (s)", row=3, col=1)
    out_mom = out_dir / f"{sid}_{condition}_{trial}_{eval_side}_joint_moments_pipeline_v3.html"
    fig_mom.write_html(str(out_mom), include_plotlyjs="cdn", full_html=True)
    saved_moments = 1

    if write_combined_html:
        fig_all = make_subplots(
            rows=6,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.035,
            subplot_titles=[*angle_dof_names, *dof_names],
        )
        for c in range(3):
            row = c + 1
            fig_all.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=y_true_ang[:, c],
                    mode="lines",
                    name="GT (IK)",
                    line=dict(color="black", width=1.5),
                    legendgroup="agt",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_all.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=pred_ang[:, c],
                    mode="lines",
                    name=_ang_pred_label,
                    line=dict(width=1.5, dash="dash", color="#2ecc71"),
                    legendgroup="a",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_all.update_yaxes(title_text=f"{angle_dof_names[c]} (rad)", row=row, col=1)
        for c in range(3):
            row = c + 4
            fig_all.add_trace(
                go.Scatter(x=t_rel, y=y_true[:, c], mode="lines", name="GT", line=dict(_MOMENT_LINE_GT_SUB), legendgroup="mgt", showlegend=(c == 0)),
                row=row,
                col=1,
            )
            fig_all.add_trace(
                go.Scatter(x=t_rel, y=pred_d[:, c], mode="lines", name="Direct LPF", line=dict(_MOMENT_LINE_DIRECT_SUB), legendgroup="md", showlegend=(c == 0)),
                row=row,
                col=1,
            )
            fig_all.add_trace(
                go.Scatter(x=t_rel, y=pred_c[:, c], mode="lines", name="Cascade LPF", line=dict(_MOMENT_LINE_CASCADE_SUB), legendgroup="mc", showlegend=(c == 0)),
                row=row,
                col=1,
            )
            fig_all.update_yaxes(title_text=f"{dof_names[c]} (N·m/kg)", row=row, col=1)
        fig_all.update_layout(
            height=1500,
            title_text=f"{sid} {condition} {trial} — {eval_side} — angles + moments ({_lpf_note})",
            template="plotly_white",
            hovermode="x unified",
        )
        fig_all.update_xaxes(title_text="Time (s)", row=6, col=1)
        fig_all.write_html(
            str(out_dir / f"{sid}_{condition}_{trial}_{eval_side}_joint_angles_and_moments_pipeline_v3.html"),
            include_plotlyjs="cdn",
            full_html=True,
        )

    with open(out_dir / "inference_manifest.json", "w") as f:
        json.dump(
            {
                **ckpt_paths,
                "pipeline_version": "V3_plot_inference_compare_pipelineV3",
                "inference_reference": "compare_pipelineV3.py",
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
                "neural_implementation": (
                    "full_sequence_causal"
                    if inference_mode == "causal"
                    else "sliding_window"
                ),
                "apply_lowpass_filter": apply_lowpass_filter,
                "lowpass_cutoff_hz": lowpass_cutoff_hz,
                "lowpass_order": lowpass_order,
                "ik_input_normalize": ik_input_normalize,
                "pipeline_zero_phase_lowpass": {
                    "apply": bool(apply_lowpass_filter),
                    "cutoff_hz": float(lowpass_cutoff_hz),
                    "order": int(lowpass_order),
                    "imu_on_load_trial": True,
                    "imu_moment_head_per_window": True,
                    "imu_angle_head_per_window": True,
                    "cascade_predicted_angles_per_window": True,
                    "cascade_angular_velocity_after_hybrid_ik": True,
                    "cascade_moment_head_per_window": True,
                },
                "n_combined_moment_figures": saved_moments,
                "n_combined_angle_figures": saved_angles,
                "angle_dof_names": angle_dof_names,
                "moment_dof_names": dof_names,
                "outputs": {
                    "joint_angles": str(out_ang.name),
                    "joint_moments": str(out_mom.name),
                    "combined_angles_and_moments": (
                        f"{sid}_{condition}_{trial}_{eval_side}_joint_angles_and_moments_pipeline_v3.html"
                        if write_combined_html
                        else None
                    ),
                },
                "write_combined_html": write_combined_html,
            },
            f,
            indent=2,
        )

    print(
        f"Saved angles: {out_ang.name} | moments: {out_mom.name} → {out_dir}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Trial plots: paired 6→3 IK, inference aligned with compare_pipelineV3 (LPF on IMU heads + cascade)"
    )
    p.add_argument("--imu-moment-ckpt", type=str, required=True)
    p.add_argument("--imu-angle-ckpt", type=str, required=True)
    p.add_argument("--ik-moment-ckpt", type=str, required=True)
    p.add_argument("--h5-dir", type=str, required=True)
    p.add_argument("--meta-root", type=str, default=None)
    p.add_argument("--subject-id", type=str, required=True)
    p.add_argument("--condition", type=str, required=True)
    p.add_argument("--trial", type=str, default="trial_01")
    p.add_argument("--output-dir", type=str, default="runs/pipeline_compare_trial_plot_v3")
    p.add_argument("--eval-side", type=str, default="right", choices=["right", "left"])
    p.add_argument(
        "--sample-rate-hz",
        type=float,
        default=200.0,
        help="Nominal Hz in manifest when checkpoint has no stored resampling.",
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
    p.add_argument(
        "--ik-input-normalize",
        action="store_true",
        default=False,
        help="Z-score IK TCN inputs (only if IK was trained with normalized inputs).",
    )
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

    print("Loading IK→moment (paired ipsilateral) …")
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
    n_sag = len(SAGITTAL_INPUT_INDICES)
    half = n_sag // 2
    if n_in != 2 * half or n_out != half:
        raise ValueError(
            f"This script expects paired sagittal IK (n_in={2*half}, n_out={half}); got n_in={n_in}, n_out={n_out} "
            f"(input_mode={input_mode!r} output_mode={output_mode!r}). Use plot_pipeline_compare_trial_inference.py for 12→6."
        )
    if input_indices is None:
        full_input_indices = list(SAGITTAL_INPUT_INDICES)
    else:
        full_input_indices = [int(i) for i in input_indices]
        if full_input_indices != list(SAGITTAL_INPUT_INDICES):
            raise ValueError(
                f"IK input_indices {full_input_indices} != sagittal {list(SAGITTAL_INPUT_INDICES)}."
            )

    side_ik_indices = _side_sagittal_ik_indices(full_input_indices, args.eval_side)

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
        side_ik_indices=side_ik_indices,
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
        ik_input_normalize=bool(args.ik_input_normalize),
        target_sample_rate_hz=imu_tgt_sr,
    )


if __name__ == "__main__":
    main()
