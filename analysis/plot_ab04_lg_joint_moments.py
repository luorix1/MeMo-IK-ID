#!/usr/bin/env python3
"""Plot joint moments for AB04_Changseob LG 1.2 m/s (awinda).

Saves an interactive Plotly HTML figure (zoom/pan enabled).

Lines per joint:
  - blue solid   : GT (OpenSim ID from Vicon)
  - red solid    : predicted (IMU IK → checkpoint ID)
  - black dotted : max performance (checkpoint ID with Vicon IK angles/velocities)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.signal import butter, find_peaks, sosfilt, sosfiltfilt

warnings.filterwarnings("ignore", message=".*NumPy.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = Path("/media/metamobility3/Samsung_T52/Results/processed")
IMU_IK_ROOT = Path("/home/metamobility3/Jinwoo/mt_processed")
CHECKPOINT = PROJECT_ROOT / "runs/0512_ik_id_all_zero_in_zero_out/best_model.pt"

SUBJECT = "AB04_Changseob"
CONDITION = "LG_1p2mps"
MASS_KG = 74.0
PKL_PATH = IMU_IK_ROOT / SUBJECT / "ik" / "VQF" / "1p2mps_lg.pkl"
ID_PATH = PROCESSED_ROOT / SUBJECT / "awinda" / "id" / f"{CONDITION}_id.sto"
IK_PATH = PROCESSED_ROOT / SUBJECT / "awinda" / "ik" / f"{CONDITION}_ik.mot"

CHANNEL_LABELS = [
    "Hip flexion (R)",
    "Knee (R)",
    "Ankle (R)",
    "Hip flexion (L)",
    "Knee (L)",
    "Ankle (L)",
]
ID_COLS = [
    "hip_flexion_r_moment",
    "knee_angle_r_moment",
    "ankle_angle_r_moment",
    "hip_flexion_l_moment",
    "knee_angle_l_moment",
    "ankle_angle_l_moment",
]

DEFAULT_FS_HZ = 100.0
IK_ALIGN_MAX_LAG_SEC = 30.0
PEAK_THRESHOLD_DEG = 15.0

LABEL_FONT = 18
TICK_FONT = 14

sys.path.insert(0, str(PROJECT_ROOT))
from dataset import IK_DOF_NAMES  # noqa: E402
from model import TCN  # noqa: E402


def parse_opensim_table(path: Path) -> pd.DataFrame:
    with open(path) as f:
        header_end = next(i for i, line in enumerate(f) if line.strip().lower() == "endheader")
    return pd.read_csv(path, sep=r"\s+", skiprows=header_end + 1).set_index("time")


def butter_lpf(x, fs_hz, cutoff_hz, order, mode="zero_phase"):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    nyq = 0.5 * fs_hz
    if cutoff_hz <= 0 or cutoff_hz >= nyq or len(x) < 4:
        return x.copy()
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, x) if mode == "zero_phase" else sosfilt(sos, x)


def lpf_mc(x, fs_hz, cutoff_hz, order, mode="zero_phase"):
    return np.column_stack(
        [butter_lpf(x[:, c], fs_hz, cutoff_hz, order, mode) for c in range(x.shape[1])]
    )


def _first_peak_idx(sig, fs_hz, threshold_rad, start_offset_s=0.25):
    x = np.asarray(sig, dtype=np.float64).copy()
    valid = np.isfinite(x)
    if valid.sum() < 20:
        raise RuntimeError("Too few finite samples for peak detection")
    x[~valid] = float(np.nanmedian(x[valid]))
    start_idx = min(len(x) - 1, max(0, int(round(start_offset_s * fs_hz))))
    span = float(np.nanmax(x) - np.nanmin(x))
    peaks, _ = find_peaks(
        x,
        prominence=max(np.deg2rad(3.0), 0.10 * span),
        distance=max(1, int(round(0.3 * fs_hz))),
    )
    peaks = peaks[(peaks >= start_idx) & (x[peaks] >= threshold_rad)]
    if len(peaks):
        return int(peaks[0])
    crossing = np.where(x[start_idx:] >= threshold_rad)[0]
    if len(crossing):
        return int(start_idx + crossing[0])
    raise RuntimeError("No right hip flexion peak found")


def estimate_lag_samples(imu_pos, mocap_pos, fs_hz, max_lag_samples, threshold_deg=15.0):
    hip_idx = IK_DOF_NAMES.index("hip_flexion_r")
    thr = np.deg2rad(threshold_deg)
    imu_idx = _first_peak_idx(imu_pos[:, hip_idx], fs_hz, thr)
    mocap_idx = _first_peak_idx(mocap_pos[:, hip_idx], fs_hz, thr)
    lag = int(imu_idx - mocap_idx)
    if lag > max_lag_samples:
        lag = max_lag_samples
    elif lag < -max_lag_samples:
        lag = -max_lag_samples
    return lag


def build_mocap_ik_rad(ik_df: pd.DataFrame) -> np.ndarray:
    pos_deg = np.full((len(ik_df), len(IK_DOF_NAMES)), np.nan)
    for j, name in enumerate(IK_DOF_NAMES):
        if name in ik_df.columns:
            pos_deg[:, j] = ik_df[name].to_numpy(dtype=np.float64)
    return np.deg2rad(pos_deg)


def build_model_input_from_pkl(imu_dict: dict) -> np.ndarray:
    n = len(next(iter(imu_dict.values())))
    pos_deg = np.zeros((n, len(IK_DOF_NAMES)), dtype=np.float64)
    key_map = {
        "hip_flexion_r": "hip_flexion_r",
        "knee_angle_r": "knee_flexion_r",
        "ankle_angle_r": "ankle_flexion_r",
        "hip_flexion_l": "hip_flexion_l",
        "knee_angle_l": "knee_flexion_l",
        "ankle_angle_l": "ankle_flexion_l",
    }
    sign_map = {"knee_angle_r": -1.0, "knee_angle_l": -1.0}
    for ik_name, pkl_name in key_map.items():
        idx = IK_DOF_NAMES.index(ik_name)
        pos_deg[:, idx] = sign_map.get(ik_name, 1.0) * np.asarray(imu_dict[pkl_name], dtype=np.float64)
    return pos_deg


def load_model(device: str):
    ckpt = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    with open(CHECKPOINT.parent / "config.json") as f:
        train_cfg = json.load(f)

    cfg = ckpt["model_config"]
    model = TCN(
        **{k: cfg[k] for k in ["n_input_channels", "n_output_channels", "hidden_channels", "n_blocks", "kernel_size", "dropout"]}
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    window_size = int(ckpt.get("window_size", 100))
    input_indices = list(ckpt.get("input_indices", [6, 9, 10, 13, 16, 17]))
    h = len(input_indices) // 2
    filters = {
        "angle_cutoff": float(train_cfg.get("lowpass_cutoff_hz", 6.0)),
        "vel_cutoff": float(train_cfg.get("velocity_lowpass_cutoff_hz", 15.0)),
        "out_cutoff": float(train_cfg.get("lowpass_cutoff_hz", 6.0)),
        "order": int(train_cfg.get("lowpass_order", 4)),
        "in_mode": str(train_cfg.get("input_lowpass_mode", "zero_phase")),
        "out_mode": str(train_cfg.get("output_lowpass_mode", "zero_phase")),
    }
    return model, window_size, input_indices[:h], input_indices[h:], filters


@torch.no_grad()
def infer_one_side(model, pos_3, vel_3, window_size, device):
    x = np.concatenate([pos_3, vel_3], axis=1).astype(np.float32)
    n, w, c_out = x.shape[0], window_size, pos_3.shape[1]

    def _fwd(start):
        xt = torch.from_numpy(np.ascontiguousarray(x[start : start + w].T)).unsqueeze(0).to(device)
        return model(xt).squeeze(0).detach().cpu().numpy().T

    pred = np.zeros((n, c_out), dtype=np.float64)
    pred[:w] = _fwd(0)
    for start in range(1, n - w + 1):
        pred[start + w - 1] = _fwd(start)[w - 1]
    return pred.astype(np.float32)


def run_bilateral_inference(model, pos_full, vel_full, window_size, idx_r, idx_l, device):
    pr = infer_one_side(model, pos_full[:, idx_r], vel_full[:, idx_r], window_size, device)
    pl = infer_one_side(model, pos_full[:, idx_l], vel_full[:, idx_l], window_size, device)
    return np.concatenate([pr, pl], axis=1)


def _infer_moments(pos_rad, fs_hz, model, window_size, idx_r, idx_l, filters, device):
    pos_f = lpf_mc(pos_rad, fs_hz, filters["angle_cutoff"], filters["order"], filters["in_mode"])
    vel_f = lpf_mc(
        np.gradient(pos_f, 1.0 / fs_hz, axis=0),
        fs_hz,
        filters["vel_cutoff"],
        filters["order"],
        filters["in_mode"],
    )
    pred = run_bilateral_inference(model, pos_f, vel_f, window_size, idx_r, idx_l, device)
    return lpf_mc(pred, fs_hz, filters["out_cutoff"], filters["order"], filters["out_mode"])


def load_trial(device: str):
    for path in (PKL_PATH, ID_PATH, IK_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    model, window_size, idx_r, idx_l, filters = load_model(device)
    imu = pickle.load(open(PKL_PATH, "rb"))
    pos_rad = np.deg2rad(build_model_input_from_pkl(imu))
    id_df = parse_opensim_table(ID_PATH)
    ik_df = parse_opensim_table(IK_PATH)
    t_id = id_df.index.to_numpy(dtype=np.float64)
    fs_hz = 1.0 / float(np.median(np.diff(t_id))) if len(t_id) > 2 else DEFAULT_FS_HZ
    mocap_pos = build_mocap_ik_rad(ik_df)

    imu_align = lpf_mc(pos_rad, fs_hz, filters["angle_cutoff"], filters["order"], filters["in_mode"])
    mocap_align = lpf_mc(mocap_pos, fs_hz, filters["angle_cutoff"], filters["order"], filters["in_mode"])
    lag = estimate_lag_samples(
        imu_align, mocap_align, fs_hz, int(round(IK_ALIGN_MAX_LAG_SEC * fs_hz)), PEAK_THRESHOLD_DEG
    )

    start_imu, start_ref = max(lag, 0), max(-lag, 0)
    n_sync = min(len(pos_rad) - start_imu, len(mocap_pos) - start_ref, len(t_id) - start_ref)
    if n_sync < window_size:
        raise RuntimeError(f"Synced window too short: {n_sync}")

    pos_sync = pos_rad[start_imu : start_imu + n_sync]
    mocap_sync = mocap_pos[start_ref : start_ref + n_sync]
    t_sync = t_id[start_ref : start_ref + n_sync] - t_id[start_ref]

    id_nm = np.column_stack(
        [
            id_df[c].to_numpy(dtype=np.float64) if c in id_df.columns else np.full(len(t_id), np.nan)
            for c in ID_COLS
        ]
    )
    id_nm_sync = id_nm[start_ref : start_ref + n_sync]
    id_nmpkg = lpf_mc(
        id_nm_sync / MASS_KG,
        fs_hz,
        filters["out_cutoff"],
        filters["order"],
        filters["out_mode"],
    )

    pred_nmpkg = _infer_moments(pos_sync, fs_hz, model, window_size, idx_r, idx_l, filters, device)
    oracle_nmpkg = _infer_moments(mocap_sync, fs_hz, model, window_size, idx_r, idx_l, filters, device)

    return {
        "t": t_sync,
        "gt_nmpkg": id_nmpkg,
        "pred_nmpkg": pred_nmpkg,
        "oracle_nmpkg": oracle_nmpkg,
        "lag_samples": lag,
        "fs_hz": fs_hz,
    }


def plot_joint_moments(result: dict, out_path: Path | None):
    t = result["t"]
    gt = result["gt_nmpkg"]
    pred = result["pred_nmpkg"]
    oracle = result["oracle_nmpkg"]

    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
        subplot_titles=CHANNEL_LABELS,
    )

    axis_style = dict(title_font=dict(size=LABEL_FONT), tickfont=dict(size=TICK_FONT))
    subplot_title_style = dict(font=dict(size=TICK_FONT))

    for i, label in enumerate(CHANNEL_LABELS):
        row, col = i // 2 + 1, i % 2 + 1
        show_legend = i == 0
        fig.add_trace(
            go.Scatter(
                x=t,
                y=gt[:, i],
                mode="lines",
                name="GT (Vicon)",
                line=dict(color="#1e88e5", width=2),
                legendgroup="gt",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=pred[:, i],
                mode="lines",
                name="Predicted",
                line=dict(color="#e53935", width=2),
                legendgroup="pred",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=oracle[:, i],
                mode="lines",
                name="Max performance",
                line=dict(color="black", width=2, dash="dot"),
                legendgroup="oracle",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        fig.add_hline(y=0, line_width=0.6, line_dash="dot", line_color="gray", row=row, col=col)
        fig.update_yaxes(title_text="Joint moment (N dot m/kg)", **axis_style, row=row, col=col)

    fig.update_xaxes(title_text="Time (s)", **axis_style, row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", **axis_style, row=3, col=2)
    for ann in fig.layout.annotations:
        ann.font = subplot_title_style["font"]

    fig.update_layout(
        height=1000,
        width=1400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=TICK_FONT)),
        margin=dict(t=80, b=60),
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
        print(f"Saved interactive figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "figures" / "ab04_lg_joint_moments.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    result = load_trial(device)
    print(f"Synced length={len(result['t'])} samples, lag={result['lag_samples']:+d} @ {result['fs_hz']:.1f} Hz")
    plot_joint_moments(result, args.out)


if __name__ == "__main__":
    main()
