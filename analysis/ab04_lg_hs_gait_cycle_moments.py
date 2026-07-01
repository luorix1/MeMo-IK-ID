#!/usr/bin/env python3
"""HS gait-cycle segmentation for AB04_Changseob LG 1.2 m/s (awinda).

Loads IMU IK, runs checkpoint ID inference (same pipeline as
process_awinda.ipynb), detects heel strikes on mocap IK,
and plots GT vs predicted mean ± std joint moments over 0–100% gait cycle.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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

CHANNELS = [
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
]
ID_COLS = [f"{c}_moment" for c in CHANNELS]
CHANNEL_LABELS = [
    "Hip flexion (R)",
    "Knee (R)",
    "Ankle (R)",
    "Hip flexion (L)",
    "Knee (L)",
    "Ankle (L)",
]

DEFAULT_FS_HZ = 100.0
IK_ALIGN_MAX_LAG_SEC = 30.0
PEAK_THRESHOLD_DEG = 15.0
N_GAIT_PTS = 101

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


# --- HS detection (from compare_jinwoo_mtw_ik_vs_opensim_id.ipynb) ---


def _interp_finite_1d(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    if good.all():
        return x.copy()
    if good.sum() == 1:
        return np.full_like(x, float(x[good][0]), dtype=float)
    idx = np.arange(x.size, dtype=float)
    return np.interp(idx, idx[good], x[good])


def lowpass(x, fs, cutoff=6, order=4):
    x = _interp_finite_1d(x)
    if x.size < max(8, order * 3) or not np.isfinite(fs) or fs <= 0:
        return x.copy()
    nyq = 0.5 * float(fs)
    cutoff_eff = float(np.clip(cutoff, 0.1, max(0.11, nyq * 0.95)))
    wn = cutoff_eff / nyq
    if wn >= 1.0:
        return x.copy()
    sos = butter(order, wn, btype="low", output="sos")
    try:
        return sosfiltfilt(sos, x)
    except ValueError:
        return x.copy()


def _first_zero_crossing_after_peak(sig, peak_idx, forward_samples):
    sig = np.asarray(sig, dtype=float)
    n = int(sig.size)
    p = int(peak_idx)
    end = min(p + 1 + max(1, int(forward_samples)), n)
    for i in range(p + 1, end):
        a, b = float(sig[i - 1]), float(sig[i])
        if a * b < 0.0 or (b == 0.0 and a != 0.0):
            return i
    return p


def _enforce_event_spacing(events, quality_signal, min_distance):
    events = np.asarray(events, dtype=int)
    if events.size == 0:
        return events
    kept = [int(events[0])]
    for idx in events[1:]:
        idx = int(idx)
        if idx - kept[-1] < int(min_distance):
            old = kept[-1]
            q_new = float(quality_signal[idx]) if 0 <= idx < len(quality_signal) else -np.inf
            q_old = float(quality_signal[old]) if 0 <= old < len(quality_signal) else -np.inf
            if q_new > q_old:
                kept[-1] = idx
            continue
        kept.append(idx)
    return np.asarray(kept, dtype=int)


def _detect_hs_from_rate_peak_zero_crossing(rate_signal, fs, min_stride_time=0.45, refine_forward_ms=200.0):
    rate = _interp_finite_1d(rate_signal)
    n = int(rate.size)
    if n < 8 or (not np.isfinite(fs)) or fs <= 0:
        return np.asarray([], dtype=int)

    fs = float(fs)
    distance = max(1, int(round(float(min_stride_time) * fs)))
    centered = rate - float(np.nanmedian(rate))
    p10, p50, p90 = np.nanpercentile(centered, [10, 50, 90])
    amp = float(max(1e-6, p90 - p10))
    peak_height = max(1e-6, p50 + 0.15 * amp)
    prominence = max(1e-6, 0.10 * amp)

    peaks, _ = find_peaks(centered, height=float(peak_height), prominence=float(prominence), distance=distance)
    if peaks.size < 2:
        peaks, _ = find_peaks(
            centered,
            height=float(peak_height),
            prominence=max(float(prominence) * 0.5, 1e-6),
            distance=distance,
        )
    if peaks.size < 2:
        peaks, _ = find_peaks(centered, prominence=max(float(prominence) * 0.35, 1e-6), distance=distance)
    if peaks.size < 2:
        return np.asarray([], dtype=int)

    forward_samples = max(2, int(np.ceil(float(refine_forward_ms) / 1000.0 * fs)))
    refined = []
    for p in peaks:
        r = int(_first_zero_crossing_after_peak(centered, int(p), forward_samples))
        if refined and r <= refined[-1]:
            continue
        refined.append(r)
    if len(refined) < 2:
        return np.asarray([], dtype=int)
    return _enforce_event_spacing(np.asarray(refined, dtype=int), centered, distance)


def _fallback_hs_from_knee_minima(knee_f, fs, min_stride_time=0.45, max_stride_time=1.5):
    fs = float(fs)
    min_stride = max(2, int(round(min_stride_time * fs)))
    knee_p10, knee_p90 = np.nanpercentile(knee_f, [10, 90])
    knee_amp = float(max(1.0, knee_p90 - knee_p10))
    prom = max(1.0, 0.10 * knee_amp)
    minima, _ = find_peaks(-knee_f, distance=max(1, int(round(0.30 * fs))), prominence=max(0.8, 0.50 * prom))
    if minima.size == 0:
        return np.asarray([], dtype=int)

    hs_events = _enforce_event_spacing(minima, -knee_f, min_stride)
    if hs_events.size >= 3:
        stride_s = np.diff(hs_events) / fs
        med = float(np.median(stride_s))
        lo, hi = max(min_stride_time, 0.60 * med), min(max_stride_time, 1.80 * med)
        hs_refined = [int(hs_events[0])]
        for idx in hs_events[1:]:
            dt = (int(idx) - hs_refined[-1]) / fs
            if lo <= dt <= hi:
                hs_refined.append(int(idx))
        hs_events = np.asarray(hs_refined, dtype=int)
    return hs_events


def detect_hs_knee_ankle(knee_angle, ankle_angle, fs, min_stride_time=0.45, max_stride_time=1.5):
    knee = _interp_finite_1d(knee_angle)
    ankle = _interp_finite_1d(ankle_angle)
    n = int(min(knee.size, ankle.size))
    if n < 8 or (not np.isfinite(fs)) or fs <= 0:
        return np.asarray([], dtype=int)

    fs = float(fs)
    knee, ankle = knee[:n], ankle[:n]
    cutoff_hz = min(8.0, max(4.0, 0.08 * fs))
    knee_f = lowpass(knee, fs, cutoff=cutoff_hz)
    ankle_f = lowpass(ankle, fs, cutoff=cutoff_hz)
    ankle_rate = np.gradient(ankle_f) * fs
    ankle_rate = lowpass(ankle_rate, fs, cutoff=min(10.0, max(5.0, 0.10 * fs)))

    hs_primary = _detect_hs_from_rate_peak_zero_crossing(
        ankle_rate, fs, min_stride_time=min_stride_time, refine_forward_ms=200.0
    )
    if hs_primary.size >= 2:
        return hs_primary
    return _fallback_hs_from_knee_minima(knee_f, fs, min_stride_time=min_stride_time, max_stride_time=max_stride_time)


def extract_hs_cycles(signals: np.ndarray, hs_events: np.ndarray, fs: float, n_pts: int = N_GAIT_PTS):
    """Resample each HS→HS cycle to 0–100% for all channels."""
    pct = np.linspace(0.0, 100.0, n_pts)
    gt_cycles, pred_cycles = [], []
    hs_events = np.asarray(hs_events, dtype=int)
    for i in range(len(hs_events) - 1):
        start, end = int(hs_events[i]), int(hs_events[i + 1])
        if end <= start:
            continue
        dur = (end - start) / fs
        if dur < 0.40 or dur > 1.5:
            continue
        seg = signals[start:end]
        if seg.shape[0] < 4 or not np.all(np.isfinite(seg)):
            continue
        src_pct = np.linspace(0.0, 100.0, seg.shape[0])
        resampled = np.column_stack([np.interp(pct, src_pct, seg[:, c]) for c in range(seg.shape[1])])
        gt_cycles.append(resampled[:, :6])
        pred_cycles.append(resampled[:, 6:])
    return np.asarray(gt_cycles), np.asarray(pred_cycles), pct


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
    return model, cfg, window_size, input_indices[:h], input_indices[h:], filters


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


def load_trial(device: str):
    for path in (PKL_PATH, ID_PATH, IK_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    model, cfg, window_size, idx_r, idx_l, filters = load_model(device)
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
    id_nm = np.column_stack(
        [
            id_df[c].to_numpy(dtype=np.float64) if c in id_df.columns else np.full(len(t_id), np.nan)
            for c in ID_COLS
        ]
    )
    id_nm_sync = id_nm[start_ref : start_ref + n_sync]
    ik_sync = ik_df.iloc[start_ref : start_ref + n_sync]

    pos_f = lpf_mc(pos_sync, fs_hz, filters["angle_cutoff"], filters["order"], filters["in_mode"])
    vel_f = lpf_mc(np.gradient(pos_f, 1.0 / fs_hz, axis=0), fs_hz, filters["vel_cutoff"], filters["order"], filters["in_mode"])
    pred_nmpkg = run_bilateral_inference(model, pos_f, vel_f, window_size, idx_r, idx_l, device)
    pred_nmpkg_f = lpf_mc(pred_nmpkg, fs_hz, filters["out_cutoff"], filters["order"], filters["out_mode"])
    id_nmpkg_f = lpf_mc(id_nm_sync / MASS_KG, fs_hz, filters["out_cutoff"], filters["order"], filters["out_mode"])

    knee_r = ik_sync["knee_angle_r"].to_numpy(dtype=np.float64)
    ankle_r = ik_sync["ankle_angle_r"].to_numpy(dtype=np.float64)
    hs_events = detect_hs_knee_ankle(knee_r, ankle_r, fs_hz)
    if hs_events.size < 3:
        raise RuntimeError(f"Too few HS events detected: {hs_events.size}")

    combined = np.concatenate([id_nmpkg_f, pred_nmpkg_f], axis=1)
    gt_cycles, pred_cycles, pct = extract_hs_cycles(combined, hs_events, fs_hz)
    if gt_cycles.size == 0:
        raise RuntimeError("No valid gait cycles after HS segmentation")

    return {
        "fs_hz": fs_hz,
        "lag_samples": lag,
        "n_hs": int(hs_events.size),
        "n_cycles": int(gt_cycles.shape[0]),
        "pct": pct,
        "gt_mean": np.nanmean(gt_cycles, axis=0),
        "gt_std": np.nanstd(gt_cycles, axis=0),
        "pred_mean": np.nanmean(pred_cycles, axis=0),
        "pred_std": np.nanstd(pred_cycles, axis=0),
    }


def plot_gait_cycles(result: dict, out_path: Path | None, show: bool):
    pct = result["pct"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axes = axes.ravel()

    for i, (ax, label) in enumerate(zip(axes, CHANNEL_LABELS)):
        gt_m, gt_s = result["gt_mean"][:, i], result["gt_std"][:, i]
        pr_m, pr_s = result["pred_mean"][:, i], result["pred_std"][:, i]
        ax.plot(pct, gt_m, color="#1f77b4", lw=2, label="GT mean")
        ax.fill_between(pct, gt_m - gt_s, gt_m + gt_s, color="#1f77b4", alpha=0.25, label="GT ± std")
        ax.plot(pct, pr_m, color="#d62728", lw=2, label="Pred mean")
        ax.fill_between(pct, pr_m - pr_s, pr_m + pr_s, color="#d62728", alpha=0.25, label="Pred ± std")
        ax.axvline(0, color="0.75", lw=0.8)
        ax.set_title(label)
        ax.set_ylabel("Moment (N·m/kg)")
        ax.grid(True, alpha=0.3)

    axes[3].set_xlabel("Gait cycle (%)")
    axes[4].set_xlabel("Gait cycle (%)")
    axes[5].set_xlabel("Gait cycle (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"{SUBJECT} {CONDITION} — HS gait cycles (n={result['n_cycles']}, lag={result['lag_samples']:+d} samples)",
        y=1.06,
        fontsize=12,
    )
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "figures" / "ab04_lg_hs_gait_cycle_moments.png",
        help="Output PNG path",
    )
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    result = load_trial(device)
    print(
        f"HS events={result['n_hs']}, valid cycles={result['n_cycles']}, "
        f"lag={result['lag_samples']:+d} samples @ {result['fs_hz']:.1f} Hz"
    )
    plot_gait_cycles(result, args.out, args.show)


if __name__ == "__main__":
    main()
