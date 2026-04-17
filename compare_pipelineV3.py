#!/usr/bin/env python3
"""
**compare_pipelineV3** — paired ipsilateral IK (**6→3** sagittal) direct vs cascade comparison with **full zero-phase Butterworth pipeline** aligned to ``ImuSagittalH5Dataset`` / IMU run config:

- IMU streams: LPF on load (``imu_sagittal.imu_sagittal_leg_dataset``).
- Ground-truth IK angles / ID moments: dataset denoising when enabled.
- **Cascade:** IMU angles → LPF → hybrid 23-DOF pose → ``_compute_velocity`` → **LPF on ω** →
  ipsilateral 3 pos + 3 vel → IK TCN.
- **Metrics:** zero-phase LPF on **direct** and **cascade** moment predictions (per window), same
  cutoff/order as above when LPF is on.

This module is **self-contained** (no import from ``compare_pipeline.py``); it only uses ``dataset``,
``imu_sagittal`` package, ``model``, and ``ik_id.test``.

For legacy **12→6** IK checkpoints use ``compare_pipeline.py``. For the older paired script without
moment/ω post-filtering, use ``compare_pipelineV2.py``.

Example::

    python compare_pipelineV3.py \\
        --imu-moment-ckpt runs/imu_moments/best_model.pt \\
        --imu-angle-ckpt runs/imu_angles/best_model.pt \\
        --ik-moment-ckpt runs/0411_jinwoo_3_EPIC/best_model.pt \\
        --test-dir /path/to/Processed/Jinwoo \\
        --output-dir results/pipeline_compare_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from dataset import (
    SAGITTAL_INPUT_INDICES,
    _compute_velocity,
    _lowpass_zero_phase,
)
from ik_id.test import load_model, load_run_config, load_subject_split, resolve_dataset_stride
from imu_sagittal.imu_sagittal_dataset import ImuSagittalH5Dataset
from imu_sagittal.imu_sagittal_eval import load_imu_checkpoint, set_global_seed
from model import TCN


def _normalize_ik_tcn_input(
    pos_rad: torch.Tensor,
    vel_rad_s: torch.Tensor,
    stats: Dict[str, np.ndarray],
    input_indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Z-score IK inputs using checkpoint ``stats`` (``normalize=True`` training only)."""
    pm = torch.as_tensor(stats["pos_mean"], device=device, dtype=torch.float32)
    ps = torch.as_tensor(stats["pos_std"], device=device, dtype=torch.float32)
    vm = torch.as_tensor(stats["vel_mean"], device=device, dtype=torch.float32)
    vs = torch.as_tensor(stats["vel_std"], device=device, dtype=torch.float32)
    idx = torch.as_tensor(list(input_indices), device=device, dtype=torch.long)

    pos_n = (pos_rad - pm[idx].view(1, -1, 1)) / ps[idx].view(1, -1, 1)
    vel_n = (vel_rad_s - vm[idx].view(1, -1, 1)) / vs[idx].view(1, -1, 1)
    return torch.cat([pos_n, vel_n], dim=1)


def _ik_moment_tcn_input(pos6: torch.Tensor, vel6: torch.Tensor) -> torch.Tensor:
    """Concatenate pos‖vel → (B, 2·N, W) like ``KineticsTCNDataset`` with ``normalize=False``."""
    return torch.cat([pos6, vel6], dim=1)


def _lowpass_window_batch(
    x: torch.Tensor,
    time_w: torch.Tensor,
    *,
    apply: bool,
    cutoff_hz: float,
    order: int,
) -> torch.Tensor:
    """Zero-phase Butterworth per batch row on (B, C, W) along time (B, W) timestamps."""
    if not apply or cutoff_hz <= 0:
        return x
    device = x.device
    dtype = x.dtype
    B, _C, _W = x.shape
    xa = x.detach().cpu().numpy().astype(np.float64)
    tw = time_w.detach().cpu().numpy().astype(np.float64)
    out = np.empty_like(xa)
    for b in range(B):
        x_tw = xa[b].T
        t = tw[b]
        xf = _lowpass_zero_phase(x_tw, t, cutoff_hz=float(cutoff_hz), order=int(order))
        out[b] = xf.T
    return torch.as_tensor(out, device=device, dtype=dtype)


def _lowpass_predicted_angles(
    pred_a: torch.Tensor,
    time_w: torch.Tensor,
    *,
    apply: bool,
    cutoff_hz: float,
    order: int,
) -> torch.Tensor:
    return _lowpass_window_batch(
        pred_a, time_w, apply=apply, cutoff_hz=cutoff_hz, order=order
    )


@torch.no_grad()
def infer_imu_head_full_sequence(
    model: torch.nn.Module,
    imu: np.ndarray,
    imu_mean: np.ndarray,
    imu_std: np.ndarray,
    time_1d: np.ndarray,
    device: str,
    *,
    pipeline_lpf_apply: bool = True,
    pipeline_lpf_cutoff_hz: float = 4.0,
    pipeline_lpf_order: int = 4,
) -> np.ndarray:
    """
    **Single-stream causal inference:** one TCN forward on ``(1, C_in, T)`` plus optional
    zero-phase Butterworth on the **full** time axis (one SciPy pass per channel).

    The causal stack sees IMU from the beginning of the trial at every frame. That differs from
    sliding-window evaluation where each window only sees ``window_size`` frames — use windowed
    helpers when you need an exact match to batched training windows.
    """
    if imu.ndim != 2:
        raise ValueError(f"imu must be (T, C), got {imu.shape}")
    n = imu.shape[0]
    if len(time_1d) != n:
        raise ValueError("time_1d length must match IMU length.")
    imu_mean = np.asarray(imu_mean, dtype=np.float64).reshape(1, -1)
    imu_std = np.asarray(imu_std, dtype=np.float64).reshape(1, -1)
    imu_n = (imu.astype(np.float64) - imu_mean) / imu_std
    dev = torch.device(device)
    x = torch.from_numpy(imu_n.T.astype(np.float32)).unsqueeze(0).to(dev)
    time_b = torch.from_numpy(time_1d.astype(np.float32)).unsqueeze(0).to(dev)
    pred = model(x)
    pred = _lowpass_window_batch(
        pred,
        time_b,
        apply=pipeline_lpf_apply,
        cutoff_hz=pipeline_lpf_cutoff_hz,
        order=pipeline_lpf_order,
    )
    return pred.squeeze(0).detach().cpu().numpy().T.astype(np.float32)


@torch.no_grad()
def infer_cascade_moments_full_sequence(
    angle_model: torch.nn.Module,
    ik_model: torch.nn.Module,
    imu: np.ndarray,
    positions_full: np.ndarray,
    time_1d: np.ndarray,
    imu_mean: np.ndarray,
    imu_std: np.ndarray,
    ik_stats: Dict[str, np.ndarray],
    side_ik_indices: List[int],
    device: str,
    eval_side: str,
    *,
    pipeline_lpf_apply: bool = True,
    pipeline_lpf_cutoff_hz: float = 4.0,
    pipeline_lpf_order: int = 4,
    ik_input_normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    **Single-stream causal cascade:** one forward each for IMU→angle and IK→moment TCNs on
    full-trial tensors, sharing the same hybrid IK / velocity / LPF path as the batched V3
    pipeline. Returns ``(moments (T, 3), imu_sagittal_angles (T, 3))``.
    """
    n = imu.shape[0]
    if positions_full.shape != (n, 23):
        raise ValueError(f"positions_full must be (T, 23), got {positions_full.shape}")
    if len(time_1d) != n:
        raise ValueError("time_1d length must match IMU length.")
    es = eval_side.lower()
    if es == "right":
        sl6 = slice(0, 3)
    elif es == "left":
        sl6 = slice(3, 6)
    else:
        raise ValueError("eval_side must be 'right' or 'left'")

    imu_mean = np.asarray(imu_mean, dtype=np.float64).reshape(1, -1)
    imu_std = np.asarray(imu_std, dtype=np.float64).reshape(1, -1)
    imu_n = (imu.astype(np.float64) - imu_mean) / imu_std
    dev = torch.device(device)
    pos23 = positions_full.astype(np.float32)
    x_imu = torch.from_numpy(imu_n.T.astype(np.float32)).unsqueeze(0).to(device)
    time_b = torch.from_numpy(time_1d.astype(np.float32)).unsqueeze(0).to(device)
    pos23_b = torch.from_numpy(pos23.T.copy()).unsqueeze(0).to(device)

    pred_a = angle_model(x_imu)
    pred_a = _lowpass_predicted_angles(
        pred_a,
        time_b,
        apply=pipeline_lpf_apply,
        cutoff_hz=pipeline_lpf_cutoff_hz,
        order=pipeline_lpf_order,
    )
    pos6, vel6 = _cascade_pos6_vel6_from_full_ik(
        pred_a,
        pos23_b,
        time_b,
        eval_side,
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
    pred_c = ik_model(x_ik)
    pred_c = _lowpass_window_batch(
        pred_c,
        time_b,
        apply=pipeline_lpf_apply,
        cutoff_hz=pipeline_lpf_cutoff_hz,
        order=pipeline_lpf_order,
    )
    pred_m = pred_c.squeeze(0).detach().cpu().numpy().T.astype(np.float32)
    pred_ang = pred_a.squeeze(0).detach().cpu().numpy().T.astype(np.float32)
    return pred_m, pred_ang


def _cascade_pos6_vel6_from_full_ik(
    pred_a: torch.Tensor,
    pos23_gt: torch.Tensor,
    time_w: torch.Tensor,
    eval_side: str,
    device: torch.device,
    *,
    vel_lowpass_apply: bool = False,
    vel_lowpass_cutoff_hz: float = 4.0,
    vel_lowpass_order: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hybrid 23-DOF positions (GT + predicted eval-leg sagittal angles), ``_compute_velocity``,
    optional zero-phase LPF on ω, then sagittal (B, 6, W) pos and vel.
    """
    if eval_side == "right":
        idx3 = tuple(SAGITTAL_INPUT_INDICES[:3])
    elif eval_side == "left":
        idx3 = tuple(SAGITTAL_INPUT_INDICES[3:])
    else:
        raise ValueError(f"eval_side must be 'right' or 'left', got {eval_side!r}")

    idx6 = list(SAGITTAL_INPUT_INDICES)
    B, n_dof, W = pos23_gt.shape
    if n_dof != 23:
        raise ValueError(f"pos23_gt must be (B, 23, W), got {pos23_gt.shape}")
    if pred_a.shape != (B, 3, W):
        raise ValueError(f"pred_a must be (B, 3, W), got {pred_a.shape}")

    pos23g = pos23_gt.detach().cpu().numpy().astype(np.float64)
    pa = pred_a.detach().cpu().numpy().astype(np.float64)
    time_np = time_w.detach().cpu().numpy().astype(np.float64)

    pos6 = np.zeros((B, 6, W), dtype=np.float64)
    vel6 = np.zeros((B, 6, W), dtype=np.float64)

    for b in range(B):
        p = pos23g[b].copy()
        for k, ik_i in enumerate(idx3):
            p[ik_i, :] = pa[b, k, :]
        pos_tw = p.T
        t = time_np[b]
        vel_tw = _compute_velocity(pos_tw, t)
        if vel_lowpass_apply and vel_lowpass_cutoff_hz > 0:
            vel_tw = _lowpass_zero_phase(
                vel_tw,
                t,
                cutoff_hz=float(vel_lowpass_cutoff_hz),
                order=int(vel_lowpass_order),
            )
        vel = vel_tw.T
        for j_out, ik_j in enumerate(idx6):
            pos6[b, j_out, :] = p[ik_j, :]
            vel6[b, j_out, :] = vel[ik_j, :]

    pos6_t = torch.from_numpy(pos6).to(device=device, dtype=torch.float32)
    vel6_t = torch.from_numpy(vel6).to(device=device, dtype=torch.float32)
    return pos6_t, vel6_t


def _resolve_eval_subjects(
    test_dir: Path,
    split_checkpoint: Path,
    eval_split: str,
    max_files: Optional[int],
) -> Tuple[List[str], str]:
    h5_subject_files = sorted([p for p in test_dir.glob("S*.h5") if p.is_file()])
    if not h5_subject_files:
        raise ValueError(f"No S*.h5 under {test_dir}")
    split = load_subject_split(str(split_checkpoint))
    subjects_in_dir = sorted([p.stem.upper() for p in h5_subject_files])
    mode = "independent"
    eval_ids: List[str]

    if split is None:
        eval_ids = subjects_in_dir
    else:
        train_s = set(split.get("train_subjects", []))
        val_s = set(split.get("val_subjects", []))
        test_s = set(split.get("test_subjects", []))
        all_s = train_s | val_s | test_s
        if eval_split == "test":
            keep = test_s if test_s else val_s
        else:
            keep = val_s
        overlap = set(subjects_in_dir) & all_s
        if not overlap:
            eval_ids = subjects_in_dir
            mode = "independent"
        else:
            eval_ids = sorted(set(subjects_in_dir) & keep)
            mode = eval_split

    if max_files is not None:
        eval_ids = eval_ids[: max_files]
    if not eval_ids:
        raise ValueError("No subjects to evaluate after split filter.")
    return eval_ids, mode


def _plot_rmse_comparison(
    dof_names: List[str],
    rmse_direct: List[float],
    rmse_cascade: List[float],
    out_path: Path,
) -> None:
    if not HAS_MPL:
        return
    x = np.arange(len(dof_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(dof_names) * 0.65), 5))
    ax.bar(x - w / 2, rmse_direct, w, label="IMU → moment (direct)", color="#3498DB")
    ax.bar(x + w / 2, rmse_cascade, w, label="IMU → angle → IK → moment", color="#E67E22")
    ax.set_xticks(x)
    ax.set_xticklabels(dof_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (N·m/kg)")
    ax.set_title("Sagittal moment error: direct vs cascade (V3)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _expected_side_ik_indices(full_indices: Sequence[int], eval_side: str) -> List[int]:
    h = len(full_indices) // 2
    if h * 2 != len(full_indices):
        raise ValueError("IK input_indices must split into equal R/L halves.")
    if eval_side == "right":
        return [int(x) for x in full_indices[:h]]
    if eval_side == "left":
        return [int(x) for x in full_indices[h:]]
    raise ValueError(f"eval_side must be 'right' or 'left', got {eval_side!r}")


def _streaming_two_preds_metrics_v3(
    direct_model: torch.nn.Module,
    angle_model: torch.nn.Module,
    ik_model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    ik_stats: Dict[str, np.ndarray],
    ik_input_indices: List[int],
    side_ik_indices: List[int],
    dof_names: List[str],
    *,
    sagittal6_slice: slice,
    eval_side: str,
    pipeline_lpf_apply: bool,
    pipeline_lpf_cutoff_hz: float,
    pipeline_lpf_order: int,
    ik_input_normalize: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Paired IK: (B,6,W) pos+vel ipsilateral → (B,3,W) moments; full pipeline LPF when enabled."""
    n_ch = len(dof_names)
    sum_sq_d = np.zeros(n_ch, dtype=np.float64)
    sum_sq_c = np.zeros(n_ch, dtype=np.float64)
    sum_abs_d = np.zeros(n_ch, dtype=np.float64)
    sum_abs_c = np.zeros(n_ch, dtype=np.float64)
    sum_t = np.zeros(n_ch, dtype=np.float64)
    sum_t2 = np.zeros(n_ch, dtype=np.float64)
    n_elem = 0

    sum_sq_all_d = sum_sq_all_c = 0.0
    sum_abs_all_d = sum_abs_all_c = 0.0
    sum_t_all = sum_t2_all = 0.0
    n_all = 0

    dev = torch.device(device)
    sl6 = sagittal6_slice

    with torch.no_grad():
        for batch in loader:
            if len(batch) != 5:
                raise ValueError(
                    "compare_pipelineV3 requires ImuSagittalH5Dataset(..., return_full_sagittal_angles=True) "
                    "(x, y, pos6, pos23, time)."
                )
            x_imu, y, _pos6_gt, pos23_gt, time_w = batch
            x_imu = x_imu.to(device)
            y = y.to(device)
            pos23_gt = pos23_gt.to(device)
            time_w = time_w.to(device)
            pred_d = direct_model(x_imu)
            pred_d = _lowpass_window_batch(
                pred_d,
                time_w,
                apply=pipeline_lpf_apply,
                cutoff_hz=pipeline_lpf_cutoff_hz,
                order=pipeline_lpf_order,
            )
            pred_a = angle_model(x_imu)
            pred_a = _lowpass_predicted_angles(
                pred_a,
                time_w,
                apply=pipeline_lpf_apply,
                cutoff_hz=pipeline_lpf_cutoff_hz,
                order=pipeline_lpf_order,
            )
            pos6, vel6 = _cascade_pos6_vel6_from_full_ik(
                pred_a,
                pos23_gt,
                time_w,
                eval_side,
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
            pred_c = ik_model(x_ik)
            pred_c = _lowpass_window_batch(
                pred_c,
                time_w,
                apply=pipeline_lpf_apply,
                cutoff_hz=pipeline_lpf_cutoff_hz,
                order=pipeline_lpf_order,
            )

            pb_d = pred_d.detach().cpu().numpy().astype(np.float64)
            pb_c = pred_c.detach().cpu().numpy().astype(np.float64)
            tb = y.detach().cpu().numpy().astype(np.float64)

            diff_d = pb_d - tb
            diff_c = pb_c - tb
            sum_sq_d += np.sum(diff_d**2, axis=(0, 2))
            sum_sq_c += np.sum(diff_c**2, axis=(0, 2))
            sum_abs_d += np.sum(np.abs(diff_d), axis=(0, 2))
            sum_abs_c += np.sum(np.abs(diff_c), axis=(0, 2))
            sum_t += np.sum(tb, axis=(0, 2))
            sum_t2 += np.sum(tb**2, axis=(0, 2))
            n_b = tb.shape[0] * tb.shape[2]
            n_elem += n_b

            sum_sq_all_d += float(np.sum(diff_d**2))
            sum_sq_all_c += float(np.sum(diff_c**2))
            sum_abs_all_d += float(np.sum(np.abs(diff_d)))
            sum_abs_all_c += float(np.sum(np.abs(diff_c)))
            sum_t_all += float(np.sum(tb))
            sum_t2_all += float(np.sum(tb**2))
            n_all += tb.size

    def finalize(
        sum_sq_ch: np.ndarray,
        sum_abs_ch: np.ndarray,
        sum_sq_g: float,
        sum_abs_g: float,
    ) -> Dict[str, Any]:
        per_ch: List[Dict[str, Any]] = []
        for c in range(n_ch):
            mse = float(sum_sq_ch[c] / max(n_elem, 1))
            rmse = float(np.sqrt(mse))
            mae = float(sum_abs_ch[c] / max(n_elem, 1))
            mean_t = sum_t[c] / max(n_elem, 1)
            ss_res = float(sum_sq_ch[c])
            ss_tot = float(sum_t2[c] - sum_t[c] * mean_t)
            r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
            name = dof_names[c] if c < len(dof_names) else f"dof_{c}"
            per_ch.append({"name": name, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2})
        overall_mse = float(sum_sq_g / max(n_all, 1))
        mean_all = sum_t_all / max(n_all, 1)
        ss_tot_all = float(sum_t2_all - sum_t_all * mean_all)
        overall_r2 = float(1.0 - sum_sq_g / (ss_tot_all + 1e-12))
        return {
            "per_channel": per_ch,
            "overall": {
                "mse": overall_mse,
                "rmse": float(np.sqrt(overall_mse)),
                "mae": float(sum_abs_g / max(n_all, 1)),
                "r2": overall_r2,
            },
        }

    met_d = finalize(sum_sq_d, sum_abs_d, sum_sq_all_d, sum_abs_all_d)
    met_c = finalize(sum_sq_c, sum_abs_c, sum_sq_all_c, sum_abs_all_c)
    return met_d, met_c


def main() -> None:
    p = argparse.ArgumentParser(
        description="V3: paired 6→3 IK compare with full pipeline zero-phase LPF (see compare_pipelineV3.py)"
    )
    p.add_argument("--imu-moment-ckpt", type=str, required=True)
    p.add_argument("--imu-angle-ckpt", type=str, required=True)
    p.add_argument("--ik-moment-ckpt", type=str, required=True, help="Paired ipsilateral IK TCN (6→3 sagittal)")
    p.add_argument("--test-dir", type=str, required=True)
    p.add_argument("--meta-root", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="results/pipeline_compare_v3")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--eval-split", type=str, default="test", choices=["test", "val"])
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--sample-rate-hz", type=float, default=200.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--walking-only", action="store_true", default=True)
    p.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    p.add_argument("--levelground-only", action="store_true", default=False)
    p.add_argument("--eval-side", type=str, default="right", choices=["right", "left"])
    p.add_argument(
        "--ik-input-normalize",
        action="store_true",
        default=False,
        help="Z-score IK inputs using checkpoint stats (only if IK was trained with dataset normalize=True).",
    )
    args = p.parse_args()

    set_global_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_root = args.meta_root or args.test_dir
    test_root = Path(args.test_dir)
    device = args.device

    print("Loading IMU→moment …")
    (
        m_direct,
        ck_m,
        schema_mr,
        schema_ml,
        tgt_m,
        out_names_mr,
        out_names_ml,
        w_imu,
        stride_imu,
        stats_imu_m,
    ) = load_imu_checkpoint(args.imu_moment_ckpt, device)
    if tgt_m != "moment":
        raise ValueError(f"--imu-moment-ckpt must be target=moment, got {tgt_m!r}")

    print("Loading IMU→angle …")
    (
        m_angle,
        ck_a,
        schema_ar,
        schema_al,
        tgt_a,
        out_names_ar,
        out_names_al,
        w_ang,
        stride_ang,
        stats_imu_a,
    ) = load_imu_checkpoint(args.imu_angle_ckpt, device)
    if tgt_a != "angle":
        raise ValueError(f"--imu-angle-ckpt must be target=angle, got {tgt_a!r}")

    if schema_mr != schema_ar or schema_ml != schema_al:
        raise ValueError("IMU moment and angle checkpoints have different paired imu_schema_right/left.")
    if not np.allclose(stats_imu_m["imu_mean"], stats_imu_a["imu_mean"], rtol=1e-5, atol=1e-8):
        raise ValueError("IMU moment/angle checkpoints have different imu_mean normalization.")
    if not np.allclose(stats_imu_m["imu_std"], stats_imu_a["imu_std"], rtol=1e-5, atol=1e-8):
        raise ValueError("IMU moment/angle checkpoints have different imu_std normalization.")
    if w_imu != w_ang:
        raise ValueError(f"IMU window_size mismatch: moment={w_imu} angle={w_ang}")

    print("Loading IK→moment (paired ipsilateral) …")
    try:
        (
            ik_model,
            ik_stats,
            dof_names_ik,
            w_ik,
            input_indices,
            _moment_indices,
            input_mode,
            output_mode,
            _stride_ik,
            _lat_ik,
            _paired_ik,
        ) = load_model(args.ik_moment_ckpt, device)
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
        if ik_stats:
            ik_stats = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in ik_stats.items()}
        dof_names_ik = ck.get("dof_names", out_names_mr)
        w_ik = int(ck.get("window_size", w_imu))
        input_indices = ck.get("input_indices")
        input_mode = ck.get("input_mode", "unknown")
        output_mode = ck.get("output_mode", "unknown")

    if ik_stats is None:
        raise ValueError("IK moment checkpoint missing normalization stats.")

    if w_ik != w_imu:
        raise ValueError(
            f"Window size mismatch: IMU models use {w_imu}, IK moment model uses {w_ik}. "
            "Retrain or pick checkpoints with the same --window-size."
        )

    n_in = ik_model.n_input_channels
    n_out = ik_model.n_output_channels
    n_sag = len(SAGITTAL_INPUT_INDICES)
    half = n_sag // 2
    if n_in != 2 * half or n_out != half:
        raise ValueError(
            "compare_pipelineV3 expects a **paired** sagittal IK TCN "
            f"(n_in=2×{half}={2*half}, n_out={half}). Got n_in={n_in}, n_out={n_out} "
            f"(input_mode={input_mode!r} output_mode={output_mode!r}). "
            "Use compare_pipeline.py for legacy 12→6 models."
        )
    if input_indices is None:
        input_indices = list(SAGITTAL_INPUT_INDICES)
    else:
        input_indices = [int(i) for i in input_indices]
        if input_indices != list(SAGITTAL_INPUT_INDICES):
            raise ValueError(
                f"IK model input_indices {input_indices} != sagittal {list(SAGITTAL_INPUT_INDICES)}."
            )

    ik_input_normalize = bool(args.ik_input_normalize)

    sagittal6_slice = slice(0, 3) if args.eval_side == "right" else slice(3, 6)
    side_ik_idx = _expected_side_ik_indices(input_indices, args.eval_side)
    dof_names = list(out_names_mr if args.eval_side == "right" else out_names_ml)
    if list(dof_names_ik) != list(dof_names):
        print(f"  [warn] IK dof_names {dof_names_ik} != IMU output names {dof_names}; using IMU names for tables.")

    run_cfg = load_run_config(args.imu_moment_ckpt)
    eval_stride = resolve_dataset_stride(
        stride_from_ckpt=stride_imu,
        run_cfg=run_cfg,
        window_size=w_imu,
        override=args.stride,
    )

    apply_lowpass_filter = True
    lowpass_cutoff_hz = 4.0
    lowpass_order = 4
    if run_cfg is not None and any(
        k in run_cfg for k in ("no_lowpass", "lowpass_cutoff_hz", "lowpass_order")
    ):
        apply_lowpass_filter = not bool(run_cfg.get("no_lowpass", False))
        lowpass_cutoff_hz = float(run_cfg.get("lowpass_cutoff_hz", 4.0))
        lowpass_order = int(run_cfg.get("lowpass_order", 4))

    _levelground_only = args.levelground_only
    if run_cfg is not None and "levelground_only" in run_cfg:
        _levelground_only = bool(run_cfg["levelground_only"])
    _walking_only = args.walking_only
    if run_cfg is not None and "walking_only" in run_cfg:
        _walking_only = bool(run_cfg["walking_only"])

    imu_tgt_sr: Optional[float] = None
    if ck_m.get("target_sample_rate_hz") is not None:
        imu_tgt_sr = float(ck_m["target_sample_rate_hz"])
    elif run_cfg is not None and run_cfg.get("target_sample_rate_hz") is not None:
        imu_tgt_sr = float(run_cfg["target_sample_rate_hz"])
    report_sr = float(imu_tgt_sr) if imu_tgt_sr is not None else float(args.sample_rate_hz)
    va = ck_a.get("target_sample_rate_hz")
    vm = ck_m.get("target_sample_rate_hz")
    if va is not None and vm is not None and float(va) != float(vm):
        print(f"  [warn] IMU angle vs moment ckpt target_sample_rate_hz differ ({va} vs {vm}); using moment ckpt / config.")

    eval_ids, mode = _resolve_eval_subjects(
        test_root,
        Path(args.imu_moment_ckpt),
        args.eval_split,
        args.max_files,
    )
    print(f"Eval subjects ({mode}): {eval_ids}")
    print(
        f"Windows: size={w_imu}  stride={eval_stride}  sample_rate_hz={report_sr}"
        + (f"  (dataset target_sample_rate_hz={imu_tgt_sr})" if imu_tgt_sr is not None else "")
    )
    print(f"Eval side: {args.eval_side}")
    print(f"IK TCN: n_in={n_in} n_out={n_out}  ik_input_normalize={ik_input_normalize}")
    print(
        f"[V3] Pipeline zero-phase LPF: apply={apply_lowpass_filter} "
        f"({lowpass_cutoff_hz} Hz, order {lowpass_order}) — "
        "IMU load + cascade angles/ω + direct/cascade moment outputs (metrics)"
    )

    ds = ImuSagittalH5Dataset(
        h5_dir=str(test_root),
        meta_root_dir=meta_root,
        subject_ids=eval_ids,
        imu_schema_right=schema_mr,
        imu_schema_left=schema_ml,
        sides=cast(Literal["right", "left"], args.eval_side),
        target="moment",
        window_size=w_imu,
        stride=eval_stride,
        walking_only=_walking_only,
        levelground_only=_levelground_only,
        normalize=True,
        stats=stats_imu_m,
        return_full_sagittal_angles=True,
        apply_lowpass_filter=apply_lowpass_filter,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        lowpass_order=lowpass_order,
        target_sample_rate_hz=imu_tgt_sr,
        preload_trials=False,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"Running V3 comparison on {len(ds):,} windows …")
    met_d, met_c = _streaming_two_preds_metrics_v3(
        m_direct,
        m_angle,
        ik_model,
        loader,
        device,
        ik_stats,
        input_indices,
        side_ik_idx,
        dof_names,
        sagittal6_slice=sagittal6_slice,
        eval_side=str(args.eval_side),
        pipeline_lpf_apply=apply_lowpass_filter,
        pipeline_lpf_cutoff_hz=lowpass_cutoff_hz,
        pipeline_lpf_order=lowpass_order,
        ik_input_normalize=ik_input_normalize,
    )

    summary = {
        "pipeline_version": "V3_paired_ik_6x3_full_pipeline_lpf",
        "test_dir": str(test_root.resolve()),
        "eval_split": args.eval_split,
        "eval_side": args.eval_side,
        "eval_mode": mode,
        "subjects": eval_ids,
        "n_windows": len(ds),
        "window_size": w_imu,
        "stride": eval_stride,
        "sample_rate_hz": report_sr,
        "target_sample_rate_hz": imu_tgt_sr,
        "imu_moment_checkpoint": str(Path(args.imu_moment_ckpt).resolve()),
        "imu_angle_checkpoint": str(Path(args.imu_angle_ckpt).resolve()),
        "ik_moment_checkpoint": str(Path(args.ik_moment_ckpt).resolve()),
        "ik_n_input_channels": n_in,
        "ik_n_output_channels": n_out,
        "ik_input_normalize": ik_input_normalize,
        "cascade_predicted_angle_lowpass": {
            "apply": bool(apply_lowpass_filter),
            "cutoff_hz": float(lowpass_cutoff_hz),
            "order": int(lowpass_order),
        },
        "pipeline_zero_phase_lowpass": {
            "apply": bool(apply_lowpass_filter),
            "cutoff_hz": float(lowpass_cutoff_hz),
            "order": int(lowpass_order),
            "imu_on_load": True,
            "cascade_predicted_angles": True,
            "cascade_angular_velocity_after_hybrid_ik": True,
            "direct_moment_predictions": True,
            "cascade_moment_predictions": True,
            "note": "Ground-truth moments/angles follow ImuSagittalH5Dataset denoising when apply is True.",
        },
        "direct_imu_to_moment": met_d,
        "cascade_imu_angle_then_ik_moment": met_c,
    }

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\nRESULTS V3 (moment RMSE / R² vs same ground truth)\n{'='*70}")
    print(f"{'DOF':<22s}  {'RMSE dir':>10s}  {'R² dir':>8s}  {'RMSE cas':>10s}  {'R² cas':>8s}")
    print("-" * 70)
    for a, b in zip(met_d["per_channel"], met_c["per_channel"]):
        print(
            f"{a['name']:<22s}  {a['rmse']:10.5f}  {a['r2']:8.4f}  {b['rmse']:10.5f}  {b['r2']:8.4f}"
        )
    od, oc = met_d["overall"], met_c["overall"]
    print("-" * 70)
    print(
        f"{'OVERALL':<22s}  {od['rmse']:10.5f}  {od['r2']:8.4f}  {oc['rmse']:10.5f}  {oc['r2']:8.4f}"
    )
    print(f"{'='*70}\nSaved {out_dir / 'comparison.json'}")

    _plot_rmse_comparison(
        dof_names,
        [c["rmse"] for c in met_d["per_channel"]],
        [c["rmse"] for c in met_c["per_channel"]],
        out_dir / "rmse_direct_vs_cascade.png",
    )


if __name__ == "__main__":
    main()
