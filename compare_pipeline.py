#!/usr/bin/env python3
"""
Compare two pipelines for sagittal joint **moments** (N·m/kg) on the same IMU windows:

1. **Direct:** IMU → moment (**3** sagittal moments on one side — default **right**).
2. **Cascade:** IMU → 3 predicted angles → optional **zero-phase Butterworth** on those angles (same cutoff/order
   as IK trial loading when LPF is enabled) → hybrid **full 23-DOF** pose → ``dataset._compute_velocity`` → IK moment TCN.

Reports per-DOF and overall **RMSE** and **R²** for both, writes ``comparison.json`` and optional plots.

Example::

    python compare_pipeline.py \\
        --imu-moment-ckpt runs/imu_moments/best_model.pt \\
        --imu-angle-ckpt runs/imu_angles/best_model.pt \\
        --ik-moment-ckpt runs/tcn_sagittal/best_model.pt \\
        --test-dir /path/to/Processed/Jinwoo \\
        --output-dir results/pipeline_compare
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from dataset import SAGITTAL_INPUT_INDICES, _compute_velocity, _lowpass_zero_phase
from ik_id.test import load_model, load_run_config, load_subject_split, resolve_dataset_stride
from imu_sagittal.imu_sagittal_dataset import ImuSagittalH5Dataset
from imu_sagittal.imu_sagittal_eval import load_imu_checkpoint, set_global_seed
from model import TCN

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _normalize_ik_tcn_input(
    pos_rad: torch.Tensor,
    vel_rad_s: torch.Tensor,
    stats: Dict[str, np.ndarray],
    input_indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Z-score IK inputs using checkpoint ``stats`` (use only if models were trained with ``normalize=True``).

    ``ik_id.train`` uses ``normalize=False`` by default; for those runs use ``_ik_moment_tcn_input`` instead.
    """
    pm = torch.as_tensor(stats["pos_mean"], device=device, dtype=torch.float32)
    ps = torch.as_tensor(stats["pos_std"], device=device, dtype=torch.float32)
    vm = torch.as_tensor(stats["vel_mean"], device=device, dtype=torch.float32)
    vs = torch.as_tensor(stats["vel_std"], device=device, dtype=torch.float32)
    idx = torch.as_tensor(list(input_indices), device=device, dtype=torch.long)

    pos_n = (pos_rad - pm[idx].view(1, -1, 1)) / ps[idx].view(1, -1, 1)
    vel_n = (vel_rad_s - vm[idx].view(1, -1, 1)) / vs[idx].view(1, -1, 1)
    return torch.cat([pos_n, vel_n], dim=1)


def _ik_moment_tcn_input(pos6: torch.Tensor, vel6: torch.Tensor) -> torch.Tensor:
    """Match ``KineticsTCNDataset`` with ``normalize=False`` (``ik_id.train`` default): (B, 12, W)."""
    return torch.cat([pos6, vel6], dim=1)


def _lowpass_predicted_angles(
    pred_a: torch.Tensor,
    time_w: torch.Tensor,
    *,
    apply: bool,
    cutoff_hz: float,
    order: int,
) -> torch.Tensor:
    """
    Zero-phase Butterworth on IMU angle-network outputs (B, 3, W), matching ``dataset._lowpass_zero_phase``
    on IK positions before hybrid merge with GT.
    """
    if not apply or cutoff_hz <= 0:
        return pred_a
    device = pred_a.device
    dtype = pred_a.dtype
    B, _C, _W = pred_a.shape
    pa = pred_a.detach().cpu().numpy().astype(np.float64)
    tw = time_w.detach().cpu().numpy().astype(np.float64)
    out = np.empty_like(pa)
    for b in range(B):
        x_tw = pa[b].T
        t = tw[b]
        xf = _lowpass_zero_phase(x_tw, t, cutoff_hz=float(cutoff_hz), order=int(order))
        out[b] = xf.T
    return torch.as_tensor(out, device=device, dtype=dtype)


def _cascade_pos6_vel6_from_full_ik(
    pred_a: torch.Tensor,
    pos23_gt: torch.Tensor,
    time_w: torch.Tensor,
    eval_side: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hybrid 23-DOF positions (GT + predicted eval-leg sagittal angles), then ``_compute_velocity``
    on (time × 23) like ``KineticsTCNDataset``; return sagittal (B, 6, W) pos and vel for the IK TCN.
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


def _streaming_two_preds_metrics(
    direct_model: torch.nn.Module,
    angle_model: torch.nn.Module,
    ik_model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    ik_stats: Dict[str, np.ndarray],
    input_indices: List[int],
    sample_rate_hz: float,
    dof_names: List[str],
    *,
    sagittal_pred_slice: slice,
    eval_side: str,
    cascade_angle_lowpass: bool,
    cascade_angle_lowpass_cutoff_hz: float,
    cascade_angle_lowpass_order: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Direct IMU→mom and cascade on **3** sagittal moments; loader yields (x, y, pos6, pos23, time)."""
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
    sl = sagittal_pred_slice
    _ = sample_rate_hz  # retained for API compatibility; velocities use window timestamps + _compute_velocity

    with torch.no_grad():
        for batch in loader:
            if len(batch) != 5:
                raise ValueError(
                    "compare_pipeline requires ImuSagittalH5Dataset(..., return_full_sagittal_angles=True) "
                    "(x, y, pos6, pos23, time)."
                )
            x_imu, y, _pos6_gt, pos23_gt, time_w = batch
            x_imu = x_imu.to(device)
            y = y.to(device)
            pos23_gt = pos23_gt.to(device)
            time_w = time_w.to(device)
            pred_d = direct_model(x_imu)
            pred_a = angle_model(x_imu)
            pred_a = _lowpass_predicted_angles(
                pred_a,
                time_w,
                apply=cascade_angle_lowpass,
                cutoff_hz=cascade_angle_lowpass_cutoff_hz,
                order=cascade_angle_lowpass_order,
            )
            pos6, vel6 = _cascade_pos6_vel6_from_full_ik(pred_a, pos23_gt, time_w, eval_side, dev)
            x_ik = _ik_moment_tcn_input(pos6, vel6)
            pred_c_full = ik_model(x_ik)
            pred_c = pred_c_full[:, sl, :]

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
    ax.set_title("Sagittal moment error: direct vs cascade")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare direct IMU→moment vs cascade IMU→angle→IK→moment")
    p.add_argument(
        "--imu-moment-ckpt",
        type=str,
        required=True,
        help="IMU→moment checkpoint (imu_sagittal/train_imu_sagittal target=moment)",
    )
    p.add_argument("--imu-angle-ckpt", type=str, required=True, help="IMU→angle checkpoint")
    p.add_argument(
        "--ik-moment-ckpt",
        type=str,
        required=True,
        help="Angle+vel→moment TCN (ik_id.train / ik_id.test checkpoint)",
    )
    p.add_argument("--test-dir", type=str, required=True, help="H5 root with S###.h5")
    p.add_argument("--meta-root", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="results/pipeline_compare")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--eval-split", type=str, default="test", choices=["test", "val"])
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--sample-rate-hz", type=float, default=200.0, help="IMU/IK timeline (Hz) for velocity from predicted angles")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--walking-only", action="store_true", default=True)
    p.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    p.add_argument("--levelground-only", action="store_true", default=False)
    p.add_argument(
        "--eval-side",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Which leg's IMU windows and moment labels (must match IMU checkpoints).",
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
        raise ValueError(
            "IMU moment and angle checkpoints have different paired imu_schema_right/left; cannot compare."
        )
    if not np.allclose(stats_imu_m["imu_mean"], stats_imu_a["imu_mean"], rtol=1e-5, atol=1e-8):
        raise ValueError("IMU moment/angle checkpoints have different imu_mean normalization.")
    if not np.allclose(stats_imu_m["imu_std"], stats_imu_a["imu_std"], rtol=1e-5, atol=1e-8):
        raise ValueError("IMU moment/angle checkpoints have different imu_std normalization.")
    if w_imu != w_ang:
        raise ValueError(f"IMU window_size mismatch: moment={w_imu} angle={w_ang}")
    if list(out_names_mr) != list(out_names_ar) or list(out_names_ml) != list(out_names_al):
        print("  [warn] output_names differ between IMU angle/moment checkpoints; using moment ckpt names.")

    print("Loading IK→moment (ik_id.train TCN) …")
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
        ck = torch.load(args.ik_moment_ckpt, map_location=device)
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
    if n_in != 2 * len(SAGITTAL_INPUT_INDICES) or n_out != len(SAGITTAL_INPUT_INDICES):
        raise ValueError(
            f"This script expects sagittal 6-DOF in/out (12 inputs, 6 outputs). "
            f"IK model has n_in={n_in}, n_out={n_out}. input_mode={input_mode!r} output_mode={output_mode!r}"
        )
    if input_indices is None:
        input_indices = list(SAGITTAL_INPUT_INDICES)
    else:
        input_indices = [int(i) for i in input_indices]
        if input_indices != list(SAGITTAL_INPUT_INDICES):
            raise ValueError(
                f"IK model input_indices {input_indices} != sagittal {list(SAGITTAL_INPUT_INDICES)}. "
                "Train with --input-mode sagittal for this comparison."
            )

    run_cfg = load_run_config(args.imu_moment_ckpt)
    eval_stride = resolve_dataset_stride(
        stride_from_ckpt=stride_imu,
        run_cfg=run_cfg,
        window_size=w_imu,
        override=args.stride,
    )

    _lat_ck = str(ck_m.get("laterality") or (run_cfg.get("laterality") if run_cfg else None) or "unilateral")
    if _lat_ck != "unilateral":
        print(f"  [warn] checkpoint laterality={_lat_ck!r}; this codebase expects unilateral IMU runs only.")

    apply_lowpass_filter = True
    lowpass_cutoff_hz = 4.0
    lowpass_order = 4
    median_kernel_samples = 0
    if run_cfg is not None and any(
        k in run_cfg for k in ("no_lowpass", "lowpass_cutoff_hz", "lowpass_order", "median_kernel_samples")
    ):
        apply_lowpass_filter = not bool(run_cfg.get("no_lowpass", False))
        lowpass_cutoff_hz = float(run_cfg.get("lowpass_cutoff_hz", 4.0))
        lowpass_order = int(run_cfg.get("lowpass_order", 4))
        median_kernel_samples = int(run_cfg.get("median_kernel_samples", 0))

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
        f"Windows: size={w_imu}  stride={eval_stride}  "
        f"sample_rate_hz={report_sr}"
        + (f"  (dataset target_sample_rate_hz={imu_tgt_sr})" if imu_tgt_sr is not None else "")
    )
    print(f"Eval side: {args.eval_side}")
    print(
        f"Cascade IMU→angle LPF (before IK TCN): apply={apply_lowpass_filter} "
        f"({lowpass_cutoff_hz} Hz, order {lowpass_order})"
    )

    sagittal_pred_slice = slice(0, 3) if args.eval_side == "right" else slice(3, 6)
    dof_names = list(out_names_mr if args.eval_side == "right" else out_names_ml)

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
        median_kernel_samples=median_kernel_samples,
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

    print(f"Running comparison on {len(ds):,} windows …")
    met_d, met_c = _streaming_two_preds_metrics(
        m_direct,
        m_angle,
        ik_model,
        loader,
        device,
        ik_stats,
        input_indices,
        report_sr,
        dof_names,
        sagittal_pred_slice=sagittal_pred_slice,
        eval_side=str(args.eval_side),
        cascade_angle_lowpass=apply_lowpass_filter,
        cascade_angle_lowpass_cutoff_hz=lowpass_cutoff_hz,
        cascade_angle_lowpass_order=lowpass_order,
    )

    summary = {
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
        "cascade_predicted_angle_lowpass": {
            "apply": bool(apply_lowpass_filter),
            "cutoff_hz": float(lowpass_cutoff_hz),
            "order": int(lowpass_order),
        },
        "direct_imu_to_moment": met_d,
        "cascade_imu_angle_then_ik_moment": met_c,
    }

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\nRESULTS (moment RMSE / R² vs same ground truth)\n{'='*70}")
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
