#!/usr/bin/env python3
"""
Compare IMU→moment **direct** vs **cascade** for **ipsilateral IK moment TCN** checkpoints
(``ik_id.train`` with ``unilateral_paired_side_windows`` / paired sagittal: **6** inputs, **3** outputs).

V1 ``compare_pipeline.py`` expects a **12→6** full sagittal IK model. This script expects **6→3**
(one leg’s pos+vel → that leg’s moments) and matches ``runs/0411_*``-style training.

Hybrid cascade: IMU predicts 3 angles → merge into full23-DOF pose (GT + predicted side) →
``_compute_velocity`` → take **only the evaluated side’s** 3 pos + 3 vel → IK TCN.

Example::

    python compare_pipelineV2.py \\
        --imu-moment-ckpt runs/imu_moments/best_model.pt \\
        --imu-angle-ckpt runs/imu_angles/best_model.pt \\
        --ik-moment-ckpt runs/0411_jinwoo_3_EPIC/best_model.pt \\
        --test-dir /path/to/Processed/Jinwoo \\
        --output-dir results/pipeline_compare_v2
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

from dataset import SAGITTAL_INPUT_INDICES
from imu_sagittal.imu_sagittal_dataset import ImuSagittalH5Dataset
from model import TCN

from compare_pipeline import (
    _cascade_pos6_vel6_from_full_ik,
    _ik_moment_tcn_input,
    _lowpass_predicted_angles,
    _normalize_ik_tcn_input,
    _plot_rmse_comparison,
    _resolve_eval_subjects,
)
from ik_id.test import load_model, load_run_config, resolve_dataset_stride
from imu_sagittal.imu_sagittal_eval import load_imu_checkpoint, set_global_seed


def _expected_side_ik_indices(full_indices: Sequence[int], eval_side: str) -> List[int]:
    h = len(full_indices) // 2
    if h * 2 != len(full_indices):
        raise ValueError("IK input_indices must split into equal R/L halves.")
    if eval_side == "right":
        return [int(x) for x in full_indices[:h]]
    if eval_side == "left":
        return [int(x) for x in full_indices[h:]]
    raise ValueError(f"eval_side must be 'right' or 'left', got {eval_side!r}")


def _streaming_two_preds_metrics_v2(
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
    cascade_angle_lowpass: bool,
    cascade_angle_lowpass_cutoff_hz: float,
    cascade_angle_lowpass_order: int,
    ik_input_normalize: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Paired IK head: (B,6,W) pos+vel ipsilateral → (B,3,W) moments."""
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
                    "compare_pipelineV2 requires ImuSagittalH5Dataset(..., return_full_sagittal_angles=True) "
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
            pos3 = pos6[:, sl6, :]
            vel3 = vel6[:, sl6, :]
            if ik_input_normalize:
                x_ik = _normalize_ik_tcn_input(pos3, vel3, ik_stats, side_ik_indices, dev)
            else:
                x_ik = _ik_moment_tcn_input(pos3, vel3)
            pred_c = ik_model(x_ik)

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
        description="Compare direct IMU→moment vs cascade (paired6→3 IK TCN; see compare_pipelineV2.py)"
    )
    p.add_argument("--imu-moment-ckpt", type=str, required=True)
    p.add_argument("--imu-angle-ckpt", type=str, required=True)
    p.add_argument("--ik-moment-ckpt", type=str, required=True, help="Paired ipsilateral IK TCN (6→3 sagittal)")
    p.add_argument("--test-dir", type=str, required=True)
    p.add_argument("--meta-root", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="results/pipeline_compare_v2")
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
            "compare_pipelineV2 expects a **paired** sagittal IK TCN "
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
        f"Cascade IMU→angle LPF: apply={apply_lowpass_filter} ({lowpass_cutoff_hz} Hz, order {lowpass_order})"
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

    print(f"Running comparison on {len(ds):,} windows …")
    met_d, met_c = _streaming_two_preds_metrics_v2(
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
        cascade_angle_lowpass=apply_lowpass_filter,
        cascade_angle_lowpass_cutoff_hz=lowpass_cutoff_hz,
        cascade_angle_lowpass_order=lowpass_order,
        ik_input_normalize=ik_input_normalize,
    )

    summary = {
        "pipeline_version": "V2_paired_ik_6x3",
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
