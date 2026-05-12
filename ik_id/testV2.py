#!/usr/bin/env python3
"""
Evaluation script V2 for trained TCN moment-prediction models.

Compatible with checkpoints from both ``train.py`` and ``trainV2.py``.

V2 additions over test.py
--------------------------
* Prints V2 training config (loss type, Huber delta, DOF weights,
  smoothness lambda, correlated-vel-noise) from ``config.json`` when present.
* Computes and reports a **prediction smoothness metric**: the mean absolute
  first temporal difference of the predicted moment sequence (N·m/kg/frame).
  A lower value means smoother predictions, which is desirable given that real
  joint moments are smooth (≤4 Hz content).  This metric can be compared
  directly between V1 and V2 runs to confirm the smoothness regularisation is
  working.

* **Rollout / rate**: ``trainV2.py`` saves ``rollout_decimate_step`` in ``config.json``;
  evaluation uses the same stride-2 decimation by default. Pass ``--rollout`` to
  force decimation, or rely on ``config.json``.

Run from ``os_kinetics/``::

    python -m ik_id.testV2 ...
    python ik_id/testV2.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from dataset import (
    KineticsTCNDataset,
    DOF_NAMES,
    normalize_laterality,
    find_trial_dirs,
    extract_subject_id,
)
from model import TCN, TransformerMoment, GaussianDiffusion1D  # noqa: F401 — all used via load_model
from training_utils import set_global_seed

# Re-use all non-main utilities from test.py to avoid duplication.
from ik_id.test import (
    resolve_unilateral_paired_for_eval,
    resolve_dataset_stride,
    load_model,
    run_inference_streaming,
    plot_per_channel_rmse,
    plot_per_channel_r2,
    plot_scatter_gt_vs_pred,
    plot_time_series,
    load_subject_split,
    load_run_config,
    resolve_test_files,
    _find_matching_local_wandb_run,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# V2-specific metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_streaming_v2(
    model: torch.nn.Module,
    loader: Any,
    device: str,
    dof_names: Any,
    *,
    n_plot_samples: int = 3,
    scatter_max_points: int = 50_000,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Like ``run_inference_streaming`` but also accumulates a smoothness metric.

    Returns the same 5 items as the base function, plus:
        smoothness_score (float): mean absolute first temporal difference of
        predictions averaged over all batches, DOFs, and time steps —
        mean(|ŷ[t] - ŷ[t-1]|).  Units: N·m/kg per sample.
    """
    sum_sq_ch = None
    sum_abs_ch = None
    sum_t_ch = None
    sum_t2_ch = None
    n_elem_ch = 0

    sum_sq_all = 0.0
    sum_abs_all = 0.0
    sum_t_all = 0.0
    sum_t2_all = 0.0
    n_all = 0

    # Smoothness accumulator: sum of |Δŷ| over all elements in (B, C, W-1)
    smooth_abs_sum = 0.0
    smooth_n = 0

    plot_pred_chunks: List[Any] = []
    plot_true_chunks: List[Any] = []
    n_plot_collected = 0
    scatter_gt_chunks: List[Any] = []
    scatter_pred_chunks: List[Any] = []
    n_scatter = 0

    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        pred = model(x)
        pb = pred.detach().cpu().numpy().astype(np.float64)
        tb = y.numpy().astype(np.float64)
        n_ch = pb.shape[1]

        if sum_sq_ch is None:
            sum_sq_ch = np.zeros(n_ch, dtype=np.float64)
            sum_abs_ch = np.zeros(n_ch, dtype=np.float64)
            sum_t_ch = np.zeros(n_ch, dtype=np.float64)
            sum_t2_ch = np.zeros(n_ch, dtype=np.float64)

        diff = pb - tb
        sum_sq_ch += np.sum(diff ** 2, axis=(0, 2))
        sum_abs_ch += np.sum(np.abs(diff), axis=(0, 2))
        sum_t_ch += np.sum(tb, axis=(0, 2))
        sum_t2_ch += np.sum(tb ** 2, axis=(0, 2))
        n_elem_ch += pb.shape[0] * pb.shape[2]

        sum_sq_all += float(np.sum(diff ** 2))
        sum_abs_all += float(np.sum(np.abs(diff)))
        sum_t_all += float(np.sum(tb))
        sum_t2_all += float(np.sum(tb ** 2))
        n_all += tb.size

        # Smoothness: mean |Δŷ| across (B, C, W-1)
        if pb.shape[2] > 1:
            delta_pred = np.abs(pb[:, :, 1:] - pb[:, :, :-1])
            smooth_abs_sum += float(np.sum(delta_pred))
            smooth_n += delta_pred.size

        if n_plot_collected < n_plot_samples:
            take = min(pb.shape[0], n_plot_samples - n_plot_collected)
            plot_pred_chunks.append(pb[:take].astype(np.float32))
            plot_true_chunks.append(tb[:take].astype(np.float32))
            n_plot_collected += take

        if n_scatter < scatter_max_points:
            need = scatter_max_points - n_scatter
            flat_g = tb.reshape(-1)
            flat_p = pb.reshape(-1)
            take = min(flat_g.size, need)
            scatter_gt_chunks.append(flat_g[:take].astype(np.float32))
            scatter_pred_chunks.append(flat_p[:take].astype(np.float32))
            n_scatter += take

    assert sum_sq_ch is not None
    per_ch: List[Dict[str, Any]] = []
    for c in range(n_ch):
        mse = float(sum_sq_ch[c] / n_elem_ch)
        rmse = float(np.sqrt(mse))
        mae = float(sum_abs_ch[c] / n_elem_ch)
        ss_res = float(sum_sq_ch[c])
        mean_t = sum_t_ch[c] / n_elem_ch
        ss_tot = float(sum_t2_ch[c] - sum_t_ch[c] * mean_t)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        name = dof_names[c] if c < len(dof_names) else f"dof_{c}"
        per_ch.append({"name": name, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2})

    overall_mse = float(sum_sq_all / max(n_all, 1))
    mean_all = sum_t_all / max(n_all, 1)
    ss_tot_all = float(sum_t2_all - sum_t_all * mean_all)
    overall_r2 = float(1.0 - sum_sq_all / (ss_tot_all + 1e-12))

    metrics: Dict[str, Any] = {
        "per_channel": per_ch,
        "overall": {
            "mse": overall_mse,
            "rmse": float(np.sqrt(overall_mse)),
            "mae": float(sum_abs_all / max(n_all, 1)),
            "r2": overall_r2,
        },
    }

    pred_plot = np.concatenate(plot_pred_chunks, 0) if plot_pred_chunks else np.zeros((0, n_ch, 0), dtype=np.float32)
    true_plot = np.concatenate(plot_true_chunks, 0) if plot_true_chunks else np.zeros((0, n_ch, 0), dtype=np.float32)
    scatter_gt = np.concatenate(scatter_gt_chunks, 0) if scatter_gt_chunks else np.zeros(0, dtype=np.float32)
    scatter_pred_arr = np.concatenate(scatter_pred_chunks, 0) if scatter_pred_chunks else np.zeros(0, dtype=np.float32)

    smoothness_score = float(smooth_abs_sum / max(smooth_n, 1))
    return metrics, pred_plot, true_plot, scatter_gt, scatter_pred_arr, smoothness_score


@torch.no_grad()
def _predict_full_trial_from_dataset(
    model: torch.nn.Module,
    dataset: KineticsTCNDataset,
    trial_idx: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict one full trial sequence from ``dataset._get_trial(trial_idx)``.
    Returns:
      pred: (C_out, T), true: (C_out, T)
    """
    trial = dataset._get_trial(trial_idx)
    pos = trial["positions"].copy()
    vel = trial["velocities"].copy()
    mom = trial["moments"].copy()

    if dataset.unilateral_paired:
        # For unilateral paired windows, plot the canonical right-side mapping
        # so full-trial visualization matches model IO channel ordering.
        in_i = getattr(dataset, "_pair_in_r")
        mom_i = getattr(dataset, "_pair_mom_r")
        if in_i is None or mom_i is None:
            raise RuntimeError("unilateral_paired is enabled but side index mapping is missing.")
        pos = pos[:, in_i]
        vel = vel[:, in_i]
        mom = mom[:, mom_i]
    else:
        if dataset.input_indices is not None:
            pos = pos[:, dataset.input_indices]
            vel = vel[:, dataset.input_indices]
        if dataset.moment_indices is not None:
            mom = mom[:, dataset.moment_indices]

    x = np.concatenate([pos, vel], axis=1).T.astype(np.float32)
    y = mom.T.astype(np.float32)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    pred = model(x_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return pred, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained TCN model (V2)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Directory with test trial directories or S*.h5 files")
    parser.add_argument("--output-dir", type=str, default="results/tcn_eval_v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--walking-only", action="store_true", default=True)
    parser.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    parser.add_argument("--levelground-only", action="store_true")
    parser.add_argument("--eval-split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-plot-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rollout",
        action="store_true",
        default=False,
        help="Use stride-2 IK/ID decimation (~100 Hz). Default: follow config.json next to checkpoint "
             "(rollout_decimate_step / rollout). Pass this to force decimated loading when the run dir "
             "has no config or to override.",
    )
    parser.add_argument(
        "--input-lowpass-mode",
        type=str,
        default=None,
        choices=["zero_phase", "causal"],
        help="Optional override for IK/velocity LPF mode used by the dataset loader.",
    )
    parser.add_argument(
        "--output-lowpass-mode",
        type=str,
        default=None,
        choices=["none", "zero_phase", "causal"],
        help="Optional override for ID moment-target LPF mode used by the dataset loader.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print(f"Loading model from {args.checkpoint}")
    (
        model, stats, dof_names, window_size, input_indices, moment_indices,
        input_mode, output_mode, laterality_ckpt, unilateral_paired_ckpt,
    ) = load_model(args.checkpoint, args.device)

    print(f"  Input mode:  {input_mode}")
    print(f"  Output mode: {output_mode}")
    print(f"  Output DOFs: {dof_names}")
    print(f"  Window size: {window_size}")

    run_cfg = load_run_config(args.checkpoint)

    # ---- Print V2 training config ----
    if run_cfg is not None:
        v2_fields = {
            "loss_type": None,
            "huber_delta": None,
            "dof_loss_weights": None,
            "smoothness_lambda": None,
            "correlated_vel_noise": None,
        }
        has_v2 = any(k in run_cfg for k in v2_fields)
        if has_v2:
            print("\n  [V2 training config]")
            loss_type = run_cfg.get("loss_type", "mse")
            huber_delta = run_cfg.get("huber_delta", 0.5)
            smoothness_lambda = run_cfg.get("smoothness_lambda", 0.0)
            correlated_vel_noise = run_cfg.get("correlated_vel_noise", False)
            dof_loss_weights = run_cfg.get("dof_loss_weights", None)
            print(f"    loss_type={loss_type}"
                  + (f"  huber_delta={huber_delta}" if loss_type == "huber" else ""))
            if smoothness_lambda:
                print(f"    smoothness_lambda={smoothness_lambda}")
            if correlated_vel_noise:
                print(f"    correlated_vel_noise=True")
            if dof_loss_weights:
                w_str = "  ".join(
                    f"{n}:{w:.2f}" for n, w in zip(dof_names, dof_loss_weights)
                )
                print(f"    dof_weights: {w_str}")

    # ---- Resolve test subjects ----
    print(f"\n{'='*70}")
    print("RESOLVING TEST FILES")
    print(f"{'='*70}")
    test_root = Path(args.test_dir)
    h5_subject_files = sorted([p for p in test_root.glob("S*.h5") if p.is_file()])
    is_h5_only_layout = len(h5_subject_files) > 0
    eval_subject_ids: Optional[List[str]] = None

    split = load_subject_split(args.checkpoint)
    if split is not None:
        # Respect train/val/test subject boundaries recorded during training.
        print(f"  Found subject_split.json next to checkpoint.")
        print(f"  Train:  {split.get('train_subjects')}")
        print(f"  Val:    {split.get('val_subjects')}")
        print(f"  Test:   {split.get('test_subjects', '(not recorded)')}")
        print(f"  Evaluating: --eval-split={args.eval_split}")
        if is_h5_only_layout:
            subjects_in_dir = sorted([p.stem.upper() for p in h5_subject_files])
            train_subjects = set(split.get("train_subjects", []))
            val_subjects = set(split.get("val_subjects", []))
            test_subjects = set(split.get("test_subjects", []))
            all_split = train_subjects | val_subjects | test_subjects

            keep = (test_subjects if test_subjects else val_subjects) if args.eval_split == "test" else val_subjects
            label = args.eval_split

            overlap = set(subjects_in_dir) & all_split
            if not overlap:
                # Independent directory: no overlap with recorded split.
                eval_subject_ids = subjects_in_dir
                mode = "independent"
                print(f"  No overlap with training split. Using all {len(subjects_in_dir)} subjects.")
            else:
                # Overlap detected: restrict strictly to requested split.
                eval_subject_ids = sorted(list(set(subjects_in_dir) & keep))
                mode = label
                print(f"  Keeping ({len(eval_subject_ids):2d}): {eval_subject_ids}")

            if args.max_files is not None:
                eval_subject_ids = eval_subject_ids[:args.max_files]
            test_files = []
        else:
            test_files, mode = resolve_test_files(
                args.test_dir, split, split_key=args.eval_split, max_files=args.max_files
            )
    else:
        print(f"  No subject_split.json. Using all files in {args.test_dir}.")
        if is_h5_only_layout:
            eval_subject_ids = sorted([p.stem.upper() for p in h5_subject_files])
            if args.max_files is not None:
                eval_subject_ids = eval_subject_ids[:args.max_files]
            test_files = []
        else:
            test_files = find_trial_dirs(args.test_dir)
            if args.max_files is not None:
                test_files = test_files[:args.max_files]
        mode = "independent"

    if is_h5_only_layout:
        subjects_used = sorted(eval_subject_ids or [])
        print(f"\n  Evaluating on {len(subjects_used)} subject h5 file(s): {subjects_used}")
    else:
        subjects_used = sorted({extract_subject_id(f) for f in test_files})
        print(f"\n  Evaluating on {len(test_files)} file(s) from subjects: {subjects_used}")

    if is_h5_only_layout and len(subjects_used) == 0:
        raise ValueError("No test subjects found.")
    if (not is_h5_only_layout) and len(test_files) == 0:
        raise ValueError("No test files found.")

    # ---- Build dataset kwargs (mirrors test.py logic) ----
    if stats is not None:
        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                stats[k] = v.numpy()

    if is_h5_only_layout:
        test_ds_kwargs: Dict[str, Any] = dict(
            data_dir=args.test_dir, h5_dir=args.test_dir, use_h5=True,
            subject_ids=eval_subject_ids, window_size=window_size, stride=1,
            walking_only=args.walking_only, normalize=False, stats=stats,
        )
    else:
        test_ds_kwargs = dict(
            data_dir=args.test_dir, b3d_files=test_files,
            window_size=window_size, stride=1,
            walking_only=args.walking_only, normalize=False, stats=stats,
        )

    _levelground_only = args.levelground_only
    if run_cfg is not None and "levelground_only" in run_cfg:
        _levelground_only = bool(run_cfg["levelground_only"])
    test_ds_kwargs["levelground_only"] = _levelground_only

    if run_cfg is not None:
        _lat = str(run_cfg.get("laterality", laterality_ckpt))
        test_ds_kwargs.update(
            input_mode=run_cfg.get("input_mode", input_mode),
            output_mode=run_cfg.get("output_mode", output_mode),
            laterality=_lat,
        )
        _cfg_paired = run_cfg.get("unilateral_paired_side_windows", None)
        if _cfg_paired is not None:
            _cfg_paired = bool(_cfg_paired)
        _paired_flag = _cfg_paired if _cfg_paired is not None else unilateral_paired_ckpt
        test_ds_kwargs["unilateral_paired_side_windows"] = resolve_unilateral_paired_for_eval(
            laterality=_lat, ckpt_flag=_paired_flag,
            n_in_model=model.n_input_channels, input_indices=input_indices,
        )
    else:
        test_ds_kwargs.update(
            input_indices=input_indices, moment_indices=moment_indices,
            laterality=laterality_ckpt,
            unilateral_paired_side_windows=resolve_unilateral_paired_for_eval(
                laterality=laterality_ckpt, ckpt_flag=unilateral_paired_ckpt,
                n_in_model=model.n_input_channels, input_indices=input_indices,
            ),
        )

    test_ds_kwargs["apply_lowpass_filter"] = True
    input_lpf_mode = "zero_phase"
    if run_cfg is not None:
        if "input_lowpass_mode" in run_cfg and run_cfg.get("input_lowpass_mode") is not None:
            input_lpf_mode = str(run_cfg.get("input_lowpass_mode"))
        if "lowpass_cutoff_hz" in run_cfg:
            test_ds_kwargs["lowpass_cutoff_hz"] = float(run_cfg["lowpass_cutoff_hz"])
        if "lowpass_order" in run_cfg:
            test_ds_kwargs["lowpass_order"] = int(run_cfg["lowpass_order"])
    if args.input_lowpass_mode is not None:
        input_lpf_mode = args.input_lowpass_mode
    test_ds_kwargs["input_lowpass_mode"] = input_lpf_mode

    output_lpf_mode = "zero_phase"
    if run_cfg is not None:
        if run_cfg.get("output_lowpass_mode") is not None:
            output_lpf_mode = str(run_cfg.get("output_lowpass_mode"))
        elif run_cfg.get("apply_moment_lowpass_filter") is not None:
            output_lpf_mode = "zero_phase" if bool(run_cfg.get("apply_moment_lowpass_filter")) else "none"
    if args.output_lowpass_mode is not None:
        output_lpf_mode = args.output_lowpass_mode
    test_ds_kwargs["apply_moment_lowpass_filter"] = bool(output_lpf_mode != "none")
    test_ds_kwargs["moment_lowpass_mode"] = (
        "zero_phase" if output_lpf_mode == "none" else output_lpf_mode
    )
    print(
        f"  Input LPF: {input_lpf_mode} ({test_ds_kwargs.get('lowpass_cutoff_hz', 4.0)} Hz, "
        f"order {test_ds_kwargs.get('lowpass_order', 4)})"
    )
    if output_lpf_mode == "none":
        print("  Output LPF (moment targets): off")
    else:
        print(
            f"  Output LPF (moment targets): {output_lpf_mode} "
            f"({test_ds_kwargs.get('lowpass_cutoff_hz', 4.0)} Hz, order {test_ds_kwargs.get('lowpass_order', 4)})"
        )

    vel_lpf_apply = False
    vel_lpf_cut = None
    vel_lpf_ord = None
    vel_lpf_mode = input_lpf_mode
    if run_cfg is not None:
        if run_cfg.get("velocity_lowpass_filter") is not None:
            vel_lpf_apply = bool(run_cfg.get("velocity_lowpass_filter"))
        vel_lpf_cut = run_cfg.get("velocity_lowpass_cutoff_hz")
        vel_lpf_ord = run_cfg.get("velocity_lowpass_order")
        if run_cfg.get("velocity_lowpass_mode") is not None:
            vel_lpf_mode = str(run_cfg.get("velocity_lowpass_mode"))
    test_ds_kwargs["apply_velocity_lowpass_filter"] = bool(vel_lpf_apply)
    test_ds_kwargs["velocity_lowpass_cutoff_hz"] = vel_lpf_cut
    test_ds_kwargs["velocity_lowpass_order"] = vel_lpf_ord
    test_ds_kwargs["velocity_lowpass_mode"] = vel_lpf_mode
    if vel_lpf_apply:
        _vcut = vel_lpf_cut or test_ds_kwargs.get("lowpass_cutoff_hz", 4.0)
        _vord = vel_lpf_ord or test_ds_kwargs.get("lowpass_order", 4)
        print(f"  Velocity LPF: on ({vel_lpf_mode}, {_vcut} Hz, order {_vord})")
    else:
        print("  Velocity LPF: off")

    if args.rollout:
        # CLI override: force stride-2 decimation regardless of config.
        rollout_step = 2
        print("  Rollout decimation: stride=2 (--rollout)")
    else:
        # Default behavior: follow checkpoint config from trainV2.
        rollout_step = 1
        if run_cfg is not None:
            rollout_step = int(run_cfg.get("rollout_decimate_step", 1))
            if rollout_step == 1 and bool(run_cfg.get("rollout")):
                rollout_step = 2
        rollout_step = max(1, rollout_step)
        if rollout_step > 1:
            print(f"  Rollout decimation: stride={rollout_step} (from config.json)")

    if rollout_step > 1:
        test_ds_kwargs["rollout_decimate_step"] = rollout_step

    print(f"\nLoading test data...")
    test_ds = KineticsTCNDataset(**test_ds_kwargs)
    print(f"  unilateral_paired_side_windows: {test_ds.unilateral_paired}")

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    # ---- Inference ----
    print("Running inference (streaming; V2 smoothness metric enabled)...")
    metrics, pred_plot, true_plot, scatter_gt, scatter_pred_arr, smoothness_score = (
        run_inference_streaming_v2(
            model, test_loader, args.device, dof_names,
            n_plot_samples=args.n_plot_samples, scatter_max_points=50_000,
        )
    )
    print(f"  Windows evaluated: {len(test_ds):,}")

    # ---- Results ----
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Overall RMSE: {metrics['overall']['rmse']:.6f}")
    print(f"  Overall MAE:  {metrics['overall']['mae']:.6f}")
    if "r2" in metrics["overall"]:
        print(f"  Overall R²:   {metrics['overall']['r2']:.6f}")
    print(f"\n  Prediction smoothness (mean |Δŷ| per frame): {smoothness_score:.6f} N·m/kg")
    print(f"  (lower = smoother; compare across V1/V2 runs to verify smoothness reg effect)")
    print(f"\n  {'DOF':<25s}  {'RMSE':>8s}  {'MAE':>8s}  {'R²':>8s}")
    print(f"  {'-'*55}")
    for ch in metrics["per_channel"]:
        print(f"  {ch['name']:<25s}  {ch['rmse']:8.4f}  {ch['mae']:8.4f}  {ch['r2']:8.4f}")
    print(f"{'='*70}")

    # ---- Save ----
    metrics["smoothness_score"] = smoothness_score
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "eval_subjects.json", "w") as f:
        json.dump({
            "test_dir": args.test_dir,
            "eval_split": args.eval_split,
            "mode": mode,
            "subjects_evaluated": subjects_used,
            "n_files": len(test_files) if not is_h5_only_layout else len(subjects_used),
            "train_subjects": split.get("train_subjects") if split else None,
            "val_subjects":   split.get("val_subjects")   if split else None,
            "test_subjects":  split.get("test_subjects")  if split else None,
        }, f, indent=2)

    # ---- Plots ----
    plot_per_channel_rmse(metrics, out_dir / "per_dof_rmse.png")
    plot_per_channel_r2(metrics, out_dir / "per_dof_r2.png")
    plot_scatter_gt_vs_pred(
        scatter_pred_arr, scatter_gt, dof_names,
        out_dir / "scatter_gt_vs_pred.png",
        overall_r2=metrics["overall"].get("r2"),
    )
    if rollout_step > 1:
        plot_sample_hz = 200.0 / float(rollout_step)
    else:
        plot_sample_hz = 200.0
    # Plot up to N full trials from one randomly selected evaluated subject.
    # This avoids mixing subjects in qualitative examples while still providing
    # randomized trial coverage across runs (seed-controlled).
    rng = random.Random(args.seed)
    chosen_subject = rng.choice(subjects_used)
    if is_h5_only_layout:
        candidate_trial_indices = [
            i for i, ref in enumerate(test_ds.h5_trial_refs)
            if ref[0].upper() == chosen_subject.upper()
        ]
    else:
        candidate_trial_indices = [
            i for i, td in enumerate(test_ds.trial_dirs)
            if extract_subject_id(td).upper() == chosen_subject.upper()
        ]
    rng.shuffle(candidate_trial_indices)
    selected_trial_indices = candidate_trial_indices[: max(0, int(args.n_plot_samples))]
    print(
        f"Plotting {len(selected_trial_indices)} full trial(s) from subject {chosen_subject} "
        f"(random, seed={args.seed})."
    )
    for k, t_idx in enumerate(selected_trial_indices):
        pred_t, true_t = _predict_full_trial_from_dataset(model, test_ds, t_idx, args.device)
        if is_h5_only_layout:
            sid, cond, trial_name, _h5_path = test_ds.h5_trial_refs[t_idx]
            trial_tag = f"{sid}_{cond}_{trial_name}"
        else:
            td = test_ds.trial_dirs[t_idx]
            trial_tag = f"{td.parent.name}_{td.name}"
        trial_tag = trial_tag.replace("/", "_")
        plot_time_series(
            pred_t[None, ...],
            true_t[None, ...],
            dof_names,
            out_dir / f"timeseries_trial_{k}_{trial_tag}.png",
            sample_idx=0,
            sample_rate_hz=plot_sample_hz,
        )

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
