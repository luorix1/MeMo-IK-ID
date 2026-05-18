#!/usr/bin/env python3
"""
Run IK->ID checkpoint evaluation for explicit subjects and explicit H5 tasks.

This script mirrors model/data loading used by existing ik_id evaluators, then
keeps only windows whose trial condition is in --conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import KineticsTCNDataset
from training_utils import set_global_seed

from ik_id.inter_dataset_eval import _accumulate_loader_metrics, _finalize_metrics
from ik_id.test import load_model, load_run_config, plot_time_series
from ik_id.test_addbiomech_repr_subjects import (
    DEFAULT_DATA_DIR,
    _assert_paths_only_under_root,
    build_h5_eval_kwargs,
)


def _parse_csv_list(value: str, *, upper: bool = False) -> List[str]:
    out = [x.strip() for x in value.split(",") if x.strip()]
    if upper:
        return [x.upper() for x in out]
    return out


def _match_condition(raw_name: str, target_set: Set[str]) -> Optional[str]:
    key = (raw_name or "").strip().lower()
    return key if key in target_set else None


def _slugify(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "trial"


def _apply_lpf_settings_from_run_cfg(
    ds_kwargs: Dict[str, Any],
    run_cfg: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Match LPF behavior used in ik_id/testV2.py for eval-time dataset loading.
    """
    input_lpf_mode = "zero_phase"
    if run_cfg is not None and run_cfg.get("input_lowpass_mode") is not None:
        input_lpf_mode = str(run_cfg.get("input_lowpass_mode"))
    ds_kwargs["input_lowpass_mode"] = input_lpf_mode

    output_lpf_mode = "zero_phase"
    if run_cfg is not None:
        if run_cfg.get("output_lowpass_mode") is not None:
            output_lpf_mode = str(run_cfg.get("output_lowpass_mode"))
        elif run_cfg.get("apply_moment_lowpass_filter") is not None:
            output_lpf_mode = (
                "zero_phase" if bool(run_cfg.get("apply_moment_lowpass_filter")) else "none"
            )
    ds_kwargs["apply_moment_lowpass_filter"] = bool(output_lpf_mode != "none")
    ds_kwargs["moment_lowpass_mode"] = (
        "zero_phase" if output_lpf_mode == "none" else output_lpf_mode
    )

    velocity_lpf_mode = input_lpf_mode
    if run_cfg is not None and run_cfg.get("velocity_lowpass_mode") is not None:
        velocity_lpf_mode = str(run_cfg.get("velocity_lowpass_mode"))
    ds_kwargs["velocity_lowpass_mode"] = velocity_lpf_mode

    return {
        "input_lowpass_mode": input_lpf_mode,
        "output_lowpass_mode": output_lpf_mode,
        "velocity_lowpass_mode": velocity_lpf_mode,
    }


@torch.no_grad()
def _predict_full_trial_with_side(
    model: torch.nn.Module,
    dataset: KineticsTCNDataset,
    trial_idx: int,
    device: str,
    side: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict one full trial, optionally selecting unilateral paired side.
    Returns (pred, true) with shape (C_out, T).
    """
    trial = dataset._get_trial(trial_idx)
    pos = trial["positions"].copy()
    vel = trial["velocities"].copy()
    mom = trial["moments"].copy()

    if dataset.unilateral_paired:
        side_eff = side or "r"
        if side_eff not in {"r", "l"}:
            raise ValueError(f"Invalid side {side_eff!r}; expected 'r' or 'l'.")
        if side_eff == "r":
            in_i = getattr(dataset, "_pair_in_r")
            mom_i = getattr(dataset, "_pair_mom_r")
        else:
            in_i = getattr(dataset, "_pair_in_l")
            mom_i = getattr(dataset, "_pair_mom_l")
        if in_i is None or mom_i is None:
            raise RuntimeError("unilateral_paired=True but side index mapping is missing.")
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


@torch.no_grad()
def _eval_window_subset(
    model: torch.nn.Module,
    ds: KineticsTCNDataset,
    window_indices: Sequence[int],
    *,
    device: str,
    batch_size: int,
    num_workers: int,
    dof_names: List[str],
) -> Dict[str, Any]:
    subset = Subset(ds, list(window_indices))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

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
    smooth_abs_sum = 0.0
    smooth_n = 0
    n_scatter = 0
    scatter_gt_chunks: List[np.ndarray] = []
    scatter_pred_chunks: List[np.ndarray] = []

    (
        sum_sq_ch,
        sum_abs_ch,
        sum_t_ch,
        sum_t2_ch,
        n_elem_ch,
        sum_sq_all,
        sum_abs_all,
        sum_t_all,
        sum_t2_all,
        n_all,
        smooth_abs_sum,
        smooth_n,
        n_scatter,
    ) = _accumulate_loader_metrics(
        model,
        loader,
        device,
        sum_sq_ch=sum_sq_ch,
        sum_abs_ch=sum_abs_ch,
        sum_t_ch=sum_t_ch,
        sum_t2_ch=sum_t2_ch,
        n_elem_ch=n_elem_ch,
        sum_sq_all=sum_sq_all,
        sum_abs_all=sum_abs_all,
        sum_t_all=sum_t_all,
        sum_t2_all=sum_t2_all,
        n_all=n_all,
        smooth_abs_sum=smooth_abs_sum,
        smooth_n=smooth_n,
        scatter_gt_chunks=scatter_gt_chunks,
        scatter_pred_chunks=scatter_pred_chunks,
        n_scatter=n_scatter,
        scatter_max_points=0,
    )
    if sum_sq_ch is None or sum_abs_ch is None or sum_t_ch is None or sum_t2_ch is None:
        raise RuntimeError("No windows available after filtering.")

    metrics = _finalize_metrics(
        dof_names=dof_names,
        sum_sq_ch=sum_sq_ch,
        sum_abs_ch=sum_abs_ch,
        sum_t_ch=sum_t_ch,
        sum_t2_ch=sum_t2_ch,
        n_elem_ch=n_elem_ch,
        sum_sq_all=sum_sq_all,
        sum_abs_all=sum_abs_all,
        sum_t_all=sum_t_all,
        sum_t2_all=sum_t2_all,
        n_all=n_all,
    )
    metrics["smoothness_score"] = float(smooth_abs_sum / max(smooth_n, 1))
    metrics["n_windows"] = int(len(window_indices))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint on explicit subjects and explicit task conditions."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/0512_ik_id_knee_causal_in_zero_out/best_model.pt",
    )
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--subjects",
        type=str,
        default="S024,S025,S026,S027",
        help="Comma-separated subject IDs.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="incline_treadmill_down10deg,incline_treadmill_up10deg",
        help="Comma-separated exact H5 condition names.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/0512_ik_id_knee_causal_in_zero_out",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--walking-only", action="store_true", default=True)
    parser.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    parser.add_argument("--levelground-only", action="store_true")
    parser.add_argument("--rollout", action="store_true", default=False)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    subjects = _parse_csv_list(args.subjects, upper=True)
    if not subjects:
        raise ValueError("No subjects provided.")
    cond_list_raw = _parse_csv_list(args.conditions, upper=False)
    if not cond_list_raw:
        raise ValueError("No conditions provided.")
    cond_key_set = {c.lower() for c in cond_list_raw}

    data_root = Path(args.data_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"--data-dir is not a directory: {data_root}")
    _assert_paths_only_under_root(data_root, subjects)

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        ckpt_try = (_ROOT / ckpt_path).resolve()
        if ckpt_try.is_file():
            ckpt_path = ckpt_try
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = (_ROOT / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    (
        model,
        stats,
        dof_names,
        window_size,
        input_indices,
        moment_indices,
        input_mode,
        output_mode,
        laterality_ckpt,
        unilateral_paired_ckpt,
    ) = load_model(str(ckpt_path), args.device)
    run_cfg = load_run_config(str(ckpt_path))
    model.eval()

    common_kw = dict(
        data_root=data_root,
        model=model,
        stats=stats,
        window_size=window_size,
        input_indices=input_indices,
        moment_indices=moment_indices,
        input_mode=input_mode,
        output_mode=output_mode,
        laterality_ckpt=laterality_ckpt,
        unilateral_paired_ckpt=unilateral_paired_ckpt,
        run_cfg=run_cfg,
        walking_only=args.walking_only,
        levelground_only_cli=args.levelground_only,
        rollout_force=args.rollout,
    )
    sample_rate_hz = 200.0
    probe_kwargs = build_h5_eval_kwargs(**common_kw, subject_ids=[subjects[0]])
    lpf_cfg = _apply_lpf_settings_from_run_cfg(probe_kwargs, run_cfg)
    rollout_step = int(probe_kwargs.get("rollout_decimate_step", 1))
    if rollout_step > 1:
        sample_rate_hz = 200.0 / float(rollout_step)

    summary: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_root),
        "subjects": subjects,
        "conditions_requested": cond_list_raw,
        "conditions_requested_normalized": sorted(cond_key_set),
        "plot_sample_rate_hz": sample_rate_hz,
        "lpf_config_used": lpf_cfg,
        "subjects_eval": {},
    }

    all_subject_ds: Dict[str, KineticsTCNDataset] = {}
    all_window_refs: List[Tuple[str, str, int]] = []  # (sid, cond_key, window_idx_in_subject_ds)
    cond_subject_hits: Dict[str, Set[str]] = {c: set() for c in cond_key_set}
    subject_trial_side_hits: Dict[str, Dict[int, Dict[str, int]]] = {}

    for sid in subjects:
        ds_kwargs = build_h5_eval_kwargs(**common_kw, subject_ids=[sid])
        _apply_lpf_settings_from_run_cfg(ds_kwargs, run_cfg)
        ds = KineticsTCNDataset(**ds_kwargs)
        all_subject_ds[sid] = ds

        hit_windows: Dict[str, List[int]] = {c: [] for c in cond_key_set}
        trial_side_hits: Dict[int, Dict[str, int]] = {}
        for widx, (trial_idx, _start, _side) in enumerate(ds.windows):
            _sid_ref, cond_name, _trial_name, _h5_path = ds.h5_trial_refs[trial_idx]
            cond_key = _match_condition(cond_name, cond_key_set)
            if cond_key is None:
                continue
            hit_windows[cond_key].append(widx)
            all_window_refs.append((sid, cond_key, widx))
            cond_subject_hits[cond_key].add(sid)
            side_key = _side if _side in {"r", "l"} else "none"
            trial_side_hits.setdefault(int(trial_idx), {})
            trial_side_hits[int(trial_idx)][side_key] = (
                trial_side_hits[int(trial_idx)].get(side_key, 0) + 1
            )
        subject_trial_side_hits[sid] = trial_side_hits

        summary["subjects_eval"][sid] = {
            "n_total_windows_subject": int(len(ds)),
            "windows_per_condition": {k: int(len(v)) for k, v in sorted(hit_windows.items())},
        }

    if not all_window_refs:
        raise RuntimeError(
            "No windows matched requested conditions for requested subjects. "
            "Check condition names and dataset filters."
        )

    # Evaluate per subject and condition.
    results_subject_condition: Dict[str, Dict[str, Any]] = {}
    for sid in subjects:
        ds = all_subject_ds[sid]
        results_subject_condition[sid] = {}
        for cond_key in sorted(cond_key_set):
            win_idx = [
                widx
                for sid_ref, cond_ref, widx in all_window_refs
                if sid_ref == sid and cond_ref == cond_key
            ]
            if not win_idx:
                continue
            metrics = _eval_window_subset(
                model,
                ds,
                win_idx,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dof_names=dof_names,
            )
            results_subject_condition[sid][cond_key] = metrics

    # Evaluate pooled by condition across subjects.
    pooled_by_condition: Dict[str, Any] = {}
    for cond_key in sorted(cond_key_set):
        refs = [(sid, widx) for sid, c, widx in all_window_refs if c == cond_key]
        if not refs:
            continue
        # Build a synthetic merged subset by concatenating cached sample indices.
        # Keep memory bounded by iterating per subject and aggregating scalar sums.
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
        smooth_abs_sum = 0.0
        smooth_n = 0
        n_scatter = 0

        for sid in subjects:
            sid_idx = [w for s, w in refs if s == sid]
            if not sid_idx:
                continue
            subset = Subset(all_subject_ds[sid], sid_idx)
            loader = DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(args.device == "cuda"),
            )
            scatter_gt_chunks: List[np.ndarray] = []
            scatter_pred_chunks: List[np.ndarray] = []
            (
                sum_sq_ch,
                sum_abs_ch,
                sum_t_ch,
                sum_t2_ch,
                n_elem_ch,
                sum_sq_all,
                sum_abs_all,
                sum_t_all,
                sum_t2_all,
                n_all,
                smooth_abs_sum,
                smooth_n,
                n_scatter,
            ) = _accumulate_loader_metrics(
                model,
                loader,
                args.device,
                sum_sq_ch=sum_sq_ch,
                sum_abs_ch=sum_abs_ch,
                sum_t_ch=sum_t_ch,
                sum_t2_ch=sum_t2_ch,
                n_elem_ch=n_elem_ch,
                sum_sq_all=sum_sq_all,
                sum_abs_all=sum_abs_all,
                sum_t_all=sum_t_all,
                sum_t2_all=sum_t2_all,
                n_all=n_all,
                smooth_abs_sum=smooth_abs_sum,
                smooth_n=smooth_n,
                scatter_gt_chunks=scatter_gt_chunks,
                scatter_pred_chunks=scatter_pred_chunks,
                n_scatter=n_scatter,
                scatter_max_points=0,
            )

        if sum_sq_ch is None or sum_abs_ch is None or sum_t_ch is None or sum_t2_ch is None:
            continue
        pooled_metrics = _finalize_metrics(
            dof_names=dof_names,
            sum_sq_ch=sum_sq_ch,
            sum_abs_ch=sum_abs_ch,
            sum_t_ch=sum_t_ch,
            sum_t2_ch=sum_t2_ch,
            n_elem_ch=n_elem_ch,
            sum_sq_all=sum_sq_all,
            sum_abs_all=sum_abs_all,
            sum_t_all=sum_t_all,
            sum_t2_all=sum_t2_all,
            n_all=n_all,
        )
        pooled_metrics["smoothness_score"] = float(smooth_abs_sum / max(smooth_n, 1))
        pooled_metrics["n_windows"] = int(len(refs))
        pooled_metrics["n_subjects"] = int(len(cond_subject_hits[cond_key]))
        pooled_metrics["subjects"] = sorted(cond_subject_hits[cond_key])
        pooled_by_condition[cond_key] = pooled_metrics

    summary["results_subject_condition"] = results_subject_condition
    summary["results_pooled_by_condition"] = pooled_by_condition

    # Plot every matched full trial (GT vs inference).
    trial_plot_root = out_root / "trial_plots"
    trial_plot_root.mkdir(parents=True, exist_ok=True)
    trial_plot_records: List[Dict[str, Any]] = []
    for sid in subjects:
        ds = all_subject_ds[sid]
        trial_side_hits = subject_trial_side_hits.get(sid, {})
        sid_plot_root = trial_plot_root / sid
        sid_plot_root.mkdir(parents=True, exist_ok=True)
        for t_idx, ref in enumerate(ds.h5_trial_refs):
            _sid_ref, cond_name, trial_name, _h5_path = ref
            cond_key = _match_condition(cond_name, cond_key_set)
            if cond_key is None:
                continue
            side_counts = trial_side_hits.get(int(t_idx), {})
            if ds.unilateral_paired:
                sides_to_plot = [s for s in ("r", "l") if side_counts.get(s, 0) > 0]
                if not sides_to_plot:
                    continue
            else:
                if int(t_idx) not in trial_side_hits:
                    continue
                sides_to_plot = [None]

            for side in sides_to_plot:
                pred_t, true_t = _predict_full_trial_with_side(
                    model, ds, t_idx, args.device, side=side
                )
                side_tag = f"__side_{side}" if side is not None else ""
                out_name = (
                    f"{t_idx:04d}__{sid}__{_slugify(cond_name)}__{_slugify(trial_name)}"
                    f"{side_tag}.png"
                )
                out_path = sid_plot_root / out_name
                plot_time_series(
                    pred_t[None, ...],
                    true_t[None, ...],
                    dof_names,
                    out_path,
                    sample_idx=0,
                    sample_rate_hz=sample_rate_hz,
                )
                trial_plot_records.append(
                    {
                        "subject_id": sid,
                        "condition": cond_name,
                        "trial_name": trial_name,
                        "trial_idx": int(t_idx),
                        "side": side,
                        "n_windows_for_side": int(side_counts.get(side or "none", 0)),
                        "plot_path": str(out_path),
                    }
                )

    summary["trial_plots"] = trial_plot_records

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results: {out_root / 'summary.json'}")
    print(f"Saved trial plots: {trial_plot_root} (n={len(trial_plot_records)})")
    for cond_key in sorted(pooled_by_condition.keys()):
        overall = pooled_by_condition[cond_key].get("overall", {})
        print(
            f"{cond_key}: "
            f"RMSE={overall.get('rmse', float('nan')):.6f} "
            f"MAE={overall.get('mae', float('nan')):.6f} "
            f"R2={overall.get('r2', float('nan')):.6f} "
            f"n_windows={pooled_by_condition[cond_key].get('n_windows', 0)}"
        )


if __name__ == "__main__":
    main()
