#!/usr/bin/env python3
"""
Evaluate an IK->ID checkpoint on one AddBiomechanics source dataset slice.

The dataset slice is selected by --dataset and resolved from JSON files in
--data-dir named:
    subject_id_map_<dataset>_No_Arm.json

Examples:
    python -m ik_id.inter_dataset_eval \
        --checkpoint runs/0427_ik_id_all_addbiomech/best_model.pt \
        --data-dir /media/metamobility3/Samsung_T5/Processed/Addbiomechanics_final \
        --dataset Carter2023
"""

from __future__ import annotations

import argparse
import json
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

from dataset import KineticsTCNDataset
from training_utils import set_global_seed

from ik_id.test import (
    load_model,
    load_run_config,
    plot_per_channel_rmse,
    plot_per_channel_r2,
    plot_scatter_gt_vs_pred,
    plot_time_series,
    resolve_unilateral_paired_for_eval,
)
from ik_id.testV2 import _predict_full_trial_from_dataset


DEFAULT_DATA_DIR = Path("/media/metamobility3/Samsung_T5/Processed/Addbiomechanics_final")


def _subject_num(subject_id: str) -> int:
    return int("".join(ch for ch in subject_id if ch.isdigit()) or 0)


@torch.no_grad()
def _accumulate_loader_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    *,
    sum_sq_ch: Optional[np.ndarray],
    sum_abs_ch: Optional[np.ndarray],
    sum_t_ch: Optional[np.ndarray],
    sum_t2_ch: Optional[np.ndarray],
    n_elem_ch: int,
    sum_sq_all: float,
    sum_abs_all: float,
    sum_t_all: float,
    sum_t2_all: float,
    n_all: int,
    smooth_abs_sum: float,
    smooth_n: int,
    scatter_gt_chunks: List[np.ndarray],
    scatter_pred_chunks: List[np.ndarray],
    n_scatter: int,
    scatter_max_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float, float, float, int, float, int, int]:
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

        if pb.shape[2] > 1:
            delta_pred = np.abs(pb[:, :, 1:] - pb[:, :, :-1])
            smooth_abs_sum += float(np.sum(delta_pred))
            smooth_n += delta_pred.size

        if n_scatter < scatter_max_points:
            need = scatter_max_points - n_scatter
            flat_g = tb.reshape(-1)
            flat_p = pb.reshape(-1)
            take = min(flat_g.size, need)
            scatter_gt_chunks.append(flat_g[:take].astype(np.float32))
            scatter_pred_chunks.append(flat_p[:take].astype(np.float32))
            n_scatter += take

    assert sum_sq_ch is not None
    assert sum_abs_ch is not None
    assert sum_t_ch is not None
    assert sum_t2_ch is not None
    return (
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
    )


def _finalize_metrics(
    *,
    dof_names: List[str],
    sum_sq_ch: np.ndarray,
    sum_abs_ch: np.ndarray,
    sum_t_ch: np.ndarray,
    sum_t2_ch: np.ndarray,
    n_elem_ch: int,
    sum_sq_all: float,
    sum_abs_all: float,
    sum_t_all: float,
    sum_t2_all: float,
    n_all: int,
) -> Dict[str, Any]:
    n_ch = int(sum_sq_ch.shape[0])
    per_ch: List[Dict[str, Any]] = []
    for c in range(n_ch):
        mse = float(sum_sq_ch[c] / n_elem_ch)
        rmse = float(np.sqrt(mse))
        mae = float(sum_abs_ch[c] / n_elem_ch)
        ss_res = float(sum_sq_ch[c])
        mean_t = float(sum_t_ch[c] / n_elem_ch)
        ss_tot = float(sum_t2_ch[c] - sum_t_ch[c] * mean_t)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        name = dof_names[c] if c < len(dof_names) else f"dof_{c}"
        per_ch.append({"name": name, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2})

    overall_mse = float(sum_sq_all / max(n_all, 1))
    mean_all = float(sum_t_all / max(n_all, 1))
    ss_tot_all = float(sum_t2_all - sum_t_all * mean_all)
    overall_r2 = float(1.0 - sum_sq_all / (ss_tot_all + 1e-12))
    return {
        "per_channel": per_ch,
        "overall": {
            "mse": overall_mse,
            "rmse": float(np.sqrt(overall_mse)),
            "mae": float(sum_abs_all / max(n_all, 1)),
            "r2": overall_r2,
        },
    }


def _discover_addbiomech_subject_maps(data_root: Path) -> Dict[str, List[str]]:
    """
    Return mapping from dataset key -> sorted S### IDs.

    Keys include both:
      - full map label, e.g. "Carter2023_No_Arm"
      - canonical name with suffix removed, e.g. "Carter2023"
    """
    mapping: Dict[str, List[str]] = {}
    for p in sorted(data_root.glob("subject_id_map_*.json")):
        stem = p.stem
        if not stem.startswith("subject_id_map_"):
            continue
        label = stem[len("subject_id_map_") :]
        try:
            obj = json.loads(p.read_text())
        except Exception as e:
            raise RuntimeError(f"Failed parsing {p.name}: {e}") from e
        if not isinstance(obj, dict):
            continue
        ids = sorted(
            [str(v).strip().upper() for v in obj.values() if str(v).strip().upper().startswith("S")],
            key=_subject_num,
        )
        if not ids:
            continue
        mapping[label] = ids
        if label.endswith("_No_Arm"):
            mapping[label[: -len("_No_Arm")]] = ids
    return mapping


def _resolve_dataset_subjects(data_root: Path, dataset_name: str) -> Tuple[str, List[str], Dict[str, List[str]]]:
    dataset_maps = _discover_addbiomech_subject_maps(data_root)
    if not dataset_maps:
        raise RuntimeError(f"No subject_id_map_*.json files found under {data_root}")

    query = dataset_name.strip()
    if query in dataset_maps:
        return query, dataset_maps[query], dataset_maps

    # Case-insensitive fallback.
    low = query.lower()
    for key, ids in dataset_maps.items():
        if key.lower() == low:
            return key, ids, dataset_maps

    keys = sorted(dataset_maps.keys())
    raise ValueError(
        f"Unknown --dataset {dataset_name!r}. Available: {keys}"
    )


def build_h5_eval_kwargs(
    *,
    data_root: Path,
    model: torch.nn.Module,
    stats: Any,
    window_size: int,
    input_indices: Any,
    moment_indices: Any,
    input_mode: str,
    output_mode: str,
    laterality_ckpt: str,
    unilateral_paired_ckpt: Optional[bool],
    run_cfg: Optional[Dict[str, Any]],
    subject_ids: List[str],
    walking_only: bool,
    levelground_only_cli: bool,
    rollout_force: bool,
) -> Dict[str, Any]:
    if stats is not None:
        for k, v in list(stats.items()):
            if isinstance(v, torch.Tensor):
                stats[k] = v.numpy()

    kwargs: Dict[str, Any] = dict(
        data_dir=str(data_root),
        h5_dir=str(data_root),
        use_h5=True,
        subject_ids=subject_ids,
        window_size=window_size,
        stride=1,
        walking_only=walking_only,
        normalize=False,
        stats=stats,
    )

    _levelground_only = levelground_only_cli
    if run_cfg is not None and "levelground_only" in run_cfg:
        _levelground_only = bool(run_cfg["levelground_only"])
    kwargs["levelground_only"] = _levelground_only

    if run_cfg is not None:
        _lat = str(run_cfg.get("laterality", laterality_ckpt))
        kwargs.update(
            input_mode=run_cfg.get("input_mode", input_mode),
            output_mode=run_cfg.get("output_mode", output_mode),
            laterality=_lat,
        )
        _cfg_paired = run_cfg.get("unilateral_paired_side_windows", None)
        if _cfg_paired is not None:
            _cfg_paired = bool(_cfg_paired)
        _paired_flag = _cfg_paired if _cfg_paired is not None else unilateral_paired_ckpt
        kwargs["unilateral_paired_side_windows"] = resolve_unilateral_paired_for_eval(
            laterality=_lat,
            ckpt_flag=_paired_flag,
            n_in_model=model.n_input_channels,
            input_indices=input_indices,
        )
    else:
        kwargs.update(
            input_indices=input_indices,
            moment_indices=moment_indices,
            laterality=laterality_ckpt,
            unilateral_paired_side_windows=resolve_unilateral_paired_for_eval(
                laterality=laterality_ckpt,
                ckpt_flag=unilateral_paired_ckpt,
                n_in_model=model.n_input_channels,
                input_indices=input_indices,
            ),
        )

    kwargs["apply_lowpass_filter"] = True
    if run_cfg is not None:
        if "lowpass_cutoff_hz" in run_cfg:
            kwargs["lowpass_cutoff_hz"] = float(run_cfg["lowpass_cutoff_hz"])
        if "lowpass_order" in run_cfg:
            kwargs["lowpass_order"] = int(run_cfg["lowpass_order"])

    vel_lpf_apply = False
    vel_lpf_cut = None
    vel_lpf_ord = None
    if run_cfg is not None:
        if run_cfg.get("velocity_lowpass_filter") is not None:
            vel_lpf_apply = bool(run_cfg.get("velocity_lowpass_filter"))
        vel_lpf_cut = run_cfg.get("velocity_lowpass_cutoff_hz")
        vel_lpf_ord = run_cfg.get("velocity_lowpass_order")
    kwargs["apply_velocity_lowpass_filter"] = bool(vel_lpf_apply)
    kwargs["velocity_lowpass_cutoff_hz"] = vel_lpf_cut
    kwargs["velocity_lowpass_order"] = vel_lpf_ord

    if rollout_force:
        rollout_step = 2
    else:
        rollout_step = 1
        if run_cfg is not None:
            rollout_step = int(run_cfg.get("rollout_decimate_step", 1))
            if rollout_step == 1 and bool(run_cfg.get("rollout")):
                rollout_step = 2
        rollout_step = max(1, rollout_step)
    if rollout_step > 1:
        kwargs["rollout_decimate_step"] = rollout_step

    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint on one AddBiomechanics dataset (all its subjects)."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Folder with S*.h5 and subject_id_map_*.json files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., Carter2023, Tan2022, Uhlrich2023, ...)",
    )
    parser.add_argument("--output-dir", type=str, default="results/inter_dataset_eval")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--walking-only", action="store_true", default=True)
    parser.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    parser.add_argument("--levelground-only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-plot-samples", type=int, default=3)
    parser.add_argument(
        "--strict-subject-files",
        action="store_true",
        default=False,
        help="Fail if any mapped subject is missing S###.h5 (default: skip missing subjects).",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        default=False,
        help="Force stride-2 IK/ID decimation (same as testV2 --rollout).",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    data_root = Path(args.data_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"--data-dir is not a directory: {data_root}")

    dataset_key, subject_ids, all_maps = _resolve_dataset_subjects(data_root, args.dataset)
    if not subject_ids:
        raise RuntimeError(f"No subjects found for dataset {dataset_key}")

    missing_h5 = [sid for sid in subject_ids if not (data_root / f"{sid}.h5").is_file()]
    if missing_h5 and args.strict_subject_files:
        raise FileNotFoundError(
            f"Dataset {dataset_key} includes missing H5 subjects under {data_root}: {missing_h5}"
        )
    subject_ids_present = [sid for sid in subject_ids if sid not in set(missing_h5)]
    if not subject_ids_present:
        raise RuntimeError(
            f"Dataset {dataset_key} has no available S*.h5 subjects under {data_root}. "
            f"Mapped subjects: {subject_ids}"
        )

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = (_ROOT / out_root).resolve()
    out_dir = out_root / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        ckpt_try = (_ROOT / ckpt_path).resolve()
        if ckpt_try.is_file():
            ckpt_path = ckpt_try
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Data root:      {data_root}")
    print(f"Checkpoint:     {ckpt_path}")
    print(f"Dataset:        {dataset_key}")
    print(f"Subjects mapped ({len(subject_ids)}): {subject_ids}")
    if missing_h5:
        print(f"[warn] Missing H5 subjects skipped ({len(missing_h5)}): {missing_h5}")
    print(f"Subjects used ({len(subject_ids_present)}): {subject_ids_present}")

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

    ds_kwargs = build_h5_eval_kwargs(
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
        subject_ids=subject_ids_present,
        walking_only=args.walking_only,
        levelground_only_cli=args.levelground_only,
        rollout_force=args.rollout,
    )

    # Subject-wise evaluation keeps memory bounded by avoiding one huge window index.
    sum_sq_ch: Optional[np.ndarray] = None
    sum_abs_ch: Optional[np.ndarray] = None
    sum_t_ch: Optional[np.ndarray] = None
    sum_t2_ch: Optional[np.ndarray] = None
    n_elem_ch = 0
    sum_sq_all = 0.0
    sum_abs_all = 0.0
    sum_t_all = 0.0
    sum_t2_all = 0.0
    n_all = 0
    smooth_abs_sum = 0.0
    smooth_n = 0
    n_windows_total = 0
    scatter_gt_chunks: List[np.ndarray] = []
    scatter_pred_chunks: List[np.ndarray] = []
    n_scatter = 0
    scatter_max_points = 50_000
    per_subject_summary: List[Dict[str, Any]] = []

    for i, sid in enumerate(subject_ids_present, start=1):
        print(f"\n[{i}/{len(subject_ids_present)}] Evaluating subject {sid}")
        sub_kwargs = dict(ds_kwargs)
        sub_kwargs["subject_ids"] = [sid]
        sub_ds = KineticsTCNDataset(**sub_kwargs)
        n_w = len(sub_ds)
        n_windows_total += n_w
        print(f"  windows: {n_w:,}")
        if n_w == 0:
            per_subject_summary.append({"subject_id": sid, "n_windows": 0, "skipped": True})
            continue

        sub_loader = DataLoader(
            sub_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )
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
            sub_loader,
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
            scatter_max_points=scatter_max_points,
        )
        per_subject_summary.append({"subject_id": sid, "n_windows": n_w, "skipped": False})

    print(f"\nWindows evaluated: {n_windows_total:,}")
    if n_elem_ch == 0 or sum_sq_ch is None or sum_abs_ch is None or sum_t_ch is None or sum_t2_ch is None:
        raise RuntimeError("No windows available for selected dataset (after filters).")

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
    smoothness = float(smooth_abs_sum / max(smooth_n, 1))
    scatter_gt = np.concatenate(scatter_gt_chunks, 0) if scatter_gt_chunks else np.zeros(0, dtype=np.float32)
    scatter_pred_arr = (
        np.concatenate(scatter_pred_chunks, 0) if scatter_pred_chunks else np.zeros(0, dtype=np.float32)
    )
    metrics["smoothness_score"] = smoothness

    print(f"Overall RMSE: {metrics['overall']['rmse']:.6f}")
    print(f"Overall MAE:  {metrics['overall']['mae']:.6f}")
    print(f"Overall R2:   {metrics['overall']['r2']:.6f}")
    print(f"Smoothness:   {smoothness:.6f}")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "eval_subjects.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_key,
                "subjects_mapped": subject_ids,
                "subjects_missing_h5": missing_h5,
                "subjects_used": subject_ids_present,
                "n_subjects_mapped": len(subject_ids),
                "n_subjects_used": len(subject_ids_present),
                "n_windows": n_windows_total,
                "per_subject_summary": per_subject_summary,
                "all_available_datasets": sorted(all_maps.keys()),
                "checkpoint": str(ckpt_path),
                "data_dir": str(data_root),
            },
            f,
            indent=2,
        )

    plot_per_channel_rmse(metrics, out_dir / "per_dof_rmse.png")
    plot_per_channel_r2(metrics, out_dir / "per_dof_r2.png")
    plot_scatter_gt_vs_pred(
        scatter_pred_arr,
        scatter_gt,
        dof_names,
        out_dir / "scatter_gt_vs_pred.png",
        overall_r2=metrics["overall"].get("r2"),
    )

    # Plot full trials from random subjects in this dataset.
    rollout_step = int(ds_kwargs.get("rollout_decimate_step", 1))
    plot_sample_hz = 200.0 / float(rollout_step) if rollout_step > 1 else 200.0
    rng = random.Random(args.seed)
    plot_subjects = subject_ids_present.copy()
    rng.shuffle(plot_subjects)
    selected_subjects = plot_subjects[: max(0, int(args.n_plot_samples))]
    for k, sid in enumerate(selected_subjects):
        sub_kwargs = dict(ds_kwargs)
        sub_kwargs["subject_ids"] = [sid]
        sub_ds = KineticsTCNDataset(**sub_kwargs)
        if len(sub_ds.h5_trial_refs) == 0:
            continue
        pred_t, true_t = _predict_full_trial_from_dataset(model, sub_ds, 0, args.device)
        sid, cond, trial_name, _h5_path = sub_ds.h5_trial_refs[0]
        trial_tag = f"{sid}_{cond}_{trial_name}".replace("/", "_")
        plot_time_series(
            pred_t[None, ...],
            true_t[None, ...],
            dof_names,
            out_dir / f"timeseries_trial_{k}_{trial_tag}.png",
            sample_idx=0,
            sample_rate_hz=plot_sample_hz,
        )

    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
