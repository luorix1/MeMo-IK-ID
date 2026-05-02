#!/usr/bin/env python3
"""
**compare_pipelineV4** — same paired **6→3** direct vs cascade setup as V3, but **evaluation is
full-trial**: one forward pass per model on ``(1, C, T)`` for each trial (no sliding windows).

- **Stride:** always evaluates **every frame** of each trial (stride 1 along time). There is no
  dataset ``stride`` / ``window_size`` tiling; TCNs see the entire contiguous trial.
- **Per-condition metrics:** ``comparison.json`` includes ``per_loc_condition`` buckets **LG** (level
  ground + treadmill + incline lab walking), **RA** / **RD** / **SA** / **SD** when names encode
  ascent/descent, plus **RAMP_AD** / **STAIR_AD** for Camargo-style combined ramp/stair recordings,
  and **OTHER**.
- **Pipeline:** identical zero-phase Butterworth / hybrid IK / ω LPF semantics to V3’s
  ``infer_*_full_sequence`` helpers.

**Filter flags vs V3:** In V3, ``--walking-only`` / ``--levelground-only`` can be **overridden** by
keys stored in the IMU checkpoint ``run_config.json``. If those keys are present, both invocations
may end up using the same filters, so **window counts match** even when you pass different CLI
flags. V4 defaults to **honoring CLI flags only**. Pass ``--dataset-filters-from-checkpoint`` to
reproduce V3’s override behavior.

Self-contained (no import from ``compare_pipeline*.py``); uses ``dataset``, ``imu_sagittal``,
``model``, ``ik_id.test``, and ``_load_trial_imu_sagittal_paired``.

Example::

    python compare_pipelineV4.py \\
        --imu-moment-ckpt runs/imu_moments/best_model.pt \\
        --imu-angle-ckpt runs/imu_angles/best_model.pt \\
        --ik-moment-ckpt runs/ik/best_model.pt \\
        --test-dir /path/to/Processed/Jinwoo \\
        --output-dir results/pipeline_compare_v4

With Plotly trial plots (under ``output_dir/S*/``)::

    python compare_pipelineV4.py ... --visualize [--visualize-combined-html]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

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
    IK_DOF_NAMES,
    MOMENT_NAMES,
    SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES,
    SAGITTAL_INPUT_INDICES,
    _compute_velocity,
    _load_subject_metadata_map,
    _lowpass_zero_phase,
    include_condition_for_dataset,
)
from ik_id.test import load_model, load_run_config, load_subject_split
from imu_sagittal.imu_sagittal_eval import load_imu_checkpoint, set_global_seed
from imu_sagittal.imu_sagittal_leg_dataset import (
    TrialRef,
    _load_trial_imu_sagittal_paired,
)
from model import TCN

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]

# Match ``plot_pipeline_compare_trial_inferenceV2.py`` moment trace styling.
_MOMENT_LINE_GT_SUB = dict(color="black", width=1.8)
_MOMENT_LINE_DIRECT_SUB = dict(color="#808080", width=1.8)
_MOMENT_LINE_CASCADE_SUB = dict(color="red", width=1.8)


def _ik_stats_as_numpy(ik_stats: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(ik_stats)
    for k, v in list(out.items()):
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
    return out


def _normalize_ik_tcn_input(
    pos_rad: torch.Tensor,
    vel_rad_s: torch.Tensor,
    stats: Dict[str, np.ndarray],
    input_indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    pm = torch.as_tensor(stats["pos_mean"], device=device, dtype=torch.float32)
    ps = torch.as_tensor(stats["pos_std"], device=device, dtype=torch.float32)
    vm = torch.as_tensor(stats["vel_mean"], device=device, dtype=torch.float32)
    vs = torch.as_tensor(stats["vel_std"], device=device, dtype=torch.float32)
    idx = torch.as_tensor(list(input_indices), device=device, dtype=torch.long)

    pos_n = (pos_rad - pm[idx].view(1, -1, 1)) / ps[idx].view(1, -1, 1)
    vel_n = (vel_rad_s - vm[idx].view(1, -1, 1)) / vs[idx].view(1, -1, 1)
    return torch.cat([pos_n, vel_n], dim=1)


def _ik_moment_tcn_input(pos6: torch.Tensor, vel6: torch.Tensor) -> torch.Tensor:
    return torch.cat([pos6, vel6], dim=1)


def _lowpass_window_batch(
    x: torch.Tensor,
    time_w: torch.Tensor,
    *,
    apply: bool,
    cutoff_hz: float,
    order: int,
) -> torch.Tensor:
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
    cascade_velocity_lowpass: bool = False,
    ik_input_normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
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
        vel_lowpass_apply=(pipeline_lpf_apply and cascade_velocity_lowpass),
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


@torch.no_grad()
def infer_ik_moments_from_gt_angles_full_sequence(
    ik_model: torch.nn.Module,
    positions_full: np.ndarray,
    time_1d: np.ndarray,
    ik_stats: Dict[str, np.ndarray],
    side_ik_indices: List[int],
    device: str,
    eval_side: str,
    *,
    pipeline_lpf_apply: bool = True,
    pipeline_lpf_cutoff_hz: float = 4.0,
    pipeline_lpf_order: int = 4,
    cascade_velocity_lowpass: bool = False,
    ik_input_normalize: bool = False,
) -> np.ndarray:
    """Run IK→moment using GT sagittal angles (mocap stand-in, no force plates)."""
    n = positions_full.shape[0]
    if positions_full.shape != (n, 23):
        raise ValueError(f"positions_full must be (T, 23), got {positions_full.shape}")
    if len(time_1d) != n:
        raise ValueError("time_1d length must match positions_full length.")

    es = eval_side.lower()
    if es == "right":
        sl6 = slice(0, 3)
    elif es == "left":
        sl6 = slice(3, 6)
    else:
        raise ValueError("eval_side must be 'right' or 'left'")

    dev = torch.device(device)
    pos23_b = torch.from_numpy(positions_full.astype(np.float32).T.copy()).unsqueeze(0).to(device)
    time_b = torch.from_numpy(time_1d.astype(np.float32)).unsqueeze(0).to(device)

    # Use GT sagittal angles as if they were delivered by an external mocap-IK pipeline.
    if es == "right":
        idx3 = tuple(SAGITTAL_INPUT_INDICES[:3])
    else:
        idx3 = tuple(SAGITTAL_INPUT_INDICES[3:])
    gt_a = pos23_b[:, idx3, :]

    pos6, vel6 = _cascade_pos6_vel6_from_full_ik(
        gt_a,
        pos23_b,
        time_b,
        eval_side,
        dev,
        vel_lowpass_apply=(pipeline_lpf_apply and cascade_velocity_lowpass),
        vel_lowpass_cutoff_hz=pipeline_lpf_cutoff_hz,
        vel_lowpass_order=pipeline_lpf_order,
    )
    pos3 = pos6[:, sl6, :]
    vel3 = vel6[:, sl6, :]
    if ik_input_normalize:
        x_ik = _normalize_ik_tcn_input(pos3, vel3, ik_stats, side_ik_indices, dev)
    else:
        x_ik = _ik_moment_tcn_input(pos3, vel3)

    pred = ik_model(x_ik)
    pred = _lowpass_window_batch(
        pred,
        time_b,
        apply=pipeline_lpf_apply,
        cutoff_hz=pipeline_lpf_cutoff_hz,
        order=pipeline_lpf_order,
    )
    return pred.squeeze(0).detach().cpu().numpy().T.astype(np.float32)


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


def _collect_trial_refs(
    h5_dir: Path,
    subject_ids: Sequence[str],
    *,
    walking_only: bool,
    levelground_only: bool,
) -> List[TrialRef]:
    import h5py

    sid_set = {s.upper() for s in subject_ids}
    refs: List[TrialRef] = []
    for h5_path in sorted(h5_dir.glob("S*.h5")):
        sid = h5_path.stem.upper()
        if sid not in sid_set:
            continue
        with h5py.File(h5_path, "r") as h5f:
            for cond in sorted(h5f.keys()):
                if not include_condition_for_dataset(
                    cond,
                    walking_only=walking_only,
                    levelground_only=levelground_only,
                ):
                    continue
                for trial_name in sorted(h5f[cond].keys()):
                    refs.append((sid, cond, trial_name, str(h5_path)))
    return refs


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
    ax.set_title("Sagittal moment error: direct vs cascade (V4 full-trial)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


LOC_BUCKET_LEGEND: Dict[str, str] = {
    "LG": "Level ground + treadmill + incline (levelground_*; treadmill or treadmill_*; incline_*)",
    "RA": "Ramp ascent",
    "RD": "Ramp descent",
    "SA": "Stair ascent",
    "SD": "Stair descent",
    "RA/RD": "All ramp trials when ascent/descent is not explicit",
    "SA/SD": "All stair trials when ascent/descent is not explicit",
    "OTHER": "Did not match the rules above (still included in overall metrics)",
}


def _load_ascent_descent_mapping(path: Optional[str]) -> Dict[Tuple[str, str, str], str]:
    """
    Load explicit ascent/descent bucket labels from JSON.

    Expected schema:
      {
        "subjects": {
          "S035": {
            "stair_bundle2": 22,
            "incline_treadmill_bundle1": 3
          }
        }
      }

    Mapping rule in this file:
      odd trial_NN -> ascent
      even trial_NN -> descent
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--loc-ascent-descent-map not found: {p}")
    doc = json.loads(p.read_text())
    subjects = doc.get("subjects", {})
    if not isinstance(subjects, dict):
        raise ValueError("Invalid mapping JSON: missing dict field 'subjects'.")

    out: Dict[Tuple[str, str, str], str] = {}
    for sid_raw, conds in subjects.items():
        if not isinstance(conds, dict):
            continue
        sid = str(sid_raw).upper()
        for cond_raw, n_raw in conds.items():
            try:
                n_trials = int(n_raw)
            except Exception:
                continue
            cond = str(cond_raw)
            if "stair" in cond.lower():
                asc_label, dsc_label = "SA", "SD"
            elif "incline" in cond.lower() or "ramp" in cond.lower():
                asc_label, dsc_label = "RA", "RD"
            else:
                continue
            for i in range(1, max(0, n_trials) + 1):
                trial = f"trial_{i:02d}"
                out[(sid, cond, trial)] = asc_label if (i % 2 == 1) else dsc_label
    return out


def _classify_loc_bucket(
    subject_id: str,
    condition: str,
    trial: str,
    loc_map: Optional[Dict[Tuple[str, str, str], str]] = None,
) -> str:
    """
    Map H5 ``condition`` group + ``trial`` key to a coarse locomotion bucket for reporting.

    Priority: trial prefix token (LG/RA/RD/SA/SD) → explicit ramp/stair naming in condition/trial
    text → LG prefixes → OTHER.
    """
    sid = (subject_id or "").strip().upper()
    c = (condition or "").strip().lower()
    tr = (trial or "").strip().lower()
    if loc_map:
        mapped = loc_map.get((sid, condition, trial))
        if mapped is not None:
            return mapped

    trial_first = (trial or "").strip().split("_")[0].upper()
    if trial_first == "LG":
        return "LG"
    if trial_first == "RA":
        return "RA"
    if trial_first == "RD":
        return "RD"
    if trial_first == "SA":
        return "SA"
    if trial_first == "SD":
        return "SD"

    if (
        "incline" in c
        or c.startswith("incline_")
        or re.match(r"^incline_\d+_[lr]$", c)
    ):
        if re.search(r"(_up|up_|ascent)", c) or re.search(r"(_up|up_|ascent)", tr):
            return "RA"
        if re.search(r"(_down|down_|descent)", c) or re.search(r"(_down|down_|descent)", tr):
            return "RD"
        return "RA/RD"

    if (
        "stair" in c
        or c.startswith("stair_")
        or re.match(r"^stair_\d+_[lr]$", c)
    ):
        if re.search(r"(_up|up_|ascent)", c) or re.search(r"(_up|up_|ascent)", tr):
            return "SA"
        if re.search(r"(_down|down_|descent)", c) or re.search(r"(_down|down_|descent)", tr):
            return "SD"
        return "SA/SD"

    if (
        c.startswith("levelground_")
        or "stair" in c
        or c.startswith("treadmill_")
    ):
        return "LG"

    # AddBiomechanics / OpenCap unified exports (dataset.classify_loc_bucket parity).
    if "uphillrun" in c:
        return "RA"
    if "downhillrun" in c:
        return "RD"
    if c.startswith("flatrun"):
        return "LG"
    if c.startswith("walk_"):
        return "LG"
    if re.match(r"^walking\d+$", c) or re.match(r"^walkingts\d+$", c):
        return "LG"
    if c in ("baseline", "fpa", "step_width", "trunk_sway"):
        return "LG"
    if re.match(r"^subj\d+_(run|walk)_\d+$", c):
        return "LG"

    return "OTHER"


def _trial_error_accumulators(n_ch: int) -> Dict[str, Any]:
    return {
        "sum_sq_d": np.zeros(n_ch, dtype=np.float64),
        "sum_sq_c": np.zeros(n_ch, dtype=np.float64),
        "sum_sq_gtang": np.zeros(n_ch, dtype=np.float64),
        "sum_abs_d": np.zeros(n_ch, dtype=np.float64),
        "sum_abs_c": np.zeros(n_ch, dtype=np.float64),
        "sum_abs_gtang": np.zeros(n_ch, dtype=np.float64),
        "sum_t": np.zeros(n_ch, dtype=np.float64),
        "sum_t2": np.zeros(n_ch, dtype=np.float64),
        "n_elem": 0,
        "sum_sq_all_d": 0.0,
        "sum_sq_all_c": 0.0,
        "sum_sq_all_gtang": 0.0,
        "sum_abs_all_d": 0.0,
        "sum_abs_all_c": 0.0,
        "sum_abs_all_gtang": 0.0,
        "sum_t_all": 0.0,
        "sum_t2_all": 0.0,
        "n_all": 0,
        "n_trials": 0,
    }


def _accum_add_trial_errors(
    acc: Dict[str, Any],
    diff_d: np.ndarray,
    diff_c: np.ndarray,
    diff_gtang: np.ndarray,
    y_b: np.ndarray,
) -> None:
    """Accumulate one trial's finite-frame errors (diff_* and GT y_b same length)."""
    sum_sq_d = acc["sum_sq_d"]
    sum_sq_c = acc["sum_sq_c"]
    sum_sq_gtang = acc["sum_sq_gtang"]
    sum_abs_d = acc["sum_abs_d"]
    sum_abs_c = acc["sum_abs_c"]
    sum_abs_gtang = acc["sum_abs_gtang"]
    sum_t = acc["sum_t"]
    sum_t2 = acc["sum_t2"]

    sum_sq_d += np.sum(diff_d**2, axis=0)
    sum_sq_c += np.sum(diff_c**2, axis=0)
    sum_sq_gtang += np.sum(diff_gtang**2, axis=0)
    sum_abs_d += np.sum(np.abs(diff_d), axis=0)
    sum_abs_c += np.sum(np.abs(diff_c), axis=0)
    sum_abs_gtang += np.sum(np.abs(diff_gtang), axis=0)
    sum_t += np.sum(y_b, axis=0)
    sum_t2 += np.sum(y_b**2, axis=0)
    n_b = int(y_b.shape[0])
    acc["n_elem"] += n_b
    acc["sum_sq_all_d"] += float(np.sum(diff_d**2))
    acc["sum_sq_all_c"] += float(np.sum(diff_c**2))
    acc["sum_sq_all_gtang"] += float(np.sum(diff_gtang**2))
    acc["sum_abs_all_d"] += float(np.sum(np.abs(diff_d)))
    acc["sum_abs_all_c"] += float(np.sum(np.abs(diff_c)))
    acc["sum_abs_all_gtang"] += float(np.sum(np.abs(diff_gtang)))
    acc["sum_t_all"] += float(np.sum(y_b))
    acc["sum_t2_all"] += float(np.sum(y_b**2))
    acc["n_all"] += int(y_b.size)
    acc["n_trials"] += 1


def _finalize_bucket_metrics(
    dof_names: List[str],
    acc: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return _finalize_metrics(
        dof_names,
        acc["sum_sq_d"],
        acc["sum_sq_c"],
        acc["sum_sq_gtang"],
        acc["sum_abs_d"],
        acc["sum_abs_c"],
        acc["sum_abs_gtang"],
        acc["sum_t"],
        acc["sum_t2"],
        int(acc["n_elem"]),
        float(acc["sum_sq_all_d"]),
        float(acc["sum_sq_all_c"]),
        float(acc["sum_sq_all_gtang"]),
        float(acc["sum_abs_all_d"]),
        float(acc["sum_abs_all_c"]),
        float(acc["sum_abs_all_gtang"]),
        float(acc["sum_t_all"]),
        float(acc["sum_t2_all"]),
        int(acc["n_all"]),
    )


def _expected_side_ik_indices(full_indices: Sequence[int], eval_side: str) -> List[int]:
    h = len(full_indices) // 2
    if h * 2 != len(full_indices):
        raise ValueError("IK input_indices must split into equal R/L halves.")
    if eval_side == "right":
        return [int(x) for x in full_indices[:h]]
    if eval_side == "left":
        return [int(x) for x in full_indices[h:]]
    raise ValueError(f"eval_side must be 'right' or 'left', got {eval_side!r}")


def _finalize_metrics(
    dof_names: List[str],
    sum_sq_d: np.ndarray,
    sum_sq_c: np.ndarray,
    sum_sq_gtang: np.ndarray,
    sum_abs_d: np.ndarray,
    sum_abs_c: np.ndarray,
    sum_abs_gtang: np.ndarray,
    sum_t: np.ndarray,
    sum_t2: np.ndarray,
    n_elem: int,
    sum_sq_all_d: float,
    sum_sq_all_c: float,
    sum_sq_all_gtang: float,
    sum_abs_all_d: float,
    sum_abs_all_c: float,
    sum_abs_all_gtang: float,
    sum_t_all: float,
    sum_t2_all: float,
    n_all: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    n_ch = len(dof_names)

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
    met_gtang = finalize(sum_sq_gtang, sum_abs_gtang, sum_sq_all_gtang, sum_abs_all_gtang)
    return met_d, met_c, met_gtang


def _run_full_trial_comparison(
    direct_model: torch.nn.Module,
    angle_model: torch.nn.Module,
    ik_model: torch.nn.Module,
    trial_refs: List[TrialRef],
    meta_map: Dict[str, Dict],
    imu_schema_right: List[Tuple[str, str]],
    imu_schema_left: List[Tuple[str, str]],
    stats_imu: Dict[str, np.ndarray],
    ik_stats: Dict[str, np.ndarray],
    side_ik_indices: List[int],
    dof_names: List[str],
    eval_side: str,
    device: str,
    *,
    apply_lpf: bool,
    lpf_hz: float,
    lpf_order: int,
    cascade_velocity_lowpass: bool,
    ik_input_normalize: bool,
    target_sample_rate_hz: Optional[float],
    rollout_decimate_step: int,
    loc_bucket_map: Optional[Dict[Tuple[str, str, str], str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], int, int, Dict[str, Any]]:
    """Returns (met_direct, met_cascade, met_ik_from_gt_angles, n_trials_used, n_frames, per_loc_condition)."""
    n_ch = len(dof_names)
    overall = _trial_error_accumulators(n_ch)
    bucket_accs: Dict[str, Dict[str, Any]] = {}

    imu_mean = stats_imu["imu_mean"]
    imu_std = stats_imu["imu_std"]

    for t_idx, ref in enumerate(trial_refs):
        if (t_idx + 1) % 200 == 0:
            print(f"  [V4] trial {t_idx + 1}/{len(trial_refs)} …")

        trial = _load_trial_imu_sagittal_paired(
            ref,
            meta_map,
            imu_schema_right,
            imu_schema_left,
            "moment",
            apply_lowpass_filter=apply_lpf,
            lowpass_cutoff_hz=lpf_hz,
            lowpass_order=lpf_order,
            target_sample_rate_hz=target_sample_rate_hz,
            rollout_decimate_step=rollout_decimate_step,
        )
        if trial is None:
            continue

        if eval_side == "right":
            imu = trial["imu_right"]
            y_gt = trial["y_right"]
        else:
            imu = trial["imu_left"]
            y_gt = trial["y_left"]
        pos23 = trial["positions"]
        time_1d = trial["time"]

        mask = (
            np.isfinite(imu).all(axis=1)
            & np.isfinite(y_gt).all(axis=1)
            & np.isfinite(pos23).all(axis=1)
            & np.isfinite(time_1d)
        )
        if not np.any(mask):
            continue

        imu = imu[mask]
        y_gt = y_gt[mask]
        pos23 = pos23[mask]
        time_1d = time_1d[mask]

        pred_d = infer_imu_head_full_sequence(
            direct_model,
            imu,
            imu_mean,
            imu_std,
            time_1d,
            device,
            pipeline_lpf_apply=apply_lpf,
            pipeline_lpf_cutoff_hz=lpf_hz,
            pipeline_lpf_order=lpf_order,
        )
        pred_c, _ = infer_cascade_moments_full_sequence(
            angle_model,
            ik_model,
            imu,
            pos23,
            time_1d,
            imu_mean,
            imu_std,
            ik_stats,
            side_ik_indices,
            device,
            eval_side,
            pipeline_lpf_apply=apply_lpf,
            pipeline_lpf_cutoff_hz=lpf_hz,
            pipeline_lpf_order=lpf_order,
            cascade_velocity_lowpass=cascade_velocity_lowpass,
            ik_input_normalize=ik_input_normalize,
        )
        pred_gtang = infer_ik_moments_from_gt_angles_full_sequence(
            ik_model,
            pos23,
            time_1d,
            ik_stats,
            side_ik_indices,
            device,
            eval_side,
            pipeline_lpf_apply=apply_lpf,
            pipeline_lpf_cutoff_hz=lpf_hz,
            pipeline_lpf_order=lpf_order,
            cascade_velocity_lowpass=cascade_velocity_lowpass,
            ik_input_normalize=ik_input_normalize,
        )

        fin = (
            np.isfinite(pred_d).all(axis=1)
            & np.isfinite(pred_c).all(axis=1)
            & np.isfinite(pred_gtang).all(axis=1)
            & np.isfinite(y_gt).all(axis=1)
        )
        if not np.any(fin):
            continue
        pred_d = pred_d[fin]
        pred_c = pred_c[fin]
        pred_gtang = pred_gtang[fin]
        y_b = y_gt[fin]

        diff_d = pred_d.astype(np.float64) - y_b.astype(np.float64)
        diff_c = pred_c.astype(np.float64) - y_b.astype(np.float64)
        diff_gtang = pred_gtang.astype(np.float64) - y_b.astype(np.float64)
        _accum_add_trial_errors(overall, diff_d, diff_c, diff_gtang, y_b)
        bk = _classify_loc_bucket(ref[0], ref[1], ref[2], loc_map=loc_bucket_map)
        if bk not in bucket_accs:
            bucket_accs[bk] = _trial_error_accumulators(n_ch)
        _accum_add_trial_errors(bucket_accs[bk], diff_d, diff_c, diff_gtang, y_b)

    n_trials_ok = int(overall["n_trials"])
    n_elem = int(overall["n_elem"])
    met_d, met_c, met_gtang = _finalize_metrics(
        dof_names,
        overall["sum_sq_d"],
        overall["sum_sq_c"],
        overall["sum_sq_gtang"],
        overall["sum_abs_d"],
        overall["sum_abs_c"],
        overall["sum_abs_gtang"],
        overall["sum_t"],
        overall["sum_t2"],
        n_elem,
        float(overall["sum_sq_all_d"]),
        float(overall["sum_sq_all_c"]),
        float(overall["sum_sq_all_gtang"]),
        float(overall["sum_abs_all_d"]),
        float(overall["sum_abs_all_c"]),
        float(overall["sum_abs_all_gtang"]),
        float(overall["sum_t_all"]),
        float(overall["sum_t2_all"]),
        int(overall["n_all"]),
    )

    per_loc: Dict[str, Any] = {}
    for bk in sorted(bucket_accs.keys()):
        acc = bucket_accs[bk]
        if int(acc["n_elem"]) == 0:
            continue
        bd, bc, bgt = _finalize_bucket_metrics(dof_names, acc)
        per_loc[bk] = {
            "description": LOC_BUCKET_LEGEND.get(bk, ""),
            "n_trials": int(acc["n_trials"]),
            "n_frames_per_channel": int(acc["n_elem"]),
            "direct_imu_to_moment": bd,
            "cascade_imu_angle_then_ik_moment": bc,
            "ik_moment_from_gt_angles": bgt,
        }

    return met_d, met_c, met_gtang, n_trials_ok, n_elem, per_loc


def _fs_safe_segment(name: str) -> str:
    return "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in (name or ""))


@torch.no_grad()
def _write_plotly_trial_v4(
    *,
    ref: TrialRef,
    meta_map: Dict[str, Dict],
    imu_schema_right: List[Tuple[str, str]],
    imu_schema_left: List[Tuple[str, str]],
    stats_imu: Dict[str, np.ndarray],
    ik_stats: Dict[str, Any],
    m_direct: torch.nn.Module,
    m_angle: torch.nn.Module,
    ik_model: torch.nn.Module,
    side_ik_indices: List[int],
    eval_side: str,
    device: str,
    subject_out: Path,
    apply_lpf: bool,
    lpf_hz: float,
    lpf_order: int,
    cascade_velocity_lowpass: bool,
    ik_input_normalize: bool,
    target_sample_rate_hz: Optional[float],
    rollout_decimate_step: int,
    write_combined_html: bool,
) -> Dict[str, Any]:
    """One trial → Plotly HTML (same layout/style as ``plot_pipeline_compare_trial_inferenceV2`` causal)."""
    if not HAS_PLOTLY:
        raise RuntimeError("plotly is required for visualization.")
    sid, condition, trial, _h5_path = ref
    sid = sid.upper()
    subject_out.mkdir(parents=True, exist_ok=True)

    trial_data = _load_trial_imu_sagittal_paired(
        ref,
        meta_map,
        imu_schema_right,
        imu_schema_left,
        "moment",
        apply_lowpass_filter=apply_lpf,
        lowpass_cutoff_hz=lpf_hz,
        lowpass_order=lpf_order,
        target_sample_rate_hz=target_sample_rate_hz,
        rollout_decimate_step=rollout_decimate_step,
        trim_nonfinite_imu_suffix=True,
    )
    if trial_data is None:
        raise RuntimeError("trial load returned None (ik+id+imu).")

    if eval_side == "right":
        imu = trial_data["imu_right"]
        y_true = trial_data["y_right"]
    else:
        imu = trial_data["imu_left"]
        y_true = trial_data["y_left"]

    pos6 = trial_data["pos_sagittal_rl"]
    time = trial_data["time"]
    t_rel = (time - time[0]).astype(np.float64)

    pred_d = infer_imu_head_full_sequence(
        m_direct,
        imu,
        stats_imu["imu_mean"],
        stats_imu["imu_std"],
        time,
        device,
        pipeline_lpf_apply=apply_lpf,
        pipeline_lpf_cutoff_hz=lpf_hz,
        pipeline_lpf_order=lpf_order,
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
        pipeline_lpf_apply=apply_lpf,
        pipeline_lpf_cutoff_hz=lpf_hz,
        pipeline_lpf_order=lpf_order,
        cascade_velocity_lowpass=cascade_velocity_lowpass,
        ik_input_normalize=ik_input_normalize,
    )
    pred_gtang = infer_ik_moments_from_gt_angles_full_sequence(
        ik_model,
        trial_data["positions"],
        trial_data["time"],
        ik_stats,
        side_ik_indices,
        device,
        eval_side,
        pipeline_lpf_apply=apply_lpf,
        pipeline_lpf_cutoff_hz=lpf_hz,
        pipeline_lpf_order=lpf_order,
        cascade_velocity_lowpass=cascade_velocity_lowpass,
        ik_input_normalize=ik_input_normalize,
    )

    if eval_side == "right":
        y_true_ang = pos6[:, :3].copy()
        angle_dof_names = [IK_DOF_NAMES[i] for i in SAGITTAL_INPUT_INDICES[:3]]
    else:
        y_true_ang = pos6[:, 3:6].copy()
        angle_dof_names = [IK_DOF_NAMES[i] for i in SAGITTAL_INPUT_INDICES[3:]]

    if eval_side == "right":
        dof_names_m = [MOMENT_NAMES[i] for i in SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES[:3]]
    else:
        dof_names_m = [MOMENT_NAMES[i] for i in SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES[3:]]

    _lpf_base = (
        f"pipeline LPF {lpf_hz} Hz, order {lpf_order}" if apply_lpf else "pipeline LPF off"
    )
    _lpf_note = f"single-stream causal + trial-axis LPF ({_lpf_base})"
    _ang_pred_label = "IMU → angle (full-sequence causal)"

    cond_s = _fs_safe_segment(condition)
    trial_s = _fs_safe_segment(trial)
    file_prefix = f"{sid}_{cond_s}_{trial_s}_{eval_side}"

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
        title_text=f"{sid} {condition} {trial} — {eval_side} — joint angles ({_lpf_note}) — compare_pipelineV4",
        template="plotly_white",
        hovermode="x unified",
    )
    fig_ang.update_xaxes(title_text="Time (s)", row=3, col=1)
    out_ang = subject_out / f"{file_prefix}_joint_angles_pipeline_v4.html"
    fig_ang.write_html(str(out_ang), include_plotlyjs="cdn", full_html=True)

    fig_mom = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=dof_names_m,
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
        fig_mom.update_yaxes(title_text=f"{dof_names_m[c]} (N·m/kg)", row=row, col=1)
    fig_mom.update_layout(
        height=900,
        title_text=f"{sid} {condition} {trial} — {eval_side} — moments ({_lpf_note}) — compare_pipelineV4",
        template="plotly_white",
        hovermode="x unified",
    )
    fig_mom.update_xaxes(title_text="Time (s)", row=3, col=1)
    out_mom = subject_out / f"{file_prefix}_joint_moments_pipeline_v4.html"
    fig_mom.write_html(str(out_mom), include_plotlyjs="cdn", full_html=True)

    fig_mom_gtang = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=dof_names_m,
    )
    for c in range(3):
        row = c + 1
        fig_mom_gtang.add_trace(
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
        fig_mom_gtang.add_trace(
            go.Scatter(
                x=t_rel,
                y=pred_gtang[:, c],
                mode="lines",
                name="IK-ID from GT angles (LPF)",
                line=dict(_MOMENT_LINE_CASCADE_SUB),
                legendgroup="gtang",
                showlegend=(c == 0),
            ),
            row=row,
            col=1,
        )
        fig_mom_gtang.update_yaxes(title_text=f"{dof_names_m[c]} (N·m/kg)", row=row, col=1)
    fig_mom_gtang.update_layout(
        height=900,
        title_text=(
            f"{sid} {condition} {trial} — {eval_side} — moments "
            f"(GT vs IK-ID from GT angles; {_lpf_note}) — compare_pipelineV4"
        ),
        template="plotly_white",
        hovermode="x unified",
    )
    fig_mom_gtang.update_xaxes(title_text="Time (s)", row=3, col=1)
    out_mom_gtang = subject_out / f"{file_prefix}_joint_moments_ik_from_gt_angles_pipeline_v4.html"
    fig_mom_gtang.write_html(str(out_mom_gtang), include_plotlyjs="cdn", full_html=True)

    combined_name: Optional[str] = None
    if write_combined_html:
        fig_all = make_subplots(
            rows=6,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.035,
            subplot_titles=[*angle_dof_names, *dof_names_m],
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
                go.Scatter(
                    x=t_rel,
                    y=y_true[:, c],
                    mode="lines",
                    name="GT",
                    line=dict(_MOMENT_LINE_GT_SUB),
                    legendgroup="mgt",
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
                    name="Direct LPF",
                    line=dict(_MOMENT_LINE_DIRECT_SUB),
                    legendgroup="md",
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
                    name="Cascade LPF",
                    line=dict(_MOMENT_LINE_CASCADE_SUB),
                    legendgroup="mc",
                    showlegend=(c == 0),
                ),
                row=row,
                col=1,
            )
            fig_all.update_yaxes(title_text=f"{dof_names_m[c]} (N·m/kg)", row=row, col=1)
        fig_all.update_layout(
            height=1500,
            title_text=f"{sid} {condition} {trial} — {eval_side} — angles + moments ({_lpf_note})",
            template="plotly_white",
            hovermode="x unified",
        )
        fig_all.update_xaxes(title_text="Time (s)", row=6, col=1)
        combined_name = f"{file_prefix}_joint_angles_and_moments_pipeline_v4.html"
        fig_all.write_html(str(subject_out / combined_name), include_plotlyjs="cdn", full_html=True)

    return {
        "subject_id": sid,
        "condition": condition,
        "trial": trial,
        "eval_side": eval_side,
        "joint_angles_html": str(out_ang.relative_to(subject_out.parent)),
        "joint_moments_html": str(out_mom.relative_to(subject_out.parent)),
        "joint_moments_ik_from_gt_angles_html": str(out_mom_gtang.relative_to(subject_out.parent)),
        "combined_html": (
            str((subject_out / combined_name).relative_to(subject_out.parent))
            if combined_name
            else None
        ),
    }


def _visualize_v4_all_subjects(
    *,
    trial_refs: List[TrialRef],
    eval_ids: Sequence[str],
    meta_map: Dict[str, Dict],
    imu_schema_right: List[Tuple[str, str]],
    imu_schema_left: List[Tuple[str, str]],
    stats_imu: Dict[str, np.ndarray],
    ik_stats: Dict[str, Any],
    m_direct: torch.nn.Module,
    m_angle: torch.nn.Module,
    ik_model: torch.nn.Module,
    side_ik_indices: List[int],
    eval_side: str,
    device: str,
    out_dir: Path,
    apply_lpf: bool,
    lpf_hz: float,
    lpf_order: int,
    cascade_velocity_lowpass: bool,
    ik_input_normalize: bool,
    target_sample_rate_hz: Optional[float],
    rollout_decimate_step: int,
    sample_rate_hz: float,
    write_combined_html: bool,
    ckpt_paths: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Write Plotly HTML under ``out_dir / S* /`` for every evaluation trial."""
    if not HAS_PLOTLY:
        raise SystemExit("Install plotly for --visualize: pip install plotly")

    ik_np = _ik_stats_as_numpy(ik_stats)
    viz_index: List[Dict[str, Any]] = []
    sid_set = {s.upper() for s in eval_ids}

    for sid in sorted(sid_set):
        subject_out = out_dir / sid
        n_written = 0
        for ref in trial_refs:
            if ref[0].upper() != sid:
                continue
            try:
                entry = _write_plotly_trial_v4(
                    ref=ref,
                    meta_map=meta_map,
                    imu_schema_right=imu_schema_right,
                    imu_schema_left=imu_schema_left,
                    stats_imu=stats_imu,
                    ik_stats=ik_np,
                    m_direct=m_direct,
                    m_angle=m_angle,
                    ik_model=ik_model,
                    side_ik_indices=side_ik_indices,
                    eval_side=eval_side,
                    device=device,
                    subject_out=subject_out,
                    apply_lpf=apply_lpf,
                    lpf_hz=lpf_hz,
                    lpf_order=lpf_order,
                    cascade_velocity_lowpass=cascade_velocity_lowpass,
                    ik_input_normalize=ik_input_normalize,
                    target_sample_rate_hz=target_sample_rate_hz,
                    rollout_decimate_step=rollout_decimate_step,
                    write_combined_html=write_combined_html,
                )
                viz_index.append(entry)
                n_written += 1
            except Exception as exc:
                print(f"  [visualize] skip {ref[0]} / {ref[1]} / {ref[2]}: {exc}")
        print(f"  [visualize] {sid}: wrote {n_written} trial figure set(s) → {subject_out}")

    manifest = {
        **ckpt_paths,
        "pipeline_version": "V4_plot_inference_compare_pipelineV4",
        "inference_reference": "compare_pipelineV4.py",
        "eval_side": eval_side,
        "sample_rate_hz": sample_rate_hz,
        "target_sample_rate_hz": target_sample_rate_hz,
        "rollout_decimate_step": int(rollout_decimate_step),
        "neural_implementation": "full_sequence_causal",
        "apply_lowpass_filter": apply_lpf,
        "lowpass_cutoff_hz": lpf_hz,
        "lowpass_order": lpf_order,
        "cascade_velocity_lowpass": cascade_velocity_lowpass,
        "ik_input_normalize": ik_input_normalize,
        "write_combined_html": write_combined_html,
        "trials": viz_index,
    }
    with open(out_dir / "visualization_index.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return viz_index, manifest


def _resolve_lpf_from_run_cfg(run_cfg: Optional[Dict[str, Any]]) -> Tuple[Optional[bool], Optional[float], Optional[int]]:
    if run_cfg is None:
        return None, None, None
    has_any = any(k in run_cfg for k in ("no_lowpass", "lowpass_cutoff_hz", "lowpass_order"))
    if not has_any:
        return None, None, None
    apply = not bool(run_cfg.get("no_lowpass", False))
    cutoff = float(run_cfg.get("lowpass_cutoff_hz", 4.0))
    order = int(run_cfg.get("lowpass_order", 4))
    return apply, cutoff, order


def _assert_lpf_consistency(
    tag: str,
    run_cfg: Optional[Dict[str, Any]],
    *,
    expected_apply: bool,
    expected_cutoff: float,
    expected_order: int,
) -> None:
    apply, cutoff, order = _resolve_lpf_from_run_cfg(run_cfg)
    if apply is None:
        print(f"  [warn] {tag}: no LPF keys in run config; cannot verify LPF consistency.")
        return
    mismatch: List[str] = []
    if bool(apply) != bool(expected_apply):
        mismatch.append(f"apply train={apply} eval={expected_apply}")
    if abs(float(cutoff) - float(expected_cutoff)) > 1e-9:
        mismatch.append(f"cutoff_hz train={cutoff} eval={expected_cutoff}")
    if int(order) != int(expected_order):
        mismatch.append(f"order train={order} eval={expected_order}")
    if mismatch:
        raise ValueError(
            f"LPF settings mismatch for {tag}: " + ", ".join(mismatch)
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="V4: full-trial direct vs cascade compare (see compare_pipelineV4.py)"
    )
    p.add_argument("--imu-moment-ckpt", type=str, required=True)
    p.add_argument("--imu-angle-ckpt", type=str, required=True)
    p.add_argument("--ik-moment-ckpt", type=str, required=True)
    p.add_argument("--test-dir", type=str, required=True)
    p.add_argument("--meta-root", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="results/pipeline_compare_v4")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--eval-split", type=str, default="test", choices=["test", "val"])
    p.add_argument(
        "--rollout",
        action="store_true",
        default=False,
        help="Use stride-2 decimation (~100 Hz). If not set, follows checkpoint config rollout_decimate_step.",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--walking-only", action="store_true", default=True)
    p.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    p.add_argument("--levelground-only", action="store_true", default=False)
    p.add_argument(
        "--dataset-filters-from-checkpoint",
        action="store_true",
        default=False,
        help="If set, walking/levelground flags follow run_config.json next to the IMU moment ckpt (V3 behavior).",
    )
    p.add_argument("--eval-side", type=str, default="right", choices=["right", "left"])
    p.add_argument(
        "--ik-input-normalize",
        action="store_true",
        default=False,
        help="Z-score IK inputs using checkpoint stats (only if IK was trained with normalize=True).",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Write Plotly HTML per trial under output_dir/S*/ (same style as plot_pipeline_compare_trial_inferenceV2).",
    )
    p.add_argument(
        "--visualize-combined-html",
        action="store_true",
        default=False,
        help="With --visualize, also write 6-row combined angles+moments HTML per trial.",
    )
    p.add_argument(
        "--cascade-velocity-lowpass",
        action="store_true",
        default=True,
        help=(
            "Apply zero-phase LPF on cascade angular velocities after _compute_velocity. "
            "Default off to match IK training data semantics."
        ),
    )
    p.add_argument(
        "--loc-ascent-descent-map",
        type=str,
        default=None,
        help=(
            "Optional JSON mapping (e.g., jinwoo_epic_ascent_descent_mapping.json) "
            "used to split per_loc_condition into SA/SD and RA/RD via explicit "
            "(subject, condition, trial) labels."
        ),
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
        _w_imu,
        _stride_imu,
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
        _w_ang,
        _stride_ang,
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

    print("Loading IK→moment (paired ipsilateral) …")
    try:
        _lm = load_model(args.ik_moment_ckpt, device)
        if len(_lm) == 11:
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
            ) = _lm
        elif len(_lm) == 10:
            (
                ik_model,
                ik_stats,
                dof_names_ik,
                w_ik,
                input_indices,
                _moment_indices,
                input_mode,
                output_mode,
                _lat_ik,
                _paired_ik,
            ) = _lm
            _stride_ik = None
        else:
            raise ValueError(f"Unexpected load_model() return length: {len(_lm)}")
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
        w_ik = int(ck.get("window_size", _w_imu))
        input_indices = ck.get("input_indices")
        input_mode = ck.get("input_mode", "unknown")
        output_mode = ck.get("output_mode", "unknown")

    if ik_stats is None:
        raise ValueError("IK moment checkpoint missing normalization stats.")

    n_in = ik_model.n_input_channels
    n_out = ik_model.n_output_channels
    n_sag = len(SAGITTAL_INPUT_INDICES)
    half = n_sag // 2
    if n_in != 2 * half or n_out != half:
        raise ValueError(
            "compare_pipelineV4 expects a paired sagittal IK TCN "
            f"(n_in=2×{half}={2*half}, n_out={half}). Got n_in={n_in}, n_out={n_out} "
            f"(input_mode={input_mode!r} output_mode={output_mode!r})."
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
    side_ik_idx = _expected_side_ik_indices(input_indices, args.eval_side)
    dof_names = list(out_names_mr if args.eval_side == "right" else out_names_ml)
    if list(dof_names_ik) != list(dof_names):
        print(f"  [warn] IK dof_names {dof_names_ik} != IMU output names {dof_names}; using IMU names for tables.")

    run_cfg = load_run_config(args.imu_moment_ckpt)
    run_cfg_angle = load_run_config(args.imu_angle_ckpt)
    run_cfg_ik = load_run_config(args.ik_moment_ckpt)
    apply_lowpass_filter = True
    lowpass_cutoff_hz = 4.0
    lowpass_order = 4
    if run_cfg is not None and any(
        k in run_cfg for k in ("no_lowpass", "lowpass_cutoff_hz", "lowpass_order")
    ):
        apply_lowpass_filter = not bool(run_cfg.get("no_lowpass", False))
        lowpass_cutoff_hz = float(run_cfg.get("lowpass_cutoff_hz", 4.0))
        lowpass_order = int(run_cfg.get("lowpass_order", 4))
    _assert_lpf_consistency(
        "imu_moment_ckpt",
        run_cfg,
        expected_apply=apply_lowpass_filter,
        expected_cutoff=lowpass_cutoff_hz,
        expected_order=lowpass_order,
    )
    _assert_lpf_consistency(
        "imu_angle_ckpt",
        run_cfg_angle,
        expected_apply=apply_lowpass_filter,
        expected_cutoff=lowpass_cutoff_hz,
        expected_order=lowpass_order,
    )
    _assert_lpf_consistency(
        "ik_moment_ckpt",
        run_cfg_ik,
        expected_apply=apply_lowpass_filter,
        expected_cutoff=lowpass_cutoff_hz,
        expected_order=lowpass_order,
    )

    _levelground_only = args.levelground_only
    _walking_only = args.walking_only
    if args.dataset_filters_from_checkpoint and run_cfg is not None:
        if "levelground_only" in run_cfg:
            _levelground_only = bool(run_cfg["levelground_only"])
        if "walking_only" in run_cfg:
            _walking_only = bool(run_cfg["walking_only"])

    imu_rollout_step = 1
    if args.rollout:
        imu_rollout_step = 2
    elif run_cfg is not None:
        imu_rollout_step = int(run_cfg.get("rollout_decimate_step", 1))
    elif ck_m.get("rollout_decimate_step") is not None:
        imu_rollout_step = int(ck_m["rollout_decimate_step"])
    imu_rollout_step = max(1, imu_rollout_step)

    imu_tgt_sr: Optional[float] = None

    if imu_rollout_step > 1:
        report_sr = 200.0 / float(imu_rollout_step)
    else:
        report_sr = 200.0

    eval_ids, mode = _resolve_eval_subjects(
        test_root,
        Path(args.imu_moment_ckpt),
        args.eval_split,
        args.max_files,
    )
    meta_map = _load_subject_metadata_map(meta_root)
    trial_refs = _collect_trial_refs(
        test_root,
        eval_ids,
        walking_only=_walking_only,
        levelground_only=_levelground_only,
    )

    print(f"Eval subjects ({mode}): {eval_ids}")
    print(
        f"Full-trial eval: stride=1 (per-frame), IK window_size={w_ik} (training only; not used for tiling)  "
        f"sample_rate_hz={report_sr}"
        + (f"  rollout_decimate_step={imu_rollout_step}" if imu_rollout_step > 1 else "")
    )
    print(f"Condition filters: walking_only={_walking_only}  levelground_only={_levelground_only}")
    if args.dataset_filters_from_checkpoint:
        print("  (filters synced from checkpoint run_config where keys exist)")
    print(f"Trial refs (pre-load filter): {len(trial_refs):,}")
    print(f"Eval side: {args.eval_side}")
    print(
        f"[V4] Pipeline zero-phase LPF: apply={apply_lowpass_filter} "
        f"({lowpass_cutoff_hz} Hz, order {lowpass_order})"
    )
    print(
        f"[V4] Cascade angular-velocity LPF after derivative: "
        f"{'on' if args.cascade_velocity_lowpass else 'off'}"
    )

    loc_bucket_map = _load_ascent_descent_mapping(args.loc_ascent_descent_map)
    if args.loc_ascent_descent_map:
        print(
            f"[V4] Loaded explicit loc ascent/descent mapping entries: {len(loc_bucket_map):,} "
            f"from {Path(args.loc_ascent_descent_map).resolve()}"
        )

    met_d, met_c, met_gtang, n_trials_ok, n_frames, per_loc = _run_full_trial_comparison(
        m_direct,
        m_angle,
        ik_model,
        trial_refs,
        meta_map,
        list(schema_mr),
        list(schema_ml),
        stats_imu_m,
        ik_stats,
        side_ik_idx,
        dof_names,
        str(args.eval_side),
        device,
        apply_lpf=apply_lowpass_filter,
        lpf_hz=lowpass_cutoff_hz,
        lpf_order=lowpass_order,
        cascade_velocity_lowpass=bool(args.cascade_velocity_lowpass),
        ik_input_normalize=ik_input_normalize,
        target_sample_rate_hz=imu_tgt_sr,
        rollout_decimate_step=imu_rollout_step,
        loc_bucket_map=loc_bucket_map,
    )

    summary = {
        "pipeline_version": "V4_paired_ik_6x3_full_trial_inference",
        "test_dir": str(test_root.resolve()),
        "eval_split": args.eval_split,
        "eval_side": args.eval_side,
        "eval_mode": mode,
        "subjects": eval_ids,
        "n_trials_evaluated": n_trials_ok,
        "n_frames_per_channel": n_frames,
        "time_stride": 1,
        "note": "Metrics aggregate all finite frames over trials (no sliding windows).",
        "condition_filters": {
            "walking_only": _walking_only,
            "levelground_only": _levelground_only,
            "dataset_filters_from_checkpoint": bool(args.dataset_filters_from_checkpoint),
        },
        "sample_rate_hz": report_sr,
        "target_sample_rate_hz": imu_tgt_sr,
        "rollout_decimate_step": int(imu_rollout_step),
        "imu_moment_checkpoint": str(Path(args.imu_moment_ckpt).resolve()),
        "imu_angle_checkpoint": str(Path(args.imu_angle_ckpt).resolve()),
        "ik_moment_checkpoint": str(Path(args.ik_moment_ckpt).resolve()),
        "ik_window_size_training": w_ik,
        "ik_n_input_channels": n_in,
        "ik_n_output_channels": n_out,
        "ik_input_normalize": ik_input_normalize,
        "pipeline_zero_phase_lowpass": {
            "apply": bool(apply_lowpass_filter),
            "cutoff_hz": float(lowpass_cutoff_hz),
            "order": int(lowpass_order),
            "imu_on_load": True,
            "cascade_predicted_angles": True,
            "cascade_angular_velocity_after_hybrid_ik": bool(args.cascade_velocity_lowpass),
            "direct_moment_predictions": True,
            "cascade_moment_predictions": True,
        },
        "direct_imu_to_moment": met_d,
        "cascade_imu_angle_then_ik_moment": met_c,
        "ik_moment_from_gt_angles": met_gtang,
        "loc_bucket_legend": LOC_BUCKET_LEGEND,
        "loc_ascent_descent_map": (
            str(Path(args.loc_ascent_descent_map).resolve())
            if args.loc_ascent_descent_map
            else None
        ),
        "per_loc_condition": per_loc,
    }

    if args.visualize:
        ckpt_paths_viz = {
            "imu_moment_checkpoint": str(Path(args.imu_moment_ckpt).resolve()),
            "imu_angle_checkpoint": str(Path(args.imu_angle_ckpt).resolve()),
            "ik_moment_checkpoint": str(Path(args.ik_moment_ckpt).resolve()),
        }
        viz_entries, _viz_manifest = _visualize_v4_all_subjects(
            trial_refs=trial_refs,
            eval_ids=eval_ids,
            meta_map=meta_map,
            imu_schema_right=list(schema_mr),
            imu_schema_left=list(schema_ml),
            stats_imu=stats_imu_m,
            ik_stats=ik_stats,
            m_direct=m_direct,
            m_angle=m_angle,
            ik_model=ik_model,
            side_ik_indices=side_ik_idx,
            eval_side=str(args.eval_side),
            device=device,
            out_dir=out_dir,
            apply_lpf=apply_lowpass_filter,
            lpf_hz=lowpass_cutoff_hz,
            lpf_order=lowpass_order,
            cascade_velocity_lowpass=bool(args.cascade_velocity_lowpass),
            ik_input_normalize=ik_input_normalize,
            target_sample_rate_hz=imu_tgt_sr,
            rollout_decimate_step=imu_rollout_step,
            sample_rate_hz=report_sr,
            write_combined_html=bool(args.visualize_combined_html),
            ckpt_paths=ckpt_paths_viz,
        )
        summary["visualization"] = {
            "n_trial_figures": len(viz_entries),
            "index_file": "visualization_index.json",
            "combined_html_enabled": bool(args.visualize_combined_html),
        }

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*96}\nRESULTS V4 full-trial (moment RMSE / R²)\n{'='*96}")
    print(f"Trials with ≥1 valid frame: {n_trials_ok:,}  |  frames summed: {n_frames:,}")
    print(
        f"{'DOF':<22s}  {'RMSE dir':>10s}  {'R² dir':>8s}  {'RMSE cas':>10s}  {'R² cas':>8s}  "
        f"{'RMSE gt-ang':>12s}  {'R² gt-ang':>10s}"
    )
    print("-" * 96)
    for a, b, g in zip(met_d["per_channel"], met_c["per_channel"], met_gtang["per_channel"]):
        print(
            f"{a['name']:<22s}  {a['rmse']:10.5f}  {a['r2']:8.4f}  {b['rmse']:10.5f}  {b['r2']:8.4f}  "
            f"{g['rmse']:12.5f}  {g['r2']:10.4f}"
        )
    od, oc, og = met_d["overall"], met_c["overall"], met_gtang["overall"]
    print("-" * 96)
    print(
        f"{'OVERALL':<22s}  {od['rmse']:10.5f}  {od['r2']:8.4f}  {oc['rmse']:10.5f}  {oc['r2']:8.4f}  "
        f"{og['rmse']:12.5f}  {og['r2']:10.4f}"
    )
    if per_loc:
        print(f"{'='*120}\nPer loc. condition (overall RMSE N·m/kg, same GT)\n{'='*120}")
        print(
            f"{'Bucket':<10s}  {'trials':>7s}  {'frames':>9s}  "
            f"{'RMSE dir':>10s}  {'R² dir':>8s}  {'RMSE cas':>10s}  {'R² cas':>8s}  "
            f"{'RMSE gt-ang':>12s}  {'R² gt-ang':>10s}"
        )
        print("-" * 120)
        for bk in sorted(per_loc.keys()):
            pl = per_loc[bk]
            d_o = pl["direct_imu_to_moment"]["overall"]
            c_o = pl["cascade_imu_angle_then_ik_moment"]["overall"]
            g_o = pl["ik_moment_from_gt_angles"]["overall"]
            print(
                f"{bk:<10s}  {pl['n_trials']:7d}  {pl['n_frames_per_channel']:9d}  "
                f"{d_o['rmse']:10.5f}  {d_o['r2']:8.4f}  {c_o['rmse']:10.5f}  {c_o['r2']:8.4f}  "
                f"{g_o['rmse']:12.5f}  {g_o['r2']:10.4f}"
            )
        print(f"{'='*96}\nPer loc. condition (per-joint RMSE / R²)\n{'='*96}")
        for bk in sorted(per_loc.keys()):
            pl = per_loc[bk]
            print(
                f"[{bk}] trials={pl['n_trials']}  frames/ch={pl['n_frames_per_channel']}"
            )
            print(
                f"{'DOF':<22s}  {'RMSE dir':>10s}  {'R² dir':>8s}  {'RMSE cas':>10s}  {'R² cas':>8s}  "
                f"{'RMSE gt-ang':>12s}  {'R² gt-ang':>10s}"
            )
            print("-" * 96)
            d_ch = pl["direct_imu_to_moment"]["per_channel"]
            c_ch = pl["cascade_imu_angle_then_ik_moment"]["per_channel"]
            g_ch = pl["ik_moment_from_gt_angles"]["per_channel"]
            for dc, cc, gc in zip(d_ch, c_ch, g_ch):
                print(
                    f"{dc['name']:<22s}  {dc['rmse']:10.5f}  {dc['r2']:8.4f}  {cc['rmse']:10.5f}  {cc['r2']:8.4f}  "
                    f"{gc['rmse']:12.5f}  {gc['r2']:10.4f}"
                )
            print("-" * 96)
            d_o = pl["direct_imu_to_moment"]["overall"]
            c_o = pl["cascade_imu_angle_then_ik_moment"]["overall"]
            g_o = pl["ik_moment_from_gt_angles"]["overall"]
            print(
                f"{'OVERALL':<22s}  {d_o['rmse']:10.5f}  {d_o['r2']:8.4f}  {c_o['rmse']:10.5f}  {c_o['r2']:8.4f}  "
                f"{g_o['rmse']:12.5f}  {g_o['r2']:10.4f}"
            )
            print()
    print(f"{'='*70}\nSaved {out_dir / 'comparison.json'}")
    if args.visualize:
        print(f"Saved visualization index: {out_dir / 'visualization_index.json'}")

    _plot_rmse_comparison(
        dof_names,
        [c["rmse"] for c in met_d["per_channel"]],
        [c["rmse"] for c in met_c["per_channel"]],
        out_dir / "rmse_direct_vs_cascade.png",
    )


if __name__ == "__main__":
    main()
