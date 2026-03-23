#!/usr/bin/env python3
"""
PCA on MeMo H5 walking trials: **separate** PCAs for IK and for ID.

- Uses only walking-like conditions (name contains user-specified keywords).
- For each trial, keeps only the first `--duration-sec` seconds (default 1.0).
- IK PCA: standardized IK angles (rad) and optionally angular velocities.
- ID PCA: standardized joint moments (N·m/kg), independent of IK.
- By default only **sagittal lower limb** (R then L: hip flexion, knee, ankle) is loaded
  from the H5 IK/ID tables; use ``--feature-set full`` for all model DOFs/moments.
- Rows with mostly-NaN features are dropped **per modality** (IK and ID can differ in N).

Outputs under `--output-dir/ik/` and `--output-dir/id/`:
  - pca_projection.npz, pca_components.npy, feature_names.json, explained_variance.json
  - scree.png, pc1_pc2_by_subject.png, loadings_heatmap.png
  - trial_outliers.json, trial_outliers.csv (subject-condition-trial outliers + H5 path + collection site).
    ID outliers use a **stricter** rule by default: higher |z| threshold, **>=2 PCs** over threshold,
    and an **IQR floor** on per-PC sigma (see CLI). IK keeps the original max-|z| rule unless
    ``--outlier-ik-min-dims`` is set.
  - trial_pc_centroids.csv (all trials: mean PC1..K in latent space for auditing)
  - ik_only_trials.json, ik_only_trials.csv (trials with usable IK rows but zero usable ID rows)

Collection site (from subject id in ``S###``):
  S001–S022 → Camargo, S023–S034 → Scherpereel, S035–S056 → Molinaro_Scherpereel

Requires: h5py, numpy, matplotlib, scikit-learn
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError as e:
    raise SystemExit("Install h5py: pip install h5py") from e

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("Install matplotlib: pip install matplotlib") from e

try:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit("Install scikit-learn: pip install scikit-learn") from e

# Reuse IK/ID definitions and H5 table reader
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dataset import (
    IK_DOF_NAMES,
    MOMENT_NAMES,
    SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES,
    SAGITTAL_INPUT_INDICES,
    _compute_velocity,
    _read_h5_opensim_table,
)

# Sagittal lower limb, R then L: hip flexion, knee angle, ankle (angles / matching moments).
SAGITTAL_LOWER_IK_INDICES: Tuple[int, ...] = tuple(SAGITTAL_INPUT_INDICES)
SAGITTAL_LOWER_MOMENT_INDICES: Tuple[int, ...] = tuple(SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES)
SAGITTAL_LOWER_IK_NAMES: Tuple[str, ...] = tuple(IK_DOF_NAMES[i] for i in SAGITTAL_LOWER_IK_INDICES)
SAGITTAL_LOWER_MOMENT_NAMES: Tuple[str, ...] = tuple(MOMENT_NAMES[i] for i in SAGITTAL_LOWER_MOMENT_INDICES)


def collection_site_from_subject_id(subject_id: str) -> str:
    """
    MeMo subject → data collection cohort folder name (per user mapping).

    S001–S022: Camargo; S023–S034: Scherpereel; S035–S056: Molinaro_Scherpereel.
    """
    m = re.match(r"^S(\d+)$", subject_id.strip().upper())
    if not m:
        return "unknown"
    n = int(m.group(1))
    if 1 <= n <= 22:
        return "Camargo"
    if 23 <= n <= 34:
        return "Scherpereel"
    if 35 <= n <= 56:
        return "Molinaro_Scherpereel"
    return "unknown"


def trial_location_fields(
    memo_root: Path,
    h5_path: Path,
    subject_id: str,
    condition: str,
    trial: str,
) -> Dict[str, str]:
    """Filesystem + HDF5 location strings for one trial."""
    h5_resolved = str(h5_path.resolve())
    group_path = f"{condition}/{trial}"
    site = collection_site_from_subject_id(subject_id)
    return {
        "subject_id": subject_id,
        "condition": condition,
        "trial": trial,
        "collection_site": site,
        "memo_root": str(memo_root.resolve()),
        "h5_file": h5_resolved,
        "hdf5_group_path": group_path,
        "h5_uri": f"{h5_resolved}::{group_path}",
        "label": f"{subject_id} / {condition} / {trial}",
    }


def _mad_scale(x: np.ndarray) -> np.ndarray:
    """Per-column robust scale: 1.4826 * median(|x - median(x)|)."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    return np.where(mad > 1e-12, 1.4826 * mad, 1.0)


def detect_trial_pc_outliers(
    Z: np.ndarray,
    row_metas: List[Dict[str, Any]],
    n_pc: int,
    mad_k: float,
    min_dims_exceeding: Optional[int] = None,
    mad_floor_iqr_frac: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Per-trial centroid in first ``n_pc`` PC scores.

    Robust z per PC: (centroid - median) / sigma, with sigma = 1.4826*MAD by default.
    If ``mad_floor_iqr_frac > 0``, sigma_j = max(MAD_scale_j, frac * IQR_j) to avoid
    tiny MAD on flat PCs blowing up z-scores.

    Outlier if either:
      - ``min_dims_exceeding`` is None: max_j |z_j| > ``mad_k`` (legacy).
      - else: count of j with |z_j| > ``mad_k`` is >= ``min_dims_exceeding``.

    Returns (outlier_records, all_trial_summaries).
    """
    n_pc = int(min(n_pc, Z.shape[1]))
    if n_pc < 1 or len(row_metas) != Z.shape[0]:
        return [], []

    if min_dims_exceeding is not None:
        min_dims_exceeding = min(int(min_dims_exceeding), n_pc)

    groups: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for i, meta in enumerate(row_metas):
        key = (meta["subject_id"], meta["condition"], meta["trial"])
        groups[key].append(i)

    trial_keys = sorted(groups.keys())
    centroids = np.zeros((len(trial_keys), n_pc), dtype=np.float64)
    meta0: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for ti, key in enumerate(trial_keys):
        idx = groups[key]
        centroids[ti] = np.mean(Z[idx, :n_pc], axis=0)
        meta0[key] = dict(row_metas[idx[0]])

    med = np.median(centroids, axis=0)
    scale = _mad_scale(centroids)
    if mad_floor_iqr_frac > 0:
        for j in range(n_pc):
            col = centroids[:, j]
            iqr = float(np.percentile(col, 75) - np.percentile(col, 25))
            scale[j] = max(float(scale[j]), mad_floor_iqr_frac * max(iqr, 1e-12))
    z_scores = (centroids - med) / scale

    all_rows: List[Dict[str, Any]] = []
    outlier_rows: List[Dict[str, Any]] = []
    for ti, key in enumerate(trial_keys):
        zs = z_scores[ti]
        worst_j = int(np.argmax(np.abs(zs)))
        worst_abs = float(np.abs(zs[worst_j]))
        n_dims_over = int(np.sum(np.abs(zs) > mad_k))
        if min_dims_exceeding is None:
            is_out = bool(worst_abs > mad_k)
            rule = (
                f"max(|robust_z|) over PC1..PC{n_pc} > {mad_k} "
                f"(MAD{' + IQR floor' if mad_floor_iqr_frac > 0 else ''} vs trial centroids)"
            )
        else:
            is_out = n_dims_over >= min_dims_exceeding
            rule = (
                f"at least {min_dims_exceeding} of PC1..PC{n_pc} have |robust_z| > {mad_k} "
                f"(MAD{' + IQR floor' if mad_floor_iqr_frac > 0 else ''})"
            )
        base = {
            **meta0[key],
            "n_frames_in_pca": len(groups[key]),
            "pc_centroid_mean": centroids[ti].tolist(),
            "robust_z_per_pc": zs.tolist(),
            "max_abs_robust_z": worst_abs,
            "worst_pc_index": worst_j,
            "worst_pc_label": f"PC{worst_j + 1}",
            "n_pc_dims_exceeding_threshold": n_dims_over,
        }
        all_rows.append({**base, "is_outlier": is_out})
        if is_out:
            outlier_rows.append(
                {
                    **base,
                    "is_outlier": True,
                    "outlier_rule": rule,
                }
            )

    return outlier_rows, all_rows


def write_trial_outlier_artifacts(
    out_subdir: Path,
    modality_label: str,
    outliers: List[Dict[str, Any]],
    all_trials: List[Dict[str, Any]],
    mad_k: float,
    n_pc: int,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> None:
    summary: Dict[str, Any] = {
        "modality": modality_label,
        "n_pc_used_for_centroid": n_pc,
        "mad_k": mad_k,
        "n_trials": len(all_trials),
        "n_outliers": len(outliers),
        "outliers": outliers,
    }
    if extra_summary:
        summary.update(extra_summary)
    with open(out_subdir / "trial_outliers.json", "w") as f:
        json.dump(summary, f, indent=2)

    cols = [
        "subject_id",
        "condition",
        "trial",
        "collection_site",
        "h5_file",
        "hdf5_group_path",
        "h5_uri",
        "label",
        "max_abs_robust_z",
        "worst_pc_label",
        "n_pc_dims_exceeding_threshold",
        "n_frames_in_pca",
    ]
    with open(out_subdir / "trial_outliers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for o in outliers:
            w.writerow({c: o.get(c, "") for c in cols})

    # All trial centroids for follow-up
    cols_all = cols + ["is_outlier", "pc_centroid_mean", "robust_z_per_pc"]
    with open(out_subdir / "trial_pc_centroids.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_all, extrasaction="ignore")
        w.writeheader()
        for o in all_trials:
            row = {c: o.get(c, "") for c in cols_all}
            row["pc_centroid_mean"] = json.dumps(o.get("pc_centroid_mean", []))
            row["robust_z_per_pc"] = json.dumps(o.get("robust_z_per_pc", []))
            w.writerow(row)


def write_ik_only_trial_list(out_dir: Path, trial_meta: List[Dict[str, Any]]) -> int:
    """
    Trials that contribute at least one row to IK PCA but no rows to ID PCA
    (``n_frames_ik > 0`` and ``n_frames_id == 0`` after the per-frame finite mask).
    Writes ``ik_only_trials.json`` and ``ik_only_trials.csv`` under ``out_dir``.
    Returns number of IK-only trials.
    """
    ik_only = [
        t
        for t in trial_meta
        if int(t.get("n_frames_ik", 0) or 0) > 0 and int(t.get("n_frames_id", 0) or 0) == 0
    ]
    summary: Dict[str, Any] = {
        "n_ik_only_trials": len(ik_only),
        "definition": (
            "Trials with at least one time sample passing the IK row mask "
            "(>50% finite IK features) and no ID row after the ID mask, including "
            "after searching for the earliest later 1s ID window with usable moments."
        ),
        "trials": ik_only,
    }
    with open(out_dir / "ik_only_trials.json", "w") as f:
        json.dump(summary, f, indent=2)

    cols = [
        "subject_id",
        "condition",
        "trial",
        "collection_site",
        "h5_file",
        "hdf5_group_path",
        "h5_uri",
        "label",
        "n_frames_ik",
        "n_frames_id",
        "id_load_mode",
        "ik_dataset",
        "id_dataset",
    ]
    with open(out_dir / "ik_only_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in ik_only:
            w.writerow({c: t.get(c, "") for c in cols})
    return len(ik_only)


def is_memo_walking_condition(condition_name: str) -> bool:
    """
    Walking / locomotion trials for MeMo-style condition folder names.

    User keywords: walk, levelground, treadmill, stair, ramp, SA, SD, RA, RD, LG
    (matched case-insensitively as substrings; short tokens may rarely false-positive).
    """
    n = condition_name.lower()
    long_kw = [
        "walk",
        "levelground",
        "treadmill",
        "stair",
        "ramp",
    ]
    short_kw = ["sa", "sd", "ra", "rd", "lg"]
    if any(k in n for k in long_kw):
        return True
    if any(k in n for k in short_kw):
        return True
    return False


def _id_moment_block_from_rows(
    id_rows: np.ndarray,
    id_cols: List[str],
    mom_names: List[str],
) -> np.ndarray:
    """Shape (n_rows, n_mom) for selected moment columns."""
    M = np.full((len(id_rows), len(mom_names)), np.nan, dtype=np.float64)
    for j, name in enumerate(mom_names):
        col = f"{name}_moment"
        if col in id_cols:
            M[:, j] = id_rows[:, id_cols.index(col)]
    return M


def _id_block_has_usable_frame(moment_block: np.ndarray) -> bool:
    if moment_block.size == 0:
        return False
    return bool(np.any(np.mean(np.isfinite(moment_block), axis=1) > 0.5))


def _first_good_id_window_mask(
    time_id: np.ndarray,
    id_data: np.ndarray,
    id_cols: List[str],
    mom_names: List[str],
    duration_sec: float,
) -> Optional[np.ndarray]:
    """
    Earliest time window of length ``duration_sec`` on the ID clock that contains
    at least one frame with >50% finite selected moments. Returns boolean mask
    into ``id_data`` / ``time_id``, or None if none found.
    """
    n = len(time_id)
    if n == 0:
        return None
    t_max = float(time_id[-1])
    for i in range(n):
        t_start = float(time_id[i])
        if t_start + duration_sec > t_max + 1e-9:
            break
        mask = (time_id >= t_start) & (time_id <= t_start + duration_sec)
        if not np.any(mask):
            continue
        M = _id_moment_block_from_rows(id_data[mask], id_cols, mom_names)
        if _id_block_has_usable_frame(M):
            return mask
    return None


def load_trial_first_seconds(
    h5_path: Path,
    condition_name: str,
    trial_name: str,
    duration_sec: float,
    feature_set: str = "sagittal_lower",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]]:
    """
    Load IK / ID from the first H5 dataset under ``trial/ik`` and ``trial/id``.

    Reads OpenSim table columns by name (same as ``dataset.py``):
      - IK: angle columns in degrees → returned as ``pos_rad`` (radians).
      - ID: ``<dof>_moment`` columns (N·m/kg).

    ``feature_set``:
      - ``sagittal_lower``: 6 angles and 6 moments — R then L: hip flexion, knee, ankle.
      - ``full``: all ``IK_DOF_NAMES`` and ``MOMENT_NAMES``.

    IK uses the first ``duration_sec`` on the shared clock
    ``t0 = max(ik_time[0], id_time[0])`` (same as before).

    ID uses the same window **if** it contains at least one usable moment frame
    (>50% finite in the selected moment columns). Otherwise scans **forward in
    time** on the ID timeline for the **earliest** window of length
    ``duration_sec`` that does.     When the primary window is usable, ID moments are **interpolated onto the IK
    time base** (same row count as IK). When a **fallback** window is used, ID
    rows are kept on the **native ID time grid** for that 1s segment (row count
    may differ from IK).

    Returns (time_ik, pos_rad, moments, ik_key, id_key, id_load_mode).
    ``id_load_mode`` is ``"primary"`` (first aligned second), ``"fallback"``
    (earliest later second with usable ID), or ``"none"`` (no usable ID window).

    ``pos_rad``: (T_ik, n_ik_features), ``moments``: (T_id, n_id_features).
    """
    with h5py.File(h5_path, "r") as h5f:
        if condition_name not in h5f or trial_name not in h5f[condition_name]:
            return None
        trial_group = h5f[condition_name][trial_name]
        if "ik" not in trial_group or "id" not in trial_group:
            return None
        ik_group = trial_group["ik"]
        id_group = trial_group["id"]
        if len(ik_group.keys()) == 0 or len(id_group.keys()) == 0:
            return None
        ik_key = sorted(list(ik_group.keys()))[0]
        id_key = sorted(list(id_group.keys()))[0]
        ik_cols, ik_data = _read_h5_opensim_table(ik_group[ik_key])
        id_cols, id_data = _read_h5_opensim_table(id_group[id_key])

    if "time" not in ik_cols or "time" not in id_cols:
        return None

    time_ik = ik_data[:, ik_cols.index("time")].astype(np.float64)
    time_id = id_data[:, id_cols.index("time")].astype(np.float64)

    if feature_set == "full":
        ik_names = list(IK_DOF_NAMES)
        mom_names = list(MOMENT_NAMES)
    elif feature_set == "sagittal_lower":
        ik_names = list(SAGITTAL_LOWER_IK_NAMES)
        mom_names = list(SAGITTAL_LOWER_MOMENT_NAMES)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")

    t0 = max(float(time_ik[0]), float(time_id[0]))
    t1 = t0 + duration_sec
    ik_mask = (time_ik >= t0) & (time_ik <= t1)
    if not np.any(ik_mask):
        return None

    time = time_ik[ik_mask]
    if len(time) < 2:
        return None

    ik_rows = ik_data[ik_mask]

    pos_deg = np.full((len(time), len(ik_names)), np.nan, dtype=np.float64)
    for j, name in enumerate(ik_names):
        if name in ik_cols:
            pos_deg[:, j] = ik_rows[:, ik_cols.index(name)]
    pos = np.deg2rad(pos_deg)

    # ID: prefer IK-aligned first second; else earliest later window with usable data.
    id_mask_primary = (time_id >= t0) & (time_id <= t1)
    id_load_mode = "none"
    moments: np.ndarray

    if np.any(id_mask_primary):
        id_rows_p = id_data[id_mask_primary]
        id_time_win = time_id[id_mask_primary]
        M_primary = _id_moment_block_from_rows(id_rows_p, id_cols, mom_names)
        if _id_block_has_usable_frame(M_primary):
            # Align ID to IK timeline (same row count as ``pos``) when primary window works.
            moments = np.full((len(time), len(mom_names)), np.nan, dtype=np.float32)
            for j, name in enumerate(mom_names):
                col = f"{name}_moment"
                if col in id_cols:
                    y_id = id_rows_p[:, id_cols.index(col)]
                    if len(id_time_win) == len(time) and np.allclose(id_time_win, time):
                        moments[:, j] = y_id.astype(np.float32)
                    else:
                        moments[:, j] = np.interp(
                            time, id_time_win, y_id, left=np.nan, right=np.nan
                        ).astype(np.float32)
            id_load_mode = "primary"
        else:
            alt = _first_good_id_window_mask(time_id, id_data, id_cols, mom_names, duration_sec)
            if alt is not None:
                moments = _id_moment_block_from_rows(id_data[alt], id_cols, mom_names).astype(np.float32)
                id_load_mode = "fallback"
            else:
                moments = np.full((0, len(mom_names)), np.nan, dtype=np.float32)
                id_load_mode = "none"
    else:
        alt = _first_good_id_window_mask(time_id, id_data, id_cols, mom_names, duration_sec)
        if alt is not None:
            moments = _id_moment_block_from_rows(id_data[alt], id_cols, mom_names).astype(np.float32)
            id_load_mode = "fallback"
        else:
            moments = np.full((0, len(mom_names)), np.nan, dtype=np.float32)
            id_load_mode = "none"

    return time.astype(np.float32), pos.astype(np.float32), moments, ik_key, id_key, id_load_mode


def ik_feature_names(include_vel: bool, feature_set: str) -> List[str]:
    if feature_set == "full":
        names = [f"ik_{n}_rad" for n in IK_DOF_NAMES]
        if include_vel:
            names.extend([f"ik_{n}_radps" for n in IK_DOF_NAMES])
    elif feature_set == "sagittal_lower":
        names = [f"ik_{n}_rad" for n in SAGITTAL_LOWER_IK_NAMES]
        if include_vel:
            names.extend([f"ik_{n}_radps" for n in SAGITTAL_LOWER_IK_NAMES])
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")
    return names


def id_feature_names(feature_set: str) -> List[str]:
    if feature_set == "full":
        return [f"id_{n}_nm_per_kg" for n in MOMENT_NAMES]
    if feature_set == "sagittal_lower":
        return [f"id_{n}_nm_per_kg" for n in SAGITTAL_LOWER_MOMENT_NAMES]
    raise ValueError(f"Unknown feature_set: {feature_set!r}")


def compute_ik_velocity(pos_rad: np.ndarray, time_s: np.ndarray, feature_set: str) -> np.ndarray:
    """
    Angular velocity (rad/s). Full model uses hybrid B-spline / rotation-matrix logic
    from ``dataset._compute_velocity`` (needs 23-DOF layout). Sagittal subset uses
    ``np.gradient`` on each angle vs time.
    """
    pos64 = pos_rad.astype(np.float64)
    t64 = time_s.astype(np.float64)
    if feature_set == "full":
        if pos64.shape[1] != len(IK_DOF_NAMES):
            raise ValueError(f"Expected {len(IK_DOF_NAMES)} IK DOFs for feature_set=full, got {pos64.shape[1]}")
        return _compute_velocity(pos64, t64).astype(np.float64)
    if feature_set == "sagittal_lower":
        if pos64.shape[1] != len(SAGITTAL_LOWER_IK_NAMES):
            raise ValueError(
                f"Expected {len(SAGITTAL_LOWER_IK_NAMES)} sagittal IK DOFs, got {pos64.shape[1]}"
            )
        return np.gradient(pos64, t64, axis=0)
    raise ValueError(f"Unknown feature_set: {feature_set!r}")


def build_ik_matrix(pos: np.ndarray, vel: np.ndarray, include_vel: bool) -> np.ndarray:
    parts = [pos]
    if include_vel:
        parts.append(vel)
    return np.concatenate(parts, axis=1)


def build_id_matrix(moments: np.ndarray) -> np.ndarray:
    return moments


def fit_save_pca_modality(
    X: np.ndarray,
    row_subject: np.ndarray,
    row_metas: List[Dict[str, Any]],
    feat_names: List[str],
    out_subdir: Path,
    modality_label: str,
    duration_sec: float,
    seed: int,
    n_components: int,
    outlier_mad_k: float,
    outlier_n_pcs: int,
    outlier_min_dims_exceeding: Optional[int] = None,
    mad_floor_iqr_frac: float = 0.0,
) -> None:
    """Impute, scale, PCA, save npz/json/plots for one modality."""
    out_subdir.mkdir(parents=True, exist_ok=True)

    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X.astype(np.float64))
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_z = scaler.fit_transform(X_imp)

    n_comp = min(n_components, X_z.shape[1], max(X_z.shape[0] - 1, 0))
    if n_comp < 1:
        raise SystemExit(f"Not enough samples or features for {modality_label} PCA.")

    pca = PCA(n_components=n_comp, random_state=seed)
    Z = pca.fit_transform(X_z)

    if len(row_metas) != X.shape[0]:
        raise ValueError("row_metas length must match number of rows in X")

    n_pc_out = min(outlier_n_pcs, n_comp)
    outliers, all_trials = detect_trial_pc_outliers(
        Z,
        row_metas,
        n_pc_out,
        outlier_mad_k,
        min_dims_exceeding=outlier_min_dims_exceeding,
        mad_floor_iqr_frac=mad_floor_iqr_frac,
    )
    extra = {
        "outlier_min_dims_exceeding": outlier_min_dims_exceeding,
        "mad_floor_iqr_frac": mad_floor_iqr_frac,
    }
    write_trial_outlier_artifacts(
        out_subdir, modality_label, outliers, all_trials, outlier_mad_k, n_pc_out, extra_summary=extra
    )
    if outlier_min_dims_exceeding is None:
        rule_s = f"max|z| over PC1..PC{n_pc_out} > {outlier_mad_k}"
    else:
        rule_s = f">= {outlier_min_dims_exceeding} PCs with |z| > {outlier_mad_k}"
    if mad_floor_iqr_frac > 0:
        rule_s += f"; sigma >= {mad_floor_iqr_frac}*IQR per PC"
    print(f"  {modality_label} trial outliers: {len(outliers)} / {len(all_trials)} ({rule_s}) -> {out_subdir / 'trial_outliers.csv'}")

    np.savez(
        out_subdir / "pca_projection.npz",
        Z=Z.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_,
        mean_=scaler.mean_,
        scale_=scaler.scale_,
    )
    np.save(out_subdir / "pca_components.npy", pca.components_)
    with open(out_subdir / "feature_names.json", "w") as f:
        json.dump(feat_names, f, indent=2)

    with open(out_subdir / "explained_variance.json", "w") as f:
        json.dump(
            {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative": np.cumsum(pca.explained_variance_ratio_).tolist(),
            },
            f,
            indent=2,
        )

    # Scree
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, n_comp + 1), pca.explained_variance_ratio_, color="steelblue", edgecolor="black")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"{modality_label} PCA scree (MeMo walking, first {duration_sec:.2f}s / trial)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_subdir / "scree.png", dpi=150)
    plt.close(fig)

    # PC1 vs PC2 by subject
    fig, ax = plt.subplots(figsize=(8, 7))
    subjects = sorted(set(row_subject.tolist()))
    try:
        cmap = matplotlib.colormaps["tab20"]
    except (AttributeError, KeyError):
        cmap = plt.get_cmap("tab20")
    for i, s in enumerate(subjects):
        m = row_subject == s
        ax.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.35, label=s, color=cmap(i % 20))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{modality_label} PCA (standardized features)")
    ax.grid(True, alpha=0.3)
    if len(subjects) <= 20:
        ax.legend(markerscale=3, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_subdir / "pc1_pc2_by_subject.png", dpi=150)
    plt.close(fig)

    # Loadings heatmap
    comp = pca.components_
    short_labels = [re.sub(r"^ik_|^id_", "", n)[:28] for n in feat_names]
    fig_h = max(6, n_comp * 0.35)
    fig_w = max(10, len(feat_names) * 0.12)
    fig, ax = plt.subplots(figsize=(min(fig_w, 24), min(fig_h, 12)))
    vmax = float(np.max(np.abs(comp))) if comp.size else 1.0
    im = ax.imshow(comp, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(n_comp))
    ax.set_yticklabels([f"PC{k+1}" for k in range(n_comp)])
    step = max(1, len(short_labels) // 40)
    ax.set_xticks(range(0, len(short_labels), step))
    ax.set_xticklabels([short_labels[i] for i in range(0, len(short_labels), step)], rotation=90, fontsize=6)
    ax.set_title(
        f"{modality_label} PCA loadings\n"
        "(sign-flip checks: compare L/R or paired DOFs across PCs)"
    )
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_subdir / "loadings_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate IK and ID PCA on MeMo walking trials (first N seconds)")
    parser.add_argument(
        "--memo-root",
        type=str,
        default="/media/metamobility3/Samsung_T51/Processed/MeMo",
        help="Directory containing S###.h5 files",
    )
    parser.add_argument("--duration-sec", type=float, default=1.0, help="Use only first N seconds of each trial")
    parser.add_argument("--max-trials", type=int, default=None, help="Cap number of trials (for speed)")
    parser.add_argument("--max-subjects", type=int, default=None, help="Cap number of subject h5 files")
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=("sagittal_lower", "full"),
        default="sagittal_lower",
        help=(
            "IK/ID columns to load: sagittal_lower = R then L hip flexion, knee, ankle (6+6); "
            "full = all IK_DOF_NAMES + MOMENT_NAMES"
        ),
    )
    parser.add_argument(
        "--include-vel",
        action="store_true",
        help="Append IK angular velocities (same length as angles) to the feature vector",
    )
    parser.add_argument("--n-components", type=int, default=10, help="Number of PCA components")
    parser.add_argument("--output-dir", type=str, default="runs/pca_memo_walking")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outlier-mad-k",
        type=float,
        default=3.5,
        help="IK: robust z threshold on trial PC centroid (max |z| over PC1..K unless --outlier-ik-min-dims is set).",
    )
    parser.add_argument(
        "--outlier-mad-k-id",
        type=float,
        default=6.5,
        help="ID: robust z threshold (higher default than IK; use with --outlier-id-min-dims and IQR floor).",
    )
    parser.add_argument(
        "--outlier-ik-min-dims",
        type=int,
        default=0,
        help="IK: if >0, require at least this many PCs with |z|>--outlier-mad-k (0 = use max-|z| rule only).",
    )
    parser.add_argument(
        "--outlier-id-min-dims",
        type=int,
        default=2,
        help="ID: require at least this many PCs with |z|>--outlier-mad-k-id (reduces spurious PC3-only flags). Use 1 to match legacy max-|z| style.",
    )
    parser.add_argument(
        "--outlier-id-mad-floor-iqr",
        type=float,
        default=0.12,
        help="ID: per-PC sigma = max(MAD_scale, frac*IQR). Set 0 to disable (not recommended for ID).",
    )
    parser.add_argument(
        "--outlier-n-pcs",
        type=int,
        default=3,
        help="Number of leading PCs used to form per-trial centroid for outlier detection",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    memo_root = Path(args.memo_root)
    if not memo_root.is_dir():
        raise SystemExit(f"memo-root not found: {memo_root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(memo_root.glob("S*.h5"))
    if args.max_subjects is not None:
        h5_files = h5_files[: args.max_subjects]

    rows_ik: List[np.ndarray] = []
    rows_id: List[np.ndarray] = []
    row_subject_ik: List[str] = []
    row_subject_id: List[str] = []
    row_meta_ik: List[Dict[str, Any]] = []
    row_meta_id: List[Dict[str, Any]] = []
    trial_meta: List[Dict] = []
    n_trials_used = 0

    for h5_path in h5_files:
        sid = h5_path.stem.upper()
        with h5py.File(h5_path, "r") as h5f:
            conds = sorted(h5f.keys())
        for cond in conds:
            if not is_memo_walking_condition(cond):
                continue
            with h5py.File(h5_path, "r") as h5f:
                if cond not in h5f:
                    continue
                trials = sorted([k for k in h5f[cond].keys()])
            for trial in trials:
                if args.max_trials is not None and n_trials_used >= args.max_trials:
                    break
                loaded = load_trial_first_seconds(
                    h5_path, cond, trial, args.duration_sec, feature_set=args.feature_set
                )
                if loaded is None:
                    continue
                time, pos, moments, ik_key, id_key, id_load_mode = loaded
                vel = compute_ik_velocity(pos, time, args.feature_set).astype(np.float32)
                X_ik = build_ik_matrix(pos, vel, args.include_vel)
                X_id = build_id_matrix(moments)
                keep_ik = np.mean(np.isfinite(X_ik), axis=1) > 0.5
                keep_id = np.mean(np.isfinite(X_id), axis=1) > 0.5
                if not np.any(keep_ik) and not np.any(keep_id):
                    continue
                n_ik = int(np.sum(keep_ik))
                n_id = int(np.sum(keep_id))
                loc = trial_location_fields(memo_root, h5_path, sid, cond, trial)
                if np.any(keep_ik):
                    rows_ik.append(X_ik[keep_ik])
                    row_subject_ik.extend([sid] * n_ik)
                    for _ in range(n_ik):
                        row_meta_ik.append(dict(loc))
                if np.any(keep_id):
                    rows_id.append(X_id[keep_id])
                    row_subject_id.extend([sid] * n_id)
                    for _ in range(n_id):
                        row_meta_id.append(dict(loc))
                trial_meta.append(
                    {
                        **loc,
                        "n_frames_ik": n_ik,
                        "n_frames_id": n_id,
                        "ik_dataset": ik_key,
                        "id_dataset": id_key,
                        "id_load_mode": id_load_mode,
                    }
                )
                n_trials_used += 1
            if args.max_trials is not None and n_trials_used >= args.max_trials:
                break
        if args.max_trials is not None and n_trials_used >= args.max_trials:
            break

    if not rows_ik and not rows_id:
        raise SystemExit("No trials collected. Check memo-root, walking filter, and H5 structure.")

    names_ik = ik_feature_names(args.include_vel, args.feature_set)
    names_id = id_feature_names(args.feature_set)

    n_trials_any = len(trial_meta)
    n_trials_ik = sum(1 for t in trial_meta if t.get("n_frames_ik", 0) > 0)
    n_trials_id = sum(1 for t in trial_meta if t.get("n_frames_id", 0) > 0)
    n_trials_ik_only = sum(
        1 for t in trial_meta if t.get("n_frames_ik", 0) > 0 and t.get("n_frames_id", 0) == 0
    )
    n_trials_id_only = sum(
        1 for t in trial_meta if t.get("n_frames_ik", 0) == 0 and t.get("n_frames_id", 0) > 0
    )

    with open(out_dir / "trials_used.json", "w") as f:
        json.dump(
            {
                "memo_root": str(memo_root),
                "duration_sec": args.duration_sec,
                "feature_set": args.feature_set,
                "ik_columns_loaded": (
                    list(SAGITTAL_LOWER_IK_NAMES)
                    if args.feature_set == "sagittal_lower"
                    else list(IK_DOF_NAMES)
                ),
                "id_moment_columns_loaded": (
                    list(SAGITTAL_LOWER_MOMENT_NAMES)
                    if args.feature_set == "sagittal_lower"
                    else list(MOMENT_NAMES)
                ),
                "include_vel": args.include_vel,
                "n_trials": len(trial_meta),
                "note": (
                    "IK and ID PCAs use separate row masks; sample counts may differ. "
                    "ID moments use the IK-aligned first second when usable; otherwise "
                    "the earliest following 1s ID window with at least one usable frame."
                ),
                "trial_counts": {
                    "with_any_usable_frame": n_trials_any,
                    "with_usable_ik_frame": n_trials_ik,
                    "with_usable_id_frame": n_trials_id,
                    "ik_only_no_id_rows": n_trials_ik_only,
                    "id_only_no_ik_rows": n_trials_id_only,
                    "ik_only_trial_list_files": ["ik_only_trials.json", "ik_only_trials.csv"],
                    "explanation": "A trial is listed if IK load succeeded (first 1s after max(ik0,id0)). ID uses that same 1s if moments are usable there, else the earliest later 1s ID segment with usable data. ID PCA excludes trials where no such ID window yields rows passing the per-frame mask.",
                },
                "collection_site_legend": {
                    "S001_S022": "Camargo",
                    "S023_S034": "Scherpereel",
                    "S035_S056": "Molinaro_Scherpereel",
                },
                "trials": trial_meta,
            },
            f,
            indent=2,
        )

    n_ik_only_written = write_ik_only_trial_list(out_dir, trial_meta)
    print(
        f"IK-only trials (no ID rows after mask): {n_ik_only_written} -> "
        f"{out_dir.resolve() / 'ik_only_trials.csv'}"
    )

    ik_min_dims = args.outlier_ik_min_dims if args.outlier_ik_min_dims > 0 else None

    if rows_ik:
        X_ik_all = np.vstack(rows_ik).astype(np.float64)
        subj_ik = np.array(row_subject_ik)
        fit_save_pca_modality(
            X_ik_all,
            subj_ik,
            row_meta_ik,
            names_ik,
            out_dir / "ik",
            "IK",
            args.duration_sec,
            args.seed,
            args.n_components,
            args.outlier_mad_k,
            args.outlier_n_pcs,
            outlier_min_dims_exceeding=ik_min_dims,
            mad_floor_iqr_frac=0.0,
        )
    else:
        print("Warning: no IK rows passed the finite-value filter; skipping IK PCA.")

    id_min_dims: Optional[int] = None if args.outlier_id_min_dims <= 0 else int(args.outlier_id_min_dims)

    if rows_id:
        X_id_all = np.vstack(rows_id).astype(np.float64)
        subj_id = np.array(row_subject_id)
        fit_save_pca_modality(
            X_id_all,
            subj_id,
            row_meta_id,
            names_id,
            out_dir / "id",
            "ID",
            args.duration_sec,
            args.seed,
            args.n_components,
            args.outlier_mad_k_id,
            args.outlier_n_pcs,
            outlier_min_dims_exceeding=id_min_dims,
            mad_floor_iqr_frac=args.outlier_id_mad_floor_iqr,
        )
    else:
        print("Warning: no ID rows passed the finite-value filter; skipping ID PCA.")

    n_ik_s = sum(int(r.shape[0]) for r in rows_ik) if rows_ik else 0
    n_id_s = sum(int(r.shape[0]) for r in rows_id) if rows_id else 0
    print(
        f"Done. Trials with ≥1 usable frame — any modality: {n_trials_any} | "
        f"IK PCA: {n_trials_ik} trials ({n_ik_s} rows) | ID PCA: {n_trials_id} trials ({n_id_s} rows). "
        f"IK-only: {n_trials_ik_only}, ID-only: {n_trials_id_only}."
    )
    print(f"  ({len(names_ik)} IK feats, {len(names_id)} ID feats per row; row kept if >50% of that modality's features are finite.)")
    print(f"Outputs in: {out_dir.resolve()} (subdirs ik/, id/)")


if __name__ == "__main__":
    main()
