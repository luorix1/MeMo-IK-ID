"""
Processed Camargo dataset loader (no nimblephysics).

Root structure:
  /media/metamobility3/Samsung_T51/Processed/Camargo
    dataset_metadata.json
    S001/<condition>/trial_01/ik/*.mot
    S001/<condition>/trial_01/id/*.sto

We use:
  - IK: joint angles (degrees) -> radians
  - Velocities: computed via finite differences (rad/s)
  - ID: net joint moments (N*m/kg per dataset_metadata.json)

Windowing:
  - Skip windows where ANY selected output sample is NaN/Inf
  - Channels-first tensors for Conv1d: x=(C_in, W), y=(C_out, W)
  - Uniform sliding-window step ``stride`` (default 1 sample between starts; no per-task variation).

Denoising (optional, for noisy IK / ID):
  - Zero-phase Butterworth low-pass only (default 4 Hz; SciPy ``sosfiltfilt``, forward-backward)
  Velocities are always computed from the final filtered positions.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from scipy.interpolate import splrep, splev
    from scipy.spatial.transform import Rotation as SciPyRotation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from scipy.signal import butter, sosfiltfilt

    HAS_SCIPY_SIGNAL = True
except ImportError:
    HAS_SCIPY_SIGNAL = False


# IK coordinate columns present in processed .mot (excluding time)
IK_DOF_NAMES: List[str] = [
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
]

# Backwards-compat alias used by older code paths.
DOF_NAMES = IK_DOF_NAMES

# ID moment channel names (subset of .sto columns, excluding pelvis translation forces)
MOMENT_NAMES: List[str] = [
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
    "knee_angle_r",
    "knee_angle_l",
    "ankle_angle_r",
    "ankle_angle_l",
    "subtalar_angle_r",
    "subtalar_angle_l",
    "mtp_angle_r",
    "mtp_angle_l",
]

# Unilateral mode: negate left hip adduction & rotation (IK, velocities, moments) so L matches R sign convention.
UNILATERAL_FLIP_IK_INDICES = (
    IK_DOF_NAMES.index("hip_adduction_l"),
    IK_DOF_NAMES.index("hip_rotation_l"),
)
UNILATERAL_FLIP_MOMENT_INDICES = (
    MOMENT_NAMES.index("hip_adduction_l"),
    MOMENT_NAMES.index("hip_rotation_l"),
)

# FIXME: Temporary IK angle sign fixes for known AddBiomechanics cohort export issues.
# Correct sign conventions in source data / preprocessing, then remove this map and callers.
# Cohort keys match labels from ``discover_dataset_slices_from_dir`` (manifests next to H5 root).
_ADDBIOMECH_TEMP_IK_ANGLE_FLIP_BY_COHORT: Dict[str, frozenset] = {
    "Camargo": frozenset({"ankle_angle_l"}),
    "Moore2015_No_Arm": frozenset(
        {"ankle_angle_l", "ankle_angle_r", "knee_angle_l", "knee_angle_r"}
    ),
    "Falisse2017": frozenset({"knee_angle_l", "knee_angle_r", "ankle_angle_l", "ankle_angle_r"}),
    "Hamner2013": frozenset({"knee_angle_l", "knee_angle_r"}),
    "Uhlrich2023_No_Arm": frozenset({"hip_flexion_l", "hip_flexion_r", "knee_angle_l", "knee_angle_r"}),
}


def _apply_temp_addbiomech_ik_angle_sign_fix(
    subject_id: str,
    pos_rad: np.ndarray,
    manifest_root: Optional[Union[str, Path]],
) -> np.ndarray:
    """FIXME: Temporary — negate selected IK columns (rad) by cohort; remove when upstream is fixed."""
    if manifest_root is None:
        return pos_rad
    root = Path(manifest_root)
    if not root.is_dir():
        return pos_rad
    try:
        from ik_id.test_addbiomech_repr_subjects import (
            discover_dataset_slices_from_dir,
            slice_label_for_subject,
        )
    except ImportError:
        return pos_rad
    try:
        slices, _ = discover_dataset_slices_from_dir(root)
        cohort = slice_label_for_subject(subject_id, slices)
    except Exception:
        return pos_rad
    if cohort in ("unmapped", "invalid_id"):
        return pos_rad
    dofs = _ADDBIOMECH_TEMP_IK_ANGLE_FLIP_BY_COHORT.get(cohort)
    if not dofs:
        return pos_rad
    out = np.asarray(pos_rad, dtype=np.float64, copy=True)
    for name in dofs:
        if name in IK_DOF_NAMES:
            j = IK_DOF_NAMES.index(name)
            out[:, j] *= -1.0
    return out


def normalize_laterality(laterality: Optional[str]) -> str:
    """
    ``bilateral`` (alias: ``both``): use all sides as stored.
    ``unilateral``: same DOFs as bilateral; left hip adduction & rotation are negated
    to align L/R sign convention (right unchanged). With symmetric R/L index lists, training can use
    **paired ipsilateral windows** (see ``KineticsTCNDataset``).
    """
    x = (laterality or "bilateral").strip().lower()
    if x in ("both", "bilateral"):
        return "bilateral"
    if x == "unilateral":
        return "unilateral"
    if x in ("right", "left"):
        raise ValueError(
            f"laterality {laterality!r} is no longer supported; use 'bilateral' or 'unilateral' "
            "(see dataset.normalize_laterality docstring)."
        )
    raise ValueError(
        f"Unknown laterality {laterality!r}; expected 'bilateral' or 'unilateral' (alias: 'both' -> bilateral)."
    )


def _apply_unilateral_left_hip_flip_inplace(
    positions: np.ndarray,
    velocities: np.ndarray,
    moments: np.ndarray,
) -> None:
    for j in UNILATERAL_FLIP_IK_INDICES:
        positions[:, j] *= -1.0
        velocities[:, j] *= -1.0
    for j in UNILATERAL_FLIP_MOMENT_INDICES:
        moments[:, j] *= -1.0


def _apply_unilateral_flip_to_trial(trial: Dict[str, Any]) -> None:
    _apply_unilateral_left_hip_flip_inplace(
        trial["positions"],
        trial["velocities"],
        trial["moments"],
    )


def _unilateral_paired_indices_eligible(
    input_indices: Optional[List[int]],
    moment_indices: Optional[List[int]],
) -> bool:
    """True if R/L halves are same length (symmetric ipsilateral chains)."""
    if input_indices is None or moment_indices is None:
        return False
    if len(input_indices) != len(moment_indices):
        return False
    if len(input_indices) < 2 or len(input_indices) % 2 != 0:
        return False
    return True


def generic_moment_names_paired(moment_indices: List[int]) -> List[str]:
    """Drop ``_r`` / ``_l`` from right-half moment names (ipsilateral head, either leg)."""
    half = moment_indices[: len(moment_indices) // 2]
    out: List[str] = []
    for i in half:
        n = MOMENT_NAMES[i]
        if n.endswith("_r") or n.endswith("_l"):
            out.append(n[:-2])
        else:
            out.append(n)
    return out


def generic_ik_names_paired(ik_indices: List[int]) -> List[str]:
    half = ik_indices[: len(ik_indices) // 2]
    out: List[str] = []
    for i in half:
        n = IK_DOF_NAMES[i]
        if n.endswith("_r") or n.endswith("_l"):
            out.append(n[:-2])
        else:
            out.append(n)
    return out


# ---- Output moment index sets (indices into MOMENT_NAMES) --------------
LOWER_LIMB_MOMENT_INDICES = [
    3, 4, 5,    # hip R
    12,         # knee R
    14,         # ankle R
    6, 7, 8,    # hip L
    13,         # knee L
    15,         # ankle L
]

HIP_KNEE_MOMENT_INDICES = [
    3, 4, 5,    # hip R
    12,         # knee R
    6, 7, 8,    # hip L
    13,         # knee L
]

SAGITTAL_HIP_KNEE_MOMENT_INDICES = [
    3,   # hip_flexion_r
    12,  # knee_angle_r
    6,   # hip_flexion_l
    13,  # knee_angle_l
]

SAGITTAL_HIP_ANKLE_MOMENT_INDICES = [
    3,   # hip_flexion_r
    14,  # ankle_angle_r
    6,   # hip_flexion_l
    15,  # ankle_angle_l
]

SAGITTAL_KNEE_ANKLE_MOMENT_INDICES = [
    12,  # knee_angle_r
    14,  # ankle_angle_r
    13,  # knee_angle_l
    15,  # ankle_angle_l
]

SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES = [
    3,   # hip_flexion_r
    12,  # knee_angle_r
    14,  # ankle_angle_r
    6,   # hip_flexion_l
    13,  # knee_angle_l
    15,  # ankle_angle_l
]

# ---- Input DOF index sets (indices into IK_DOF_NAMES) ------------------
LOWER_LIMB_INPUT_INDICES = [
    6, 7, 8,    # hip R
    9,          # knee R
    10,         # ankle R
    13, 14, 15, # hip L
    16,         # knee L
    17,         # ankle L
]

SAGITTAL_INPUT_INDICES = [
    6,   # hip_flexion_r
    9,   # knee_angle_r
    10,  # ankle_angle_r
    13,  # hip_flexion_l
    16,  # knee_angle_l
    17,  # ankle_angle_l
]

SAGITTAL_HIP_KNEE_INPUT_INDICES = [
    6,   # hip_flexion_r
    9,   # knee_angle_r
    13,  # hip_flexion_l
    16,  # knee_angle_l
]

SAGITTAL_HIP_ANKLE_INPUT_INDICES = [
    6,   # hip_flexion_r
    10,  # ankle_angle_r
    13,  # hip_flexion_l
    17,  # ankle_angle_l
]

SAGITTAL_KNEE_ANKLE_INPUT_INDICES = [
    9,   # knee_angle_r
    10,  # ankle_angle_r
    16,  # knee_angle_l
    17,  # ankle_angle_l
]

# Single sagittal DOF type × R/L (paired with ``laterality=unilateral`` → ipsilateral 1→1 windows).
SAGITTAL_HIP_FLEXION_INPUT_INDICES = [6, 13] # hip_flexion_r, hip_flexion_l
SAGITTAL_KNEE_INPUT_INDICES = [9, 16]  # knee_angle_r, knee_angle_l
SAGITTAL_ANKLE_INPUT_INDICES = [10, 17]  # ankle_angle_r, ankle_angle_l

SAGITTAL_HIP_FLEXION_MOMENT_INDICES = [3, 6]  # hip_flexion_r, hip_flexion_l
SAGITTAL_KNEE_MOMENT_INDICES = [12, 13]  # knee_angle_r, knee_angle_l
SAGITTAL_ANKLE_MOMENT_INDICES = [14, 15]  # ankle_angle_r, ankle_angle_l


INPUT_MODE_INDICES = {
    "full": None,
    "lower_limb": LOWER_LIMB_INPUT_INDICES,
    "sagittal": SAGITTAL_INPUT_INDICES,
    "sagittal_hip_knee": SAGITTAL_HIP_KNEE_INPUT_INDICES,
    "sagittal_hip_ankle": SAGITTAL_HIP_ANKLE_INPUT_INDICES,
    "sagittal_knee_ankle": SAGITTAL_KNEE_ANKLE_INPUT_INDICES,
    "sagittal_hip_flexion": SAGITTAL_HIP_FLEXION_INPUT_INDICES,
    "sagittal_knee": SAGITTAL_KNEE_INPUT_INDICES,
    "sagittal_ankle": SAGITTAL_ANKLE_INPUT_INDICES,
}

OUTPUT_MODE_INDICES = {
    "all": None,
    "lower_limb": LOWER_LIMB_MOMENT_INDICES,
    "hip_knee": HIP_KNEE_MOMENT_INDICES,
    "sagittal_hip_knee": SAGITTAL_HIP_KNEE_MOMENT_INDICES,
    "sagittal_hip_ankle": SAGITTAL_HIP_ANKLE_MOMENT_INDICES,
    "sagittal_knee_ankle": SAGITTAL_KNEE_ANKLE_MOMENT_INDICES,
    "sagittal_hip_knee_ankle": SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES,
    "sagittal_hip_flexion": SAGITTAL_HIP_FLEXION_MOMENT_INDICES,
    "sagittal_knee": SAGITTAL_KNEE_MOMENT_INDICES,
    "sagittal_ankle": SAGITTAL_ANKLE_MOMENT_INDICES,
}


def extract_subject_id(path: Path) -> str:
    m = re.search(r"(S\d{3})", str(path), flags=re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def is_walking_condition(condition_dir: Path) -> bool:
    name = condition_dir.name.lower()
    # Camargo: levelground_* / ramp_* / stair_* / treadmill
    # Scherpereel & Molinaro_Scherpereel: normal_walk_*, dynamic_walk_*, incline_walk_*, ...
    include = (
        ("levelground" in name)
        or ("incline" in name)
        or ("stair" in name)
        or ("treadmill" in name)
    )
    # Common non-walking conditions.
    exclude = (
        ("static" in name)
        or ("dynamic" in name)
        or ("backward" in name)
    )
    return bool(include and not exclude)


def is_levelground_subset_condition(condition_name: str) -> bool:
    """
    True for "level-included" task names (case-insensitive):

      levelground_*, treadmill_normal_walk*, treadmill_transient*, treadmill_0p*,
      treadmill_1p*, treadmill_2p*, treadmill_unspecified_speed*
    """
    n = (condition_name or "").lower()
    if n.startswith("levelground_"):
        return True
    if n.startswith("treadmill_normal_walk"):
        return True
    if n.startswith("treadmill_transient"):
        return True
    if n.startswith("treadmill_0p"):
        return True
    if n.startswith("treadmill_1p"):
        return True
    if n.startswith("treadmill_2p"):
        return True
    if n.startswith("treadmill_unspecified_speed"):
        return True
    return False


def _subject_num(subject_id: str) -> int:
    """Return the integer number from a subject ID string like 'S042', or 0."""
    m = re.search(r"\d+", subject_id)
    return int(m.group()) if m else 0


def _subject_id_excluded_temp_broken_h5(subject_id: str) -> bool:
    """Remove known bad-quality datasets from AddBiomechanics during loading."""
    n = _subject_num(subject_id)
    # Falisse2017, Hamner2013, Moore2015, Tan2021
    if 386 <= n <= 426:
        return True
    # Tiziana2019
    if 444 <= n <= 493:
        return True
    # vanderZee2022
    if 513 <= n <= 522:
        return True
    return False


def include_condition_for_dataset(
    condition_name: str,
    *,
    walking_only: bool,
    levelground_only: bool,
    subject_id: str = "",
) -> bool:
    """Whether to include an H5 group / trial-folder condition in KineticsTCNDataset.

    The is_walking_condition() name filter is only applied for S001–S056 (b3d
    datasets with heterogeneous condition naming).  Subjects beyond S056 use
    canonical unified condition names and are filtered by
    is_levelground_subset_condition() when levelground_only=True, or kept
    unconditionally when walking_only=True.
    """
    if levelground_only:
        return is_levelground_subset_condition(condition_name)
    if walking_only and _subject_num(subject_id) <= 56:
        return is_walking_condition(Path(condition_name))
    return True


# Buckets aligned with compare_pipelineV4._classify_loc_bucket / per_loc_condition reporting.
LOC_BUCKET_OVERSAMPLE_KEYS: Tuple[str, ...] = ("LG", "SA", "SD", "RA", "RD")


def load_loc_ascent_descent_mapping(path: Optional[str]) -> Dict[Tuple[str, str, str], str]:
    """
    Load explicit ascent/descent bucket labels from JSON (same schema as compare_pipelineV4).

    Odd trial_NN -> ascent (SA or RA), even -> descent (SD or RD) for listed condition groups.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"loc_ascent_descent_map not found: {p}")
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
            cl = cond.lower()
            if "stair" in cl:
                asc_label, dsc_label = "SA", "SD"
            elif "incline" in cl or "ramp" in cl:
                asc_label, dsc_label = "RA", "RD"
            else:
                continue
            for i in range(1, max(0, n_trials) + 1):
                trial = f"trial_{i:02d}"
                out[(sid, cond, trial)] = asc_label if (i % 2 == 1) else dsc_label
    return out


def classify_loc_bucket(
    subject_id: str,
    condition: str,
    trial: str,
    loc_map: Optional[Dict[Tuple[str, str, str], str]] = None,
) -> str:
    """
    Map condition group + trial key to a coarse locomotion bucket (LG / RA / RD / SA / SD / …).

    Priority: optional JSON map → trial prefix token (LG/RA/RD/SA/SD) → ramp/stair naming in
    condition/trial text → Camargo LG prefixes → AddBiomechanics/OpenCap slope & walk groups
    (``uphillrun`` / ``downhillrun`` / ``flatrun_*`` / ``walk_*`` / …) → OTHER.
    Keep compare_pipelineV4._classify_loc_bucket aligned when editing.
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

    # AddBiomechanics / OpenCap unified exports (typically S057+): slope runs and level walking.
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


# MeMo / Camargo-style locomotion prefixes (see memo_task_duration_composition.py).
LOC_CONDITION_FAMILY_KEYS: Tuple[str, ...] = ("treadmill", "incline", "stair", "levelground")


def classify_loc_condition_family(condition_name: str) -> Optional[str]:
    """
    Map condition group / folder name to a coarse family for stride balancing, or None.

    Prefixes (case-insensitive): treadmill_, incline_, stair_, levelground_.
    """
    n = (condition_name or "").lower()
    if n.startswith("treadmill_"):
        return "treadmill"
    if n.startswith("incline_"):
        return "incline"
    if n.startswith("stair_"):
        return "stair"
    if n.startswith("levelground_"):
        return "levelground"
    return None


def enumerate_walking_trials_for_stride_plan(
    *,
    data_dir: str,
    h5_dir: Optional[str],
    use_h5: bool,
    subject_ids: Optional[List[str]],
    b3d_files: Optional[List[Path]],
    walking_only: bool,
    levelground_only: bool = False,
    max_files: Optional[int],
) -> List[Union[Tuple[str, str, str, str], Path]]:
    """
    List the same trial sources KineticsTCNDataset would use (condition filter, subject filter),
    without loading full IK/ID. Used to total time per loc-condition family.
    """
    subject_ids_norm: Optional[List[str]] = None
    if subject_ids is not None:
        subject_ids_norm = [s.upper() for s in subject_ids]

    requested_h5_dir = h5_dir or _default_h5_dir_from_processed_root(data_dir)
    use_h5_eff = bool(use_h5 and HAS_H5PY and Path(requested_h5_dir).exists())
    candidates: List[Union[Tuple[str, str, str, str], Path]] = []

    if use_h5_eff:
        if b3d_files is not None:
            h5_root = Path(requested_h5_dir)
            for p in sorted([Path(x) for x in b3d_files]):
                sid = extract_subject_id(p)
                if _subject_id_excluded_temp_broken_h5(sid):
                    continue
                cond = p.parent.name
                if not include_condition_for_dataset(
                    cond, walking_only=walking_only, levelground_only=levelground_only,
                    subject_id=sid,
                ):
                    continue
                tr = p.name
                candidates.append((sid, cond, tr, str(h5_root / f"{sid}.h5")))
        else:
            for subject_h5_path in sorted(Path(requested_h5_dir).glob("S*.h5")):
                sid = subject_h5_path.stem.upper()
                if _subject_id_excluded_temp_broken_h5(sid):
                    continue
                if subject_ids_norm is not None and sid not in set(subject_ids_norm):
                    continue
                with h5py.File(subject_h5_path, "r") as h5f:
                    for cond in sorted(h5f.keys()):
                        if not include_condition_for_dataset(
                            cond, walking_only=walking_only, levelground_only=levelground_only,
                            subject_id=sid,
                        ):
                            continue
                        for trial in sorted(h5f[cond].keys()):
                            candidates.append((sid, cond, trial, str(subject_h5_path)))
        if max_files is not None:
            candidates = candidates[: int(max_files)]
    else:
        if b3d_files is not None:
            trial_dirs = sorted([Path(p) for p in b3d_files])
        else:
            trial_dirs = find_trial_dirs(data_dir)
        trial_dirs = [
            td
            for td in trial_dirs
            if (not _subject_id_excluded_temp_broken_h5(extract_subject_id(td)))
            and include_condition_for_dataset(
                td.parent.name, walking_only=walking_only, levelground_only=levelground_only,
                subject_id=extract_subject_id(td),
            )
        ]
        if max_files is not None:
            trial_dirs = trial_dirs[: int(max_files)]
        candidates = list(trial_dirs)

    return candidates


def ik_time_span_seconds_h5_ref(ref: Tuple[str, str, str, str]) -> Optional[float]:
    """Duration from first IK table: last(time) - first(time)."""
    _sid, cond_name, trial_name, subject_h5_path = ref
    h5_path = Path(subject_h5_path)
    if not h5_path.exists():
        return None
    try:
        with h5py.File(h5_path, "r") as h5f:
            if cond_name not in h5f or trial_name not in h5f[cond_name]:
                return None
            trial_group = h5f[cond_name][trial_name]
            if "ik" not in trial_group or len(trial_group["ik"].keys()) == 0:
                return None
            ik_key = sorted(list(trial_group["ik"].keys()))[0]
            ik_cols, ik_data = _read_h5_opensim_table(trial_group["ik"][ik_key])
    except Exception:
        return None
    if "time" not in ik_cols or ik_data.shape[0] < 2:
        return None
    t = ik_data[:, ik_cols.index("time")].astype(np.float64)
    span = float(t[-1] - t[0])
    return span if np.isfinite(span) and span > 0 else None


def ik_time_span_seconds_trial_dir(trial_dir: Path) -> Optional[float]:
    """Duration from first IK .mot under trial_dir/ik/."""
    ik_dir = trial_dir / "ik"
    if not ik_dir.is_dir():
        return None
    ik_files = sorted(ik_dir.glob("*.mot"))
    if not ik_files:
        return None
    try:
        ik_cols, ik_data = _read_opensim_table(ik_files[0])
    except Exception:
        return None
    if "time" not in ik_cols or ik_data.shape[0] < 2:
        return None
    t = ik_data[:, ik_cols.index("time")].astype(np.float64)
    span = float(t[-1] - t[0])
    return span if np.isfinite(span) and span > 0 else None


def summarize_loc_condition_family_times(
    *,
    data_dir: str,
    h5_dir: Optional[str],
    use_h5: bool,
    subject_ids: Optional[List[str]],
    b3d_files: Optional[List[Path]],
    walking_only: bool,
    levelground_only: bool = False,
    max_files: Optional[int],
) -> Dict[str, float]:
    """Total IK time (seconds) per loc family on the listed trials; only keys with >0 time."""
    candidates = enumerate_walking_trials_for_stride_plan(
        data_dir=data_dir,
        h5_dir=h5_dir,
        use_h5=use_h5,
        subject_ids=subject_ids,
        b3d_files=b3d_files,
        walking_only=walking_only,
        levelground_only=levelground_only,
        max_files=max_files,
    )
    seconds = {k: 0.0 for k in LOC_CONDITION_FAMILY_KEYS}
    for c in candidates:
        if isinstance(c, tuple):
            dur = ik_time_span_seconds_h5_ref(c)
            cond_name = c[1]
        else:
            dur = ik_time_span_seconds_trial_dir(c)
            cond_name = c.parent.name
        if dur is None:
            continue
        fam = classify_loc_condition_family(cond_name)
        if fam is None:
            continue
        seconds[fam] += float(dur)
    return seconds


def compute_balanced_strides_for_loc_families(
    seconds_per_family: Dict[str, float],
    base_stride: float,
) -> Dict[str, int]:
    """
    Per-family stride = floor(base_stride * T_f / median(T_present)), min 1.

    Longer total-time families get a larger stride (fewer windows); shorter ones
    get a smaller stride (denser sampling).
    """
    base_stride = float(base_stride)
    present = {
        k: float(seconds_per_family.get(k, 0.0))
        for k in LOC_CONDITION_FAMILY_KEYS
        if float(seconds_per_family.get(k, 0.0)) > 0.0
    }
    if not present:
        return {}
    t_ref = float(np.median(list(present.values())))
    if t_ref <= 0 or not math.isfinite(t_ref):
        s0 = max(1, int(math.floor(base_stride)))
        return {k: s0 for k in present}
    out: Dict[str, int] = {}
    for fam, T in present.items():
        s_raw = base_stride * (T / t_ref)
        out[fam] = max(1, int(math.floor(s_raw)))
    return out


def plan_loc_condition_family_stride_balance(
    *,
    data_dir: str,
    h5_dir: Optional[str],
    use_h5: bool,
    subject_ids: Optional[List[str]],
    b3d_files: Optional[List[Path]],
    walking_only: bool,
    levelground_only: bool = False,
    max_files: Optional[int],
    base_stride: float,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Compute total time per family and balanced integer strides (floored)."""
    secs = summarize_loc_condition_family_times(
        data_dir=data_dir,
        h5_dir=h5_dir,
        use_h5=use_h5,
        subject_ids=subject_ids,
        b3d_files=b3d_files,
        walking_only=walking_only,
        levelground_only=levelground_only,
        max_files=max_files,
    )
    strides = compute_balanced_strides_for_loc_families(secs, base_stride)
    return secs, strides


def find_trial_dirs(root_dir: str) -> List[Path]:
    """
    Find all <root>/S###/<condition>/trial_## directories.
    """
    trial_dirs: List[Path] = []
    for root, _dirs, _files in os.walk(root_dir):
        if Path(root).name.startswith("trial_"):
            trial_dirs.append(Path(root))
    return sorted(trial_dirs)


def _load_subject_metadata_map(root_dir: str) -> Dict[str, Dict]:
    meta_path = Path(root_dir) / "dataset_metadata.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    out: Dict[str, Dict] = {}
    for s in meta.get("subjects", []):
        sid = s.get("subject_id")
        if sid:
            out[sid.upper()] = s
    return out


def _read_opensim_table(path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Read OpenSim .mot/.sto (endheader + tab-separated table).
    Returns (columns, data) where data is float ndarray.
    """
    lines = path.read_text().splitlines()
    end = None
    for i, l in enumerate(lines):
        if l.strip().lower() == "endheader":
            end = i
            break
    if end is None:
        raise ValueError(f"Could not find endheader in {path}")
    cols = lines[end + 1].strip().split()
    data = np.genfromtxt(lines[end + 2 :], delimiter="\t")
    if data.ndim == 1:
        data = data[None, :]
    return cols, data


def _compute_velocity(pos: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Hybrid velocity computation:
      - 1D joints: B-spline derivative of angle trajectory.
      - 3D joints (pelvis, lumbar/trunk, hips): convert Euler->R, finite-difference
        R(t), then omega from skew( Rdot * R^T ).
      - Fallback: finite-difference with np.gradient when SciPy is unavailable.
    """
    if not HAS_SCIPY:
        return np.gradient(pos, time, axis=0)

    n_t, n_dof = pos.shape
    if n_t < 5:
        return np.gradient(pos, time, axis=0)

    vel = np.zeros_like(pos, dtype=np.float64)

    # 3D joint triplets in IK_DOF_NAMES order (x, y, z-like channels).
    joint_3d_triplets = [
        (0, 1, 2),      # pelvis_tilt, pelvis_list, pelvis_rotation
        (6, 7, 8),      # hip_r
        (13, 14, 15),   # hip_l
        (20, 21, 22),   # lumbar/trunk
    ]
    three_d_indices = {idx for triplet in joint_3d_triplets for idx in triplet}

    # 1D angular joints explicitly called out.
    one_d_angular_indices = [9, 10, 11, 16, 17, 18]  # knee/ankle/subtalar R/L

    # 1) 1D joints with B-spline derivatives.
    for j in one_d_angular_indices:
        y = pos[:, j].astype(np.float64)
        tck = splrep(time, y, s=0, k=3)
        vel[:, j] = splev(time, tck, der=1)

    # 2) 3D joints: Euler -> R, finite-difference R, omega extraction.
    dt = np.gradient(time)
    eps = 1e-8
    dt = np.where(np.abs(dt) < eps, eps, dt)
    for a, b, c in joint_3d_triplets:
        euler = pos[:, [a, b, c]].astype(np.float64)
        rot_mats = SciPyRotation.from_euler("xyz", euler, degrees=False).as_matrix()  # (T,3,3)
        rdot = np.gradient(rot_mats, axis=0) / dt[:, None, None]
        omega_mat = np.einsum("tij,tkj->tik", rdot, rot_mats)  # Rdot * R^T
        # omega from skew-symmetric matrix:
        # [  0   -wz   wy ]
        # [ wz    0   -wx ]
        # [-wy   wx    0  ]
        wx = 0.5 * (omega_mat[:, 2, 1] - omega_mat[:, 1, 2])
        wy = 0.5 * (omega_mat[:, 0, 2] - omega_mat[:, 2, 0])
        wz = 0.5 * (omega_mat[:, 1, 0] - omega_mat[:, 0, 1])
        vel[:, a] = wx
        vel[:, b] = wy
        vel[:, c] = wz

    # 3) Remaining channels (e.g., pelvis translations, mtp) use spline derivative.
    for j in range(n_dof):
        if j in three_d_indices or j in one_d_angular_indices:
            continue
        y = pos[:, j].astype(np.float64)
        tck = splrep(time, y, s=0, k=3)
        vel[:, j] = splev(time, tck, der=1)

    return vel.astype(np.float64)


def _lowpass_zero_phase(
    data: np.ndarray,
    time: np.ndarray,
    cutoff_hz: float = 4.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase low-pass filter along time axis for each channel.

    Implemented with ``scipy.signal.sosfiltfilt`` (second-order sections, forward-backward).
    Training/eval IK and ID denoising must keep this zero-phase form; do not swap in
    one-pass ``sosfilt`` / ``lfilter`` here.
    Falls back to input unchanged when SciPy signal is unavailable or filtering settings
    are invalid for the current trial length/sample rate.
    """
    if not HAS_SCIPY_SIGNAL:
        return data
    if data.ndim != 2 or len(time) != data.shape[0] or data.shape[0] < 8:
        return data

    dt = np.diff(time.astype(np.float64))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return data

    fs = 1.0 / float(np.median(dt))
    nyquist = 0.5 * fs
    if not np.isfinite(nyquist) or nyquist <= 0:
        return data
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        return data

    try:
        sos = butter(order, cutoff_hz / nyquist, btype="low", output="sos")
        # Forward-backward => zero phase (training/eval IK+ID denoise must not use one-pass filters).
        return sosfiltfilt(sos, data.astype(np.float64), axis=0)
    except Exception:
        return data


def _denoise_pos_and_moments(
    pos: np.ndarray,
    moments: np.ndarray,
    time: np.ndarray,
    *,
    apply_lowpass_filter: bool,
    lowpass_cutoff_hz: float,
    lowpass_order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Denoise pipeline for IK positions (rad) and ID moments.

    Optional **zero-phase** Butterworth low-pass only (``_lowpass_zero_phase`` / ``scipy.signal.sosfiltfilt``).
    """
    pos = pos.astype(np.float64)
    moments = moments.astype(np.float64)

    if apply_lowpass_filter:
        pos = _lowpass_zero_phase(pos, time, cutoff_hz=lowpass_cutoff_hz, order=lowpass_order)
        finite_mom = np.isfinite(moments)
        moments_fill = np.where(finite_mom, moments, 0.0)
        moments_filt = _lowpass_zero_phase(
            moments_fill, time, cutoff_hz=lowpass_cutoff_hz, order=lowpass_order
        )
        moments = np.where(finite_mom, moments_filt, np.nan)

    return pos, moments


def _lowpass_trial_channels(
    x: np.ndarray,
    time: np.ndarray,
    *,
    apply_lowpass_filter: bool,
    lowpass_cutoff_hz: float,
    lowpass_order: int,
) -> np.ndarray:
    """
    Zero-phase Butterworth along time for each column (e.g. IMU features).

    Non-finite samples are masked out during filtering (filled with 0, then restored to NaN),
    matching the moment-channel path in ``_denoise_pos_and_moments``.
    """
    if not apply_lowpass_filter:
        return x
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(time):
        return x
    finite = np.isfinite(arr)
    fill = np.where(finite, arr, 0.0)
    filt = _lowpass_zero_phase(
        fill, time, cutoff_hz=float(lowpass_cutoff_hz), order=int(lowpass_order)
    )
    out = np.where(finite, filt, np.nan)
    if np.issubdtype(x.dtype, np.floating):
        return out.astype(x.dtype, copy=False)
    return out.astype(np.float32)


def _read_h5_opensim_table(dset) -> Tuple[List[str], np.ndarray]:
    """
    Read one OpenSim table stored in H5 with:
      - values in dataset array
      - JSON columns in dataset attr 'columns'
    """
    if "columns" not in dset.attrs:
        raise KeyError(f"Missing H5 attribute 'columns' for dataset {dset.name}")
    columns = json.loads(dset.attrs["columns"])
    data = dset[()]
    if data.ndim == 1:
        data = data[None, :]
    return columns, data


def _ik_time_and_pos_deg(
    ik_cols: List[str], ik_data: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Build time and joint positions in degrees for all IK_DOF_NAMES.

    Some exports (e.g. MeMo / partial Theia IK) omit pelvis translation, lumbar,
    or MTP columns. Missing coordinates are filled with 0 deg so the stacked
    shape stays (T, len(IK_DOF_NAMES)) and matches full-body pipelines.
    """
    if "time" not in ik_cols:
        return None
    time = ik_data[:, ik_cols.index("time")]
    n = len(time)
    pos_deg = np.zeros((n, len(IK_DOF_NAMES)), dtype=np.float64)
    for j, name in enumerate(IK_DOF_NAMES):
        if name in ik_cols:
            pos_deg[:, j] = ik_data[:, ik_cols.index(name)]
    return time, pos_deg


def _default_h5_dir_from_processed_root(data_dir: str) -> str:
    d = Path(data_dir)
    return str(d.with_name(f"{d.name}_h5"))


def resample_trial_to_uniform_hz(
    time: np.ndarray,
    pos: np.ndarray,
    moments: np.ndarray,
    target_hz: float,
    *,
    native_hz_tolerance: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uniformly resample IK positions and ID moments (same time base) to ``target_hz`` via linear interpolation.

    If the median native sample rate is within ``native_hz_tolerance`` of ``target_hz``, returns inputs
    unchanged (avoids unnecessary regridding when data already match). Typical Camargo H5 IK is ~200 Hz;
    use e.g. ``target_hz=100`` for half-rate training windows.
    """
    if target_hz <= 0:
        raise ValueError("target_hz must be positive.")
    t = np.asarray(time, dtype=np.float64).ravel()
    if t.size < 2:
        return time, pos, moments
    p = np.asarray(pos, dtype=np.float64)
    m = np.asarray(moments, dtype=np.float64)
    if p.shape[0] != t.size or m.shape[0] != t.size:
        raise ValueError(
            f"time length {t.size} must match pos rows {p.shape[0]} and moments rows {m.shape[0]}."
        )

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return time, pos, moments
    fs_native = 1.0 / float(np.median(dt))
    if abs(fs_native - float(target_hz)) <= float(native_hz_tolerance):
        return time, pos, moments

    t0, t1 = float(t[0]), float(t[-1])
    span = t1 - t0
    if span <= 0:
        return time, pos, moments

    n_out = max(2, int(round(span * float(target_hz))) + 1)
    t_new = np.linspace(t0, t1, n_out, dtype=np.float64)

    def _interp_cols(t_new_: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty((n_out, y.shape[1]), dtype=np.float64)
        for j in range(y.shape[1]):
            col = y[:, j]
            fin = np.isfinite(col)
            if not np.any(fin):
                out[:, j] = np.nan
            else:
                out[:, j] = np.interp(t_new_, t[fin], col[fin])
        return out

    p_new = _interp_cols(t_new, p)
    m_new = _interp_cols(t_new, m)
    return (
        t_new.astype(np.float32),
        p_new.astype(np.float32),
        m_new.astype(np.float32),
    )


def decimate_trial_aligned(
    time: np.ndarray,
    pos: np.ndarray,
    moments: np.ndarray,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep every ``step``-th sample along time (indices 0, step, 2*step, ...).

    Use ``step=2`` on ~200 Hz native data to obtain ~100 Hz without interpolation
    (avoids regridding artifacts from ``resample_trial_to_uniform_hz``).
    """
    if step < 1:
        raise ValueError("decimate step must be >= 1")
    if step == 1:
        return time, pos, moments
    t = np.asarray(time, dtype=np.float64).ravel()
    p = np.asarray(pos, dtype=np.float64)
    m = np.asarray(moments, dtype=np.float64)
    if t.size < 2:
        return time, pos, moments
    if p.shape[0] != t.size or m.shape[0] != t.size:
        raise ValueError(
            f"time length {t.size} must match pos rows {p.shape[0]} and moments rows {m.shape[0]}."
        )
    idx = np.arange(0, t.size, int(step), dtype=np.int64)
    return (
        t[idx].astype(np.float32),
        p[idx].astype(np.float32),
        m[idx].astype(np.float32),
    )


def _coerce_weight_kg(weight_kg_value) -> float:
    """
    Convert metadata weight/mass value into a single float.
    Some datasets store a dict like {'no_exo': ..., 'w_exo': ...}.
    """
    if weight_kg_value is None:
        return float("nan")
    if isinstance(weight_kg_value, (int, float)):
        return float(weight_kg_value)
    if isinstance(weight_kg_value, str):
        try:
            return float(weight_kg_value)
        except ValueError:
            return float("nan")
    if isinstance(weight_kg_value, dict):
        # Common MeMo schema keys.
        for k in ["w_exo", "with_exo", "exo", "with_exoskeleton", "with_exos"]:
            if k in weight_kg_value and isinstance(weight_kg_value[k], (int, float)):
                return float(weight_kg_value[k])
        for k in ["no_exo", "without_exo", "no_exoskeleton"]:
            if k in weight_kg_value and isinstance(weight_kg_value[k], (int, float)):
                return float(weight_kg_value[k])
        # As a last resort, pick the first numeric value.
        for _k, v in weight_kg_value.items():
            if isinstance(v, (int, float)):
                return float(v)
        return float("nan")
    # Unknown type
    try:
        return float(weight_kg_value)
    except Exception:
        return float("nan")


def load_trial_from_processed(
    trial_dir: Path,
    root_dir: str,
    meta_map: Dict[str, Dict],
    apply_lowpass_filter: bool = True,
    lowpass_cutoff_hz: float = 4.0,
    lowpass_order: int = 4,
    target_sample_rate_hz: Optional[float] = None,
    rollout_decimate_step: int = 1,
    apply_velocity_lowpass_filter: bool = False,
    velocity_lowpass_cutoff_hz: Optional[float] = None,
    velocity_lowpass_order: Optional[int] = None,
) -> Optional[Dict]:
    """
    Load one processed trial directory into arrays:
      positions: (T, 23) rad
      velocities: (T, 23) rad/s
      moments: (T, 20) N*m/kg
    """
    ik_dir = trial_dir / "ik"
    id_dir = trial_dir / "id"
    if not ik_dir.exists() or not id_dir.exists():
        return None

    ik_files = sorted(ik_dir.glob("*.mot"))
    id_files = sorted(id_dir.glob("*.sto"))
    if not ik_files or not id_files:
        return None

    ik_path = ik_files[0]
    id_path = id_files[0]

    subj_id = extract_subject_id(trial_dir)
    mass_val = meta_map.get(subj_id, {}).get(
        "weight_kg",
        meta_map.get(subj_id, {}).get("mass_kg", np.nan),
    )
    mass = _coerce_weight_kg(mass_val)
    # Moments are stored as N*m/kg, so we don't strictly require mass.
    # Keep mass for completeness, but don't reject trials when metadata is missing.
    if not np.isfinite(mass) or mass <= 0:
        mass = np.nan

    # IK (allow partial column sets; pad missing DOFs with 0 deg)
    ik_cols, ik_data = _read_opensim_table(ik_path)
    ik_tp = _ik_time_and_pos_deg(ik_cols, ik_data)
    if ik_tp is None:
        return None
    time, pos_deg = ik_tp
    pos = np.deg2rad(pos_deg)
    # ID
    id_cols, id_data = _read_opensim_table(id_path)
    if "time" not in id_cols:
        return None
    id_time = id_data[:, id_cols.index("time")]

    n = min(len(time), len(id_time))
    time = time[:n]
    pos = pos[:n]
    id_data = id_data[:n]

    pos = _apply_temp_addbiomech_ik_angle_sign_fix(subj_id, pos, Path(root_dir))

    moments = np.full((n, len(MOMENT_NAMES)), np.nan, dtype=np.float64)
    for j, name in enumerate(MOMENT_NAMES):
        col = f"{name}_moment"
        if col in id_cols:
            moments[:, j] = id_data[:, id_cols.index(col)]
        else:
            # If a moment channel is missing entirely, leave NaNs.
            # Windowing will drop windows where selected output channels are non-finite.
            pass

    if rollout_decimate_step > 1:
        time, pos, moments = decimate_trial_aligned(
            time, pos, moments, int(rollout_decimate_step)
        )
    elif target_sample_rate_hz is not None and target_sample_rate_hz > 0:
        time, pos, moments = resample_trial_to_uniform_hz(
            time, pos, moments, float(target_sample_rate_hz)
        )

    if apply_lowpass_filter:
        pos, moments = _denoise_pos_and_moments(
            pos,
            moments,
            time,
            apply_lowpass_filter=apply_lowpass_filter,
            lowpass_cutoff_hz=lowpass_cutoff_hz,
            lowpass_order=lowpass_order,
        )

    vel = _compute_velocity(pos, time)
    if apply_velocity_lowpass_filter:
        vel_cutoff = (
            float(velocity_lowpass_cutoff_hz)
            if velocity_lowpass_cutoff_hz is not None
            else float(lowpass_cutoff_hz)
        )
        vel_order = (
            int(velocity_lowpass_order)
            if velocity_lowpass_order is not None
            else int(lowpass_order)
        )
        vel = _lowpass_zero_phase(vel, time, cutoff_hz=vel_cutoff, order=vel_order)

    trial_name = f"{trial_dir.parent.name}/{trial_dir.name}"
    return {
        "positions": pos.astype(np.float32),
        "velocities": vel.astype(np.float32),
        "moments": moments.astype(np.float32),  # N*m/kg
        "moments_unit": "N*m/kg",
        "mass": mass,
        "subject_id": subj_id,
        "trial_name": trial_name,
        "ik_path": str(ik_path),
        "id_path": str(id_path),
        "time": time.astype(np.float32),
    }


class KineticsTCNDataset(Dataset):
    """
    Windowed dataset for TCN:
      x: (C_in, W)  where C_in = 2 * n_input_dofs
      y: (C_out, W) where C_out = n_output_dofs

    With ``laterality=unilateral`` and symmetric R/L ``input_indices`` / ``moment_indices`` (same length,
    even count), each time step range can yield **two** training windows (right chain and left chain), doubling
    data like the IMU ipsilateral pipeline. Use ``unilateral_paired_side_windows=False`` (or
    ``--legacy-unilateral-full-window`` in ``ik_id.train``) to keep one full R+L window per start (old behavior).

    ``target_sample_rate_hz``: if set (e.g. 100), resample IK/ID to a uniform grid at that rate before
    denoising and velocity estimation. Default ``None`` keeps the native timeline (~200 Hz for typical H5).
    Do not combine with ``rollout_decimate_step > 1`` (mutually exclusive).

    ``rollout_decimate_step``: if > 1 (typically ``2``), keep every ``step``-th sample after IK/ID alignment
    (no interpolation). ``step=2`` approximates 200 Hz → 100 Hz. Applied before denoising, same stage as
    resampling.

    Optional Butterworth low-pass on IK positions and ID moments is always **zero-phase**
    (``_lowpass_zero_phase`` / ``sosfiltfilt``), same as ``_denoise_pos_and_moments``.

    ``balance_loc_buckets_oversample``: after windowing, optionally upsample with replacement so buckets
    LG, SA, SD, RA, RD (see ``classify_loc_bucket``, same rules as ``compare_pipelineV4``) each contribute
    equally among buckets that appear in the loader. Windows mapped to OTHER / RA/RD / SA/SD keep one copy.
    """

    _UNSET = object()

    @staticmethod
    def _window_ok_for_training(
        trial: Dict[str, Any],
        start: int,
        end: int,
        input_indices: Optional[List[int]],
        moment_indices: Optional[List[int]],
    ) -> bool:
        """
        Require finite IK positions, velocities, and selected moments over the window.

        MeMo / partial IK exports can leave NaNs in angle columns or in velocities from
        splines; moments alone were previously checked, which let NaN inputs through and
        caused NaN MSE during training.
        """
        pos_w = trial["positions"][start:end]
        vel_w = trial["velocities"][start:end]
        mom_w = trial["moments"][start:end]
        if input_indices is not None:
            pos_w = pos_w[:, input_indices]
            vel_w = vel_w[:, input_indices]
        if moment_indices is not None:
            mom_w = mom_w[:, moment_indices]
        return (
            np.all(np.isfinite(pos_w))
            and np.all(np.isfinite(vel_w))
            and np.all(np.isfinite(mom_w))
        )

    @staticmethod
    def _row_finite_mask_for_training(
        trial: Dict[str, Any],
        input_indices: Optional[List[int]],
        moment_indices: Optional[List[int]],
    ) -> np.ndarray:
        """
        Per-frame finite mask equivalent to ``_window_ok_for_training`` checks.

        A frame is valid iff all selected IK positions, velocities, and moments are finite.
        """
        pos = trial["positions"]
        vel = trial["velocities"]
        mom = trial["moments"]
        if input_indices is not None:
            pos = pos[:, input_indices]
            vel = vel[:, input_indices]
        if moment_indices is not None:
            mom = mom[:, moment_indices]
        return (
            np.isfinite(pos).all(axis=1)
            & np.isfinite(vel).all(axis=1)
            & np.isfinite(mom).all(axis=1)
        )

    @staticmethod
    def _window_valid_flags_from_row_mask(
        row_ok: np.ndarray,
        window_size: int,
        starts: np.ndarray,
    ) -> np.ndarray:
        """
        For each ``start`` in ``starts``, return whether ``row_ok[start:start+window_size]`` is all True.
        """
        if starts.size == 0:
            return np.zeros((0,), dtype=bool)
        bad = (~row_ok).astype(np.int64, copy=False)
        csum = np.empty((bad.shape[0] + 1,), dtype=np.int64)
        csum[0] = 0
        np.cumsum(bad, out=csum[1:])
        bad_in_window = csum[starts + int(window_size)] - csum[starts]
        return bad_in_window == 0

    def _append_windows_for_trial(
        self,
        trial: Dict[str, Any],
        t_idx: int,
        stride_eff: int,
    ) -> None:
        """
        Append valid windows for one trial into ``self.windows``.
        Preserves legacy append order exactly: for paired windows, r then l per start.
        """
        n = trial["positions"].shape[0]
        if n < self.window_size:
            return
        starts = np.arange(0, n - self.window_size + 1, int(stride_eff), dtype=np.int64)
        if starts.size == 0:
            return

        if self._unilateral_paired:
            assert self._pair_in_r is not None and self._pair_mom_r is not None
            assert self._pair_in_l is not None and self._pair_mom_l is not None
            r_row_ok = self._row_finite_mask_for_training(
                trial, self._pair_in_r, self._pair_mom_r
            )
            l_row_ok = self._row_finite_mask_for_training(
                trial, self._pair_in_l, self._pair_mom_l
            )
            r_valid = self._window_valid_flags_from_row_mask(
                r_row_ok, self.window_size, starts
            )
            l_valid = self._window_valid_flags_from_row_mask(
                l_row_ok, self.window_size, starts
            )
            for i, st in enumerate(starts.tolist()):
                if bool(r_valid[i]):
                    self.windows.append((t_idx, int(st), "r"))
                if bool(l_valid[i]):
                    self.windows.append((t_idx, int(st), "l"))
            return

        row_ok = self._row_finite_mask_for_training(
            trial, self.input_indices, self.moment_indices
        )
        valid = self._window_valid_flags_from_row_mask(
            row_ok, self.window_size, starts
        )
        for i, st in enumerate(starts.tolist()):
            if bool(valid[i]):
                self.windows.append((t_idx, int(st), None))

    def _trial_loc_bucket(self, t_idx: int) -> str:
        if self.use_h5:
            sid, cond, trial, _ = self.h5_trial_refs[t_idx]
            return classify_loc_bucket(sid, cond, trial, self._loc_bucket_map or None)
        td = self.trial_dirs[t_idx]
        sid = extract_subject_id(td)
        cond = td.parent.name
        trial = td.name
        return classify_loc_bucket(sid, cond, trial, self._loc_bucket_map or None)

    def _apply_loc_bucket_oversample(self) -> None:
        """
        Duplicate windows so each of LG, SA, SD, RA, RD present in the loader contributes equally
        (same target count = max per-bucket count). Other buckets keep a single copy per window.
        """
        if not self.windows:
            return

        bucket_lists: Dict[str, List[Tuple[int, int, Optional[str]]]] = {
            k: [] for k in LOC_BUCKET_OVERSAMPLE_KEYS
        }
        other_wins: List[Tuple[int, int, Optional[str]]] = []
        for win in self.windows:
            b = self._trial_loc_bucket(win[0])
            if b in bucket_lists:
                bucket_lists[b].append(win)
            else:
                other_wins.append(win)

        present = {k: v for k, v in bucket_lists.items() if len(v) > 0}
        if len(present) < 2:
            print(
                "[KineticsTCNDataset] balance_loc_buckets_oversample: need ≥2 of "
                f"{LOC_BUCKET_OVERSAMPLE_KEYS} with windows (got {list(present.keys())}); skipping."
            )
            return

        target = max(len(v) for v in present.values())
        seed = self.loc_bucket_balance_seed
        rng = np.random.default_rng(int(seed) if seed is not None else None)

        balanced: List[Tuple[int, int, Optional[str]]] = []
        counts_before = {k: len(bucket_lists[k]) for k in LOC_BUCKET_OVERSAMPLE_KEYS}
        for k in LOC_BUCKET_OVERSAMPLE_KEYS:
            wins = bucket_lists[k]
            if not wins:
                continue
            n = len(wins)
            idx = rng.choice(n, size=int(target), replace=True)
            balanced.extend(wins[i] for i in idx.tolist())

        n_before = len(self.windows)
        self.windows = balanced + other_wins
        rng.shuffle(self.windows)
        n_after = len(self.windows)
        counts_after = {k: int(target) if counts_before[k] > 0 else 0 for k in LOC_BUCKET_OVERSAMPLE_KEYS}
        print(
            "[KineticsTCNDataset] balance_loc_buckets_oversample: "
            f"per-bucket window counts before {counts_before} → "
            f"balanced (each non-empty) ≈ {counts_after}; "
            f"other_bucket_windows={len(other_wins)}; "
            f"total_windows {n_before} → {n_after} "
            f"(seed={seed})"
        )

    def __init__(
        self,
        data_dir: Optional[str] = None,
        h5_dir: Optional[str] = None,
        use_h5: bool = True,
        window_size: int = 200,
        stride: int = 1,
        walking_only: bool = True,
        levelground_only: bool = False,
        normalize: bool = True,
        max_files: Optional[int] = None,
        stats: Optional[Dict] = None,
        moment_indices: Union[Optional[List[int]], object] = _UNSET,
        input_indices: Union[Optional[List[int]], object] = _UNSET,
        input_mode: str = "lower_limb",
        output_mode: str = "lower_limb",
        laterality: str = "bilateral",
        unilateral_paired_side_windows: Optional[bool] = None,
        subject_ids: Optional[List[str]] = None,
        b3d_files: Optional[List[Path]] = None,  # interpreted as explicit trial_dirs if provided
        preload_trials: bool = False,
        apply_lowpass_filter: bool = True,
        lowpass_cutoff_hz: float = 4.0,
        lowpass_order: int = 4,
        target_sample_rate_hz: Optional[float] = None,
        rollout_decimate_step: int = 1,
        apply_velocity_lowpass_filter: bool = False,
        velocity_lowpass_cutoff_hz: Optional[float] = None,
        velocity_lowpass_order: Optional[int] = None,
        balance_loc_buckets_oversample: bool = False,
        loc_bucket_balance_seed: Optional[int] = None,
        loc_ascent_descent_map: Optional[str] = None,
    ):
        if data_dir is None:
            raise ValueError("data_dir must point to processed Camargo root")

        self.window_size = window_size
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        self.normalize = normalize
        self.preload_trials = preload_trials
        self.apply_lowpass_filter = bool(apply_lowpass_filter)
        self.lowpass_cutoff_hz = float(lowpass_cutoff_hz)
        self.lowpass_order = int(lowpass_order)
        self.apply_velocity_lowpass_filter = bool(apply_velocity_lowpass_filter)
        self.velocity_lowpass_cutoff_hz = (
            None if velocity_lowpass_cutoff_hz is None else float(velocity_lowpass_cutoff_hz)
        )
        self.velocity_lowpass_order = (
            None if velocity_lowpass_order is None else int(velocity_lowpass_order)
        )
        self.target_sample_rate_hz: Optional[float] = (
            None if target_sample_rate_hz is None else float(target_sample_rate_hz)
        )
        self.rollout_decimate_step = max(1, int(rollout_decimate_step))
        if self.rollout_decimate_step > 1 and self.target_sample_rate_hz is not None:
            raise ValueError(
                "Use either rollout_decimate_step>1 (stride subsample) or target_sample_rate_hz "
                "(interpolation resample), not both."
            )

        self.laterality = normalize_laterality(laterality)
        self._use_unilateral_flip = self.laterality == "unilateral"
        self.walking_only = bool(walking_only)

        self._pair_in_r: Optional[List[int]] = None
        self._pair_in_l: Optional[List[int]] = None
        self._pair_mom_r: Optional[List[int]] = None
        self._pair_mom_l: Optional[List[int]] = None
        self.levelground_only = bool(levelground_only)
        self.balance_loc_buckets_oversample = bool(balance_loc_buckets_oversample)
        self.loc_bucket_balance_seed = (
            None if loc_bucket_balance_seed is None else int(loc_bucket_balance_seed)
        )
        self._loc_bucket_map: Dict[Tuple[str, str, str], str] = (
            load_loc_ascent_descent_mapping(loc_ascent_descent_map)
            if loc_ascent_descent_map
            else {}
        )

        subject_ids_norm: Optional[List[str]] = None
        if subject_ids is not None:
            subject_ids_norm = [s.upper() for s in subject_ids]

        # If indices are not provided, default to lower body hip(3DoF) + knee(1DoF) + ankle(1DoF).
        # Bilateral vs unilateral uses the same DOF sets; unilateral applies a sign flip per trial.
        if moment_indices is self._UNSET:
            base = OUTPUT_MODE_INDICES.get(output_mode, LOWER_LIMB_MOMENT_INDICES)
            if base is None:
                self.moment_indices = list(range(len(MOMENT_NAMES)))
            else:
                self.moment_indices = list(base)
        else:
            # can be None (= all outputs)
            self.moment_indices = moment_indices

        if input_indices is self._UNSET:
            base = INPUT_MODE_INDICES.get(input_mode, LOWER_LIMB_INPUT_INDICES)
            if base is None:
                self.input_indices = list(range(len(IK_DOF_NAMES)))
            else:
                self.input_indices = list(base)
        else:
            # can be None (= all inputs)
            self.input_indices = input_indices

        _eligible = _unilateral_paired_indices_eligible(
            self.input_indices if isinstance(self.input_indices, list) else None,
            self.moment_indices if isinstance(self.moment_indices, list) else None,
        )
        if unilateral_paired_side_windows is None:
            self._unilateral_paired = bool(self._use_unilateral_flip and _eligible)
        else:
            if unilateral_paired_side_windows and not self._use_unilateral_flip:
                raise ValueError(
                    "unilateral_paired_side_windows=True requires laterality=unilateral."
                )
            if unilateral_paired_side_windows and not _eligible:
                raise ValueError(
                    "unilateral_paired_side_windows=True requires symmetric R/L input and moment index lists."
                )
            self._unilateral_paired = bool(unilateral_paired_side_windows)
        if self._unilateral_paired:
            assert self.input_indices is not None and self.moment_indices is not None
            h = len(self.input_indices) // 2
            self._pair_in_r = self.input_indices[:h]
            self._pair_in_l = self.input_indices[h:]
            self._pair_mom_r = self.moment_indices[:h]
            self._pair_mom_l = self.moment_indices[h:]

        self.data_dir = data_dir
        self.meta_map = _load_subject_metadata_map(data_dir)
        self._trial_cache: Dict[int, Dict] = {}
        self.trial_dirs: List[Path] = []
        self.h5_trial_refs: List[Tuple[str, str, str, str]] = []
        self.trials: List[Dict] = []  # only used if preload_trials=True
        self.use_h5 = False
        self.h5_dir: Optional[str] = None

        requested_h5_dir = h5_dir or _default_h5_dir_from_processed_root(data_dir)
        if use_h5 and HAS_H5PY and Path(requested_h5_dir).exists():
            self.use_h5 = True
            self.h5_dir = requested_h5_dir
        elif use_h5 and not HAS_H5PY:
            print("[KineticsTCNDataset] h5py not available; falling back to text .mot/.sto.")
        elif use_h5 and not Path(requested_h5_dir).exists():
            print(f"[KineticsTCNDataset] H5 dir not found ({requested_h5_dir}); falling back to text .mot/.sto.")
        if self.apply_lowpass_filter and not HAS_SCIPY_SIGNAL:
            print(
                "[KineticsTCNDataset] scipy.signal unavailable; "
                "zero-phase low-pass denoising disabled."
            )

        if self.use_h5:
            if b3d_files is not None:
                # Input is expected as processed trial dirs; convert to H5 refs.
                refs: List[Tuple[str, str, str, str]] = []
                for p in sorted([Path(x) for x in b3d_files]):
                    sid = extract_subject_id(p)
                    if _subject_id_excluded_temp_broken_h5(sid):
                        continue
                    cond = p.parent.name
                    if not include_condition_for_dataset(
                        cond,
                        walking_only=self.walking_only,
                        levelground_only=self.levelground_only,
                        subject_id=sid,
                    ):
                        continue
                    trial = p.name
                    subject_h5_path = Path(self.h5_dir) / f"{sid}.h5"
                    refs.append((sid, cond, trial, str(subject_h5_path)))
                self.h5_trial_refs = refs
                source_desc = "explicit trial list (mapped to h5)"
            else:
                refs = []
                for subject_h5_path in sorted(Path(self.h5_dir).glob("S*.h5")):
                    sid = subject_h5_path.stem.upper()
                    if _subject_id_excluded_temp_broken_h5(sid):
                        continue
                    if subject_ids_norm is not None and sid not in set(subject_ids_norm):
                        continue
                    with h5py.File(subject_h5_path, "r") as h5f:
                        for cond in sorted(h5f.keys()):
                            if not include_condition_for_dataset(
                                cond,
                                walking_only=self.walking_only,
                                levelground_only=self.levelground_only,
                                subject_id=sid,
                            ):
                                continue
                            for trial in sorted(h5f[cond].keys()):
                                refs.append((sid, cond, trial, str(subject_h5_path)))
                self.h5_trial_refs = refs
                source_desc = self.h5_dir

            if max_files is not None:
                self.h5_trial_refs = self.h5_trial_refs[:max_files]

            print(f"[KineticsTCNDataset] Found {len(self.h5_trial_refs)} H5 trials in {source_desc}")
            if self.levelground_only:
                print(
                    "  [KineticsTCNDataset] levelground_only=True "
                    "(level-included tasks only; see is_levelground_subset_condition)"
                )
        else:
            if b3d_files is not None:
                trial_dirs = sorted([Path(p) for p in b3d_files])
                source_desc = "explicit trial list"
            else:
                trial_dirs = find_trial_dirs(data_dir)
                source_desc = data_dir

            trial_dirs = [
                td
                for td in trial_dirs
                if (not _subject_id_excluded_temp_broken_h5(extract_subject_id(td)))
                and include_condition_for_dataset(
                    td.parent.name,
                    walking_only=self.walking_only,
                    levelground_only=self.levelground_only,
                    subject_id=extract_subject_id(td),
                )
            ]
            if max_files is not None:
                trial_dirs = trial_dirs[:max_files]

            self.trial_dirs = trial_dirs
            print(f"[KineticsTCNDataset] Found {len(self.trial_dirs)} trials in {source_desc}")
            if self.levelground_only:
                print(
                    "  [KineticsTCNDataset] levelground_only=True "
                    "(level-included tasks only; see is_levelground_subset_condition)"
                )

        if stats is not None:
            self.pos_mean = stats["pos_mean"]
            self.pos_std = stats["pos_std"]
            self.vel_mean = stats["vel_mean"]
            self.vel_std = stats["vel_std"]
            total_frames_for_stats = None
            # We'll still need to validate windows, but no need to recompute stats.
            compute_stats = False
        else:
            # Streaming stats to avoid concatenating all trials at once.
            compute_stats = True
            total_frames_for_stats = 0
            sum_pos = np.zeros(len(IK_DOF_NAMES), dtype=np.float64)
            sumsq_pos = np.zeros(len(IK_DOF_NAMES), dtype=np.float64)
            sum_vel = np.zeros(len(IK_DOF_NAMES), dtype=np.float64)
            sumsq_vel = np.zeros(len(IK_DOF_NAMES), dtype=np.float64)

        # Build window index once during init.
        # Tuple is (trial_idx, start_frame, side) with side None (bilateral / legacy full window) or "r"/"l".
        self.windows: List[Tuple[int, int, Optional[str]]] = []
        stride_eff = self._window_stride()

        candidates = self.h5_trial_refs if self.use_h5 else self.trial_dirs
        valid_trial_dirs: List[Path] = []
        valid_h5_refs: List[Tuple[str, str, str, str]] = []
        for i, ref in enumerate(candidates):
            if (i + 1) % 200 == 0 or i == 0:
                print(f"  Loading trial {i+1}/{len(candidates)}: {ref}")
            if self.use_h5:
                trial = self._load_trial_from_h5_ref(ref)
                _sid, cond_name, _trial_name, _hp = ref
            else:
                trial = load_trial_from_processed(
                    ref,
                    root_dir=data_dir,
                    meta_map=self.meta_map,
                    apply_lowpass_filter=self.apply_lowpass_filter,
                    lowpass_cutoff_hz=self.lowpass_cutoff_hz,
                    lowpass_order=self.lowpass_order,
                    target_sample_rate_hz=self.target_sample_rate_hz,
                    rollout_decimate_step=self.rollout_decimate_step,
                )
                cond_name = ref.parent.name
            if trial is None:
                continue

            if self._use_unilateral_flip:
                _apply_unilateral_flip_to_trial(trial)

            if self.use_h5:
                t_idx = len(valid_h5_refs)
                valid_h5_refs.append(ref)
            else:
                t_idx = len(valid_trial_dirs)
                valid_trial_dirs.append(ref)

            if compute_stats:
                pos = trial["positions"].astype(np.float64)
                vel = trial["velocities"].astype(np.float64)
                T = pos.shape[0]
                good = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1)
                if np.any(good):
                    pos = pos[good]
                    vel = vel[good]
                    n_g = float(pos.shape[0])
                    total_frames_for_stats += n_g
                    sum_pos += pos.sum(axis=0)
                    sumsq_pos += np.square(pos).sum(axis=0)
                    sum_vel += vel.sum(axis=0)
                    sumsq_vel += np.square(vel).sum(axis=0)

            # Create windows and validate outputs for those windows.
            self._append_windows_for_trial(trial, t_idx, stride_eff)

            if self.preload_trials:
                self.trials.append(trial)

            # If we don't preload, we drop the arrays immediately (lazy loading in workers).
            if not self.preload_trials:
                del trial

        if self.use_h5:
            self.h5_trial_refs = valid_h5_refs
        else:
            self.trial_dirs = valid_trial_dirs

        if self.balance_loc_buckets_oversample:
            self._apply_loc_bucket_oversample()

        n_valid_trials = len(self.h5_trial_refs) if self.use_h5 else len(self.trial_dirs)
        if n_valid_trials == 0:
            raise ValueError(f"No valid trials found in {source_desc}")

        if compute_stats:
            n_total = float(total_frames_for_stats if total_frames_for_stats is not None else 1.0)
            self.pos_mean = sum_pos / n_total
            self.vel_mean = sum_vel / n_total
            pos_var = sumsq_pos / n_total - np.square(self.pos_mean)
            vel_var = sumsq_vel / n_total - np.square(self.vel_mean)
            self.pos_std = np.sqrt(np.maximum(pos_var, 0.0)) + 1e-8
            self.vel_std = np.sqrt(np.maximum(vel_var, 0.0)) + 1e-8

        n_windows = len(getattr(self, "windows", []))
        _pair_note = f", unilateral_paired_side_windows={self._unilateral_paired}" if self._use_unilateral_flip else ""
        print(
            f"  Loaded {n_valid_trials} valid trials, "
            f"created {n_windows} windows (window={window_size}, stride={stride}, "
            f"preload_trials={self.preload_trials}, use_h5={self.use_h5}{_pair_note})"
        )
        if n_windows == 0 and n_valid_trials > 0:
            raise ValueError(
                "No training windows: trials may be shorter than window_size, or all windows were "
                "dropped because IK/velocity/moment channels had non-finite values (common with "
                "partial MeMo H5 exports). See check_memo_nan_windows.py."
            )

    def _load_trial_from_h5_ref(self, ref: Tuple[str, str, str, str]) -> Optional[Dict]:
        """
        Load trial arrays from H5 tuple:
          (subject_id, condition_name, trial_name, subject_h5_path)
        """
        subj_id, cond_name, trial_name, subject_h5_path = ref
        h5_path = Path(subject_h5_path)
        if not h5_path.exists():
            return None

        with h5py.File(h5_path, "r") as h5f:
            if cond_name not in h5f:
                return None
            cond_group = h5f[cond_name]
            if trial_name not in cond_group:
                return None
            trial_group = cond_group[trial_name]
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

        if "time" not in id_cols:
            return None

        ik_tp = _ik_time_and_pos_deg(ik_cols, ik_data)
        if ik_tp is None:
            return None
        time, pos_deg = ik_tp
        pos = np.deg2rad(pos_deg)

        mass_val = self.meta_map.get(subj_id, {}).get(
            "weight_kg",
            self.meta_map.get(subj_id, {}).get("mass_kg", np.nan),
        )
        mass = _coerce_weight_kg(mass_val)
        # Moments are stored as N*m/kg, so we don't strictly require mass.
        # Keep mass for completeness, but don't reject trials when metadata is missing.
        if not np.isfinite(mass) or mass <= 0:
            mass = np.nan

        id_time = id_data[:, id_cols.index("time")]
        n = min(len(time), len(id_time))
        time = time[:n]
        pos = pos[:n]
        id_data = id_data[:n]

        _manifest = Path(self.h5_dir) if self.h5_dir else Path(self.data_dir)
        pos = _apply_temp_addbiomech_ik_angle_sign_fix(subj_id, pos, _manifest)

        moments = np.full((n, len(MOMENT_NAMES)), np.nan, dtype=np.float64)
        for j, name in enumerate(MOMENT_NAMES):
            col = f"{name}_moment"
            if col in id_cols:
                moments[:, j] = id_data[:, id_cols.index(col)]
            else:
                # If a moment channel is missing entirely, leave NaNs.
                # Windowing will drop windows where selected output channels are non-finite.
                pass

        if self.rollout_decimate_step > 1:
            time, pos, moments = decimate_trial_aligned(
                time, pos, moments, int(self.rollout_decimate_step)
            )
        elif self.target_sample_rate_hz is not None and self.target_sample_rate_hz > 0:
            time, pos, moments = resample_trial_to_uniform_hz(
                time, pos, moments, float(self.target_sample_rate_hz)
            )

        if self.apply_lowpass_filter:
            pos, moments = _denoise_pos_and_moments(
                pos,
                moments,
                time,
                apply_lowpass_filter=self.apply_lowpass_filter,
                lowpass_cutoff_hz=self.lowpass_cutoff_hz,
                lowpass_order=self.lowpass_order,
            )

        vel = _compute_velocity(pos, time)
        if self.apply_velocity_lowpass_filter:
            vel_cutoff = (
                float(self.velocity_lowpass_cutoff_hz)
                if self.velocity_lowpass_cutoff_hz is not None
                else float(self.lowpass_cutoff_hz)
            )
            vel_order = (
                int(self.velocity_lowpass_order)
                if self.velocity_lowpass_order is not None
                else int(self.lowpass_order)
            )
            vel = _lowpass_zero_phase(vel, time, cutoff_hz=vel_cutoff, order=vel_order)

        trial_name_full = f"{subj_id}/{cond_name}/{trial_name}"
        return {
            "positions": pos.astype(np.float32),
            "velocities": vel.astype(np.float32),
            "moments": moments.astype(np.float32),
            "moments_unit": "N*m/kg",
            "mass": mass,
            "subject_id": subj_id,
            "trial_name": trial_name_full,
            "time": time.astype(np.float32),
        }

    def _get_trial(self, t_idx: int) -> Dict:
        if self.preload_trials:
            return self.trials[t_idx]

        if t_idx in self._trial_cache:
            return self._trial_cache[t_idx]

        if self.use_h5:
            ref = self.h5_trial_refs[t_idx]
            trial = self._load_trial_from_h5_ref(ref)
            label = ref
        else:
            td = self.trial_dirs[t_idx]
            trial = load_trial_from_processed(
                td,
                root_dir=self.data_dir,
                meta_map=self.meta_map,
                apply_lowpass_filter=self.apply_lowpass_filter,
                lowpass_cutoff_hz=self.lowpass_cutoff_hz,
                lowpass_order=self.lowpass_order,
                target_sample_rate_hz=self.target_sample_rate_hz,
                rollout_decimate_step=self.rollout_decimate_step,
                apply_velocity_lowpass_filter=self.apply_velocity_lowpass_filter,
                velocity_lowpass_cutoff_hz=self.velocity_lowpass_cutoff_hz,
                velocity_lowpass_order=self.velocity_lowpass_order,
            )
            label = td
        if trial is None:
            raise RuntimeError(f"Failed to reload trial: {label}")

        if self._use_unilateral_flip:
            _apply_unilateral_flip_to_trial(trial)

        self._trial_cache[t_idx] = trial
        return trial

    def _compute_stats(self):
        all_pos = np.concatenate([t["positions"] for t in self.trials], axis=0)
        all_vel = np.concatenate([t["velocities"] for t in self.trials], axis=0)
        self.pos_mean = all_pos.mean(axis=0)
        self.pos_std = all_pos.std(axis=0) + 1e-8
        self.vel_mean = all_vel.mean(axis=0)
        self.vel_std = all_vel.std(axis=0) + 1e-8

    def get_stats(self) -> Dict:
        return {
            "pos_mean": self.pos_mean,
            "pos_std": self.pos_std,
            "vel_mean": self.vel_mean,
            "vel_std": self.vel_std,
        }

    def _window_stride(self) -> int:
        """Sliding-window step in samples (same for every condition)."""
        return max(1, int(self.stride))

    def _build_index(self):
        self.windows: List[Tuple[int, int, Optional[str]]] = []
        stride_eff = self._window_stride()
        for t_idx, trial in enumerate(self.trials):
            self._append_windows_for_trial(trial, t_idx, stride_eff)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        t_idx, start, side = self.windows[idx]
        end = start + self.window_size
        trial = self._get_trial(t_idx)

        pos = trial["positions"][start:end].copy()
        vel = trial["velocities"][start:end].copy()
        mom = trial["moments"][start:end].copy()  # N*m/kg for MOMENT_NAMES

        if self.normalize:
            pos = (pos - self.pos_mean) / self.pos_std
            vel = (vel - self.vel_mean) / self.vel_std

        if side is not None:
            assert self._unilateral_paired
            assert self._pair_in_r is not None and self._pair_in_l is not None
            assert self._pair_mom_r is not None and self._pair_mom_l is not None
            in_i = self._pair_in_r if side == "r" else self._pair_in_l
            mom_i = self._pair_mom_r if side == "r" else self._pair_mom_l
            pos = pos[:, in_i]
            vel = vel[:, in_i]
            mom = mom[:, mom_i]
        else:
            if self.input_indices is not None:
                pos = pos[:, self.input_indices]
                vel = vel[:, self.input_indices]

            if self.moment_indices is not None:
                mom = mom[:, self.moment_indices]

        x = np.concatenate([pos, vel], axis=1).T.astype(np.float32)
        y = mom.T.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def unilateral_paired(self) -> bool:
        return self._unilateral_paired

    @property
    def n_input_channels(self):
        if self._unilateral_paired and self.input_indices is not None:
            n = len(self.input_indices) // 2
        else:
            n = len(self.input_indices) if self.input_indices is not None else len(IK_DOF_NAMES)
        return n * 2

    @property
    def n_output_channels(self):
        if self._unilateral_paired and self.moment_indices is not None:
            return len(self.moment_indices) // 2
        if self.moment_indices is not None:
            return len(self.moment_indices)
        return len(MOMENT_NAMES)

    @property
    def input_dof_names(self) -> List[str]:
        if self._unilateral_paired and self.input_indices is not None:
            return generic_ik_names_paired(self.input_indices)
        if self.input_indices is not None:
            return [IK_DOF_NAMES[i] for i in self.input_indices]
        return list(IK_DOF_NAMES)

    @property
    def output_dof_names(self) -> List[str]:
        if self._unilateral_paired and self.moment_indices is not None:
            return generic_moment_names_paired(self.moment_indices)
        if self.moment_indices is not None:
            return [MOMENT_NAMES[i] for i in self.moment_indices]
        return list(MOMENT_NAMES)

