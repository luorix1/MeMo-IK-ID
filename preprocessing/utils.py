"""Shared constants and utilities for the cascade OpenSim processing pipeline."""
from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Folder-type → output folder name
# ---------------------------------------------------------------------------
FOLDER_TYPE_TO_OUTPUT: dict[str, str] = {
    "awinda":    "awinda",
    "hip_exo":   "hip-exo",
    "knee_exo":  "knee-exo",
}

# Sub-folders created inside each trial-type directory
OUTPUT_SUBFOLDERS = ["marker", "grf", "mocap", "ik", "id"]

# ---------------------------------------------------------------------------
# OpenSim marker weights per trial type
# Tuple: (zero_weight, tiny_weight, small_weight)
#   zero  = 0.00  — marker is hidden/unusable (e.g. under exo frame)
#   tiny  = 0.01  — marker sits on the exo and drifts relative to the body
#   small = 0.10  — marker is partially obstructed / less reliable
# ---------------------------------------------------------------------------
MARKER_WEIGHTS: dict[str, tuple[set[str], set[str], set[str]]] = {
    #              zero              tiny              small
    "awinda":   (set(),             set(),             set()),
    "hip_exo":  ({"LPSI", "RPSI"}, {"LGTR", "RGTR"}, {"RLFC", "LLFC"}),
    "knee_exo": (set(),             {"RLFC"},           set()),
}

# ---------------------------------------------------------------------------
# Model search patterns per trial type (checked in priority order)
# ---------------------------------------------------------------------------
MODEL_SEARCH_PATTERNS: dict[str, list[str]] = {
    "awinda":   ["*_no_exo.osim", "*_no_exo_*.osim"],
    "hip_exo":  ["*_hip_exo_added_inertia.osim", "*_hip_exo*.osim"],
    "knee_exo": [
        "*_knee_exo_added_inertia.osim",
        "*_knee_exo*.osim",
        "*_hip_exo_added_inertia.osim",
        "*_hip_exo*.osim",
    ],
}

# ---------------------------------------------------------------------------
# OpenSim processing constants
# ---------------------------------------------------------------------------
LOWPASS_CUTOFF_HZ: float = 6.0
FORCES_TO_EXCLUDE: list[str] = ["Muscles"]
IK_ACCURACY: float = 1e-5

# Added exoskeleton mass (kg) per OpenSim body, keyed by exo type.
# Diagonal inertia moments are scaled proportionally to the new total mass;
# off-diagonal products of inertia are left unchanged.
# The correct table is selected automatically from the model filename.
ADDED_MASS_BY_EXO_TYPE: dict[str, dict[str, float]] = {
    "hip_exo": {
        "pelvis":   2.00,
        "femur_r":  0.50,
        "femur_l":  0.50,
        "torso":    1.30,
    },
    "knee_exo": {
        "pelvis":   0.90,
        "femur_r":  0.91,
        "tibia_r":  0.39,
    },
}

# ---------------------------------------------------------------------------
# File extensions to skip (Vicon / session metadata)
# ---------------------------------------------------------------------------
SKIP_EXTENSIONS: set[str] = {
    ".c3d", ".x1d", ".x2d", ".xcp",
    ".history", ".enf", ".system",
    ".mp", ".vsk", ".patient",
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_trial_dirs(subject_root: Path) -> dict[str, Path]:
    """Return {folder_type: path} for every awinda/hip_exo/knee_exo subdirectory."""
    result: dict[str, Path] = {}
    for subdir in sorted(subject_root.iterdir()):
        if not subdir.is_dir():
            continue
        name_lower = subdir.name.lower()
        for suffix in FOLDER_TYPE_TO_OUTPUT:
            if name_lower.endswith(f"_{suffix}") or name_lower == suffix:
                if suffix not in result:
                    result[suffix] = subdir
                break
    return result


def find_model(opensim_dir: Path, folder_type: str) -> Path | None:
    """Return the first matching .osim model for *folder_type* inside *opensim_dir*."""
    for pattern in MODEL_SEARCH_PATTERNS.get(folder_type, []):
        matches = sorted(opensim_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Trial-name extraction
# ---------------------------------------------------------------------------

def format_trial_name(raw: str) -> str:
    """Reformat a raw Vicon trial identifier into display form.

    The Vicon naming convention is ``{speed}_{direction}``; this function
    inverts that to ``{DIRECTION}_{speed}`` so the task type comes first.

    Examples
    --------
    0p8mps_lg  →  LG_0p8mps
    0p8mps_ra  →  RA_0p8mps
    0p8mps_rd  →  RD_0p8mps
    1p2mps_lg  →  LG_1p2mps
    static     →  static      (unchanged)
    """
    if raw == "static":
        return "static"
    tokens = raw.split("_")
    if len(tokens) >= 2:
        direction = tokens[-1].upper()
        speed = "_".join(tokens[:-1])
        return f"{direction}_{speed}"
    return raw


def extract_trial_name(stem: str, folder_type: str, subject_lower: str) -> str | None:
    """Return the formatted trial name from a source file stem, or None if unrecognised.

    Examples
    --------
    awinda   : ab01_jinwoo_awinda_0p8mps_lg        → LG_0p8mps
    hip_exo  : ab01_jinwoo_hip_0p8mps_lg_exo_on_1  → LG_0p8mps
    hip_exo  : ab01_jinwoo_hip_exo_on_static        → static
    hip_exo  : ab02_oscar_hip_static                → static
    hip_exo  : ab01_jinwoo_hip_0p8mps_lg_exo_on_ik → LG_0p8mps  (pre-computed IK)
    knee_exo : ab01_jinwoo_knee_0p8mps_rd_exo_on_1 → RD_0p8mps
    knee_exo : ab01_jinwoo_knee_exo_on_static       → static
    knee_exo : ab02_oscar_knee_static               → static
    """
    s = re.escape(subject_lower)

    if folder_type == "awinda":
        m = re.match(rf"^{s}_awinda_(.+)$", stem)
        if m:
            # Strip optional _no_exo[_rep] suffix (present in some subjects)
            raw = re.sub(r"_no_exo(?:_\d+)?$", "", m.group(1))
            return format_trial_name(raw)

    elif folder_type == "hip_exo":
        if re.match(rf"^{s}_hip_(?:exo_on_)?static\d*$", stem):
            return "static"
        # dynamic rep: …_hip_{trial}_exo_on_{rep}
        m = re.match(rf"^{s}_hip_(.+)_exo_on_(\d+)$", stem)
        if m:
            return format_trial_name(m.group(1))
        # pre-computed IK: …_hip_{trial}_exo_on_ik
        m = re.match(rf"^{s}_hip_(.+)_exo_on_ik$", stem)
        if m:
            return format_trial_name(m.group(1))

    elif folder_type == "knee_exo":
        if re.match(rf"^{s}_knee_(?:exo_on_)?static\d*$", stem):
            return "static"
        # dynamic rep: …_knee_{trial}_exo_on_{rep}
        m = re.match(rf"^{s}_knee_(.+)_exo_on_(\d+)$", stem)
        if m:
            return format_trial_name(m.group(1))

    return None


def is_static_trial(trial_name: str) -> bool:
    return "static" in trial_name.lower()
