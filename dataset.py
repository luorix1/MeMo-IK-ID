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

Denoising (optional, for noisy IK / ID):
  - Temporal median filter (impulse / double-peak spikes)
  - Zero-phase Butterworth low-pass (default 4 Hz)
  Velocities are always computed from the final filtered positions.
"""

from __future__ import annotations

import json
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
    from scipy.signal import butter, medfilt, sosfiltfilt

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


def normalize_laterality(laterality: Optional[str]) -> str:
    """
    ``bilateral`` (alias: ``both``): use all sides as stored.
    ``unilateral``: same DOFs as bilateral; left hip adduction & rotation are negated
    to align L/R sign convention (right unchanged).
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


INPUT_MODE_INDICES = {
    "full": None,
    "lower_limb": LOWER_LIMB_INPUT_INDICES,
    "sagittal": SAGITTAL_INPUT_INDICES,
}

OUTPUT_MODE_INDICES = {
    "all": None,
    "lower_limb": LOWER_LIMB_MOMENT_INDICES,
    "hip_knee": HIP_KNEE_MOMENT_INDICES,
    "sagittal_hip_knee": SAGITTAL_HIP_KNEE_MOMENT_INDICES,
    "sagittal_hip_knee_ankle": SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES,
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
        or ("ramp" in name)
        or ("incline" in name)
        or ("stair" in name)
        or ("treadmill" in name)
        or ("normal_walk" in name)
        or ("dynamic_walk" in name)
    )
    # Common non-walking conditions.
    exclude = ("static" in name)
    return bool(include and not exclude)


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
    Falls back to input unchanged when SciPy signal is unavailable or
    filtering settings are invalid for the current trial length/sample rate.
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
        return sosfiltfilt(sos, data.astype(np.float64), axis=0)
    except Exception:
        return data


def _median_filter_timeseries(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Short temporal median filter per channel (removes impulse / spike noise).
    ``kernel_size`` is coerced to an odd integer in [3, T].
    NaNs are filled with the column median for filtering only, then restored.
    """
    if not HAS_SCIPY_SIGNAL or kernel_size < 3:
        return data
    if data.ndim != 2 or data.shape[0] < 3:
        return data
    T = data.shape[0]
    ks = int(kernel_size)
    if ks % 2 == 0:
        ks += 1
    ks = max(3, min(ks, T if T % 2 == 1 else T - 1))
    if ks < 3:
        return data

    x = data.astype(np.float64)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        col = x[:, j].copy()
        if not np.any(np.isfinite(col)):
            out[:, j] = col
            continue
        nan_m = ~np.isfinite(col)
        fill = float(np.nanmedian(col))
        col[nan_m] = fill
        try:
            out[:, j] = medfilt(col, kernel_size=ks)
        except Exception:
            out[:, j] = x[:, j]
        out[nan_m, j] = np.nan
    return out


def _denoise_pos_and_moments(
    pos: np.ndarray,
    moments: np.ndarray,
    time: np.ndarray,
    *,
    median_kernel_samples: int,
    apply_lowpass_filter: bool,
    lowpass_cutoff_hz: float,
    lowpass_order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Denoise pipeline for IK positions (rad) and ID moments.

    Order: optional median → optional zero-phase Butterworth low-pass.
    """
    pos = pos.astype(np.float64)
    moments = moments.astype(np.float64)

    def _mom_median(m: np.ndarray) -> np.ndarray:
        finite = np.isfinite(m)
        m_fill = np.where(finite, m, 0.0)
        m_f = _median_filter_timeseries(m_fill, median_kernel_samples)
        return np.where(finite, m_f, np.nan)

    if median_kernel_samples >= 3:
        pos = _median_filter_timeseries(pos, median_kernel_samples)
        moments = _mom_median(moments)

    if apply_lowpass_filter:
        pos = _lowpass_zero_phase(pos, time, cutoff_hz=lowpass_cutoff_hz, order=lowpass_order)
        finite_mom = np.isfinite(moments)
        moments_fill = np.where(finite_mom, moments, 0.0)
        moments_filt = _lowpass_zero_phase(
            moments_fill, time, cutoff_hz=lowpass_cutoff_hz, order=lowpass_order
        )
        moments = np.where(finite_mom, moments_filt, np.nan)

    return pos, moments


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
    apply_lowpass_filter: bool = False,
    lowpass_cutoff_hz: float = 4.0,
    lowpass_order: int = 4,
    median_kernel_samples: int = 0,
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

    moments = np.full((n, len(MOMENT_NAMES)), np.nan, dtype=np.float64)
    for j, name in enumerate(MOMENT_NAMES):
        col = f"{name}_moment"
        if col in id_cols:
            moments[:, j] = id_data[:, id_cols.index(col)]
        else:
            # If a moment channel is missing entirely, leave NaNs.
            # Windowing will drop windows where selected output channels are non-finite.
            pass

    if median_kernel_samples >= 3 or apply_lowpass_filter:
        pos, moments = _denoise_pos_and_moments(
            pos,
            moments,
            time,
            median_kernel_samples=median_kernel_samples,
            apply_lowpass_filter=apply_lowpass_filter,
            lowpass_cutoff_hz=lowpass_cutoff_hz,
            lowpass_order=lowpass_order,
        )

    vel = _compute_velocity(pos, time)

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
    """

    _UNSET = object()

    def __init__(
        self,
        data_dir: Optional[str] = None,
        h5_dir: Optional[str] = None,
        use_h5: bool = True,
        window_size: int = 200,
        stride: int = 50,
        walking_only: bool = True,
        normalize: bool = True,
        max_files: Optional[int] = None,
        stats: Optional[Dict] = None,
        moment_indices: Union[Optional[List[int]], object] = _UNSET,
        input_indices: Union[Optional[List[int]], object] = _UNSET,
        input_mode: str = "lower_limb",
        output_mode: str = "lower_limb",
        laterality: str = "bilateral",
        subject_ids: Optional[List[str]] = None,
        b3d_files: Optional[List[Path]] = None,  # interpreted as explicit trial_dirs if provided
        preload_trials: bool = False,
        apply_lowpass_filter: bool = True,
        lowpass_cutoff_hz: float = 4.0,
        lowpass_order: int = 4,
        median_kernel_samples: int = 0,
    ):
        if data_dir is None:
            raise ValueError("data_dir must point to processed Camargo root")

        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.preload_trials = preload_trials
        self.apply_lowpass_filter = bool(apply_lowpass_filter)
        self.lowpass_cutoff_hz = float(lowpass_cutoff_hz)
        self.lowpass_order = int(lowpass_order)
        self.median_kernel_samples = int(median_kernel_samples)

        self.laterality = normalize_laterality(laterality)
        self._use_unilateral_flip = self.laterality == "unilateral"

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
        if (self.apply_lowpass_filter or self.median_kernel_samples >= 3) and not HAS_SCIPY_SIGNAL:
            print(
                "[KineticsTCNDataset] scipy.signal unavailable; "
                "low-pass / median denoising disabled."
            )

        if self.use_h5:
            if b3d_files is not None:
                # Input is expected as processed trial dirs; convert to H5 refs.
                refs: List[Tuple[str, str, str, str]] = []
                for p in sorted([Path(x) for x in b3d_files]):
                    sid = extract_subject_id(p)
                    cond = p.parent.name
                    trial = p.name
                    subject_h5_path = Path(self.h5_dir) / f"{sid}.h5"
                    refs.append((sid, cond, trial, str(subject_h5_path)))
                self.h5_trial_refs = refs
                source_desc = "explicit trial list (mapped to h5)"
            else:
                refs = []
                for subject_h5_path in sorted(Path(self.h5_dir).glob("S*.h5")):
                    sid = subject_h5_path.stem.upper()
                    if subject_ids_norm is not None and sid not in set(subject_ids_norm):
                        continue
                    with h5py.File(subject_h5_path, "r") as h5f:
                        for cond in sorted(h5f.keys()):
                            if walking_only and not is_walking_condition(Path(cond)):
                                continue
                            for trial in sorted(h5f[cond].keys()):
                                refs.append((sid, cond, trial, str(subject_h5_path)))
                self.h5_trial_refs = refs
                source_desc = self.h5_dir

            if max_files is not None:
                self.h5_trial_refs = self.h5_trial_refs[:max_files]

            print(f"[KineticsTCNDataset] Found {len(self.h5_trial_refs)} H5 trials in {source_desc}")
        else:
            if b3d_files is not None:
                trial_dirs = sorted([Path(p) for p in b3d_files])
                source_desc = "explicit trial list"
            else:
                trial_dirs = find_trial_dirs(data_dir)
                source_desc = data_dir

            if walking_only:
                trial_dirs = [td for td in trial_dirs if is_walking_condition(td.parent)]
            if max_files is not None:
                trial_dirs = trial_dirs[:max_files]

            self.trial_dirs = trial_dirs
            print(f"[KineticsTCNDataset] Found {len(self.trial_dirs)} trials in {source_desc}")

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

        # Build window index once during init (then only (trial_idx, start_frame) is stored).
        self.windows: List[Tuple[int, int]] = []

        candidates = self.h5_trial_refs if self.use_h5 else self.trial_dirs
        valid_trial_dirs: List[Path] = []
        valid_h5_refs: List[Tuple[str, str, str, str]] = []
        for i, ref in enumerate(candidates):
            if (i + 1) % 200 == 0 or i == 0:
                print(f"  Loading trial {i+1}/{len(candidates)}: {ref}")
            if self.use_h5:
                trial = self._load_trial_from_h5_ref(ref)
            else:
                trial = load_trial_from_processed(
                    ref,
                    root_dir=data_dir,
                    meta_map=self.meta_map,
                    apply_lowpass_filter=self.apply_lowpass_filter,
                    lowpass_cutoff_hz=self.lowpass_cutoff_hz,
                    lowpass_order=self.lowpass_order,
                    median_kernel_samples=self.median_kernel_samples,
                )
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
                total_frames_for_stats += T
                sum_pos += pos.sum(axis=0)
                sumsq_pos += np.square(pos).sum(axis=0)
                sum_vel += vel.sum(axis=0)
                sumsq_vel += np.square(vel).sum(axis=0)

            # Create windows and validate outputs for those windows.
            n = trial["positions"].shape[0]
            for start in range(0, n - self.window_size + 1, self.stride):
                end = start + self.window_size
                mom_w = trial["moments"][start:end]
                if self.moment_indices is not None:
                    mom_w = mom_w[:, self.moment_indices]
                if np.all(np.isfinite(mom_w)):
                    # Store (trial_index, start_frame) to avoid pickling large arrays.
                    self.windows.append((t_idx, start))

            if self.preload_trials:
                self.trials.append(trial)

            # If we don't preload, we drop the arrays immediately (lazy loading in workers).
            if not self.preload_trials:
                del trial

        if self.use_h5:
            self.h5_trial_refs = valid_h5_refs
        else:
            self.trial_dirs = valid_trial_dirs

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
        print(
            f"  Loaded {n_valid_trials} valid trials, "
            f"created {n_windows} windows (window={window_size}, stride={stride}, "
            f"preload_trials={self.preload_trials}, use_h5={self.use_h5})"
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

        moments = np.full((n, len(MOMENT_NAMES)), np.nan, dtype=np.float64)
        for j, name in enumerate(MOMENT_NAMES):
            col = f"{name}_moment"
            if col in id_cols:
                moments[:, j] = id_data[:, id_cols.index(col)]
            else:
                # If a moment channel is missing entirely, leave NaNs.
                # Windowing will drop windows where selected output channels are non-finite.
                pass

        if self.median_kernel_samples >= 3 or self.apply_lowpass_filter:
            pos, moments = _denoise_pos_and_moments(
                pos,
                moments,
                time,
                median_kernel_samples=self.median_kernel_samples,
                apply_lowpass_filter=self.apply_lowpass_filter,
                lowpass_cutoff_hz=self.lowpass_cutoff_hz,
                lowpass_order=self.lowpass_order,
            )

        vel = _compute_velocity(pos, time)

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
                median_kernel_samples=self.median_kernel_samples,
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

    def _build_index(self):
        self.windows: List[Tuple[int, int]] = []
        for t_idx, trial in enumerate(self.trials):
            n = trial["positions"].shape[0]
            for start in range(0, n - self.window_size + 1, self.stride):
                end = start + self.window_size
                mom_w = trial["moments"][start:end]
                if self.moment_indices is not None:
                    mom_w = mom_w[:, self.moment_indices]
                if np.all(np.isfinite(mom_w)):
                    self.windows.append((t_idx, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        t_idx, start = self.windows[idx]
        end = start + self.window_size
        trial = self._get_trial(t_idx)

        pos = trial["positions"][start:end].copy()
        vel = trial["velocities"][start:end].copy()
        mom = trial["moments"][start:end].copy()  # N*m/kg for MOMENT_NAMES

        if self.normalize:
            pos = (pos - self.pos_mean) / self.pos_std
            vel = (vel - self.vel_mean) / self.vel_std


        if self.input_indices is not None:
            pos = pos[:, self.input_indices]
            vel = vel[:, self.input_indices]

        if self.moment_indices is not None:
            mom = mom[:, self.moment_indices]

        x = np.concatenate([pos, vel], axis=1).T.astype(np.float32)
        y = mom.T.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def n_input_channels(self):
        n = len(self.input_indices) if self.input_indices is not None else len(IK_DOF_NAMES)
        return n * 2

    @property
    def n_output_channels(self):
        if self.moment_indices is not None:
            return len(self.moment_indices)
        return len(MOMENT_NAMES)

    @property
    def input_dof_names(self) -> List[str]:
        if self.input_indices is not None:
            return [IK_DOF_NAMES[i] for i in self.input_indices]
        return list(IK_DOF_NAMES)

    @property
    def output_dof_names(self) -> List[str]:
        if self.moment_indices is not None:
            return [MOMENT_NAMES[i] for i in self.moment_indices]
        return list(MOMENT_NAMES)

