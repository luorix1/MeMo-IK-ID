"""
IMU → sagittal lower-limb dataset (H5), **separate from** ``dataset.py``.

**Paired unilateral chains** (24 IMU channels → 3 sagittal labels each), with ``ik_id.train``-style unilateral IK flip:

- **Right sample:** IMU = **pelvis + right thigh + right shank + right foot** → targets = **right** hip flexion,
  knee, ankle.
- **Left sample:** IMU = **pelvis + left thigh + left shank + left foot** → targets = **left** hip flexion,
  knee, ankle.

Each time window yields **up to two** examples (right and/or left) when ``sides="both"`` (training default).
Pelvis channels are the **same** signals in both chains; segment column counts must match so each chain has
``IMU_UNILATERAL_N_CHANNELS`` (24 with 6 signals × 4 segments).

IK positions, ID moments, and **IMU channels** use the same optional denoising as ``KineticsTCNDataset``:
**zero-phase** Butterworth via ``dataset._denoise_pos_and_moments`` and ``dataset._lowpass_trial_channels``
(``sosfiltfilt``), after alignment / optional resampling to the IK time base.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from dataset import (
    IK_DOF_NAMES,
    MOMENT_NAMES,
    SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES,
    SAGITTAL_INPUT_INDICES,
    _apply_unilateral_left_hip_flip_inplace,
    _coerce_weight_kg,
    _compute_velocity,
    _denoise_pos_and_moments,
    _ik_time_and_pos_deg,
    _lowpass_trial_channels,
    _load_subject_metadata_map,
    _read_h5_opensim_table,
    include_condition_for_dataset,
    resample_trial_to_uniform_hz,
)

TrialRef = Tuple[str, str, str, str]  # subject_id, condition, trial_name, h5_path

_IMU_RIGHT_CHAIN: Tuple[str, ...] = (
    "pelvis",
    "right_thigh",
    "right_shank",
    "right_foot",
)
_IMU_LEFT_CHAIN: Tuple[str, ...] = (
    "pelvis",
    "left_thigh",
    "left_shank",
    "left_foot",
)

IMU_UNILATERAL_N_CHANNELS: int = 24

_SAGITTAL_RIGHT_IK_IDX: Tuple[int, ...] = tuple(SAGITTAL_INPUT_INDICES[:3])
_SAGITTAL_LEFT_IK_IDX: Tuple[int, ...] = tuple(SAGITTAL_INPUT_INDICES[3:])
_SAGITTAL_RIGHT_MOM_IDX: Tuple[int, ...] = tuple(SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES[:3])
_SAGITTAL_LEFT_MOM_IDX: Tuple[int, ...] = tuple(SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES[3:])


def molinaro_subject_ids() -> List[str]:
    """Subjects S035–S056 (inclusive), zero-padded."""
    return [f"S{i:03d}" for i in range(35, 57)]


def imu_paired_chain_orders() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(pelvis+R thigh/shank/foot, pelvis+L thigh/shank/foot)."""
    return _IMU_RIGHT_CHAIN, _IMU_LEFT_CHAIN


def imu_unilateral_24_segment_order() -> Tuple[str, ...]:
    """Backward-compat: unique segments across both chains (logging only)."""
    return _IMU_RIGHT_CHAIN + ("left_thigh", "left_shank", "left_foot")


def imu_lower_limb_segment_order() -> Tuple[str, ...]:
    return imu_unilateral_24_segment_order()


def imu_segment_order_for_laterality(laterality: Optional[str] = None) -> Tuple[str, ...]:
    _ = laterality
    return imu_unilateral_24_segment_order()


def _match_imu_segment_key(imu_group: Any, logical: str) -> Optional[str]:
    """Resolve ``logical`` (e.g. ``right_thigh``) to an existing child name of ``imu_group``."""
    logical_l = logical.lower().strip()
    keys = [str(k) for k in imu_group.keys()]
    if logical in imu_group:
        return logical
    low_map = {str(k).lower(): str(k) for k in keys}
    if logical_l in low_map:
        return low_map[logical_l]
    alt = logical_l.replace(" ", "_")
    if alt in low_map:
        return low_map[alt]
    return None


def sorted_imu_schema_for_leg(
    imu_group: Any,
    segment_order: Sequence[str],
) -> Optional[List[Tuple[str, str]]]:
    """
    Build feature list in ``segment_order``: for each segment, all non-``time`` columns, sorted by name.
    Tuple is ``(resolved_h5_key, column_name)`` for use with ``_imu_matrix_for_schema``.
    Returns ``None`` if any required segment is missing.
    """
    schema: List[Tuple[str, str]] = []
    for logical in segment_order:
        key = _match_imu_segment_key(imu_group, logical)
        if key is None:
            return None
        dset = imu_group[key]
        cols, _ = _read_h5_opensim_table(dset)
        for c in sorted(cols):
            if c != "time":
                schema.append((key, c))
    return schema if schema else None


def _imu_matrix_for_schema(
    imu_group: Any,
    time_ik: np.ndarray,
    schema: Sequence[Tuple[str, str]],
) -> Optional[np.ndarray]:
    """Shape (T, len(schema)) float64; None if any channel missing or unusable."""
    t = time_ik.astype(np.float64)
    n = len(t)
    out = np.full((n, len(schema)), np.nan, dtype=np.float64)
    for j, (seg, col) in enumerate(schema):
        if seg not in imu_group:
            return None
        dset = imu_group[seg]
        cols, data = _read_h5_opensim_table(dset)
        if "time" not in cols or col not in cols:
            return None
        t_imu = data[:, cols.index("time")].astype(np.float64)
        y = data[:, cols.index(col)].astype(np.float64)
        if len(t_imu) < 2:
            return None
        if np.any(np.diff(t_imu) <= 0):
            uniq, idx = np.unique(t_imu, return_index=True)
            t_imu = uniq
            y = y[idx]
        out[:, j] = np.interp(t, t_imu, y, left=np.nan, right=np.nan)
    return out


def _load_trial_imu_sagittal_paired(
    ref: TrialRef,
    meta_map: Dict[str, Dict],
    imu_schema_right: Sequence[Tuple[str, str]],
    imu_schema_left: Sequence[Tuple[str, str]],
    target: str,
    *,
    apply_lowpass_filter: bool,
    lowpass_cutoff_hz: float,
    lowpass_order: int,
    target_sample_rate_hz: Optional[float] = None,
    rollout_decimate_step: int = 1,
    trim_nonfinite_imu_suffix: bool = False,
) -> Optional[Dict[str, Any]]:
    subj_id, cond_name, trial_name, subject_h5_path = ref
    h5_path = Path(subject_h5_path)
    if not h5_path.exists():
        return None

    with h5py.File(h5_path, "r") as h5f:
        if cond_name not in h5f or trial_name not in h5f[cond_name]:
            return None
        trial_group = h5f[cond_name][trial_name]
        if "ik" not in trial_group or "id" not in trial_group or "imu" not in trial_group:
            return None
        imu_group = trial_group["imu"]
        if len(imu_group.keys()) == 0:
            return None
        ik_group = trial_group["ik"]
        id_group = trial_group["id"]
        if len(ik_group.keys()) == 0 or len(id_group.keys()) == 0:
            return None
        ik_key = sorted(list(ik_group.keys()))[0]
        id_key = sorted(list(id_group.keys()))[0]
        ik_cols, ik_data = _read_h5_opensim_table(ik_group[ik_key])
        id_cols, id_data = _read_h5_opensim_table(id_group[id_key])
        t_ik = ik_data[:, ik_cols.index("time")]
        imu_r = _imu_matrix_for_schema(imu_group, t_ik, imu_schema_right)
        imu_l = _imu_matrix_for_schema(imu_group, t_ik, imu_schema_left)
        if imu_r is None or imu_l is None:
            return None
        if trim_nonfinite_imu_suffix:
            ok_row = np.isfinite(imu_r).all(axis=1) & np.isfinite(imu_l).all(axis=1)
            if not np.any(ok_row):
                return None
            first_bad = np.flatnonzero(~ok_row)
            n_keep = int(first_bad[0]) if first_bad.size else int(imu_r.shape[0])
            if n_keep < 2:
                return None
            ik_data = ik_data[:n_keep]
            id_data = id_data[:n_keep]
            imu_r = imu_r[:n_keep]
            imu_l = imu_l[:n_keep]
        elif not np.all(np.isfinite(imu_r)) or not np.all(np.isfinite(imu_l)):
            return None

    if "time" not in id_cols:
        return None
    ik_tp = _ik_time_and_pos_deg(ik_cols, ik_data)
    if ik_tp is None:
        return None
    time, pos_deg = ik_tp
    pos = np.deg2rad(pos_deg)

    id_time = id_data[:, id_cols.index("time")]
    n = min(len(time), len(id_time))
    time = time[:n]
    pos = pos[:n]
    id_data = id_data[:n]
    imu_r = imu_r[:n]
    imu_l = imu_l[:n]

    moments = np.full((n, len(MOMENT_NAMES)), np.nan, dtype=np.float64)
    for j, name in enumerate(MOMENT_NAMES):
        col = f"{name}_moment"
        if col in id_cols:
            moments[:, j] = id_data[:, id_cols.index(col)]

    if rollout_decimate_step > 1 and target_sample_rate_hz is not None:
        raise ValueError(
            "Use either rollout_decimate_step>1 (subsample) or target_sample_rate_hz (interpolate), not both."
        )

    if rollout_decimate_step > 1:
        idx = np.arange(0, n, int(rollout_decimate_step), dtype=np.int64)
        time = time[idx]
        pos = pos[idx]
        moments = moments[idx]
        imu_r = imu_r[idx]
        imu_l = imu_l[idx]
    elif target_sample_rate_hz is not None and target_sample_rate_hz > 0:
        t_src = time.astype(np.float64)
        time_rs, pos_rs, moments_rs = resample_trial_to_uniform_hz(
            t_src, pos, moments, float(target_sample_rate_hz)
        )
        t_new = time_rs.astype(np.float64)
        imu_rn = np.empty((t_new.shape[0], imu_r.shape[1]), dtype=np.float32)
        imu_ln = np.empty_like(imu_rn)
        for j in range(imu_r.shape[1]):
            imu_rn[:, j] = np.interp(
                t_new, t_src, imu_r[:, j].astype(np.float64)
            ).astype(np.float32)
            imu_ln[:, j] = np.interp(
                t_new, t_src, imu_l[:, j].astype(np.float64)
            ).astype(np.float32)
        time = time_rs
        pos = pos_rs.astype(np.float64)
        moments = moments_rs.astype(np.float64)
        imu_r = imu_rn
        imu_l = imu_ln

    if apply_lowpass_filter:
        pos, moments = _denoise_pos_and_moments(
            pos,
            moments,
            time,
            apply_lowpass_filter=apply_lowpass_filter,
            lowpass_cutoff_hz=lowpass_cutoff_hz,
            lowpass_order=lowpass_order,
        )
        imu_r = _lowpass_trial_channels(
            imu_r,
            time,
            apply_lowpass_filter=apply_lowpass_filter,
            lowpass_cutoff_hz=lowpass_cutoff_hz,
            lowpass_order=lowpass_order,
        )
        imu_l = _lowpass_trial_channels(
            imu_l,
            time,
            apply_lowpass_filter=apply_lowpass_filter,
            lowpass_cutoff_hz=lowpass_cutoff_hz,
            lowpass_order=lowpass_order,
        )

    vel = _compute_velocity(pos, time)
    _apply_unilateral_left_hip_flip_inplace(pos, vel, moments)

    pos_s = pos[:, SAGITTAL_INPUT_INDICES].astype(np.float32)
    mom_s = moments[:, SAGITTAL_HIP_KNEE_ANKLE_MOMENT_INDICES].astype(np.float32)

    if target == "angle":
        y_r = pos_s[:, :3].copy()
        y_l = pos_s[:, 3:6].copy()
    elif target == "moment":
        y_r = mom_s[:, :3].copy()
        y_l = mom_s[:, 3:6].copy()
    else:
        raise ValueError(f"Unknown target {target!r}")

    trial_name_full = f"{subj_id}/{cond_name}/{trial_name}"
    return {
        "imu_right": imu_r.astype(np.float32),
        "imu_left": imu_l.astype(np.float32),
        "y_right": y_r,
        "y_left": y_l,
        "pos_sagittal_rl": pos_s,
        "positions": pos.astype(np.float32),
        "subject_id": subj_id,
        "trial_name": trial_name_full,
        "time": time.astype(np.float32),
    }


class ImuSagittalH5Dataset(Dataset):
    """
    Sliding-window: **24** IMU channels (one pelvis+limb chain) → **3** sagittal angles or moments.

    ``sides``: ``"both"`` doubles windows (right + left); ``"right"`` / ``"left"`` keeps one chain for eval.
    """

    def __init__(
        self,
        h5_dir: str,
        meta_root_dir: str,
        subject_ids: Sequence[str],
        imu_schema_right: Sequence[Tuple[str, str]],
        imu_schema_left: Sequence[Tuple[str, str]],
        target: str,
        window_size: int = 200,
        stride: int = 1,
        *,
        sides: Literal["both", "right", "left"] = "both",
        walking_only: bool = True,
        levelground_only: bool = False,
        normalize: bool = True,
        stats: Optional[Dict[str, np.ndarray]] = None,
        return_full_sagittal_angles: bool = False,
        apply_lowpass_filter: bool = True,
        lowpass_cutoff_hz: float = 4.0,
        lowpass_order: int = 4,
        preload_trials: bool = False,
        max_trials: Optional[int] = None,
        target_sample_rate_hz: Optional[float] = None,
        rollout_decimate_step: int = 1,
    ) -> None:
        if not HAS_H5PY:
            raise RuntimeError("imu_sagittal_leg_dataset requires h5py.")
        if target not in ("angle", "moment"):
            raise ValueError("target must be 'angle' or 'moment'")
        self.imu_schema_right = list(imu_schema_right)
        self.imu_schema_left = list(imu_schema_left)
        if len(self.imu_schema_right) != IMU_UNILATERAL_N_CHANNELS or len(self.imu_schema_left) != IMU_UNILATERAL_N_CHANNELS:
            raise ValueError(
                f"Each IMU chain must have {IMU_UNILATERAL_N_CHANNELS} channels; got "
                f"right={len(self.imu_schema_right)} left={len(self.imu_schema_left)}."
            )
        if sides not in ("both", "right", "left"):
            raise ValueError("sides must be 'both', 'right', or 'left'")
        self._sides = sides

        self.h5_dir = Path(h5_dir)
        self.meta_map = _load_subject_metadata_map(meta_root_dir)
        self.target = target
        self.window_size = int(window_size)
        self.stride = max(1, int(stride))
        self.normalize = normalize
        self.preload_trials = preload_trials
        self.walking_only = walking_only
        self.levelground_only = levelground_only
        self.apply_lowpass_filter = apply_lowpass_filter
        self.lowpass_cutoff_hz = lowpass_cutoff_hz
        self.lowpass_order = lowpass_order
        self.return_full_sagittal_angles = bool(return_full_sagittal_angles)
        self.target_sample_rate_hz: Optional[float] = (
            None if target_sample_rate_hz is None else float(target_sample_rate_hz)
        )
        self.rollout_decimate_step = max(1, int(rollout_decimate_step))
        if self.rollout_decimate_step > 1 and self.target_sample_rate_hz is not None:
            raise ValueError(
                "Use either rollout_decimate_step>1 (subsample) or target_sample_rate_hz (interpolate), not both."
            )

        self._trial_refs: List[TrialRef] = []
        self.trials: List[Dict[str, Any]] = []
        self.windows: List[Tuple[int, int, str]] = []
        self._trial_cache: Dict[int, Dict[str, Any]] = {}

        sid_set = {s.upper() for s in subject_ids}
        for h5_path in sorted(self.h5_dir.glob("S*.h5")):
            sid = h5_path.stem.upper()
            if sid not in sid_set:
                continue
            with h5py.File(h5_path, "r") as h5f:
                for cond in sorted(h5f.keys()):
                    for trial_name in sorted(h5f[cond].keys()):
                        if not include_condition_for_dataset(
                            cond,
                            walking_only=self.walking_only,
                            levelground_only=self.levelground_only,
                            subject_id=sid,
                            trial_name=trial_name,
                        ):
                            continue
                        self._trial_refs.append((sid, cond, trial_name, str(h5_path)))

        if max_trials is not None:
            self._trial_refs = self._trial_refs[: max_trials]

        compute_stats = stats is None
        if not compute_stats:
            self.imu_mean = np.asarray(stats["imu_mean"], dtype=np.float64)
            self.imu_std = np.asarray(stats["imu_std"], dtype=np.float64)
        else:
            sum_imu = np.zeros(IMU_UNILATERAL_N_CHANNELS, dtype=np.float64)
            sumsq_imu = np.zeros(IMU_UNILATERAL_N_CHANNELS, dtype=np.float64)
            total_frames = 0.0

        valid_refs: List[TrialRef] = []
        for t_idx, ref in enumerate(self._trial_refs):
            if (t_idx + 1) % 500 == 0:
                print(f"  [ImuSagittalH5Dataset] scanning trial {t_idx+1}/{len(self._trial_refs)} …")

            trial = _load_trial_imu_sagittal_paired(
                ref,
                self.meta_map,
                self.imu_schema_right,
                self.imu_schema_left,
                self.target,
                apply_lowpass_filter=self.apply_lowpass_filter,
                lowpass_cutoff_hz=self.lowpass_cutoff_hz,
                lowpass_order=self.lowpass_order,
                target_sample_rate_hz=self.target_sample_rate_hz,
                rollout_decimate_step=self.rollout_decimate_step,
            )
            if trial is None:
                continue

            trial_index = len(valid_refs)
            valid_refs.append(ref)

            if compute_stats:
                ir = trial["imu_right"].astype(np.float64)
                il = trial["imu_left"].astype(np.float64)
                T = ir.shape[0]
                sum_imu += ir.sum(axis=0) + il.sum(axis=0)
                sumsq_imu += np.square(ir).sum(axis=0) + np.square(il).sum(axis=0)
                total_frames += float(2 * T)

            if self.preload_trials:
                self.trials.append(trial)

            n = trial["imu_right"].shape[0]
            for start in range(0, n - self.window_size + 1, self.stride):
                end = start + self.window_size
                pos6_w = trial["pos_sagittal_rl"][start:end]
                if not np.all(np.isfinite(pos6_w)):
                    continue
                if self._sides in ("both", "right"):
                    yr = trial["y_right"][start:end]
                    xr = trial["imu_right"][start:end]
                    if np.all(np.isfinite(yr)) and np.all(np.isfinite(xr)):
                        self.windows.append((trial_index, start, "r"))
                if self._sides in ("both", "left"):
                    yl = trial["y_left"][start:end]
                    xl = trial["imu_left"][start:end]
                    if np.all(np.isfinite(yl)) and np.all(np.isfinite(xl)):
                        self.windows.append((trial_index, start, "l"))

        self._trial_refs = valid_refs
        if not self.preload_trials:
            self.trials = []

        if len(self._trial_refs) == 0:
            raise ValueError(
                "No valid IMU+IK+ID trials for paired pelvis+limb IMU layout. "
                f"Need segments R={_IMU_RIGHT_CHAIN} L={_IMU_LEFT_CHAIN} with {IMU_UNILATERAL_N_CHANNELS} ch each."
            )
        if len(self.windows) == 0:
            raise ValueError(
                "No valid windows: trial length < window_size or NaNs in IMU/targets."
            )

        if compute_stats:
            n_total = max(total_frames, 1.0)
            self.imu_mean = sum_imu / n_total
            imu_var = sumsq_imu / n_total - np.square(self.imu_mean)
            self.imu_std = np.sqrt(np.maximum(imu_var, 0.0)) + 1e-8

        print(
            f"  [ImuSagittalH5Dataset] convention=unilateral  sides={self._sides!r}  "
            f"imu_R={_IMU_RIGHT_CHAIN}  imu_L={_IMU_LEFT_CHAIN}  target={self.target}  "
            f"trials={len(self._trial_refs)}  windows={len(self.windows)}  imu_dim={IMU_UNILATERAL_N_CHANNELS}"
        )

    def get_stats(self) -> Dict[str, np.ndarray]:
        return {"imu_mean": self.imu_mean, "imu_std": self.imu_std}

    @property
    def n_input_channels(self) -> int:
        return IMU_UNILATERAL_N_CHANNELS

    @property
    def n_output_channels(self) -> int:
        return 3

    @property
    def output_names(self) -> List[str]:
        """Right-leg names (legacy alias; prefer ``output_names_for_side``)."""
        return self.output_names_right

    @property
    def output_names_right(self) -> List[str]:
        if self.target == "angle":
            return [IK_DOF_NAMES[i] for i in _SAGITTAL_RIGHT_IK_IDX]
        return [MOMENT_NAMES[i] for i in _SAGITTAL_RIGHT_MOM_IDX]

    @property
    def output_names_left(self) -> List[str]:
        if self.target == "angle":
            return [IK_DOF_NAMES[i] for i in _SAGITTAL_LEFT_IK_IDX]
        return [MOMENT_NAMES[i] for i in _SAGITTAL_LEFT_MOM_IDX]

    def output_names_for_side(self, side: str) -> List[str]:
        s = side.lower()
        if s in ("r", "right"):
            return self.output_names_right
        if s in ("l", "left"):
            return self.output_names_left
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")

    def _get_trial(self, t_idx: int) -> Dict[str, Any]:
        if self.preload_trials:
            return self.trials[t_idx]
        if t_idx in self._trial_cache:
            return self._trial_cache[t_idx]
        ref = self._trial_refs[t_idx]
        trial = _load_trial_imu_sagittal_paired(
            ref,
            self.meta_map,
            self.imu_schema_right,
            self.imu_schema_left,
            self.target,
            apply_lowpass_filter=self.apply_lowpass_filter,
            lowpass_cutoff_hz=self.lowpass_cutoff_hz,
            lowpass_order=self.lowpass_order,
            target_sample_rate_hz=self.target_sample_rate_hz,
            rollout_decimate_step=self.rollout_decimate_step,
        )
        if trial is None:
            raise RuntimeError(f"Failed to reload trial {ref}")
        self._trial_cache[t_idx] = trial
        return trial

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        t_idx, start, side = self.windows[idx]
        end = start + self.window_size
        trial = self._get_trial(t_idx)
        if side == "r":
            imu = trial["imu_right"][start:end].copy()
            y = trial["y_right"][start:end].copy()
        else:
            imu = trial["imu_left"][start:end].copy()
            y = trial["y_left"][start:end].copy()
        if self.normalize:
            imu = (imu - self.imu_mean) / self.imu_std
        x = imu.T.astype(np.float32)
        y_t = torch.from_numpy(y.T.astype(np.float32))
        if self.return_full_sagittal_angles:
            pos6 = trial["pos_sagittal_rl"][start:end].copy()
            pos23 = trial["positions"][start:end].copy()
            tw = trial["time"][start:end].copy()
            aux6 = torch.from_numpy(pos6.T.astype(np.float32))
            aux23 = torch.from_numpy(pos23.T.astype(np.float32))
            time_t = torch.from_numpy(tw.astype(np.float32))
            return torch.from_numpy(x), y_t, aux6, aux23, time_t
        return torch.from_numpy(x), y_t


def discover_imu_schemas_paired_first_trial(
    h5_dir: str,
    subject_ids: Sequence[str],
    *,
    walking_only: bool = True,
    levelground_only: bool = False,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """First trial where R and L pelvis+limb chains each yield ``IMU_UNILATERAL_N_CHANNELS`` columns."""
    if not HAS_H5PY:
        raise RuntimeError("h5py required.")
    sid_set = {s.upper() for s in subject_ids}
    root = Path(h5_dir)
    for h5_path in sorted(root.glob("S*.h5")):
        sid = h5_path.stem.upper()
        if sid not in sid_set:
            continue
        with h5py.File(h5_path, "r") as h5f:
            for cond in sorted(h5f.keys()):
                for trial_name in sorted(h5f[cond].keys()):
                    if not include_condition_for_dataset(
                        cond,
                        walking_only=walking_only,
                        levelground_only=levelground_only,
                        subject_id=sid,
                        trial_name=trial_name,
                    ):
                        continue
                    tg = h5f[cond][trial_name]
                    if "imu" not in tg or len(tg["imu"].keys()) == 0:
                        continue
                    sr = sorted_imu_schema_for_leg(tg["imu"], _IMU_RIGHT_CHAIN)
                    sl = sorted_imu_schema_for_leg(tg["imu"], _IMU_LEFT_CHAIN)
                    if (
                        sr is not None
                        and sl is not None
                        and len(sr) == len(sl) == IMU_UNILATERAL_N_CHANNELS
                    ):
                        return sr, sl
    raise ValueError(
        f"No trial with paired IMU chains R={_IMU_RIGHT_CHAIN} L={_IMU_LEFT_CHAIN} "
        f"({IMU_UNILATERAL_N_CHANNELS} ch each) under {h5_dir} for subjects {sorted(sid_set)}"
    )


def discover_imu_schema_first_trial(
    h5_dir: str,
    subject_ids: Sequence[str],
    *,
    walking_only: bool = True,
    levelground_only: bool = False,
) -> List[Tuple[str, str]]:
    """Deprecated alias: returns **right** chain schema only. Prefer ``discover_imu_schemas_paired_first_trial``."""
    sr, _ = discover_imu_schemas_paired_first_trial(
        h5_dir,
        subject_ids,
        walking_only=walking_only,
        levelground_only=levelground_only,
    )
    return sr
