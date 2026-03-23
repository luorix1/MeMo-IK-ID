"""
Benchmark dataset loading speed:
  1) `KineticsTCNDataset` (text `.mot/.sto` from `Processed/Camargo`)
  2) `KineticsTCNH5Dataset` (arrays from `Processed/Camargo_h5`)

The H5-backed dataset mirrors the processed dataset logic:
  - IK positions (deg -> rad)
  - velocities via finite differences (np.gradient)
  - ID moments (N*m/kg) assembled to match `MOMENT_NAMES`
  - windowing + skipping windows with any non-finite output
  - streaming mean/std computation during init (unless `stats` provided)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Allow running this file directly via:
#   python os_kinetics/benchmark_dataset_loading_camargo_h5.py
# without needing PYTHONPATH configuration.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from os_kinetics.dataset import (
    IK_DOF_NAMES,
    INPUT_MODE_INDICES,
    LOWER_LIMB_INPUT_INDICES,
    LOWER_LIMB_MOMENT_INDICES,
    KineticsTCNDataset,
    MOMENT_NAMES,
    OUTPUT_MODE_INDICES,
    _load_subject_metadata_map,
)


def _is_walking_condition_name(name: str) -> bool:
    n = name.lower()
    include = ("levelground" in n) or ("ramp" in n) or ("stair" in n) or ("treadmill" in n)
    exclude = ("static" in n)
    return include and not exclude


def _read_h5_opensim_table(dset: h5py.Dataset) -> Tuple[List[str], np.ndarray]:
    """
    H5 conversion stores:
      - values: df.values (float table)
      - attr 'columns': JSON list of column names
    """
    if "columns" not in dset.attrs:
        raise KeyError(f"Missing H5 attribute 'columns' for dataset {dset.name}")
    columns = json.loads(dset.attrs["columns"])
    data = dset[()]
    if data.ndim == 1:
        data = data[None, :]
    return columns, data


@dataclass(frozen=True)
class _TrialRef:
    subject_id: str
    subject_h5_path: str
    condition_name: str
    trial_name: str


class KineticsTCNH5Dataset(Dataset):
    """
    Windowed dataset for TCN where input/output are assembled from Camargo_h5.
    Mirrors `KineticsTCNDataset`, but reads from the HDF5 structure instead of
    parsing OpenSim tables from disk each time.
    """

    _UNSET = object()

    def __init__(
        self,
        h5_dir: str,
        meta_root_dir: str,
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
        preload_trials: bool = False,
    ):
        if h5_dir is None:
            raise ValueError("h5_dir must point to processed Camargo_h5 folder")
        if meta_root_dir is None:
            raise ValueError("meta_root_dir must point to processed Camargo folder containing dataset_metadata.json")

        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.preload_trials = preload_trials

        if moment_indices is self._UNSET:
            self.moment_indices = OUTPUT_MODE_INDICES.get(output_mode, LOWER_LIMB_MOMENT_INDICES)
        else:
            self.moment_indices = moment_indices  # can be None (= all outputs)

        if input_indices is self._UNSET:
            self.input_indices = INPUT_MODE_INDICES.get(input_mode, LOWER_LIMB_INPUT_INDICES)
        else:
            self.input_indices = input_indices  # can be None (= all inputs)

        self.h5_dir = h5_dir
        self.meta_map = _load_subject_metadata_map(meta_root_dir)
        self._trial_cache: Dict[int, Dict] = {}

        self._trial_refs: List[_TrialRef] = []
        self.trials: List[Dict] = []  # only used if preload_trials=True
        self.windows: List[Tuple[int, int]] = []

        # 1) Enumerate trial refs by scanning H5 group structure.
        subject_paths = sorted(
            [
                p
                for p in Path(h5_dir).iterdir()
                if p.is_file() and p.suffix == ".h5" and p.stem.upper().startswith("S")
            ]
        )
        if not subject_paths:
            raise ValueError(f"No .h5 subject files found under: {h5_dir}")

        trial_refs: List[_TrialRef] = []
        for subject_h5_path in subject_paths:
            subject_id = subject_h5_path.stem.upper()
            with h5py.File(subject_h5_path, "r") as h5f:
                for condition_name in sorted(h5f.keys()):
                    if walking_only and not _is_walking_condition_name(condition_name):
                        continue
                    cond_group = h5f[condition_name]
                    for trial_name in sorted(cond_group.keys()):
                        trial_refs.append(
                            _TrialRef(
                                subject_id=subject_id,
                                subject_h5_path=str(subject_h5_path),
                                condition_name=condition_name,
                                trial_name=trial_name,
                            )
                        )

        if max_files is not None:
            trial_refs = trial_refs[:max_files]

        print(f"[KineticsTCNH5Dataset] Scanning complete. Candidate trials: {len(trial_refs)}")

        # 2) Stream stats and build windows, mirroring processed dataset behavior.
        if stats is not None:
            self.pos_mean = stats["pos_mean"]
            self.pos_std = stats["pos_std"]
            self.vel_mean = stats["vel_mean"]
            self.vel_std = stats["vel_std"]
            compute_stats = False
        else:
            compute_stats = True
            total_frames_for_stats = 0.0
            sum_pos = np.zeros(len(IK_DOF_NAMES), dtype=np.float64)
            sumsq_pos = np.zeros(len(IK_DOF_NAMES), dtype=np.float64)
            sum_vel = np.zeros(len(IK_DOF_NAMES), dtype=np.float64)
            sumsq_vel = np.zeros(len(IK_DOF_NAMES), dtype=np.float64)

        for i, tr in enumerate(trial_refs):
            if (i + 1) % 200 == 0 or i == 0:
                print(f"  Loading trial {i+1}/{len(trial_refs)}: {tr.subject_id} {tr.condition_name}/{tr.trial_name}")

            trial = self._load_trial_from_h5(tr)
            if trial is None:
                continue

            t_idx = len(self._trial_refs)
            self._trial_refs.append(tr)

            if compute_stats:
                pos = trial["positions"].astype(np.float64)
                vel = trial["velocities"].astype(np.float64)
                T = pos.shape[0]
                total_frames_for_stats += float(T)
                sum_pos += pos.sum(axis=0)
                sumsq_pos += np.square(pos).sum(axis=0)
                sum_vel += vel.sum(axis=0)
                sumsq_vel += np.square(vel).sum(axis=0)

            n = trial["positions"].shape[0]
            for start in range(0, n - self.window_size + 1, self.stride):
                end = start + self.window_size
                mom_w = trial["moments"][start:end]
                if self.moment_indices is not None:
                    mom_w = mom_w[:, self.moment_indices]
                if np.all(np.isfinite(mom_w)):
                    self.windows.append((t_idx, start))

            if self.preload_trials:
                self.trials.append(trial)
            else:
                # Drop arrays immediately; lazy reload in __getitem__.
                del trial

        if len(self._trial_refs) == 0:
            raise ValueError("No valid trials found in H5 dataset scan (after validation).")

        if compute_stats:
            n_total = float(total_frames_for_stats if total_frames_for_stats is not None else 1.0)
            self.pos_mean = sum_pos / n_total
            self.vel_mean = sum_vel / n_total
            pos_var = sumsq_pos / n_total - np.square(self.pos_mean)
            vel_var = sumsq_vel / n_total - np.square(self.vel_mean)
            self.pos_std = np.sqrt(np.maximum(pos_var, 0.0)) + 1e-8
            self.vel_std = np.sqrt(np.maximum(vel_var, 0.0)) + 1e-8

        print(
            f"  Loaded {len(self._trial_refs)} valid trials, created {len(self.windows)} windows "
            f"(window={window_size}, stride={stride}, preload_trials={self.preload_trials})"
        )

    def _load_trial_from_h5(self, tr: _TrialRef) -> Optional[Dict]:
        with h5py.File(tr.subject_h5_path, "r") as h5f:
            if tr.condition_name not in h5f:
                return None
            cond_group = h5f[tr.condition_name]
            if tr.trial_name not in cond_group:
                return None
            trial_group = cond_group[tr.trial_name]

            if "ik" not in trial_group or "id" not in trial_group:
                return None
            if "imu" in trial_group:
                # IMU not used by current kinetics model.
                pass

            ik_group = trial_group["ik"]
            id_group = trial_group["id"]
            if len(ik_group.keys()) == 0 or len(id_group.keys()) == 0:
                return None

            # Mirror processed loader: "use the first mot/sto file found".
            ik_dataset_key = sorted(list(ik_group.keys()))[0]
            id_dataset_key = sorted(list(id_group.keys()))[0]

            ik_dset = ik_group[ik_dataset_key]
            id_dset = id_group[id_dataset_key]

            ik_cols, ik_data = _read_h5_opensim_table(ik_dset)
            id_cols, id_data = _read_h5_opensim_table(id_dset)

            if "time" not in ik_cols or "time" not in id_cols:
                return None

            subj_id = tr.subject_id
            mass = float(self.meta_map.get(subj_id, {}).get("weight_kg", np.nan))
            if not np.isfinite(mass) or mass <= 0:
                return None

            time = ik_data[:, ik_cols.index("time")]
            pos_deg = []
            for name in IK_DOF_NAMES:
                if name not in ik_cols:
                    return None
                pos_deg.append(ik_data[:, ik_cols.index(name)])
            pos_deg = np.stack(pos_deg, axis=1)  # (T, 23)
            pos = np.deg2rad(pos_deg)
            vel = np.gradient(pos, time, axis=0)

            id_time = id_data[:, id_cols.index("time")]
            n = min(len(time), len(id_time))
            time = time[:n]
            pos = pos[:n]
            vel = vel[:n]
            id_data = id_data[:n]

            moments = np.full((n, len(MOMENT_NAMES)), np.nan, dtype=np.float64)
            for j, name in enumerate(MOMENT_NAMES):
                col = f"{name}_moment"
                if col in id_cols:
                    moments[:, j] = id_data[:, id_cols.index(col)]
                else:
                    # If a moment channel is missing entirely, skip this trial.
                    return None

            trial_name = f"{tr.subject_id}/{tr.condition_name}/{tr.trial_name}"
            return {
                "positions": pos.astype(np.float32),
                "velocities": vel.astype(np.float32),
                "moments": moments.astype(np.float32),
                "moments_unit": "N*m/kg",
                "mass": mass,
                "subject_id": subj_id,
                "trial_name": trial_name,
                "time": time.astype(np.float32),
            }

    def _get_trial(self, t_idx: int) -> Dict:
        if self.preload_trials:
            return self.trials[t_idx]
        if t_idx in self._trial_cache:
            return self._trial_cache[t_idx]

        tr = self._trial_refs[t_idx]
        trial = self._load_trial_from_h5(tr)
        if trial is None:
            raise RuntimeError(f"Failed to reload trial from H5: {tr}")

        self._trial_cache[t_idx] = trial
        return trial

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        t_idx, start = self.windows[idx]
        end = start + self.window_size
        trial = self._get_trial(t_idx)

        pos = trial["positions"][start:end].copy()
        vel = trial["velocities"][start:end].copy()
        mom = trial["moments"][start:end].copy()

        if self.normalize:
            pos = (pos - self.pos_mean) / self.pos_std
            vel = (vel - self.vel_mean) / self.vel_std

        if self.input_indices is not None:
            pos = pos[:, self.input_indices]
            vel = vel[:, self.input_indices]

        if self.moment_indices is not None:
            mom = mom[:, self.moment_indices]

        # Channels-first for Conv1d: x=(C_in, W)
        x = np.concatenate([pos, vel], axis=1).T.astype(np.float32)
        y = mom.T.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def n_input_channels(self) -> int:
        n = len(self.input_indices) if self.input_indices is not None else len(IK_DOF_NAMES)
        return n * 2

    @property
    def n_output_channels(self) -> int:
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


def _time_getitems(ds: Dataset, n: int) -> float:
    n = min(n, len(ds))
    start = time.perf_counter()
    for i in range(n):
        _ = ds[i]
    end = time.perf_counter()
    return end - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Camargo kinetics dataset loading")
    parser.add_argument(
        "--processed_root",
        default="/media/metamobility3/Samsung_T51/Processed/Camargo",
        help="Processed Camargo root containing dataset_metadata.json and per-trial folders.",
    )
    parser.add_argument(
        "--h5_root",
        default="/media/metamobility3/Samsung_T51/Processed/Camargo_h5",
        help="Output folder containing per-subject .h5 files (created by convert_dataset_to_h5.py).",
    )
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--walking_only", action="store_true", help="Use only walking trials/conditions.")
    parser.add_argument("--normalize", action="store_true", help="Apply pos/vel normalization.")
    parser.add_argument("--max_trials", type=int, default=30, help="Max number of trial groups to scan/load (for speed).")
    parser.add_argument("--num_getitems", type=int, default=100, help="Number of __getitem__ calls to time.")
    parser.add_argument("--warmup_getitems", type=int, default=5, help="Warmup __getitem__ calls before timing.")
    args = parser.parse_args()

    # Default behavior in dataset.py is walking_only=True and normalize=True; mimic that unless flags are passed.
    walking_only = args.walking_only if args.walking_only else True
    normalize = args.normalize if args.normalize else True

    print("=== Dataset Init Benchmark ===")
    print(f"processed_root: {args.processed_root}")
    print(f"h5_root:         {args.h5_root}")
    print(f"window_size: {args.window_size} stride: {args.stride} max_trials: {args.max_trials}")

    t0 = time.perf_counter()
    ds_processed = KineticsTCNDataset(
        data_dir=args.processed_root,
        window_size=args.window_size,
        stride=args.stride,
        walking_only=walking_only,
        normalize=normalize,
        max_files=args.max_trials,
        preload_trials=False,
    )
    t1 = time.perf_counter()
    print(f"[processed] init: {t1 - t0:.3f}s len={len(ds_processed)}")

    t2 = time.perf_counter()
    ds_h5 = KineticsTCNH5Dataset(
        h5_dir=args.h5_root,
        meta_root_dir=args.processed_root,
        window_size=args.window_size,
        stride=args.stride,
        walking_only=walking_only,
        normalize=normalize,
        max_files=args.max_trials,
        preload_trials=False,
    )
    t3 = time.perf_counter()
    print(f"[h5] init: {t3 - t2:.3f}s len={len(ds_h5)}")

    print("\n=== __getitem__ Benchmark (sequential) ===")
    # Warmup (helps JIT/caches; keeps timing stable).
    _ = _time_getitems(ds_processed, args.warmup_getitems)
    _ = _time_getitems(ds_h5, args.warmup_getitems)

    dt_processed = _time_getitems(ds_processed, args.num_getitems)
    dt_h5 = _time_getitems(ds_h5, args.num_getitems)

    print(f"[processed] {min(args.num_getitems, len(ds_processed))} getitems: {dt_processed:.3f}s")
    print(f"[h5]         {min(args.num_getitems, len(ds_h5))} getitems: {dt_h5:.3f}s")
    if dt_h5 > 0:
        print(f"Speedup (processed/h5): {dt_processed / dt_h5:.2f}x")


if __name__ == "__main__":
    main()

