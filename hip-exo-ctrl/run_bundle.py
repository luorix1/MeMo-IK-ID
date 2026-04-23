"""
Load training-run metadata for deployment (normalization, IK indices, window size).

Same contract as ``knee-exo-ctrl/run_bundle.py`` for ``ik_id/trainV2.py`` checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

OS_KINETICS_ROOT = Path(__file__).resolve().parents[1]


def resolve_run_dir(run_dir: Path | str) -> Path:
    p = Path(run_dir).expanduser()
    if p.is_absolute():
        return p.resolve()
    trial = (Path.cwd() / p).resolve()
    if trial.exists():
        return trial
    return (OS_KINETICS_ROOT / p).resolve()


def load_train_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_metadata(ckpt_path: Path) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as e:
        raise ImportError("Loading best_model.pt requires PyTorch (`pip install torch`).") from e

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    need = ("normalization", "model_config", "input_indices", "window_size")
    missing = [k for k in need if k not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint {ckpt_path} missing keys: {missing}")
    return ckpt


def ik_indices_unilateral_paired(input_indices: List[int]) -> Tuple[int, int]:
    """Return (ik_dof_index_right, ik_dof_index_left) for symmetric paired modes."""
    if len(input_indices) != 2 or len(input_indices) % 2 != 0:
        raise ValueError(f"Expected paired input_indices length 2 for paired sagittal DOF, got {input_indices}")
    h = len(input_indices) // 2
    r = input_indices[:h]
    l = input_indices[h:]
    return int(r[0]), int(l[0])


def normalization_for_dof(
    norm: Dict[str, np.ndarray], ik_idx: int
) -> Tuple[float, float, float, float]:
    pm = np.asarray(norm["pos_mean"], dtype=np.float64)
    ps = np.asarray(norm["pos_std"], dtype=np.float64)
    vm = np.asarray(norm["vel_mean"], dtype=np.float64)
    vs = np.asarray(norm["vel_std"], dtype=np.float64)
    return float(pm[ik_idx]), float(ps[ik_idx]), float(vm[ik_idx]), float(vs[ik_idx])
