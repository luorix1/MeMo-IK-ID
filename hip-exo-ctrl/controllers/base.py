from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class Sensors:
    """One control tick: dual hip motors + pelvis / thigh IMUs (accel xyz + gyro xyz)."""

    imu_P: np.ndarray  # (6,) pelvis
    imu_L: np.ndarray  # (6,) left thigh
    imu_R: np.ndarray  # (6,) right thigh
    pos_L: float  # hip motor angle, degrees (device frame)
    pos_R: float
    vel_L: float  # hip motor velocity, deg/s
    vel_R: float


@dataclass(slots=True)
class CtrlResult:
    model_out_R: float
    model_out_L: float
    applied_R: float
    applied_L: float
    extra: dict


class BaseController:
    name: str = "base"

    def start(self) -> None:
        pass

    def step(self, s: Sensors) -> CtrlResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


class RollingWindow:
    """Shape (C, T): oldest at [:, 0], newest at [:, -1]."""

    def __init__(self, shape: Tuple[int, ...], dtype=np.float32):
        shape = tuple(int(x) for x in shape)
        if len(shape) < 2:
            raise ValueError("RollingWindow shape must be at least 2D: (..., T)")
        self.buf = np.zeros(shape, dtype=dtype)
        self._last_slice_shape = self.buf.shape[:-1]

    def push_last(self, arr_last: np.ndarray) -> np.ndarray:
        arr_last = np.asarray(arr_last, dtype=self.buf.dtype)
        arr_last = arr_last.reshape(self._last_slice_shape)
        self.buf[..., :-1] = self.buf[..., 1:]
        self.buf[..., -1] = arr_last
        return self.buf
