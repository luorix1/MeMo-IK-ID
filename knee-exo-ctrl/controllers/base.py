from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class Sensors:
    """Snapshot of one control tick (dual leg)."""

    imu_L1: np.ndarray  # (6,) accel xyz + gyro xyz, thigh left
    imu_L2: np.ndarray  # (6,) shank left
    imu_R1: np.ndarray  # (6,) thigh right
    imu_R2: np.ndarray  # (6,) shank right
    pos_L: float  # motor encoder knee angle, degrees (device frame)
    pos_R: float
    vel_L: float  # motor encoder knee velocity, deg/s
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
