from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Sensors:
    imu_P: np.ndarray
    imu_L: np.ndarray
    imu_R: np.ndarray
    pos_L: float
    pos_R: float
    vel_L: float
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

    def start(self):
        pass

    def step(self, s: Sensors) -> CtrlResult:
        raise NotImplementedError

    def close(self):
        pass


class RollingWindow:
    def __init__(self, shape, dtype=np.float32):
        shape = tuple(int(x) for x in shape)
        if len(shape) < 2:
            raise ValueError("RollingWindow shape must be at least 2D: (..., T)")
        self.buf = np.zeros(shape, dtype=dtype)
        self._last_slice_shape = self.buf.shape[:-1]

    def push_last(self, arr_last):
        arr_last = np.asarray(arr_last, dtype=self.buf.dtype)
        arr_last = arr_last.reshape(self._last_slice_shape)

        self.buf[..., :-1] = self.buf[..., 1:]
        self.buf[..., -1] = arr_last
        return self.buf
