# controllers/base.py
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Sensors:
    """Per-tick sensor packet for ankle control.

    IMU slots are reserved for future wireless Muse (shank/foot). For the
    encoder-only bring-up they are filled with zeros.

    Convention (match hip cascade):
      pos_* / vel_* in radians (and rad/s).
      Right-side encoder is sign-corrected in the runner before Sensors is built
      so both sides share a flexion-positive convention inside controllers.
    """

    imu_L1: np.ndarray  # left shank (future Muse), shape (6,)
    imu_L2: np.ndarray  # left foot  (future Muse), shape (6,)
    imu_R1: np.ndarray  # right shank
    imu_R2: np.ndarray  # right foot
    pos_L: float
    pos_R: float
    vel_L: float  # time derivative of encoder (or motor-reported; see cfg)
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
