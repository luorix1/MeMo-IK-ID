# controllers/base.py
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class Sensors:
    """Raw sensor readings for one control cycle.

    IMU arrays are [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z] in SI units.
    Motor positions are in degrees, velocities in deg/s.
    """
    imu_P: np.ndarray   # pelvis IMU (6,)
    imu_L: np.ndarray   # left thigh IMU (6,)
    imu_R: np.ndarray   # right thigh IMU (6,)
    pos_L: float        # left motor position (deg), positive = flexion
    pos_R: float        # right motor position (deg), positive = flexion
    vel_L: float        # left motor velocity (deg/s)
    vel_R: float        # right motor velocity (deg/s)


@dataclass(slots=True)
class CtrlResult:
    model_out_R: float   # raw model output, right (Nm/kg)
    model_out_L: float   # raw model output, left (Nm/kg)
    applied_R: float     # final torque command, right (Nm)
    applied_L: float     # final torque command, left (Nm)
    extra: dict          # arbitrary per-controller diagnostics


class BaseController:
    name: str = "base"

    def start(self): pass

    def step(self, s: Sensors) -> CtrlResult:
        raise NotImplementedError

    def close(self): pass


class RollingWindow:
    """Fixed-size FIFO buffer with the newest sample at index [..., -1]."""

    def __init__(self, shape, dtype=np.float32):
        shape = tuple(int(x) for x in shape)
        if len(shape) < 2:
            raise ValueError("RollingWindow shape must be at least 2D: (..., T)")
        self.buf = np.zeros(shape, dtype=dtype)
        self._last_slice_shape = self.buf.shape[:-1]

    def push_last(self, arr_last):
        arr_last = np.asarray(arr_last, dtype=self.buf.dtype).reshape(self._last_slice_shape)
        self.buf[..., :-1] = self.buf[..., 1:]
        self.buf[..., -1] = arr_last
        return self.buf
