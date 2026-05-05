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
