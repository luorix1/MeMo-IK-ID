import numpy as np
from collections import deque
from .base import BaseController, Sensors, CtrlResult

class TEST(BaseController):
    name = "TEST"

    def __init__(self, config: dict):
        pass

    def step(self, s: Sensors) -> CtrlResult:
        tau_r = 0.5
        tau_l = -0.5
        return CtrlResult(tau_r, tau_l, tau_r, tau_l, extra={})