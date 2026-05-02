import numpy as np
from collections import deque

from .base import BaseController, CtrlResult, Sensors


class DOFC(BaseController):
    name = "DOFC"

    def __init__(
        self,
        fs: float = 100.0,
        delay: float = 0.25,
        lpf_tau: float = 0.15,
        kappa: float = 10.0,
    ):
        self.fs = float(fs)
        self.dt = 1.0 / self.fs

        self.lpf_tau = float(lpf_tau)
        if self.lpf_tau <= 0:
            self.alpha = 1.0
        else:
            self.alpha = max(0.0, min(1.0, self.dt / self.lpf_tau))

        self.delay = float(delay)
        self.delay_samples = max(1, int(round(self.delay * self.fs)))
        self.y_buffer = deque([0.0] * self.delay_samples, maxlen=self.delay_samples)

        self.kappa = float(kappa)
        self.r_pos_old = 0.0
        self.l_pos_old = 0.0
        self.prev = np.zeros(2, dtype=np.float32)

    def start(self):
        self.r_pos_old = 0.0
        self.l_pos_old = 0.0
        self.y_buffer.clear()
        for _ in range(self.delay_samples):
            self.y_buffer.append(0.0)
        self.prev[:] = 0.0

    def step(self, s: Sensors) -> CtrlResult:
        r_raw = -float(s.pos_R)
        l_raw = float(s.pos_L)

        r_filt = (1.0 - self.alpha) * self.r_pos_old + self.alpha * r_raw
        l_filt = (1.0 - self.alpha) * self.l_pos_old + self.alpha * l_raw
        self.r_pos_old, self.l_pos_old = r_filt, l_filt

        y_now = np.sin(r_filt) - np.sin(l_filt)

        y_delayed = self.y_buffer[0]
        self.y_buffer.append(y_now)

        tau = self.kappa * y_delayed

        tau_r = -tau
        tau_l = tau

        self.prev[0], self.prev[1] = tau_r, tau_l

        return CtrlResult(
            model_out_R=tau_r,
            model_out_L=tau_l,
            applied_R=tau_r,
            applied_L=tau_l,
            extra={},
        )

    def close(self):
        pass
