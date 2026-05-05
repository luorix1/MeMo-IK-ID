"""Small utilities (aligned with hip-exo-ctrl-V1 `utils/utils.py` layout)."""

from __future__ import annotations

import time
import typing

import numpy as np
from scipy import signal as sp_signal
from scipy.signal import butter


def fast_roll(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr[:-1] = arr[1:]
        arr[-1] = 0
    elif arr.ndim == 2:
        arr[:, :-1] = arr[:, 1:]
        arr[:, -1] = 0
    elif arr.ndim == 3:
        arr[:, :, :-1] = arr[:, :, 1:]
        arr[:, :, -1] = 0
    else:
        raise ValueError(f"fast_roll: unsupported ndim {arr.ndim}")
    return arr


class LowpassFilter:
    """Streaming SOS Butterworth (same calling convention as legacy HelperFunc.lowpass_filter)."""

    def __init__(self, order: int = 2, cutoff: float = 10.0, fs: float = 100.0):
        self.cutoff = float(cutoff)
        self.fs = float(fs)
        self.order = int(order)
        self.sos = butter(order, cutoff, btype="low", fs=fs, output="sos")

    def realtimeButterworth(self, data: typing.Union[float, np.ndarray], zi=None, reset=False):
        x = np.asarray(data, dtype=float)
        squeeze_out = False
        if x.ndim == 0:
            x = x.reshape(1)
            squeeze_out = True
        elif x.ndim != 1:
            raise ValueError("data must be scalar or 1D array")

        if zi is None:
            zi = sp_signal.sosfilt_zi(self.sos) * x[0]

        y, zf = sp_signal.sosfilt(self.sos, x, zi=zi)
        if squeeze_out:
            return float(y[0]), zf
        return y, zf


class RateKeeper:
    """Fixed-rate scheduler (hip-exo-ctrl-V1 `utils.utils.RateKeeper`)."""

    def __init__(self, hz: float, catchup_cycles: int = 3, spin_ns: int = 50_000):
        self.period_ns = int(1e9 / float(hz))
        self.next_ns = None
        self.tick = 0
        self.catchup_cycles = int(catchup_cycles)
        self.spin_ns = int(spin_ns)

    def start(self):
        self.next_ns = time.perf_counter_ns()

    def wait(self):
        self.next_ns += self.period_ns
        while True:
            now_ns = time.perf_counter_ns()
            dt_ns = self.next_ns - now_ns
            if dt_ns <= 0:
                late_cycles = (-dt_ns) // self.period_ns
                if late_cycles > self.catchup_cycles:
                    self.next_ns += late_cycles * self.period_ns
                    self.tick += late_cycles
                break
            if dt_ns > self.spin_ns:
                time.sleep((dt_ns - self.spin_ns) / 1e9)
            else:
                while (self.next_ns - time.perf_counter_ns()) > 0:
                    pass
                break
        sched_s = self.tick * (self.period_ns / 1e9)
        overrun_s = max(0.0, -dt_ns / 1e9)
        self.tick += 1
        return overrun_s, sched_s, self.tick - 1
