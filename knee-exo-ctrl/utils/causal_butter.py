"""
Causal Butterworth low-pass for real-time knee model I/O (sample-by-sample).

Uses ``scipy.signal.lfilter`` state so each ``step()`` is causal (no zero-phase
``filtfilt``). Intended to match training-style lowpass (e.g. 4 Hz, 4th order)
on control rate ``fs``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class CausalButterworthLowpass:
    """One scalar stream; ``step`` returns filtered sample."""

    b: object
    a: object
    zi: object
    disabled: bool

    @classmethod
    def create(cls, fs_hz: float, cutoff_hz: float, order: int, *, enabled: bool) -> CausalButterworthLowpass:
        if not enabled or cutoff_hz <= 0.0 or order < 1:
            return cls(b=None, a=None, zi=None, disabled=True)
        from scipy import signal

        nyq = 0.5 * float(fs_hz)
        wn = min(float(cutoff_hz) / nyq, 0.499)
        b, a = signal.butter(int(order), wn, btype="low", analog=False)
        zi = signal.lfilter_zi(b, a)
        return cls(b=b, a=a, zi=zi, disabled=False)

    def step(self, x: float) -> float:
        if self.disabled:
            return float(x)
        from scipy import signal

        y, self.zi = signal.lfilter(self.b, self.a, [float(x)], zi=self.zi)
        return float(y[0])


def make_model_io_filter_bank(
    fs_hz: float,
    *,
    cutoff_hz: float,
    order: int,
    enabled: bool,
) -> Dict[str, CausalButterworthLowpass]:
    """Six independent streams: q and qd per leg, moment per leg."""
    f = lambda: CausalButterworthLowpass.create(fs_hz, cutoff_hz, order, enabled=enabled)
    return {
        "q_r": f(),
        "q_l": f(),
        "qd_r": f(),
        "qd_l": f(),
        "m_r": f(),
        "m_l": f(),
    }
