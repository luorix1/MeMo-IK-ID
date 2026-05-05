"""Helpers matching State2Torque `HelperFunc` (fast roll, SOS Butterworth, UDP teleplot batch)."""

from __future__ import annotations

import socket
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
    """Same API as HelperFunc.lowpass_filter (`realtimeButterworth`)."""

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


class TeleplotBatch:
    def __init__(self, ip: str = "127.0.0.1", port: int = 47269):
        self.addr = (ip, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data_dict: dict) -> bool:
        now = time.time() * 1000
        try:
            for name, value in data_dict.items():
                msg = f"{name}:{now}:{value}|g"
                self._sock.sendto(msg.encode(), self.addr)
            return True
        except OSError as e:
            print(f"[TeleplotBatch] send error: {e}")
            return False

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass
