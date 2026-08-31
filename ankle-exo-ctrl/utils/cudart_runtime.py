"""Minimal CUDA runtime via ctypes (Jetson-safe; no torch.cuda)."""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Optional

import numpy as np


def _load_cudart() -> ctypes.CDLL:
    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/lib/aarch64-linux-gnu/libcudart.so",
    ]
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.insert(0, found)
    last_err: Optional[OSError] = None
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            last_err = e
    raise RuntimeError(
        f"Could not load libcudart (CUDA runtime). Last error: {last_err}"
    )


class CudaRuntime:
    """CUDA malloc / memcpy / stream without PyTorch."""

    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2

    def __init__(self) -> None:
        self.lib = _load_cudart()
        self.backend = "libcudart"

        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.lib.cudaMemset.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaDeviceSynchronize.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int

        stream = ctypes.c_void_p()
        err = self.lib.cudaStreamCreate(ctypes.byref(stream))
        if err != 0:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")
        self.stream = stream

    def _check(self, err: int, what: str) -> None:
        if err != 0:
            raise RuntimeError(f"{what} failed: cudaError={err}")

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return int(ptr.value)

    def free(self, ptr: int) -> None:
        self._check(self.lib.cudaFree(ctypes.c_void_p(ptr)), "cudaFree")

    def memset(self, dptr: int, value: int, nbytes: int) -> None:
        self._check(
            self.lib.cudaMemset(ctypes.c_void_p(dptr), int(value), int(nbytes)),
            "cudaMemset",
        )

    def h2d(self, host: np.ndarray, dptr: int) -> None:
        host = np.ascontiguousarray(host)
        self._check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(dptr),
                host.ctypes.data_as(ctypes.c_void_p),
                host.nbytes,
                self.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "cudaMemcpyAsync H2D",
        )

    def d2h(self, dptr: int, host: np.ndarray) -> None:
        host = np.ascontiguousarray(host)
        self._check(
            self.lib.cudaMemcpyAsync(
                host.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(dptr),
                host.nbytes,
                self.cudaMemcpyDeviceToHost,
                self.stream,
            ),
            "cudaMemcpyAsync D2H",
        )

    def sync(self) -> None:
        self._check(self.lib.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

    def stream_handle(self) -> int:
        return int(self.stream.value) if self.stream.value else 0
