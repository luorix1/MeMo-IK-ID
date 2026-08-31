#!/usr/bin/env python3
"""
bench_trt_throughput.py — measure TensorRT ankle TCN inference latency / FPS.

Does **not** require torch.cuda (avoids Jetson pip-torch / driver mismatches).
GPU buffers via libcudart (ctypes), or optionally cuda-python / pycuda.

Expected engine I/O (cascade_uni):
  input  : (1, C, T) float32
  output : (1, O)   float32

Examples (on Jetson, from ankle-exo-ctrl root):

    python utils/bench_trt_throughput.py --trt best_model.trt
    python utils/benchmark.py --trt best_model.trt --cfg cfg/final.yaml \\
        --warmup 50 --iters 500 --host-copy
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Tuple

import numpy as np
import yaml

# Allow `python utils/bench_*.py` from repo root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.cudart_runtime import CudaRuntime  # noqa: E402


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark ankle-exo-ctrl TensorRT engine throughput (no torch.cuda required).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trt", required=True, help="Path to TensorRT engine (.trt).")
    p.add_argument(
        "--cfg",
        default=None,
        help="Optional YAML (uses frame_length / input_size / output_size).",
    )
    p.add_argument("--batch", type=int, default=1, help="Batch size (cascade_uni uses 1).")
    p.add_argument("--channels", type=int, default=None, help="Input channels C (default 2).")
    p.add_argument("--window", type=int, default=None, help="Sequence length T (default 100).")
    p.add_argument("--outputs", type=int, default=None, help="Output channels O (default 1).")
    p.add_argument("--warmup", type=int, default=50, help="Warmup iterations (not timed).")
    p.add_argument("--iters", type=int, default=500, help="Timed iterations.")
    p.add_argument(
        "--host-copy",
        action="store_true",
        help="H2D input + D2H output each iter (closer to controller path).",
    )
    return p.parse_args()


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _find_io_names(engine, trt) -> Tuple[str, str]:
    in_name = out_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            in_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            out_name = name
    if in_name is None or out_name is None:
        raise RuntimeError("Could not find TRT input/output tensor names.")
    return in_name, out_name


def main() -> None:
    args = parse_args()

    channels = args.channels
    window = args.window
    outputs = args.outputs
    if args.cfg and os.path.isfile(args.cfg):
        cfg = load_yaml(args.cfg)
        if channels is None:
            channels = int(cfg.get("input_size", 2))
        if window is None:
            window = int(cfg.get("frame_length", cfg.get("window_size", 100)))
        if outputs is None:
            outputs = int(cfg.get("output_size", 1))

    channels = 2 if channels is None else int(channels)
    window = 100 if window is None else int(window)
    outputs = 1 if outputs is None else int(outputs)
    batch = int(args.batch)

    in_shape = (batch, channels, window)
    out_shape = (batch, outputs)

    import tensorrt as trt

    mem = CudaRuntime()
    print(f"[INFO] CUDA backend: {mem.backend} (torch not required)")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    trt_path = os.path.abspath(args.trt)
    print(f"[INFO] Loading engine: {trt_path}")
    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TRT engine.")

    context = engine.create_execution_context()
    in_name, out_name = _find_io_names(engine, trt)

    eng_in = tuple(engine.get_tensor_shape(in_name))
    if -1 in eng_in:
        ok = context.set_input_shape(in_name, in_shape)
        if ok is False:
            raise RuntimeError(f"TRT refused input shape {in_shape}")
    else:
        if eng_in != in_shape:
            print(f"[WARN] Engine input {eng_in} != requested {in_shape}; using engine shape.")
            in_shape = eng_in
            try:
                context.set_input_shape(in_name, in_shape)
            except Exception:
                pass

    eng_out = tuple(context.get_tensor_shape(out_name))
    if -1 not in eng_out:
        out_shape = eng_out

    print(f"[INFO] input  shape: {in_shape}")
    print(f"[INFO] output shape: {out_shape}")
    print(f"[INFO] warmup={args.warmup}  iters={args.iters}  host_copy={args.host_copy}")

    h_in = np.empty(in_shape, dtype=np.float32)
    h_out = np.empty(out_shape, dtype=np.float32)
    d_in = mem.malloc(h_in.nbytes)
    d_out = mem.malloc(h_out.nbytes)

    context.set_tensor_address(in_name, d_in)
    context.set_tensor_address(out_name, d_out)

    rng = np.random.default_rng(0)

    def rand_in() -> None:
        h_in[...] = rng.standard_normal(size=in_shape).astype(np.float32, copy=False)

    rand_in()
    mem.h2d(h_in, d_in)
    mem.sync()

    def infer_once() -> None:
        if args.host_copy:
            rand_in()
            mem.h2d(h_in, d_in)
        context.execute_async_v3(stream_handle=mem.stream_handle())
        if args.host_copy:
            mem.d2h(d_out, h_out)
        mem.sync()

    for _ in range(max(0, args.warmup)):
        infer_once()

    lat_ms: list[float] = []
    t0 = time.perf_counter()
    for _ in range(max(1, args.iters)):
        t_i = time.perf_counter()
        infer_once()
        lat_ms.append((time.perf_counter() - t_i) * 1e3)
    total_s = time.perf_counter() - t0

    mem.d2h(d_out, h_out)
    mem.sync()

    lat_sorted = sorted(lat_ms)
    mean_ms = float(statistics.mean(lat_ms))
    std_ms = statistics.pstdev(lat_ms) if len(lat_ms) > 1 else 0.0
    fps = len(lat_ms) / total_s if total_s > 0 else float("nan")

    print("\n=== TRT throughput ===")
    print(f"samples     : {len(lat_ms)}")
    print(f"total time  : {total_s:.3f} s")
    print(f"throughput  : {fps:.1f} Hz")
    print(f"latency ms  : mean={mean_ms:.3f}  std={std_ms:.3f}")
    print(
        f"             p50={_percentile(lat_sorted, 0.50):.3f}  "
        f"p90={_percentile(lat_sorted, 0.90):.3f}  "
        f"p95={_percentile(lat_sorted, 0.95):.3f}  "
        f"p99={_percentile(lat_sorted, 0.99):.3f}  "
        f"max={lat_sorted[-1]:.3f}"
    )
    print(f"sample out[0] : {float(h_out.reshape(-1)[0]): .6f}")

    budget_ms = 10.0
    print(
        f"\n[HINT] 100 Hz control budget = {budget_ms:.1f} ms/tick; "
        f"mean latency is {mean_ms / budget_ms * 100:.1f}% of that budget."
    )

    try:
        mem.free(d_in)
        mem.free(d_out)
    except Exception:
        pass


if __name__ == "__main__":
    main()
