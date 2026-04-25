#!/usr/bin/env python3
"""
Smoke-test a **serialized TensorRT engine** (.trt) for the knee IK→ID model.

Unlike ``run_onnx.py`` (ONNX Runtime), this loads the native TRT engine produced by
``convert_to_trt.py`` — the same path used by ``ik_id_knee_trt`` at runtime.

Modes
-----
verify  — PyTorch vs TRT on the same random input (requires tensorrt + pycuda).
bench   — Latency benchmark with ``execute_v2`` (GPU), same shape as deployment.

Examples
--------
  python knee-exo-ctrl/run_trt.py verify \\
      --trt knee-exo-ctrl/best_model_0423_knee.trt \\
      --checkpoint runs/0423_ik_id_knee_huber_noise/best_model.pt

  python knee-exo-ctrl/run_trt.py bench \\
      --trt knee-exo-ctrl/best_model_0423_knee.trt \\
      --checkpoint runs/0423_ik_id_knee_huber_noise/best_model.pt \\
      --warmup 50 --n-reps 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

_KNEE_ROOT = Path(__file__).resolve().parent
_OS_KIN_ROOT = _KNEE_ROOT.parent
for _p in (_KNEE_ROOT, _OS_KIN_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from convert_to_trt import _build_model, _load_raw


def _load_meta(ckpt_path: Path) -> Tuple[int, int, int]:
    raw = _load_raw(ckpt_path)
    if not isinstance(raw, dict) or "model_config" not in raw:
        raise ValueError(f"{ckpt_path} missing model_config")
    cfg = raw["model_config"]
    n_in = int(cfg["n_input_channels"])
    n_out = int(cfg["n_output_channels"])
    ws = int(raw.get("window_size", 100))
    return n_in, n_out, ws


class _TrtSession:
    """Engine loaded once; reuse buffers for bench."""

    def __init__(self, trt_path: Path, n_in: int, n_out: int, seq_len: int):
        import tensorrt as trt  # type: ignore[import]
        import pycuda.autoinit  # type: ignore[import]  # noqa: F401
        import pycuda.driver as cuda  # type: ignore[import]

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(trt_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"deserialize_cuda_engine failed for {trt_path}")
        self.context = engine.create_execution_context()
        self.context.set_input_shape("input", (1, n_in, seq_len))
        x_nbytes = int(np.prod((1, n_in, seq_len)) * np.dtype(np.float32).itemsize)
        y_nbytes = int(np.prod((1, n_out, seq_len)) * np.dtype(np.float32).itemsize)
        self.d_input = cuda.mem_alloc(x_nbytes)
        self.d_output = cuda.mem_alloc(y_nbytes)
        self.y_np = np.empty((1, n_out, seq_len), dtype=np.float32)

    def infer(self, x_np: np.ndarray) -> np.ndarray:
        cuda = self._cuda
        cuda.memcpy_htod(self.d_input, x_np)
        self.context.execute_v2([int(self.d_input), int(self.d_output)])
        cuda.memcpy_dtoh(self.y_np, self.d_output)
        return self.y_np


def cmd_verify(args: argparse.Namespace) -> None:
    import torch

    trt_path = Path(args.trt).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not trt_path.is_file():
        raise FileNotFoundError(f"TRT engine not found: {trt_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    n_in, n_out, ws = _load_meta(ckpt_path)
    seq_len = args.seq_len if args.seq_len else ws

    raw = _load_raw(ckpt_path)
    model = _build_model(raw)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()

    print(f"TRT      : {trt_path}")
    print(f"Ckpt     : {ckpt_path}")
    print(f"Shape    : (1, {n_in}, {seq_len}) → (1, {n_out}, {seq_len})")

    sess = _TrtSession(trt_path, n_in, n_out, seq_len)
    rng = np.random.default_rng(0)
    max_abs_all = []
    print(f"\nComparing PyTorch vs serialized TRT on {args.n_verify} random windows…\n")
    for i in range(args.n_verify):
        x_np = rng.standard_normal((1, n_in, seq_len)).astype(np.float32)
        x_t = torch.from_numpy(x_np)
        with torch.no_grad():
            y_pt = model(x_t).cpu().numpy()
        y_trt = sess.infer(x_np)
        max_abs = float(np.max(np.abs(y_pt - y_trt)))
        max_abs_all.append(max_abs)
        status = "OK" if max_abs <= args.atol else "FAIL"
        print(f"  [{i+1:3d}] max |torch − trt| = {max_abs:.3e}  {status}")

    overall = max(max_abs_all)
    print(f"\n  Overall max |Δ| = {overall:.3e}  (atol = {args.atol:.3e})")
    if overall <= args.atol:
        print("  ✓ TRT engine matches PyTorch within tolerance.")
    else:
        print("  ✗ TRT diverges from PyTorch — rebuild engine or relax --atol for FP16.")
        sys.exit(1)


def cmd_bench(args: argparse.Namespace) -> None:
    trt_path = Path(args.trt).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not trt_path.is_file():
        raise FileNotFoundError(f"TRT engine not found: {trt_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    n_in, n_out, ws = _load_meta(ckpt_path)
    seq_len = args.seq_len if args.seq_len else ws

    print(f"TRT      : {trt_path}")
    print(f"Ckpt     : {ckpt_path}")
    print(f"Shape    : (1, {n_in}, {seq_len})")
    print(f"Warm-up: {args.warmup}  Timed: {args.n_reps}")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, n_in, seq_len)).astype(np.float32)

    sess = _TrtSession(trt_path, n_in, n_out, seq_len)
    for _ in range(args.warmup):
        _ = sess.infer(x)

    times_ms = []
    for _ in range(args.n_reps):
        t0 = time.perf_counter()
        _ = sess.infer(x)
        times_ms.append((time.perf_counter() - t0) * 1e3)

    times_ms = np.array(times_ms)
    budget_ms = 1e3 / args.fs

    print(f"\n── TRT latency (ms) over {args.n_reps} runs ─────────────────────────")
    print(f"  mean  : {times_ms.mean():.3f}")
    print(f"  std   : {times_ms.std():.3f}")
    print(f"  min   : {times_ms.min():.3f}")
    print(f"  p50   : {np.percentile(times_ms, 50):.3f}")
    print(f"  p99   : {np.percentile(times_ms, 99):.3f}")
    print(f"  max   : {times_ms.max():.3f}")
    print(f"  budget: {budget_ms:.3f}  (@ {args.fs} Hz control loop)")
    headroom = budget_ms - np.percentile(times_ms, 99)
    flag = "✓ OK" if headroom > 0 else "✗ OVER BUDGET"
    print(f"  p99 headroom: {headroom:.3f} ms  {flag}")
    print("────────────────────────────────────────────────────────────────────")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify / benchmark a native TensorRT (.trt) knee engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    def _common(sp):
        sp.add_argument(
            "--trt", "-t", type=str, required=True,
            help="Path to the serialized .trt engine.",
        )
        sp.add_argument(
            "--checkpoint", "-c", type=str,
            default=str(_OS_KIN_ROOT / "runs/0423_ik_id_knee_huber_noise/best_model.pt"),
            help="Path to best_model.pt (architecture + weights for PyTorch compare).",
        )
        sp.add_argument(
            "--seq-len", type=int, default=None,
            help="Override sequence length (default: window_size from checkpoint).",
        )
        sp.add_argument("--fs", type=float, default=100.0, help="Control rate Hz (for bench budget).")

    sv = sub.add_parser("verify", help="Numeric check: serialized TRT vs PyTorch.")
    _common(sv)
    sv.add_argument("--n-verify", type=int, default=20, help="Random windows (default: 20).")
    sv.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance (default: 1e-2).")

    sb = sub.add_parser("bench", help="GPU latency benchmark on the .trt engine.")
    _common(sb)
    sb.add_argument("--warmup", type=int, default=50, help="Warm-up iterations (default: 50).")
    sb.add_argument("--n-reps", type=int, default=200, help="Timed iterations (default: 200).")

    return p.parse_args()


def main() -> None:
    args = _parse()
    if args.mode == "verify":
        cmd_verify(args)
    elif args.mode == "bench":
        cmd_bench(args)


if __name__ == "__main__":
    main()
