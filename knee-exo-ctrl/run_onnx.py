#!/usr/bin/env python3
"""
Standalone ONNX inference script for the knee IK→ID model.

Runs three modes:

  bench   — latency benchmark: feeds random windows repeatedly and reports
             mean / std / p99 inference time.  Use this to confirm the model
             meets the control-loop deadline before deploying.

  step    — simulates the rolling-window control loop: feeds one frame at a
             time (like the real controller), prints the moment output each
             tick, and reports latency.  Useful for end-to-end timing on the
             Jetson.

  verify  — loads the original .pt checkpoint, runs PyTorch and ONNX on the
             same random input, and checks numeric agreement.

Examples
--------
  # Latency benchmark (200 warm-up, 1000 timed, CPU provider)
  python knee-exo-ctrl/run_onnx.py bench \\
      --onnx knee-exo-ctrl/best_model_0423_knee.onnx \\
      --checkpoint runs/0423_ik_id_knee_huber_noise/best_model.pt

  # Step-wise simulation (100 frames at 100 Hz)
  python knee-exo-ctrl/run_onnx.py step \\
      --onnx knee-exo-ctrl/best_model_0423_knee.onnx \\
      --checkpoint runs/0423_ik_id_knee_huber_noise/best_model.pt \\
      --n-frames 100

  # Numeric verification vs PyTorch
  python knee-exo-ctrl/run_onnx.py verify \\
      --onnx knee-exo-ctrl/best_model_0423_knee.onnx \\
      --checkpoint runs/0423_ik_id_knee_huber_noise/best_model.pt

  # CUDA/TensorRT execution providers (on Jetson)
  python knee-exo-ctrl/run_onnx.py bench \\
      --onnx knee-exo-ctrl/best_model_0423_knee.trt \\
      --checkpoint runs/0423_ik_id_knee_huber_noise/best_model.pt \\
      --providers TensorrtExecutionProvider CUDAExecutionProvider
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── path setup ───────────────────────────────────────────────────────────────
_KNEE_ROOT = Path(__file__).resolve().parent
_OS_KIN_ROOT = _KNEE_ROOT.parent
for _p in (_KNEE_ROOT, _OS_KIN_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# ─────────────────────────────────────────────────────────────────────────────

from run_bundle import (
    ik_indices_unilateral_paired,
    load_checkpoint_metadata,
    load_train_config,
    normalization_for_dof,
    resolve_run_dir,
)


# ---------------------------------------------------------------------------
# ONNX session helpers
# ---------------------------------------------------------------------------

def _load_session(onnx_path: Path, providers: List[str]):
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required.  Install with:\n"
            "  pip install onnxruntime        # CPU\n"
            "  pip install onnxruntime-gpu    # CUDA (non-Jetson)\n"
            "  # Jetson: pre-installed with JetPack"
        )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    return sess


def _infer(sess, x: np.ndarray) -> np.ndarray:
    """x: (1, 2, T) float32 → (1, 1, T) float32."""
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess.run([out_name], {in_name: x})[0]


def _last_moment(out: np.ndarray) -> float:
    """Extract the scalar moment (N·m/kg) from the last time step."""
    if out.ndim == 3:
        return float(out[0, 0, -1])
    if out.ndim == 2:
        return float(out[0, -1])
    raise RuntimeError(f"Unexpected output shape {out.shape}")


# ---------------------------------------------------------------------------
# Checkpoint / normalization helpers
# ---------------------------------------------------------------------------

def _load_norm(ckpt_path: Path):
    """Return (pr_m, pr_s, vr_m, vr_s, pl_m, pl_s, vl_m, vl_s, window_size)."""
    ckpt = load_checkpoint_metadata(ckpt_path)
    inp_idx = list(ckpt["input_indices"])
    ik_r, ik_l = ik_indices_unilateral_paired(inp_idx)
    norm = ckpt["normalization"]
    pr_m, pr_s, vr_m, vr_s = normalization_for_dof(norm, ik_r)
    pl_m, pl_s, vl_m, vl_s = normalization_for_dof(norm, ik_l)
    ws = int(ckpt["window_size"])
    return pr_m, pr_s, vr_m, vr_s, pl_m, pl_s, vl_m, vl_s, ws


def _normalize(q: float, qd: float, pm: float, ps: float, vm: float, vs: float) -> Tuple[float, float]:
    return (q - pm) / ps, (qd - vm) / vs


# ---------------------------------------------------------------------------
# Mode: bench
# ---------------------------------------------------------------------------

def cmd_bench(args: argparse.Namespace) -> None:
    onnx_path = Path(args.onnx).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()

    print(f"ONNX   : {onnx_path}")
    print(f"Ckpt   : {ckpt_path}")
    print(f"Providers: {args.providers}")

    *_, ws = _load_norm(ckpt_path)
    seq_len = args.seq_len if args.seq_len else ws
    sess = _load_session(onnx_path, args.providers)

    print(f"\nInput shape: (1, 2, {seq_len})")
    print(f"Warm-up: {args.warmup}  Timed: {args.n_reps}")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 2, seq_len)).astype(np.float32)

    # Warm-up
    for _ in range(args.warmup):
        _infer(sess, x)

    # Timed runs
    times_ms = []
    for _ in range(args.n_reps):
        t0 = time.perf_counter()
        out = _infer(sess, x)
        times_ms.append((time.perf_counter() - t0) * 1e3)

    times_ms = np.array(times_ms)
    budget_ms = 1e3 / args.fs

    print(f"\n── Latency (ms) over {args.n_reps} runs ─────────────────────────────")
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


# ---------------------------------------------------------------------------
# Mode: step
# ---------------------------------------------------------------------------

def cmd_step(args: argparse.Namespace) -> None:
    onnx_path = Path(args.onnx).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()

    print(f"ONNX   : {onnx_path}")
    print(f"Ckpt   : {ckpt_path}")
    print(f"Providers: {args.providers}")

    pr_m, pr_s, vr_m, vr_s, pl_m, pl_s, vl_m, vl_s, ws = _load_norm(ckpt_path)
    seq_len = args.seq_len if args.seq_len else ws
    sess = _load_session(onnx_path, args.providers)

    # Rolling windows for each leg
    buf_r = np.zeros((1, 2, seq_len), dtype=np.float32)
    buf_l = np.zeros((1, 2, seq_len), dtype=np.float32)

    rng = np.random.default_rng(42)
    dt = 1.0 / args.fs

    # Synthetic walking signal: ~1 Hz sinusoid ± small noise (mimics knee angle/vel)
    t_arr = np.arange(args.n_frames) * dt
    q_walk = 0.3 * np.sin(2 * np.pi * 0.9 * t_arr)          # ~0.3 rad peak angle
    qd_walk = 0.3 * 2 * np.pi * 0.9 * np.cos(2 * np.pi * 0.9 * t_arr)

    times_ms = []
    moments_r, moments_l = [], []

    print(f"\nSimulating {args.n_frames} frames @ {args.fs} Hz (seq_len={seq_len})…\n")
    print(f"{'Frame':>6}  {'q_r (rad)':>10}  {'qd_r (rad/s)':>13}  {'m_r (N·m/kg)':>13}  {'lat (ms)':>9}")
    print("─" * 62)

    for i in range(args.n_frames):
        q_r  = float(q_walk[i]  + rng.normal(0, 0.005))
        qd_r = float(qd_walk[i] + rng.normal(0, 0.02))
        q_l  = float(-q_walk[i] + rng.normal(0, 0.005))   # approximate contralateral
        qd_l = float(-qd_walk[i]+ rng.normal(0, 0.02))

        # Normalize
        qn_r, qdn_r = _normalize(q_r, qd_r, pr_m, pr_s, vr_m, vr_s)
        qn_l, qdn_l = _normalize(q_l, qd_l, pl_m, pl_s, vl_m, vl_s)

        # Roll windows
        buf_r[0, :, :-1] = buf_r[0, :, 1:]
        buf_r[0, 0, -1]  = qn_r
        buf_r[0, 1, -1]  = qdn_r
        buf_l[0, :, :-1] = buf_l[0, :, 1:]
        buf_l[0, 0, -1]  = qn_l
        buf_l[0, 1, -1]  = qdn_l

        t0 = time.perf_counter()
        out_r = _infer(sess, buf_r)
        out_l = _infer(sess, buf_l)
        lat = (time.perf_counter() - t0) * 1e3

        m_r = _last_moment(out_r)
        m_l = _last_moment(out_l)
        moments_r.append(m_r)
        moments_l.append(m_l)
        times_ms.append(lat)

        if i < 10 or (i + 1) % 10 == 0:
            print(f"{i+1:>6}  {q_r:>10.4f}  {qd_r:>13.4f}  {m_r:>13.4f}  {lat:>9.3f}")

    times_ms_arr = np.array(times_ms)
    print("─" * 62)
    print(f"\nPer-step latency (both legs, ms):")
    print(f"  mean {times_ms_arr.mean():.3f}  |  p99 {np.percentile(times_ms_arr,99):.3f}"
          f"  |  max {times_ms_arr.max():.3f}  (budget {1e3/args.fs:.1f} ms @ {args.fs} Hz)")
    print(f"\nRight leg moment  — mean {np.mean(moments_r):.4f}  std {np.std(moments_r):.4f} N·m/kg")
    print(f"Left  leg moment  — mean {np.mean(moments_l):.4f}  std {np.std(moments_l):.4f} N·m/kg")


# ---------------------------------------------------------------------------
# Mode: verify
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> None:
    import torch

    onnx_path = Path(args.onnx).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()

    print(f"ONNX   : {onnx_path}")
    print(f"Ckpt   : {ckpt_path}")
    print(f"Providers: {args.providers}")

    sys.path.insert(0, str(_OS_KIN_ROOT))
    from model import TCN, TransformerMoment

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = raw["model_config"]
    ws = int(raw["window_size"])
    seq_len = args.seq_len if args.seq_len else ws
    model_type = cfg.get("model_type", "tcn")

    if model_type == "diffusion":
        raise RuntimeError("GaussianDiffusion1D is not supported for ONNX verification.")

    if model_type == "transformer":
        model = TransformerMoment(
            n_input_channels=int(cfg["n_input_channels"]),
            n_output_channels=int(cfg["n_output_channels"]),
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            n_layers=int(cfg["n_layers"]),
            d_ff=int(cfg["d_ff"]),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    else:
        model = TCN(
            n_input_channels=int(cfg["n_input_channels"]),
            n_output_channels=int(cfg["n_output_channels"]),
            hidden_channels=int(cfg["hidden_channels"]),
            n_blocks=int(cfg["n_blocks"]),
            kernel_size=int(cfg["kernel_size"]),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    model.load_state_dict(raw["model_state_dict"])
    model.eval()

    sess = _load_session(onnx_path, args.providers)

    n_in = int(cfg["n_input_channels"])
    rng = np.random.default_rng(0)

    print(f"\nComparing PyTorch vs ONNX on {args.n_verify} random windows "
          f"(shape 1×{n_in}×{seq_len})…\n")

    max_abs_all, rel_all = [], []
    for i in range(args.n_verify):
        x_np = rng.standard_normal((1, n_in, seq_len)).astype(np.float32)
        x_t  = torch.from_numpy(x_np)
        with torch.no_grad():
            y_pt = model(x_t).numpy()
        y_ort = _infer(sess, x_np)

        max_abs = float(np.max(np.abs(y_pt - y_ort)))
        rel = float(np.max(np.abs(y_pt - y_ort) / (np.abs(y_pt) + 1e-8)))
        max_abs_all.append(max_abs)
        rel_all.append(rel)
        status = "OK" if max_abs < args.atol else "FAIL"
        print(f"  [{i+1:3d}] max |Δ| = {max_abs:.3e}   max rel = {rel:.3e}   {status}")

    overall = max(max_abs_all)
    print(f"\n  Overall max |Δ| = {overall:.3e}   (atol = {args.atol:.3e})")
    if overall < args.atol:
        print("  ✓ ONNX matches PyTorch within tolerance.")
    else:
        print("  ✗ ONNX diverges — check export opset or model type.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run / benchmark the knee ONNX model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── shared args ──────────────────────────────────────────────────────────
    def _common(sp):
        sp.add_argument("--onnx", "-m", type=str,
                        default=str(_KNEE_ROOT / "best_model_0423_knee.onnx"),
                        help="Path to .onnx (or .trt) file.")
        sp.add_argument("--checkpoint", "-c", type=str,
                        default=str(_OS_KIN_ROOT / "runs/0423_ik_id_knee_huber_noise/best_model.pt"),
                        help="Path to best_model.pt (for normalization stats + window_size).")
        sp.add_argument("--seq-len", type=int, default=None,
                        help="Override sequence length (default: training window_size from checkpoint).")
        sp.add_argument("--providers", nargs="+",
                        default=["CPUExecutionProvider"],
                        help="OnnxRuntime execution providers (default: CPUExecutionProvider). "
                             "On Jetson try: TensorrtExecutionProvider CUDAExecutionProvider.")
        sp.add_argument("--fs", type=float, default=100.0,
                        help="Control loop rate in Hz (default: 100).")

    # bench
    sb = sub.add_parser("bench", help="Latency benchmark.")
    _common(sb)
    sb.add_argument("--warmup", type=int, default=200, help="Warm-up iterations (default: 200).")
    sb.add_argument("--n-reps", type=int, default=1000, help="Timed iterations (default: 1000).")

    # step
    ss = sub.add_parser("step", help="Frame-by-frame rolling-window simulation.")
    _common(ss)
    ss.add_argument("--n-frames", type=int, default=200,
                    help="Number of control ticks to simulate (default: 200).")

    # verify
    sv = sub.add_parser("verify", help="Numerical check: ONNX vs PyTorch.")
    _common(sv)
    sv.add_argument("--n-verify", type=int, default=20,
                    help="Number of random windows to compare (default: 20).")
    sv.add_argument("--atol", type=float, default=1e-4,
                    help="Absolute tolerance (default: 1e-4).")

    return p.parse_args()


def main() -> None:
    args = _parse()
    if args.mode == "bench":
        cmd_bench(args)
    elif args.mode == "step":
        cmd_step(args)
    elif args.mode == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
