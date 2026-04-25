#!/usr/bin/env python3
"""
Convert a trained ``.pt`` checkpoint to a TensorRT engine (``.trt``) for
low-latency Jetson inference.

Pipeline
--------
  .pt  ──► ONNX (intermediate)  ──► .trt  (serialised TRT engine)

The ONNX step reuses or creates ``<checkpoint_stem>.onnx`` in the same
directory unless ``--onnx`` is specified.  Pass ``--keep-onnx`` to keep the
intermediate file after the engine is built.

TensorRT note
-------------
TensorRT is only installed on the target Jetson (via JetPack).  Run this
script **on the Jetson** (or in an environment that mirrors the Jetson's CUDA
/ TRT version).  Building on an x86 host will produce a non-portable engine.

Supported model types
---------------------
* **TCN**              — fully supported (causal convolutions + BN export cleanly)
* **TransformerMoment** — supported (static input, bidirectional attention)
* **GaussianDiffusion1D** — NOT supported (DDIM inference loop is dynamic Python
  control flow; it cannot be captured as a static ONNX/TRT graph).  Train a
  TCN or TransformerMoment for deployment.

Precision
---------
* ``--precision fp32``  — safe default; always available
* ``--precision fp16``  — recommended for Jetson Orin/Xavier (~2× speedup,
  minimal accuracy loss for gait moments)

Jetson / TRT build OOM
----------------------
If you see ``Cuda Runtime (out of memory)`` or ``NvMapMemAllocInternalTagged … error 12``
even for tiny allocations, the GPU has **no free VRAM** (desktop compositor, another
process, or a stale CUDA context).  Try: close other GPU apps; reboot; then rebuild
with a **smaller** workspace, e.g. ``--workspace-gb 0.25`` or ``--workspace-gb 0.125``.
The default workspace is kept modest for edge devices; increase only on a clean
desktop GPU if the builder reports tactic timeouts.

Examples
--------
  # Basic: TCN knee model, FP16
  python knee-exo-ctrl/convert_to_trt.py \\
      --checkpoint runs/0423_ik_id_knee_huber_noise/best_model.pt \\
      --seq-len 100 --precision fp16 --verify

  # Reuse an existing ONNX (skip re-export)
  python knee-exo-ctrl/convert_to_trt.py \\
      --onnx runs/0423_ik_id_knee_huber_noise/best_model.onnx \\
      --checkpoint runs/0423_ik_id_knee_huber_noise/best_model.pt \\
      --precision fp16 --verify

  # Print the equivalent trtexec command and exit (no TRT required)
  python knee-exo-ctrl/convert_to_trt.py \\
      --checkpoint runs/0423_ik_id_knee_huber_noise/best_model.pt \\
      --precision fp16 --print-trtexec
"""

from __future__ import annotations

import argparse
import gc
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ── path setup ───────────────────────────────────────────────────────────────
_KNEE_ROOT = Path(__file__).resolve().parent           # os_kinetics/knee-exo-ctrl/
_OS_KIN_ROOT = _KNEE_ROOT.parent                       # os_kinetics/
for _p in (_KNEE_ROOT, _OS_KIN_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch

from model import GaussianDiffusion1D, TCN, TransformerMoment


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_raw(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_model(ckpt: Dict[str, Any]) -> torch.nn.Module:
    cfg = ckpt.get("model_config", {})
    model_type = cfg.get("model_type", "tcn")

    if model_type == "diffusion":
        raise RuntimeError(
            "GaussianDiffusion1D cannot be exported to TensorRT.\n"
            "The DDIM inference loop is dynamic Python control flow and cannot be\n"
            "captured as a static ONNX/TRT graph.  Use a TCN or TransformerMoment\n"
            "for real-time Jetson deployment."
        )

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
    else:  # tcn (default / legacy checkpoints without model_type key)
        required = ("n_input_channels", "n_output_channels", "hidden_channels", "n_blocks", "kernel_size")
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ValueError(f"model_config missing keys: {missing}")
        model = TCN(
            n_input_channels=int(cfg["n_input_channels"]),
            n_output_channels=int(cfg["n_output_channels"]),
            hidden_channels=int(cfg["hidden_channels"]),
            n_blocks=int(cfg["n_blocks"]),
            kernel_size=int(cfg["kernel_size"]),
            dropout=float(cfg.get("dropout", 0.1)),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_model_and_meta(ckpt_path: Path) -> Tuple[torch.nn.Module, int, int, int, str]:
    """Return (model, n_in, n_out, window_size, model_type)."""
    raw = _load_raw(ckpt_path)
    if not isinstance(raw, dict) or "model_config" not in raw:
        raise ValueError(
            f"{ckpt_path} does not look like a repo checkpoint "
            "(missing 'model_config').  Re-save with trainV2.py."
        )
    model = _build_model(raw)
    cfg = raw["model_config"]
    n_in = int(cfg["n_input_channels"])
    n_out = int(cfg["n_output_channels"])
    window_size = int(raw.get("window_size", 100))
    model_type = cfg.get("model_type", "tcn")
    return model, n_in, n_out, window_size, model_type


# ---------------------------------------------------------------------------
# ONNX export helper
# ---------------------------------------------------------------------------

def _export_to_onnx(
    model: torch.nn.Module,
    n_in: int,
    seq_len: int,
    onnx_path: Path,
    opset: int,
    dynamic: bool,
) -> None:
    model.eval()
    model.cpu()
    dummy = torch.zeros(1, n_in, seq_len, dtype=torch.float32, device="cpu")
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    if dynamic:
        dynamic_axes = {
            "input":  {0: "batch", 2: "seq"},
            "output": {0: "batch", 2: "seq"},
        }
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
    print(f"  Exported ONNX: {onnx_path}")


# ---------------------------------------------------------------------------
# TensorRT engine builder
# ---------------------------------------------------------------------------

def _trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _release_cuda_before_trt_build() -> None:
    """Best-effort: drop PyTorch CUDA caches so TRT has a clean pool on Jetson."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _build_trt_engine(
    onnx_path: Path,
    trt_path: Path,
    n_in: int,
    seq_len: int,
    precision: str,
    workspace_gb: float,
    verbose: bool,
) -> None:
    """Parse ONNX and build a static-shape TensorRT engine."""
    import tensorrt as trt  # type: ignore[import]

    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(log_level)

    with (
        trt.Builder(logger) as builder,
        builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network,
        trt.OnnxParser(network, logger) as parser,
    ):
        config = builder.create_builder_config()

        # Workspace memory
        workspace_bytes = int(workspace_gb * (1 << 30))
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        except AttributeError:
            # TRT < 8.4 uses builder.max_workspace_size
            builder.max_workspace_size = workspace_bytes  # type: ignore[attr-defined]

        # Precision flags
        if precision == "fp16":
            if not builder.platform_has_fast_fp16:
                print("  WARNING: platform does not advertise fast FP16; using FP32.")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
                print("  Precision: FP16")
        else:
            print("  Precision: FP32")

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            onnx_bytes = f.read()
        if not parser.parse(onnx_bytes):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parsing failed:\n" + "\n".join(errors))

        # Static optimization profile (batch=1, fixed seq_len)
        profile = builder.create_optimization_profile()
        shape = (1, n_in, seq_len)
        profile.set_shape("input", min=shape, opt=shape, max=shape)
        config.add_optimization_profile(profile)

        print(f"  Building engine (this may take several minutes on first build)…")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(
                "TensorRT engine build failed (build_serialized_network returned None). "
                "TRT logs usually show Cuda OOM or NvMap errors when the GPU has no free "
                "memory. Free VRAM (close apps using the GPU, reboot the Jetson), then "
                "retry with a smaller builder workspace, e.g. "
                "`--workspace-gb 0.25` or `--workspace-gb 0.125`. "
                "See script docstring section 'Jetson / TRT build OOM'."
            )

    trt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trt_path, "wb") as f:
        f.write(bytes(serialized))

    size_mb = trt_path.stat().st_size / (1 << 20)
    print(f"  Saved TRT engine: {trt_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify(
    ckpt_path: Path,
    trt_path: Path,
    model: torch.nn.Module,
    n_in: int,
    seq_len: int,
    n_out: int,
    atol: float,
) -> None:
    """Compare PyTorch output vs TRT engine output on a random input."""
    try:
        import tensorrt as trt  # type: ignore[import]
        import pycuda.autoinit  # type: ignore[import]  # noqa: F401
        import pycuda.driver as cuda  # type: ignore[import]
    except ImportError as e:
        print(f"  [verify] Skipped — missing dependency for TRT inference: {e}")
        print("  Install pycuda: pip install pycuda")
        return

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((1, n_in, seq_len)).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    model.eval()
    with torch.no_grad():
        y_torch = model(x_t).cpu().numpy()

    # TRT inference via pycuda
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_input_shape("input", (1, n_in, seq_len))

    d_input = cuda.mem_alloc(x_np.nbytes)
    y_np = np.empty((1, n_out, seq_len), dtype=np.float32)
    d_output = cuda.mem_alloc(y_np.nbytes)

    cuda.memcpy_htod(d_input, x_np)
    context.execute_v2([int(d_input), int(d_output)])
    cuda.memcpy_dtoh(y_np, d_output)

    max_abs = float(np.max(np.abs(y_torch - y_np)))
    status = "PASS" if max_abs <= atol else "FAIL"
    print(f"  [verify] max |torch − trt| = {max_abs:.3e}  (atol={atol:.3e})  → {status}")


# ---------------------------------------------------------------------------
# trtexec command printer
# ---------------------------------------------------------------------------

def _print_trtexec_cmd(
    onnx_path: Path,
    trt_path: Path,
    n_in: int,
    seq_len: int,
    precision: str,
    workspace_gb: float,
) -> None:
    ws_mb = int(workspace_gb * 1024)
    fp16_flag = "--fp16 \\" if precision == "fp16" else ""
    print("\n── equivalent trtexec command ──────────────────────────────────────────")
    print(f"trtexec \\")
    print(f"  --onnx={onnx_path} \\")
    print(f"  --saveEngine={trt_path} \\")
    print(f"  --shapes=input:1x{n_in}x{seq_len} \\")
    print(f"  --workspace={ws_mb} \\")
    if fp16_flag:
        print(f"  {fp16_flag}")
    print("────────────────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert .pt checkpoint → TensorRT engine (.trt) for Jetson deployment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to the .pt checkpoint produced by trainV2.py.",
    )
    p.add_argument(
        "--onnx", type=str, default=None,
        help="Existing ONNX file to use instead of re-exporting from the checkpoint. "
             "If omitted, a .onnx is created next to the .pt file.",
    )
    p.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output .trt path. Default: same dir as checkpoint, .trt extension.",
    )
    p.add_argument(
        "--seq-len", type=int, default=None,
        help="Sequence length (frames).  Defaults to the training window_size stored in the checkpoint.",
    )
    p.add_argument(
        "--precision", type=str, default="fp16", choices=["fp32", "fp16"],
        help="TRT engine precision.  fp16 is recommended for Jetson (default: fp16).",
    )
    p.add_argument(
        "--workspace-gb", type=float, default=0.5,
        help="TRT builder workspace cap in GB (default: 0.5). Small IK/ID nets rarely "
             "need more; on Jetson use 0.25–0.5 if build fails with CUDA OOM.",
    )
    p.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset for export (default: 17; use 13 for older TRT/parsers).",
    )
    p.add_argument(
        "--keep-onnx", action="store_true",
        help="Keep the intermediate .onnx file after the TRT engine is built.",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="Numerically verify TRT engine output against PyTorch (requires pycuda).",
    )
    p.add_argument(
        "--verify-atol", type=float, default=1e-2,
        help="Absolute tolerance for --verify (default: 1e-2; FP16 may need 5e-2).",
    )
    p.add_argument(
        "--print-trtexec", action="store_true",
        help="Print the equivalent trtexec command and exit (no TRT build performed).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable TensorRT verbose logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── load model & metadata ─────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    model, n_in, n_out, window_size, model_type = _load_model_and_meta(ckpt_path)
    seq_len = args.seq_len if args.seq_len is not None else window_size
    print(f"  model_type : {model_type}")
    print(f"  n_input    : {n_in}")
    print(f"  n_output   : {n_out}")
    print(f"  seq_len    : {seq_len}  (training window_size={window_size})")

    # ── resolve paths ─────────────────────────────────────────────────────
    onnx_provided = args.onnx is not None
    onnx_path = (
        Path(args.onnx).expanduser().resolve()
        if onnx_provided
        else ckpt_path.with_suffix(".onnx")
    )
    trt_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else ckpt_path.with_suffix(".trt")
    )

    # ── ONNX export (skip if an existing file was provided) ───────────────
    if onnx_provided and onnx_path.is_file():
        print(f"Reusing ONNX: {onnx_path}")
    else:
        if onnx_provided and not onnx_path.is_file():
            raise FileNotFoundError(f"--onnx path does not exist: {onnx_path}")
        print("Exporting to ONNX…")
        _export_to_onnx(
            model=model,
            n_in=n_in,
            seq_len=seq_len,
            onnx_path=onnx_path,
            opset=args.opset,
            dynamic=False,  # static shape is optimal for real-time Jetson inference
        )

    # ── print trtexec command (optional) ─────────────────────────────────
    _print_trtexec_cmd(onnx_path, trt_path, n_in, seq_len, args.precision, args.workspace_gb)
    if args.print_trtexec:
        print("(--print-trtexec: skipping TRT build)")
        return

    # ── build TRT engine ─────────────────────────────────────────────────
    if not _trt_available():
        print(
            "TensorRT Python package not found.\n"
            "Install TensorRT on Jetson via JetPack, or install the TRT wheel:\n"
            "  pip install tensorrt\n\n"
            "You can also build the engine using the trtexec command printed above."
        )
        sys.exit(1)

    print(f"\nBuilding TRT engine → {trt_path}")
    _release_cuda_before_trt_build()
    _build_trt_engine(
        onnx_path=onnx_path,
        trt_path=trt_path,
        n_in=n_in,
        seq_len=seq_len,
        precision=args.precision,
        workspace_gb=args.workspace_gb,
        verbose=args.verbose,
    )

    # ── clean up intermediate ONNX ────────────────────────────────────────
    if not onnx_provided and not args.keep_onnx:
        onnx_path.unlink(missing_ok=True)
        print(f"  Removed intermediate ONNX (use --keep-onnx to retain).")

    # ── verify ───────────────────────────────────────────────────────────
    if args.verify:
        print("\nVerifying TRT engine against PyTorch…")
        _verify(
            ckpt_path=ckpt_path,
            trt_path=trt_path,
            model=model,
            n_in=n_in,
            seq_len=seq_len,
            n_out=n_out,
            atol=args.verify_atol,
        )

    print(f"\nDone.  Engine ready at: {trt_path}")
    print(
        "\nTo use at runtime, load the engine with tensorrt.Runtime.deserialize_cuda_engine()\n"
        "and call context.execute_v2([d_input, d_output]) in your control loop.\n"
        "See the pycuda / trt inference examples in the TensorRT developer guide."
    )


if __name__ == "__main__":
    main()
