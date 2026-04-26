"""
pt2trt.py  —  PyTorch → ONNX → TensorRT conversion for knee-exo-ctrl TCN models.

Usage (from the repo root):
    python utils/pt2trt.py --pt <path/to/model.pt> [options]

Examples:
    # FP32, output .trt next to the .pt file, config from default YAML
    python utils/pt2trt.py --pt /path/to/uni_model.pt

    # FP16 with an explicit output path and a custom config
    python utils/pt2trt.py --pt /path/to/uni_model.pt \\
                           --trt /path/to/uni_model.trt \\
                           --cfg cfg/cascade_0425.yaml \\
                           --fp16
"""

import argparse
import os
import sys

import torch
import tensorrt as trt
import yaml

# ---------------------------------------------------------------------------
# Resolve paths relative to the repo root so the script works from anywhere.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tcn_model"))

from TCN_Header_Model import TCNModel  # noqa: E402  (import after sys.path patch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_list(value: str) -> list:
    """Parse a comma-separated string into a list of ints (e.g. '80,80,80')."""
    return [int(v.strip()) for v in value.split(",")]


def load_yaml(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def pt_to_trt(
    pt_model_path: str,
    trt_engine_path: str,
    hyperparam_config: dict,
    fp16_mode: bool = False,
) -> None:
    """Convert a .pt checkpoint to a TensorRT .trt engine via ONNX."""

    input_size  = int(hyperparam_config["input_size"])
    window_size = int(hyperparam_config["window_size"])

    # ------------------------------------------------------------------
    # 1. Load PyTorch model
    # ------------------------------------------------------------------
    print(f"[INFO] Loading PyTorch model from: {pt_model_path}")
    model = TCNModel(hyperparam_config).eval()

    import numpy as np
    safe_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.core.multiarray.scalar,
    ]
    with torch.serialization.safe_globals(safe_globals):
        state_dict = torch.load(pt_model_path, map_location="cpu", weights_only=True)

    model.load_state_dict(state_dict)
    model.cuda()

    # ------------------------------------------------------------------
    # 2. Export to ONNX (dynamic batch axis)
    # ------------------------------------------------------------------
    onnx_path = trt_engine_path.replace(".trt", ".onnx")
    dummy_input = torch.randn(2, input_size, window_size, device="cuda")

    print(f"[INFO] Exporting ONNX model to: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes={
                "input":  {0: "batch"},
                "output": {0: "batch"},
            },
        )
    print(f"[INFO] ONNX model saved to: {onnx_path}")

    # ------------------------------------------------------------------
    # 3. Parse ONNX with TensorRT
    # ------------------------------------------------------------------
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(onnx_path):
        print("[ERROR] Failed to parse ONNX model:")
        for i in range(parser.num_errors):
            print(f"  {parser.get_error(i)}")
        raise RuntimeError("ONNX model parsing failed.")

    input_tensor  = network.get_input(0)
    output_tensor = network.get_output(0)
    print(f"[INFO] Network input  — name: {input_tensor.name},  shape: {input_tensor.shape}")
    print(f"[INFO] Network output — name: {output_tensor.name}, shape: {output_tensor.shape}")

    # ------------------------------------------------------------------
    # 4. Builder config
    # ------------------------------------------------------------------
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    if fp16_mode:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] Building engine in FP16 mode.")
        else:
            print("[WARNING] FP16 not supported on this platform — falling back to FP32.")

    # Dynamic-batch optimization profile: batch 1 (min / opt) → 2 (max)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        min=(1, input_size, window_size),
        opt=(1, input_size, window_size),
        max=(2, input_size, window_size),
    )
    config.add_optimization_profile(profile)

    # ------------------------------------------------------------------
    # 5. Build and serialize
    # ------------------------------------------------------------------
    print("[INFO] Building TensorRT engine (this may take a few minutes) …")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed.")

    os.makedirs(os.path.dirname(os.path.abspath(trt_engine_path)), exist_ok=True)
    with open(trt_engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"[SUCCESS] TensorRT engine saved: {trt_engine_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_cfg = os.path.join(REPO_ROOT, "cfg", "cascade_0425.yaml")

    p = argparse.ArgumentParser(
        description="Convert a TCN .pt checkpoint to a TensorRT .trt engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--pt",  required=True, help="Path to the input .pt model file.")

    # Optional — I/O paths
    p.add_argument(
        "--trt",
        default=None,
        help="Path for the output .trt engine. Defaults to <pt_path>.trt",
    )
    p.add_argument(
        "--cfg",
        default=default_cfg,
        help="Path to the YAML config file (reads input_size / frame_length).",
    )

    # Build options
    p.add_argument("--fp16", action="store_true", help="Build engine in FP16 mode.")

    # TCN architecture overrides (rarely needed; the defaults match training configs)
    p.add_argument("--num-channels", default="80,80,80,80,80",
                   help="Comma-separated channel list per TCN block.")
    p.add_argument("--kernel-size",  type=int, default=5)
    p.add_argument("--num-layers",   type=int, default=2,
                   help="Number of conv layers inside each TemporalBlock.")
    p.add_argument("--dropout",      type=float, default=0.15)
    p.add_argument("--dilations",    default="1,2,4,8,16",
                   help="Comma-separated dilation per TCN block.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve .trt output path
    pt_path  = os.path.abspath(args.pt)
    trt_path = os.path.abspath(args.trt) if args.trt else pt_path.replace(".pt", ".trt")

    # Load YAML config
    cfg = load_yaml(args.cfg)
    print(f"[INFO] Loaded config from: {args.cfg}")

    # frame_length in YAML is the TCN window (sequence length)
    window_size = int(cfg.get("frame_length", cfg.get("window_size", 100)))
    input_size  = int(cfg.get("input_size",  2))
    output_size = int(cfg.get("output_size", 2))

    hyperparam_config = {
        # Model I/O (from YAML)
        "input_size":      input_size,
        "output_size":     output_size,
        "window_size":     window_size,
        # TCN architecture (CLI overrides or defaults)
        "num_channels":    _parse_list(args.num_channels),
        "kernel_size":     args.kernel_size,
        "number_of_layers": args.num_layers,
        "dropout":         args.dropout,
        "dilations":       _parse_list(args.dilations),
    }

    print("[INFO] Hyperparam config:")
    for k, v in hyperparam_config.items():
        print(f"       {k}: {v}")

    pt_to_trt(
        pt_model_path=pt_path,
        trt_engine_path=trt_path,
        hyperparam_config=hyperparam_config,
        fp16_mode=args.fp16,
    )


if __name__ == "__main__":
    main()
