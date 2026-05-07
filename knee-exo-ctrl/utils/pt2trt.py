"""
pt2trt.py  —  PyTorch → ONNX → TensorRT conversion for knee-exo-ctrl TCN models.

Supports checkpoints saved by the os_kinetics training pipeline, which store:
  - model_state_dict  : TCN weights
  - model_config      : architecture hyper-parameters
  - window_size       : sequence length used during training

The TRT engine produced matches what TRTWorkerUni expects:
  input  shape : (1, n_input_channels, window_size)
  output shape : (1, n_output_channels)   ← last timestep of TCN output

Usage (from knee-exo-ctrl root or anywhere):
    python utils/pt2trt.py --pt /path/to/best_model.pt [options]

Examples:
    # FP32, .trt written next to the .pt file
    python utils/pt2trt.py --pt /path/to/best_model.pt

    # FP16, explicit output path, custom YAML for window_size override
    python utils/pt2trt.py --pt /path/to/best_model.pt \\
                           --trt /path/to/model.trt    \\
                           --cfg cfg/cascade_0425.yaml  \\
                           --fp16
"""

import argparse
import inspect
import os
import sys

import torch
import torch.nn as nn
import tensorrt as trt
import yaml

# ---------------------------------------------------------------------------
# Locate model.py: it lives one directory above knee-exo-ctrl (the repo root).
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))       # .../knee-exo-ctrl/utils
_CTRL_ROOT   = os.path.dirname(_SCRIPT_DIR)                      # .../knee-exo-ctrl
_PARENT_ROOT = os.path.dirname(_CTRL_ROOT)                       # .../MeMo-IK-ID  (has model.py)

sys.path.insert(0, _PARENT_ROOT)

try:
    from model import TCN  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"Could not import TCN from model.py. "
        f"Expected model.py at: {_PARENT_ROOT}\n"
        f"Original error: {e}"
    )


# ---------------------------------------------------------------------------
# Wrapper: slice last timestep so output matches TRTWorkerUni expectation.
#   TCN forward : (B, C_in, T) → (B, C_out, T)
#   Wrapper     : (B, C_in, T) → (B, C_out)
# ---------------------------------------------------------------------------
class _TCNLastStep(nn.Module):
    def __init__(self, tcn: TCN) -> None:
        super().__init__()
        self.tcn = tcn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x)[:, :, -1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _tcn_ctor_kwargs(cfg: dict) -> dict:
    """Drop checkpoint-only keys (e.g. ``model_type``) before ``TCN(**kwargs)``."""
    allowed = {k for k in inspect.signature(TCN.__init__).parameters if k != "self"}
    return {k: v for k, v in cfg.items() if k in allowed}


def load_checkpoint(pt_path: str) -> dict:
    """Load checkpoint regardless of whether numpy globals are embedded."""
    return torch.load(pt_path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def pt_to_trt(
    pt_model_path: str,
    trt_engine_path: str,
    window_size: int,
    fp16_mode: bool = False,
) -> None:
    """Convert an os_kinetics TCN checkpoint to a TensorRT engine."""

    # ------------------------------------------------------------------
    # 1. Load checkpoint and extract components
    # ------------------------------------------------------------------
    print(f"[INFO] Loading checkpoint: {pt_model_path}")
    ckpt = load_checkpoint(pt_model_path)

    if "model_state_dict" not in ckpt:
        raise KeyError(
            "Expected 'model_state_dict' in checkpoint. "
            f"Found keys: {list(ckpt.keys())}"
        )

    model_config = ckpt["model_config"]
    print(f"[INFO] model_config from checkpoint: {model_config}")

    mt = model_config.get("model_type", "tcn")
    if mt != "tcn":
        raise ValueError(
            f"This script only converts TCN checkpoints; got model_type={mt!r}. "
            "Use a checkpoint trained with --model-type tcn."
        )

    # window_size: prefer checkpoint value, allow CLI override
    ckpt_window = ckpt.get("window_size", None)
    if ckpt_window is not None and ckpt_window != window_size:
        print(
            f"[INFO] Checkpoint window_size={ckpt_window} overrides "
            f"supplied window_size={window_size}."
        )
        window_size = int(ckpt_window)
    print(f"[INFO] window_size (sequence length): {window_size}")

    n_input_channels  = model_config["n_input_channels"]
    n_output_channels = model_config["n_output_channels"]

    # ------------------------------------------------------------------
    # 2. Build model and load weights
    # ------------------------------------------------------------------
    tcn_kw = _tcn_ctor_kwargs(model_config)
    tcn = TCN(**tcn_kw).eval()
    tcn.load_state_dict(ckpt["model_state_dict"])
    model = _TCNLastStep(tcn).eval().cuda()
    print(f"[INFO] Model built — parameters: {sum(p.numel() for p in tcn.parameters()):,}")

    # ------------------------------------------------------------------
    # 3. Export to ONNX (static batch=1 to match TRTWorkerUni)
    # ------------------------------------------------------------------
    onnx_path   = trt_engine_path.replace(".trt", ".onnx")
    in_shape    = (1, n_input_channels, window_size)
    dummy_input = torch.randn(*in_shape, device="cuda")

    print(f"[INFO] Exporting ONNX → {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            do_constant_folding=True,
        )
    print(f"[INFO] ONNX saved  — input {in_shape} → output (1, {n_output_channels})")

    # ------------------------------------------------------------------
    # 4. Parse ONNX with TensorRT
    # ------------------------------------------------------------------
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(onnx_path):
        print("[ERROR] ONNX parsing failed:")
        for i in range(parser.num_errors):
            print(f"  {parser.get_error(i)}")
        raise RuntimeError("ONNX model parsing failed.")

    net_in  = network.get_input(0)
    net_out = network.get_output(0)
    print(f"[INFO] TRT network input  : {net_in.name}  shape={net_in.shape}")
    print(f"[INFO] TRT network output : {net_out.name} shape={net_out.shape}")

    # ------------------------------------------------------------------
    # 5. Builder config  (static batch=1 — no dynamic profile needed)
    # ------------------------------------------------------------------
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    if fp16_mode:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] Building in FP16 mode.")
        else:
            print("[WARNING] FP16 not supported on this platform — using FP32.")

    # ------------------------------------------------------------------
    # 6. Build and serialize
    # ------------------------------------------------------------------
    print("[INFO] Building TensorRT engine (may take several minutes) …")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed.")

    os.makedirs(os.path.dirname(os.path.abspath(trt_engine_path)), exist_ok=True)
    with open(trt_engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"[SUCCESS] TRT engine saved : {trt_engine_path}")
    print(f"          Engine I/O       : {in_shape} → (1, {n_output_channels})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_cfg = os.path.join(_CTRL_ROOT, "cfg", "cascade_0425.yaml")

    p = argparse.ArgumentParser(
        description="Convert an os_kinetics TCN .pt checkpoint to a TensorRT engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pt",  required=True, help="Path to the input .pt checkpoint.")
    p.add_argument(
        "--trt", default=None,
        help="Output .trt path. Defaults to <pt_path>.trt (replaces .pt extension).",
    )
    p.add_argument(
        "--cfg", default=default_cfg,
        help="YAML config for fallback window_size (frame_length). "
             "Ignored if checkpoint contains window_size.",
    )
    p.add_argument(
        "--window-size", type=int, default=None,
        help="Sequence length override. Checkpoint value takes precedence if present.",
    )
    p.add_argument("--fp16", action="store_true", help="Build engine in FP16 mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pt_path  = os.path.abspath(args.pt)
    trt_path = (
        os.path.abspath(args.trt)
        if args.trt
        else pt_path.replace(".pt", ".trt")
    )

    # Resolve window_size: CLI → YAML → default 100
    if args.window_size is not None:
        window_size = args.window_size
    elif os.path.isfile(args.cfg):
        cfg = load_yaml(args.cfg)
        window_size = int(cfg.get("frame_length", cfg.get("window_size", 100)))
        print(f"[INFO] window_size={window_size} from config: {args.cfg}")
    else:
        window_size = 100
        print(f"[INFO] Config not found — using default window_size={window_size}")

    pt_to_trt(
        pt_model_path=pt_path,
        trt_engine_path=trt_path,
        window_size=window_size,
        fp16_mode=args.fp16,
    )


if __name__ == "__main__":
    main()
