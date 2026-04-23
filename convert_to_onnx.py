#!/usr/bin/env python3
"""
Convert TCN PyTorch checkpoints (.pt/.pth) to ONNX for Jetson inference.

Supported input checkpoint formats:
1) Dict checkpoint with "model_config" + "model_state_dict" (default in this repo)
2) Serialized torch.nn.Module

Examples:
  python convert_to_onnx.py \
    --checkpoint runs/0422_ik_id_all_huber_noise/best_model.pt \
    --seq-len 200

  python convert_to_onnx.py \
    --checkpoint runs/0422_ik_id_all_huber_noise/best_model.pt \
    --output runs/0422_ik_id_all_huber_noise/best_model.onnx \
    --opset 13 \
    --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from model import TCN


def _load_checkpoint(path: Path, device: str) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older torch versions without weights_only.
        return torch.load(path, map_location=device)


def _build_model_from_repo_checkpoint(ckpt: Dict[str, Any]) -> TCN:
    if "model_config" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError(
            "Checkpoint missing required keys: expected 'model_config' and 'model_state_dict'."
        )

    cfg = ckpt["model_config"]
    required = (
        "n_input_channels",
        "n_output_channels",
        "hidden_channels",
        "n_blocks",
        "kernel_size",
        "dropout",
    )
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"model_config missing keys: {missing}")

    model = TCN(
        n_input_channels=int(cfg["n_input_channels"]),
        n_output_channels=int(cfg["n_output_channels"]),
        hidden_channels=int(cfg["hidden_channels"]),
        n_blocks=int(cfg["n_blocks"]),
        kernel_size=int(cfg["kernel_size"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _resolve_model_and_channels(loaded: Any) -> Tuple[torch.nn.Module, int]:
    if isinstance(loaded, torch.nn.Module):
        model = loaded.eval()
    elif isinstance(loaded, dict):
        model = _build_model_from_repo_checkpoint(loaded)
    else:
        raise ValueError(
            f"Unsupported checkpoint type: {type(loaded)}. "
            "Expected torch.nn.Module or dict checkpoint."
        )

    if not hasattr(model, "n_input_channels"):
        raise ValueError(
            "Model does not expose 'n_input_channels'. "
            "This exporter currently expects the repo TCN model."
        )
    n_input_channels = int(model.n_input_channels)  # type: ignore[attr-defined]
    return model, n_input_channels


def _validate_with_onnxruntime(
    onnx_path: Path,
    model: torch.nn.Module,
    batch_size: int,
    n_input_channels: int,
    seq_len: int,
) -> None:
    import numpy as np  # Local import so verify remains optional dependency usage.
    import onnxruntime as ort

    x = torch.randn(batch_size, n_input_channels, seq_len, dtype=torch.float32)
    with torch.no_grad():
        y_torch = model(x).detach().cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    y_onnx = sess.run(None, {input_name: x.cpu().numpy()})[0]

    max_abs = float(np.max(np.abs(y_torch - y_onnx)))
    print(f"[verify] max |torch - onnx| = {max_abs:.6e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert .pt checkpoint to .onnx for Jetson inference.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt/.pth checkpoint")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX path (default: same name as checkpoint with .onnx extension)",
    )
    p.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13)")
    p.add_argument("--batch-size", type=int, default=1, help="Dummy export batch size")
    p.add_argument("--seq-len", type=int, default=200, help="Dummy export sequence length")
    p.add_argument(
        "--static-shape",
        action="store_true",
        help="Export fixed input/output shape (disables dynamic batch/sequence axes).",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Run a quick PyTorch vs ONNXRuntime numerical sanity check.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path = Path(args.output).expanduser().resolve() if args.output else ckpt_path.with_suffix(".onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.batch_size < 1 or args.seq_len < 1:
        raise ValueError("--batch-size and --seq-len must be >= 1.")

    loaded = _load_checkpoint(ckpt_path, device="cpu")
    model, n_input_channels = _resolve_model_and_channels(loaded)
    model.to("cpu")
    model.eval()

    dummy_input = torch.randn(args.batch_size, n_input_channels, args.seq_len, dtype=torch.float32)
    dynamic_axes = None
    if not args.static_shape:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "sequence_length"},
            "output": {0: "batch_size", 2: "sequence_length"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=int(args.opset),
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    print(f"Exported ONNX: {out_path}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  input:  (batch, channels, seq) = ({args.batch_size}, {n_input_channels}, {args.seq_len})")
    print(f"  dynamic: {'no' if args.static_shape else 'yes (batch, seq)'}")
    print(f"  opset: {args.opset}")

    if args.verify:
        try:
            _validate_with_onnxruntime(
                out_path,
                model=model,
                batch_size=args.batch_size,
                n_input_channels=n_input_channels,
                seq_len=args.seq_len,
            )
        except ModuleNotFoundError as exc:
            missing = str(exc).split("'")[1] if "'" in str(exc) else "dependency"
            raise ModuleNotFoundError(
                f"--verify requested, but missing dependency: {missing}. "
                "Install with: pip install onnxruntime"
            ) from exc


if __name__ == "__main__":
    main()
