"""
Compare outputs from a TensorRT engine and a PyTorch checkpoint on identical inputs.

Supports sample windows taken from a logged knee-exo .npz file.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import torch
import tensorrt as trt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CTRL_ROOT = os.path.dirname(SCRIPT_DIR)           # .../knee-exo-ctrl
PROJECT_ROOT = os.path.dirname(CTRL_ROOT)         # .../os_kinetics
sys.path.insert(0, PROJECT_ROOT)

from model import TCN  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare TRT vs PT outputs on sample inputs.")
    p.add_argument("--trt", required=True, help="Path to TensorRT engine (.trt)")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--npz", required=True, help="Path to runtime log .npz")
    p.add_argument("--side", choices=["right", "left"], default="right")
    p.add_argument("--vel-key", choices=["gyr", "enc"], default="gyr")
    p.add_argument("--angle-unit", choices=["rad", "deg"], default="rad")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _find_io_names(engine) -> tuple[str, str]:
    in_name, out_name = None, None
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


def build_samples(
    npz_path: str,
    side: str,
    vel_key: str,
    angle_unit: str,
    window_size: int,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    data = np.load(npz_path)
    side_sfx = "r" if side == "right" else "l"
    angle = np.asarray(data[f"knee_angle_{side_sfx}"], dtype=np.float32)
    vel_name = f"knee_angle_{side_sfx}_u_gyr" if vel_key == "gyr" else f"knee_angle_{side_sfx}_u"
    vel = np.asarray(data[vel_name], dtype=np.float32)

    n = min(len(angle), len(vel))
    angle = angle[:n]
    vel = vel[:n]
    if angle_unit == "deg":
        angle = np.deg2rad(angle)

    valid_t = np.arange(window_size - 1, n)
    if len(valid_t) == 0:
        raise ValueError(f"Not enough frames ({n}) for window_size={window_size}")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_t, size=min(n_samples, len(valid_t)), replace=False)
    chosen.sort()

    xs: List[np.ndarray] = []
    for t in chosen:
        s = t - window_size + 1
        x = np.stack([angle[s : t + 1], vel[s : t + 1]], axis=0)  # (2, W)
        xs.append(x.astype(np.float32))
    return np.stack(xs, axis=0)  # (N, 2, W)


def run_pt(ckpt_path: str, x_np: np.ndarray) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_config"]
    model = TCN(**model_cfg).eval().cuda()
    model.load_state_dict(ckpt["model_state_dict"])

    with torch.no_grad():
        xt = torch.from_numpy(x_np).cuda()             # (N,2,W)
        y_seq = model(xt)                              # (N,1,W)
        y_last = y_seq[:, :, -1].squeeze(1).cpu().numpy()
    return y_last


def run_trt(trt_path: str, x_np: np.ndarray) -> np.ndarray:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TRT engine.")

    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    in_name, out_name = _find_io_names(engine)

    outputs = []
    for x in x_np:
        xb = x[None, ...]  # (1,2,W)
        in_shape = tuple(xb.shape)
        out_shape = tuple(engine.get_tensor_shape(out_name))
        # resolve dynamic dims if any
        if -1 in out_shape:
            context.set_input_shape(in_name, in_shape)
            out_shape = tuple(context.get_tensor_shape(out_name))
        else:
            context.set_input_shape(in_name, in_shape)

        d_in = torch.from_numpy(np.ascontiguousarray(xb)).cuda()
        d_out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        context.set_tensor_address(in_name, int(d_in.data_ptr()))
        context.set_tensor_address(out_name, int(d_out.data_ptr()))
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        y = d_out.detach().cpu().numpy().reshape(-1)[0]
        outputs.append(y)
    return np.asarray(outputs, dtype=np.float32)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    window_size = int(ckpt.get("window_size", 100))
    model_cfg = ckpt["model_config"]
    print("[INFO] window_size:", window_size)
    print("[INFO] model_config:", model_cfg)

    x_np = build_samples(
        npz_path=args.npz,
        side=args.side,
        vel_key=args.vel_key,
        angle_unit=args.angle_unit,
        window_size=window_size,
        n_samples=args.num_samples,
        seed=args.seed,
    )
    print("[INFO] sample batch shape:", x_np.shape)

    y_pt = run_pt(args.ckpt, x_np)
    y_trt = run_trt(args.trt, x_np)

    abs_err = np.abs(y_trt - y_pt)
    denom = np.maximum(np.abs(y_pt), 1e-6)
    rel_err = abs_err / denom

    print("\n=== Comparison (Nm/kg output) ===")
    print(f"samples: {len(y_pt)}")
    print(f"PT  : min={y_pt.min(): .6f} max={y_pt.max(): .6f} mean={y_pt.mean(): .6f}")
    print(f"TRT : min={y_trt.min(): .6f} max={y_trt.max(): .6f} mean={y_trt.mean(): .6f}")
    print(f"abs err: mean={abs_err.mean(): .6e} max={abs_err.max(): .6e}")
    print(f"rel err: mean={rel_err.mean(): .6e} max={rel_err.max(): .6e}")

    print("\nidx | y_pt | y_trt | abs_err")
    for i in range(min(10, len(y_pt))):
        print(f"{i:3d} | {y_pt[i]: .6f} | {y_trt[i]: .6f} | {abs_err[i]: .6e}")


if __name__ == "__main__":
    main()

