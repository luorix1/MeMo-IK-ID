"""
Offline evaluator for os_kinetics knee model checkpoints.

Loads:
  - checkpoint (.pt) from ik_id training
  - runtime log (.npz) from knee-exo-ctrl

Builds the TCN from checkpoint metadata and runs sliding-window inference on:
  - knee angle
  - knee angular velocity (from gyro stand-in or encoder velocity key)

Outputs:
  - console summary stats / correlations
  - optional .npz with per-frame predictions
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CTRL_ROOT = os.path.dirname(SCRIPT_DIR)           # .../knee-exo-ctrl
PROJECT_ROOT = os.path.dirname(CTRL_ROOT)         # .../os_kinetics
sys.path.insert(0, PROJECT_ROOT)

from model import TCN  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run checkpoint inference directly on logged knee-exo .npz signals."
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Path to training checkpoint (.pt), e.g. runs/.../best_model.pt",
    )
    p.add_argument(
        "--npz",
        required=True,
        help="Path to runtime log .npz, e.g. knee-exo-ctrl/cascade_0425.npz",
    )
    p.add_argument(
        "--side",
        choices=["right", "left"],
        default="right",
        help="Which side signals to use from the .npz file.",
    )
    p.add_argument(
        "--vel-key",
        choices=["gyr", "enc"],
        default="gyr",
        help="Use gyro stand-in velocity (*_u_gyr) or encoder velocity (*_u).",
    )
    p.add_argument(
        "--angle-unit",
        choices=["rad", "deg"],
        default="rad",
        help="Unit of knee angle in npz log. Use 'deg' for logs that stored raw motor degrees.",
    )
    p.add_argument(
        "--mass",
        type=float,
        default=88.0,
        help="Subject mass (kg) for Nm/kg -> Nm conversion.",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Additional runtime scale multiplier.",
    )
    p.add_argument(
        "--save",
        default=None,
        help="Optional output .npz path for predictions.",
    )
    p.add_argument(
        "--plot",
        default=None,
        help="Optional output PNG path for plots.",
    )
    return p.parse_args()


def get_signals(log: np.lib.npyio.NpzFile, side: str, vel_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    side_sfx = "r" if side == "right" else "l"

    angle_key = f"knee_angle_{side_sfx}"
    if vel_key == "gyr":
        vel_name = f"knee_angle_{side_sfx}_u_gyr"
    else:
        vel_name = f"knee_angle_{side_sfx}_u"
    cmd_key = "cmd_R" if side == "right" else "cmd_L"

    missing = [k for k in (angle_key, vel_name, cmd_key) if k not in log.files]
    if missing:
        raise KeyError(f"Missing required keys in npz: {missing}. Available: {log.files}")

    angle = np.asarray(log[angle_key], dtype=np.float32)
    vel = np.asarray(log[vel_name], dtype=np.float32)
    cmd = np.asarray(log[cmd_key], dtype=np.float32)
    return angle, vel, cmd


def run_inference(
    model: TCN,
    angle: np.ndarray,
    vel: np.ndarray,
    window_size: int,
    device: str,
) -> np.ndarray:
    """Sliding window inference producing per-frame Nm/kg prediction."""
    n = int(min(len(angle), len(vel)))
    pred = np.zeros(n, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for t in range(n):
            start = max(0, t - window_size + 1)
            valid = t - start + 1

            x = np.zeros((2, window_size), dtype=np.float32)
            x[0, -valid:] = angle[start : t + 1]
            x[1, -valid:] = vel[start : t + 1]

            xt = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,2,W)
            y = model(xt)  # (1,1,W)
            pred[t] = float(y[0, 0, -1].item())
    return pred


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den == 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    model_cfg = ckpt["model_config"]
    window_size = int(ckpt.get("window_size", 100))
    print("[INFO] model_config:", model_cfg)
    print("[INFO] window_size:", window_size)

    model = TCN(**model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("[INFO] device:", device)

    log = np.load(args.npz)
    angle, vel, cmd = get_signals(log, side=args.side, vel_key=args.vel_key)
    n = int(min(len(angle), len(vel), len(cmd)))
    angle = angle[:n]
    vel = vel[:n]
    cmd = cmd[:n]
    if args.angle_unit == "deg":
        angle = np.deg2rad(angle)

    pred_nmpkg = run_inference(model, angle, vel, window_size, device=device)
    pred_nm = pred_nmpkg * float(args.mass) * float(args.scale)

    print("\n=== Signal Summary ===")
    print(f"n={n}")
    print(f"angle: min={angle.min(): .4f} max={angle.max(): .4f} mean={angle.mean(): .4f} std={angle.std(): .4f}")
    print(f"vel  : min={vel.min(): .4f} max={vel.max(): .4f} mean={vel.mean(): .4f} std={vel.std(): .4f}")
    print(f"pred_nmpkg: min={pred_nmpkg.min(): .4f} max={pred_nmpkg.max(): .4f} mean={pred_nmpkg.mean(): .4f} std={pred_nmpkg.std(): .4f}")
    print(f"pred_nm   : min={pred_nm.min(): .4f} max={pred_nm.max(): .4f} mean={pred_nm.mean(): .4f} std={pred_nm.std(): .4f}")
    print(f"cmd_nm    : min={cmd.min(): .4f} max={cmd.max(): .4f} mean={cmd.mean(): .4f} std={cmd.std(): .4f}")
    print(f"corr(pred_nm, cmd_nm) = {corr(pred_nm, cmd): .4f}")
    print(f"MAE(pred_nm vs cmd_nm)= {float(np.mean(np.abs(pred_nm - cmd))): .4f}")

    if args.save:
        out = os.path.abspath(args.save)
        np.savez(
            out,
            pred_nmpkg=pred_nmpkg,
            pred_nm=pred_nm,
            cmd_nm=cmd,
            knee_angle=angle,
            knee_vel=vel,
        )
        print(f"[INFO] Saved predictions to: {out}")

    if args.plot:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        x = np.arange(n)

        axes[0].plot(x, pred_nm, label="pred_nm (offline from ckpt)", linewidth=1.5)
        axes[0].plot(x, cmd, label="cmd_nm (logged)", linewidth=1.0, alpha=0.85)
        axes[0].set_ylabel("Torque (Nm)")
        axes[0].set_title("Offline Checkpoint Prediction vs Logged Command")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, pred_nmpkg, color="tab:orange", linewidth=1.2)
        axes[1].set_ylabel("Nm/kg")
        axes[1].set_title("Model Output (Nm/kg)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(x, angle, color="tab:green", linewidth=1.2)
        axes[2].set_ylabel("rad")
        axes[2].set_title("Input Channel 0: Knee Angle")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(x, vel, color="tab:red", linewidth=1.2)
        axes[3].set_ylabel("rad/s")
        axes[3].set_xlabel("Frame")
        axes[3].set_title("Input Channel 1: Knee Angular Velocity")
        axes[3].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.abspath(args.plot)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
