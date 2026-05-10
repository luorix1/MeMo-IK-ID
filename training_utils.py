"""Shared TCN training loop helpers (IK→ID and IMU sagittal trainers)."""

from __future__ import annotations

import math
import random
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and Torch (CPU & CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MomentLoss(nn.Module):
    """
    Composite loss for joint moment prediction with optional Huber robustness and
    per-DOF channel weighting.

    Args:
        loss_type: ``"mse"`` (standard L2) or ``"huber"`` (smooth-L1 / Huber). Huber
            is less sensitive to large residuals at transition events (stair/ramp).
        huber_delta: Threshold below which Huber behaves as L2, above as L1
            (in N·m/kg). Typical gait RMSE is 0.1–0.2, so ``delta=0.5`` means the
            majority of residuals use the L2 branch.
        dof_weights: 1-D tensor of length ``n_output_channels``. Each channel's
            per-element loss is multiplied by the corresponding weight before
            averaging. If ``None``, all channels have weight 1.0 (identical to
            unweighted MSE/Huber).
    """

    def __init__(
        self,
        loss_type: str = "mse",
        huber_delta: float = 0.5,
        dof_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if loss_type not in ("mse", "huber"):
            raise ValueError(f"loss_type must be 'mse' or 'huber', got {loss_type!r}")
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        if dof_weights is not None:
            self.register_buffer("dof_weights", dof_weights.float())
        else:
            self.register_buffer("dof_weights", None)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: ``(B, C, W)`` – batch × output DOFs × window length.
        Returns:
            Scalar loss.
        """
        if self.loss_type == "huber":
            diff = pred - target
            abs_diff = diff.abs()
            delta = self.huber_delta
            elem = torch.where(
                abs_diff <= delta,
                0.5 * diff ** 2,
                delta * (abs_diff - 0.5 * delta),
            )
        else:
            elem = (pred - target) ** 2

        if self.dof_weights is not None:
            # Mean over B and W per DOF, then weighted average over DOFs.
            per_dof = elem.mean(dim=(0, 2))          # (C,)
            w = self.dof_weights.to(pred.device)
            return (w * per_dof).sum() / w.sum()
        return elem.mean()


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    grad_clip: float = 1.0,
    input_noise_std: float = 0.0,
    angle_jitter_std: float = 0.0,
    n_position_channels: Optional[int] = None,
    *,
    angle_input_offset_augment_deg: float = 0.0,
    correlated_vel_noise: bool = False,
    sample_rate_hz: float = 200.0,
    smoothness_lambda: float = 0.0,
) -> float:
    """
    Train for one epoch.

    New keyword-only arguments (all backward-compatible with defaults):

    correlated_vel_noise:
        When ``True`` and ``angle_jitter_std > 0``, the velocity channels are
        *recomputed* from the noisy position channels via finite difference instead
        of leaving the pre-loaded (optionally LPF'd) velocities unchanged.  This
        reproduces the cascade noise model where the IMU estimator delivers noisy
        angles and the downstream module numerically differentiates them — meaning
        velocity noise is correlated with and amplified from angle noise.
    sample_rate_hz:
        Sample rate used to scale the finite-difference velocity (rad/s).
        Must match the dataset's effective sample rate (after optional resampling).
    smoothness_lambda:
        If > 0, a temporal smoothness penalty ``λ · mean(||Δŷ||²)`` is added to
        the primary loss.  Joint moments are inherently smooth (≤4 Hz content);
        this term discourages high-frequency prediction jitter, especially
        important when the model input carries noise from the cascade.
    angle_input_offset_augment_deg:
        If > 0, adds independent ``Uniform(-d, +d)`` degree bias per batch sample
        and angle DOF (constant over the time window) to position channels only;
        velocity channels are unchanged.  Set to 0 to disable.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # ---- Input augmentation ----
        _need_n_pos = angle_input_offset_augment_deg > 0 or angle_jitter_std > 0
        if _need_n_pos:
            n_pos = n_position_channels if n_position_channels is not None else x.shape[1] // 2
        else:
            n_pos = 0

        if angle_input_offset_augment_deg > 0 and n_pos > 0:
            half_rad = math.radians(angle_input_offset_augment_deg)
            off = (
                torch.rand(x.shape[0], n_pos, 1, device=x.device, dtype=x.dtype) * 2.0 - 1.0
            ) * half_rad
            x = x.clone()
            x[:, :n_pos, :] = x[:, :n_pos, :] + off

        if angle_jitter_std > 0:
            if n_pos > 0:
                noise = torch.randn(
                    x.shape[0], n_pos, x.shape[2], device=x.device, dtype=x.dtype
                ) * angle_jitter_std
                noisy_pos = x[:, :n_pos, :] + noise
                if correlated_vel_noise:
                    # Recompute velocity from noisy positions (finite difference, no LPF).
                    # Mimics cascade: IMU angle → finite_diff → velocity, so noise in
                    # velocity is directly derived from angle perturbation.
                    diff = noisy_pos[:, :, 1:] - noisy_pos[:, :, :-1]   # (B, n_pos, W-1)
                    # Replicate first frame so output length stays W.
                    noisy_vel = torch.cat([diff[:, :, :1], diff], dim=2) * sample_rate_hz
                    # Rebuild input: replace first 2*n_pos channels; keep any extra.
                    rest = x[:, 2 * n_pos:, :]
                    x = torch.cat([noisy_pos, noisy_vel, rest], dim=1) if rest.shape[1] > 0 else torch.cat([noisy_pos, noisy_vel], dim=1)
                else:
                    x = x.clone()
                    x[:, :n_pos, :] = noisy_pos

        if input_noise_std > 0:
            x = x + torch.randn_like(x) * input_noise_std

        # ---- Forward + loss ----
        pred = model(x)
        loss = criterion(pred, y)

        if smoothness_lambda > 0.0:
            delta_pred = pred[:, :, 1:] - pred[:, :, :-1]   # (B, C, W-1)
            loss = loss + smoothness_lambda * (delta_pred ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, np.ndarray, float, np.ndarray]:
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_pred, all_true = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        running_loss += loss.item()
        n_batches += 1
        all_pred.append(pred.cpu())
        all_true.append(y.cpu())

    avg_loss = running_loss / max(n_batches, 1)

    all_pred = torch.cat(all_pred, dim=0)  # (N, C_out, W)
    all_true = torch.cat(all_true, dim=0)

    per_ch_mse = ((all_pred - all_true) ** 2).mean(dim=(0, 2))  # (C_out,)
    per_ch_rmse = per_ch_mse.sqrt().numpy()

    ss_res = ((all_pred - all_true) ** 2).sum(dim=(0, 2))  # (C_out,)
    t_mean = all_true.mean(dim=(0, 2), keepdim=True)  # (1, C_out, 1)
    ss_tot = ((all_true - t_mean) ** 2).sum(dim=(0, 2))
    per_ch_r2 = torch.where(
        ss_tot > 0,
        1.0 - ss_res / ss_tot,
        torch.full_like(ss_res, float("nan")),
    )
    per_ch_r2_np = per_ch_r2.numpy()

    flat_p = all_pred.reshape(-1)
    flat_t = all_true.reshape(-1)
    ss_res_g = ((flat_p - flat_t) ** 2).sum()
    ss_tot_g = ((flat_t - flat_t.mean()) ** 2).sum()
    if ss_tot_g.item() > 0:
        r2_global = float((1.0 - ss_res_g / ss_tot_g).item())
    else:
        r2_global = float("nan")

    return avg_loss, per_ch_rmse, r2_global, per_ch_r2_np
