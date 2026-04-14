"""Shared TCN training loop helpers (IK→ID and IMU sagittal trainers)."""

from __future__ import annotations

import random
from typing import Any, Optional, Tuple

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
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if angle_jitter_std > 0:
            n_pos = n_position_channels
            if n_pos is None:
                n_pos = x.shape[1] // 2
            if n_pos > 0:
                x[:, :n_pos, :] += (
                    torch.randn(
                        x.shape[0], n_pos, x.shape[2], device=x.device, dtype=x.dtype
                    )
                    * angle_jitter_std
                )
        if input_noise_std > 0:
            x = x + torch.randn_like(x) * input_noise_std
        pred = model(x)
        loss = criterion(pred, y)

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
