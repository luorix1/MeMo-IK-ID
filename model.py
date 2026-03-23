"""
Temporal Convolutional Network (TCN) for joint moment prediction.

Architecture:
  - Stack of residual TCN blocks with exponentially increasing dilation
  - Each block: two dilated causal Conv1d + BatchNorm + ReLU + Dropout
  - 1x1 skip-connection when channel counts differ
  - Final 1x1 Conv1d projects to output channels

With kernel_size=3 and dilations [1,2,4,8,16,32,64], the receptive field is
1 + 2*(1+2+4+8+16+32+64) = 255, which covers a 200-frame window.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CausalConv1d(nn.Module):
    """Conv1d with causal (left) padding so output length == input length."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Residual block:
      x -> CausalConv -> BN -> ReLU -> Dropout
        -> CausalConv -> BN -> ReLU -> Dropout
      + residual (with optional 1x1 projection)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Full TCN model.

    Args:
        n_input_channels: number of input features per timestep (e.g. 46)
        n_output_channels: number of output features per timestep (e.g. 23)
        hidden_channels: width of hidden TCN layers
        n_blocks: number of residual TCN blocks (dilation doubles each block)
        kernel_size: temporal kernel size (default 3)
        dropout: dropout rate
    """

    def __init__(
        self,
        n_input_channels: int = 46,
        n_output_channels: int = 23,
        hidden_channels: int = 64,
        n_blocks: int = 7,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels

        layers = []
        for i in range(n_blocks):
            dilation = 2 ** i
            in_ch = n_input_channels if i == 0 else hidden_channels
            layers.append(TCNBlock(in_ch, hidden_channels, kernel_size, dilation, dropout))

        self.tcn = nn.Sequential(*layers)
        self.output_proj = nn.Conv1d(hidden_channels, n_output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_input_channels, seq_len)
        Returns:
            (batch, n_output_channels, seq_len)
        """
        h = self.tcn(x)
        return self.output_proj(h)

    @property
    def receptive_field(self) -> int:
        n_blocks = len(self.tcn)
        k = self.tcn[0].conv1.conv.kernel_size[0]
        return 1 + 2 * sum(2 ** i for i in range(n_blocks)) * (k - 1) // (k - 1)
        # Simplified: 1 + 2 * (2^n_blocks - 1)
