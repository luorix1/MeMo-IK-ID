"""
Models for joint moment prediction from joint angles and angular velocities.

TCN
---
Temporal Convolutional Network with causal (left-padded) dilated convolutions.
Suitable for online/streaming inference.  Receptive field grows exponentially
with n_blocks, so keep n_blocks small relative to window_size.

TransformerMoment
-----------------
Bidirectional Transformer encoder with global self-attention over the full
input window.  Not causal — requires the full sequence before producing output.
Use this when offline accuracy matters more than real-time usability.  Avoids
the TCN receptive-field / window-size mismatch that causes cascade failures
when n_blocks is large.

Both models share the same input/output convention:
    x : (batch, n_input_channels, seq_len)  →  (batch, n_output_channels, seq_len)
"""

import math

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
        # Each block contributes 2*(k-1)*dilation; dilation=2^i for block i.
        return 1 + sum(2 * (k - 1) * (2 ** i) for i in range(n_blocks))


# ---------------------------------------------------------------------------
# TransformerMoment
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    """
    Fixed sinusoidal positional encoding, pre-computed up to ``max_len``.
    If the input sequence exceeds ``max_len`` at runtime (e.g. full-trial
    inference), the encoding is extended on-the-fly without recomputing the
    full buffer.
    """

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer("pe", self._make_pe(d_model, max_len))

    @staticmethod
    def _make_pe(d_model: int, length: int) -> torch.Tensor:
        pe = torch.zeros(1, length, d_model)
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[: d_model // 2])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.shape[1]
        if T > self.pe.shape[1]:
            # Extend on-the-fly for long sequences (e.g. full-trial inference).
            pe_ext = self._make_pe(self.d_model, T).to(x.device)
            x = x + pe_ext
        else:
            x = x + self.pe[:, :T, :]
        return self.dropout(x)


class TransformerMoment(nn.Module):
    """
    Bidirectional Transformer encoder for joint moment prediction.

    Unlike the causal TCN this model applies **global** self-attention over the
    entire input window, so it is not suitable for real-time streaming.  It
    does not suffer from the TCN receptive-field / training-window mismatch
    that causes cascade failures with large n_blocks.

    Designed to match the GaitDynamics refinement architecture
    (``TransformerEncoderArchitecture``) while fitting the existing training
    pipeline: same ``(B, C_in, T) → (B, C_out, T)`` interface as ``TCN``.

    Args:
        n_input_channels:  Number of input features per timestep (e.g. 6).
        n_output_channels: Number of output features per timestep (e.g. 3).
        d_model:           Internal embedding dimension (default 256).
        n_heads:           Number of attention heads (default 8).
                           Must evenly divide ``d_model``.
        n_layers:          Number of stacked encoder blocks (default 6).
        d_ff:              Feed-forward hidden dimension (default 1024).
        dropout:           Dropout applied inside attention, FFN, and on PE.
        max_seq_len:       Maximum sequence length for positional encoding
                           pre-computation (default 4096 ≈ 40 s at 100 Hz).
    """

    def __init__(
        self,
        n_input_channels: int = 6,
        n_output_channels: int = 3,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        inference_chunk_size: int = 1024,
        inference_overlap: int = 64,
    ) -> None:
        super().__init__()
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.d_model = d_model
        self.inference_chunk_size = inference_chunk_size
        self.inference_overlap = inference_overlap

        self.input_proj = nn.Linear(n_input_channels, d_model)
        self.pos_enc = _SinusoidalPE(d_model, max_seq_len, dropout=0.0)

        # Pre-norm (norm_first=True) is more training-stable for deeper models.
        # enable_nested_tensor is disabled because PyTorch does not support it
        # with norm_first=True; this suppresses the associated UserWarning.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.output_proj = nn.Linear(d_model, n_output_channels)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Core forward pass — expects (B, C_in, T) with manageable T."""
        x = x.permute(0, 2, 1)       # (B, T, C_in)
        x = self.input_proj(x)        # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)           # bidirectional
        x = self.output_proj(x)       # (B, T, C_out)
        return x.permute(0, 2, 1)     # (B, C_out, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_input_channels, seq_len)
        Returns:
            (batch, n_output_channels, seq_len)

        For sequences longer than ``inference_chunk_size`` (set at construction
        time, default 1024), inference is performed with overlapping chunks to
        keep attention memory bounded.  Edges of each chunk are trimmed by
        ``inference_overlap`` frames to reduce boundary artefacts before the
        chunks are stitched.
        """
        T = x.shape[2]
        if T <= self.inference_chunk_size:
            return self._forward_single(x)

        # Chunked inference for long sequences (e.g. full-trial evaluation).
        chunk = self.inference_chunk_size
        overlap = self.inference_overlap
        step = chunk - 2 * overlap

        out_chunks = []
        start = 0
        while start < T:
            end = min(start + chunk, T)
            chunk_in = x[:, :, start:end]
            chunk_out = self._forward_single(chunk_in)  # (B, C_out, end-start)

            # Trim overlap from both sides, except at sequence boundaries.
            trim_left = overlap if start > 0 else 0
            trim_right = overlap if end < T else 0
            c_len = chunk_out.shape[2]
            r = c_len - trim_right if trim_right > 0 else c_len
            out_chunks.append(chunk_out[:, :, trim_left:r])

            start += step
            if start >= T:
                break

        return torch.cat(out_chunks, dim=2)


# ---------------------------------------------------------------------------
# GaussianDiffusion1D  (GaitDynamics-style conditional diffusion)
# ---------------------------------------------------------------------------

class _SinusoidalTimestepEmb(nn.Module):
    """Sinusoidal embedding for integer diffusion timesteps (B,) → (B, dim)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / (half - 1)
        )
        emb = t.float()[:, None] * freqs[None, :]   # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


class _FiLMDense(nn.Module):
    """
    Project a timestep embedding into a (scale, shift) pair for FiLM
    conditioning, following GaitDynamics' DenseFiLM.
    """

    def __init__(self, embed_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Mish(), nn.Linear(embed_dim, out_dim * 2))

    def forward(self, emb: torch.Tensor):
        out = self.proj(emb).unsqueeze(1)       # (B, 1, out_dim*2)
        scale, shift = out.chunk(2, dim=-1)     # each (B, 1, out_dim)
        return scale, shift


class _FiLMTransformerBlock(nn.Module):
    """
    Bidirectional Transformer encoder block with FiLM timestep conditioning,
    mirroring GaitDynamics' FiLMTransformerDecoderLayer (self-attention only).
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)
        self.film1 = _FiLMDense(d_model * 4, d_model)
        self.film2 = _FiLMDense(d_model * 4, d_model)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model), t_emb: (B, d_model*4)
        scale1, shift1 = self.film1(t_emb)
        h = self.norm1(x)
        h_attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + (1.0 + scale1) * self.drop(h_attn) + shift1

        scale2, shift2 = self.film2(t_emb)
        x = x + (1.0 + scale2) * self.drop(self.ff(self.norm2(x))) + shift2
        return x


class _DiffusionBackbone(nn.Module):
    """
    FiLM-Transformer denoiser: maps (noisy_moments ∥ clean_condition, t) → predicted noise.
    """

    def __init__(
        self,
        n_noisy_ch: int,
        n_cond_ch: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_noisy_ch + n_cond_ch, d_model)
        self.pos_enc = _SinusoidalPE(d_model)
        self.time_emb = nn.Sequential(
            _SinusoidalTimestepEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
        )
        self.blocks = nn.ModuleList(
            [_FiLMTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, n_noisy_ch)

    def forward(
        self, y_noisy: torch.Tensor, condition: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([y_noisy, condition], dim=1)  # (B, C_out+C_in, T)
        x = x.permute(0, 2, 1)                      # (B, T, C_out+C_in)
        x = self.input_proj(x)                       # (B, T, d_model)
        x = self.pos_enc(x)
        t_emb = self.time_emb(t)                     # (B, d_model*4)
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = self.out_norm(x)
        x = self.output_proj(x)                      # (B, T, C_out)
        return x.permute(0, 2, 1)                    # (B, C_out, T)


class GaussianDiffusion1D(nn.Module):
    """
    Conditional DDPM for joint moment generation, adapted from GaitDynamics.

    The condition (joint angles + angular velocities) is always clean.
    Only the moments are noised during training and iteratively denoised at
    inference.

    **Training**: call ``p_losses(y0, condition)`` — returns a scalar MSE loss.

    **Inference**: ``forward(condition)`` runs DDIM sampling (``n_inference_steps``
    deterministic steps) and returns sampled moments.  The input/output
    signature matches TCN and TransformerMoment, so the eval pipeline needs
    no changes.

    Args:
        n_input_channels:   Condition channels (angles + vels, e.g. 6).
        n_output_channels:  Target channels (moments, e.g. 3).
        d_model:            Denoiser embedding dimension (default 256).
        n_heads:            Attention heads (default 8).
        n_layers:           FiLM-Transformer blocks (default 6).
        d_ff:               Feed-forward hidden dim inside each block (default 1024).
        dropout:            Dropout rate (default 0.1).
        n_timesteps:        Total diffusion steps T (default 1000).
        schedule:           Beta schedule: ``"linear"`` or ``"cosine"`` (default).
        predict_epsilon:    Predict noise ε (True, default) or clean x0 directly.
        n_inference_steps:  DDIM steps at inference (default 50).
        inference_chunk_size: Chunk long sequences to bound O(T²) memory.
        inference_overlap:  Overlap frames trimmed from chunk boundaries.
    """

    def __init__(
        self,
        n_input_channels: int = 6,
        n_output_channels: int = 3,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_timesteps: int = 1000,
        schedule: str = "cosine",
        predict_epsilon: bool = True,
        n_inference_steps: int = 50,
        inference_chunk_size: int = 1024,
        inference_overlap: int = 64,
    ) -> None:
        super().__init__()
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.n_timesteps = n_timesteps
        self.predict_epsilon = predict_epsilon
        self.n_inference_steps = n_inference_steps
        self.inference_chunk_size = inference_chunk_size
        self.inference_overlap = inference_overlap

        self.backbone = _DiffusionBackbone(
            n_noisy_ch=n_output_channels,
            n_cond_ch=n_input_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        betas = self._make_betas(schedule, n_timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = torch.cat([torch.ones(1), acp[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("alphas_cumprod_prev", acp_prev)
        self.register_buffer("sqrt_acp", acp.sqrt())
        self.register_buffer("sqrt_one_minus_acp", (1.0 - acp).sqrt())
        self.register_buffer("sqrt_recip_acp", (1.0 / acp).sqrt())
        self.register_buffer("sqrt_recipm1_acp", (1.0 / acp - 1.0).sqrt())

    @staticmethod
    def _make_betas(schedule: str, n: int) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(1e-4, 2e-2, n, dtype=torch.float32)
        if schedule == "cosine":
            steps = torch.arange(n + 1, dtype=torch.float64) / n + 8e-3
            alphas = (steps / (1 + 8e-3) * math.pi / 2).cos().pow(2)
            alphas = alphas / alphas[0]
            betas = 1.0 - alphas[1:] / alphas[:-1]
            return betas.clamp(0.0, 0.999).float()
        raise ValueError(f"Unknown diffusion schedule {schedule!r}. Use 'linear' or 'cosine'.")

    def _gather(self, buf: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        out = buf.gather(0, t)
        return out.view(t.shape[0], *([1] * (ndim - 1)))

    def q_sample(
        self, y0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion: add Gaussian noise to clean moments at step t."""
        if noise is None:
            noise = torch.randn_like(y0)
        s = self._gather(self.sqrt_acp, t, y0.ndim)
        s1m = self._gather(self.sqrt_one_minus_acp, t, y0.ndim)
        return s * y0 + s1m * noise

    def p_losses(
        self,
        y0: torch.Tensor,
        condition: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training loss (MSE on predicted noise or x0).

        Args:
            y0:        (B, C_out, T) clean moments.
            condition: (B, C_in,  T) clean angles + angular velocities.
            t:         (B,) integer timesteps; sampled uniformly if None.
            noise:     (B, C_out, T) Gaussian noise; sampled if None.
        Returns:
            Scalar MSE loss.
        """
        B, device = y0.shape[0], y0.device
        if t is None:
            t = torch.randint(0, self.n_timesteps, (B,), device=device)
        if noise is None:
            noise = torch.randn_like(y0)
        y_t = self.q_sample(y0, t, noise)
        pred = self.backbone(y_t, condition, t)
        target = noise if self.predict_epsilon else y0
        return nn.functional.mse_loss(pred, target)

    def _predict_x0(
        self, y_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor
    ) -> torch.Tensor:
        if not self.predict_epsilon:
            return pred
        recip = self._gather(self.sqrt_recip_acp, t, y_t.ndim)
        recipm1 = self._gather(self.sqrt_recipm1_acp, t, y_t.ndim)
        return (recip * y_t - recipm1 * pred).clamp(-5.0, 5.0)

    @torch.no_grad()
    def _ddim_sample(self, condition: torch.Tensor) -> torch.Tensor:
        """DDIM sampling on a single (B, C_in, T) condition tensor (eta=0)."""
        B, _, T = condition.shape
        device = condition.device
        times = torch.linspace(self.n_timesteps - 1, 0, self.n_inference_steps + 1).long()
        pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))

        y_t = torch.randn(B, self.n_output_channels, T, device=device)
        for t_now, t_next in pairs:
            t_b = torch.full((B,), t_now, device=device, dtype=torch.long)
            pred = self.backbone(y_t, condition, t_b)
            x0 = self._predict_x0(y_t, t_b, pred)
            if t_next < 0:
                return x0
            alpha = self.alphas_cumprod[t_now]
            alpha_next = self.alphas_cumprod[t_next]
            eps = (y_t - alpha.sqrt() * x0) / (1.0 - alpha).sqrt().clamp(min=1e-8)
            y_t = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * eps
        return y_t

    @torch.no_grad()
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Generate moments via DDIM given clean condition.

        Args:
            condition: (B, C_in, T) — identical input format to TCN/TransformerMoment.
        Returns:
            (B, C_out, T) sampled moments.
        """
        T = condition.shape[2]
        if T <= self.inference_chunk_size:
            return self._ddim_sample(condition)

        chunk = self.inference_chunk_size
        overlap = self.inference_overlap
        step = chunk - 2 * overlap
        out_chunks: List[torch.Tensor] = []
        start = 0
        while start < T:
            end = min(start + chunk, T)
            chunk_out = self._ddim_sample(condition[:, :, start:end])
            trim_left = overlap if start > 0 else 0
            trim_right = overlap if end < T else 0
            c_len = chunk_out.shape[2]
            r = c_len - trim_right if trim_right > 0 else c_len
            out_chunks.append(chunk_out[:, :, trim_left:r])
            start += step
            if start >= T:
                break
        return torch.cat(out_chunks, dim=2)
