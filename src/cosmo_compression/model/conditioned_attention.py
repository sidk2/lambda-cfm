"""AdaLN-Zero conditioned self-attention for timestep-aware spatial mixing."""

import torch
import torch.nn as nn


class ConditionedSelfAttention(nn.Module):
    """Multi-head self-attention with AdaLN-Zero timestep conditioning.

    Each sub-block (MHA and FFN) is gated by a learnable, zero-initialized
    scalar so the module starts as identity and gracefully learns when to
    activate spatial mixing at each flow-matching timestep.

    Args:
        channels: Number of input feature-map channels.
        time_dim: Dimensionality of the sinusoidal timestep embedding.
    """

    def __init__(self, channels: int, time_dim: int = 256):
        super().__init__()
        self.channels = channels

        # ---------- attention sub-block ----------
        self.ln_attn = nn.LayerNorm([channels])
        self.mha = nn.MultiheadAttention(channels, 1, batch_first=True)

        # ---------- feed-forward sub-block ----------
        self.ln_ff = nn.LayerNorm([channels])
        self.ff = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        # ---------- AdaLN projections from timestep ----------
        # Each sub-block gets (scale, shift, gate) = 3 * channels
        self.adaLN_proj = nn.Linear(time_dim, 6 * channels)

        # Zero-init so gate starts at 0 → module is identity at init
        nn.init.zeros_(self.adaLN_proj.weight)
        nn.init.zeros_(self.adaLN_proj.bias)

    def _adaln_modulate(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        ln: nn.LayerNorm,
    ) -> torch.Tensor:
        """Apply adaptive layer norm: scale * LN(x) + shift."""
        return scale * ln(x) + shift

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map ``[B, C, H, W]``.
            t: Timestep embedding ``[B, time_dim]``.

        Returns:
            Modulated feature map ``[B, C, H, W]``.
        """
        B, C, H, W = x.shape

        # Project timestep → 6 modulation vectors
        t_mod = self.adaLN_proj(t)  # [B, 6C]
        scale_attn, shift_attn, gate_attn, scale_ff, shift_ff, gate_ff = (
            t_mod.chunk(6, dim=-1)
        )

        # Reshape for sequence operations: [B, C, H, W] → [B, H*W, C]
        x_seq = x.view(B, C, H * W).permute(0, 2, 1)

        # --- Attention sub-block with AdaLN-Zero ---
        x_norm = self._adaln_modulate(
            x_seq, 1 + scale_attn.unsqueeze(1), shift_attn.unsqueeze(1), self.ln_attn,
        )
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x_seq = x_seq + gate_attn.unsqueeze(1) * attn_out

        # --- FFN sub-block with AdaLN-Zero ---
        x_norm = self._adaln_modulate(
            x_seq, 1 + scale_ff.unsqueeze(1), shift_ff.unsqueeze(1), self.ln_ff,
        )
        ff_out = self.ff(x_norm)
        x_seq = x_seq + gate_ff.unsqueeze(1) * ff_out

        # Reshape back: [B, H*W, C] → [B, C, H, W]
        return x_seq.permute(0, 2, 1).view(B, C, H, W)
