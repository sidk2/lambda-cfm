"""Diffusion Transformer (DiT) backbone for conditional flow-matching velocity prediction.

Replaces UNet as the velocity model in FlowMatching.  The interface is identical:
    forward(x, t, z) -> velocity tensor of the same shape as x

Conditioning strategy (following adaLN-Zero from Peebles & Xie 2023):
  * Timestep  → sinusoidal embedding → MLP → hidden_size vector  (t_emb)
  * Spatial latent z [B, C, H_z, W_z]
      - optional time-dependent channel masking (same as UNet)
      - adaptive avg-pool → flatten → Linear → hidden_size vector  (z_emb)
  * c = t_emb + z_emb  (fused conditioning token used in every DiTBlock)

Positional encoding: fixed 2-D sin-cos, not learned (same as original DiT).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: x * (1 + scale) + shift  (broadcast over sequence dim)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Return (grid_size**2, embed_dim) fixed 2-D sin-cos positional embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # (grid_size, grid_size) each
    grid = np.stack([grid_w, grid_h], axis=0).reshape(2, 1, grid_size, grid_size)

    assert embed_dim % 2 == 0
    half = embed_dim // 2
    emb_h = _1d_sincos(half, grid[0].reshape(-1))  # (H*W, D/2)
    emb_w = _1d_sincos(half, grid[1].reshape(-1))  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def _1d_sincos(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Scalar timestep → hidden_size vector via sinusoidal + MLP."""

    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.freq_embed_size = freq_embed_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def sinusoidal(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t can be [B] or [B, 1]
        t = t.squeeze(-1) if t.dim() == 2 else t
        return self.mlp(self.sinusoidal(t, self.freq_embed_size))


class PatchEmbed(nn.Module):
    """Image → sequence of patch tokens (no class token)."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → [B, N, D]
        return self.proj(x).flatten(2).transpose(1, 2)


class DiTBlock(nn.Module):
    """Single DiT block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        # Projects conditioning vector c → 6 * hidden_size (shift/scale/gate × 2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final adaLN + linear that maps tokens back to pixel patches."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """Diffusion Transformer velocity model for conditional flow matching.

    Drop-in replacement for ``UNet`` – exposes the same ``forward(x, t, z)``
    signature so it can be used with ``FlowMatching`` and ``CosmoFlow`` without
    any changes to the training harness.

    Args:
        n_channels: Number of input/output image channels (usually 1).
        img_size: Spatial resolution of the noisy input ``x`` (e.g. 256).
        patch_size: Side length of each square patch token.
        hidden_size: Transformer width (embedding dimension).
        depth: Number of DiTBlock layers.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden-dim expansion factor inside each block.
        latent_img_channels: Number of channels in the encoder spatial latent ``z``.
        use_temporal_masking: If True, apply the same time-dependent channel
            masking as the UNet before projecting the latent.
    """

    def __init__(
        self,
        n_channels: int = 1,
        img_size: int = 256,
        patch_size: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        latent_img_channels: int = 32,
        use_temporal_masking: bool = True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.latent_img_channels = latent_img_channels
        self.use_temporal_masking = use_temporal_masking

        num_patches = (img_size // patch_size) ** 2

        # ── Input side ────────────────────────────────────────────────────────
        self.x_embedder = PatchEmbed(img_size, patch_size, n_channels, hidden_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # ── Conditioning ──────────────────────────────────────────────────────
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Pool the spatial latent to a vector, then project to hidden_size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.z_proj = nn.Sequential(
            nn.Linear(latent_img_channels, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # ── Output ────────────────────────────────────────────────────────────
        self.final_layer = FinalLayer(hidden_size, patch_size, n_channels)

        self._initialize_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_weights(self) -> None:
        def _basic_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_basic_init)

        # Fixed sin-cos positional embedding
        grid_size = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Patch projection like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj.bias)

        # Timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation (gate init = 0 → identity at start)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    # ------------------------------------------------------------------
    # Patch helpers
    # ------------------------------------------------------------------

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N, patch**2 * C) → (B, C, H, W)."""
        c = self.n_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], "Sequence length must be a perfect square."
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the velocity field.

        Args:
            x: Noisy image ``[B, C, H, W]``.
            t: Flow-matching timestep ``[B]`` or ``[B, 1]`` in [0, 1].
            z: Spatial latent from encoder ``[B, C_latent, H_z, W_z]``.

        Returns:
            Velocity tensor ``[B, C, H, W]``.
        """
        assert t is not None and z is not None

        B, C_z, _, _ = z.shape
        spatial = z.clone()

        # Normalise t to [B]
        t_scalar = t.squeeze(-1) if t.dim() == 2 else t
        if t_scalar.shape[0] != B:
            t_scalar = t_scalar.expand(B)

        # ── Time-dependent channel masking ────────────────────────────────────
        if self.use_temporal_masking:
            for b in range(B):
                t_val = float(t_scalar[b])
                num_mask = int(C_z * t_val)
                unmasked = C_z - num_mask
                if num_mask > 0:
                    spatial[b, unmasked:] = 0.0

        # ── Conditioning vector c = t_emb + z_emb ────────────────────────────
        t_emb = self.t_embedder(t_scalar)                                # [B, D]
        z_vec = self.pool(spatial).flatten(1)                            # [B, C_z]
        z_emb = self.z_proj(z_vec)                                       # [B, D]
        c = t_emb + z_emb                                                # [B, D]

        # ── Patch embedding + positional encoding ─────────────────────────────
        x = self.x_embedder(x) + self.pos_embed                         # [B, N, D]

        # ── Transformer ───────────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x, c)

        # ── Decode back to pixels ─────────────────────────────────────────────
        x = self.final_layer(x, c)                                       # [B, N, p*p*C]
        return self.unpatchify(x)                                         # [B, C, H, W]
