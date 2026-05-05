"""UNet architecture for conditional flow-matching velocity prediction."""

import torch
import torch.nn as nn

from cosmo_compression.model import gdn
from cosmo_compression.model.conditioned_attention import ConditionedSelfAttention

def compute_groups(channels: int) -> int:
    """Compute the number of groups for GroupNorm."""
    num_groups = 1
    c = channels
    while c % 2 == 0:
        c //= 2
        num_groups *= 2
    return min(num_groups, 8)

class AdaGN(nn.Module):
    """Adaptive GroupNorm – modulates layer activations with conditioning latent."""

    def __init__(self, num_channels: int, num_groups: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_channels=num_channels, num_groups=num_groups)

    def forward(
        self,
        x: torch.Tensor,
        t_s: torch.Tensor | None = None,
        t_b: torch.Tensor | None = None,
        z_s: torch.Tensor | None = None,
        z_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t_s is None or t_b is None:
            return self.gn(x)
        if z_s is None or z_b is None:
            return t_s[:, :, None, None] * self.gn(x) + t_b[:, :, None, None]
        return (
            z_s[:, :, None, None]
            * (t_s[:, :, None, None] * self.gn(x) + t_b[:, :, None, None])
            + z_b[:, :, None, None]
        )


class SelfAttention(nn.Module):
    """Multi-head self-attention block for spatial feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 1, batch_first=True)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_seq = x.view(B, self.channels, H * W).swapaxes(1, 2)
        attn_out, _ = self.mha(x_seq, x_seq, x_seq)
        x_seq = x_seq + attn_out
        x_seq = self.ff_self(x_seq) + x_seq
        return x_seq.swapaxes(2, 1).view(B, self.channels, H, W)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1),
        nn.PixelShuffle(r),
    )


class UpsamplingUNetConv(nn.Module):
    """Two convolutions (second is sub-pixel upsampling) with AdaGN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        int_channels: int | None = None,
        residual: bool = False,
        time_dim: int | None = 256,
        latent_vec_dim: int | None = None,
    ):
        super().__init__()
        self.residual = residual
        if not int_channels:
            int_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, int_channels, kernel_size=3, padding=1)
        self.gn_1 = AdaGN(num_channels=int_channels, num_groups=compute_groups(int_channels))
        self.gelu = nn.GELU()
        self.conv2 = subpel_conv3x3(in_ch=int_channels, out_ch=out_channels, r=2)
        self.gn_2 = AdaGN(num_channels=out_channels, num_groups=compute_groups(out_channels))

        # Residual skip path: upsample 2x to match sub-pixel conv, project channels if needed.
        if residual:
            skip_layers: list[nn.Module] = [nn.Upsample(scale_factor=2, mode="nearest")]
            if in_channels != out_channels:
                skip_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
            self.skip_proj = nn.Sequential(*skip_layers)
        else:
            self.skip_proj = None

        self.t_scale_proj_1 = nn.Linear(time_dim, int_channels) if time_dim is not None else None
        self.t_bias_proj_1 = nn.Linear(time_dim, int_channels) if time_dim is not None else None
        self.t_scale_proj_2 = nn.Linear(time_dim, out_channels) if time_dim is not None else None
        self.t_bias_proj_2 = nn.Linear(time_dim, out_channels) if time_dim is not None else None

        self.z_scale_proj_1 = nn.Linear(latent_vec_dim, int_channels) if latent_vec_dim is not None else None
        self.z_bias_proj_1 = nn.Linear(latent_vec_dim, int_channels) if latent_vec_dim is not None else None
        self.z_scale_proj_2 = nn.Linear(latent_vec_dim, out_channels) if latent_vec_dim is not None else None
        self.z_bias_proj_2 = nn.Linear(latent_vec_dim, out_channels) if latent_vec_dim is not None else None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_s1 = self.t_scale_proj_1(t) if self.t_scale_proj_1 is not None and t is not None else None
        t_b1 = self.t_bias_proj_1(t) if self.t_bias_proj_1 is not None and t is not None else None
        t_s2 = self.t_scale_proj_2(t) if self.t_scale_proj_2 is not None and t is not None else None
        t_b2 = self.t_bias_proj_2(t) if self.t_bias_proj_2 is not None and t is not None else None

        z_s1 = self.z_scale_proj_1(z) if self.z_scale_proj_1 is not None and z is not None else None
        z_b1 = self.z_bias_proj_1(z) if self.z_bias_proj_1 is not None and z is not None else None
        z_s2 = self.z_scale_proj_2(z) if self.z_scale_proj_2 is not None and z is not None else None
        z_b2 = self.z_bias_proj_2(z) if self.z_bias_proj_2 is not None and z is not None else None

        identity = x
        x = self.conv1(x)
        x = self.gn_1(x, t_s1, t_b1, z_s1, z_b1)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gn_2(x, t_s2, t_b2, z_s2, z_b2)
        if self.residual and self.skip_proj is not None:
            x = x + self.skip_proj(identity)
        return self.gelu(x)


class UNetConv(nn.Module):
    """Two circular-padded convolutions with AdaGN. Basic UNet building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int | None,
        int_channels: int | None = None,
        residual: bool = False,
        latent_vec_dim: int | None = None,
    ):
        super().__init__()
        self.residual = residual
        if not int_channels:
            int_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels, int_channels, kernel_size=3, padding=1, bias=False, padding_mode="circular",
        )
        self.gn_1 = AdaGN(num_channels=int_channels, num_groups=compute_groups(int_channels))
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(
            int_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode="circular",
        )
        self.gn_2 = AdaGN(num_channels=out_channels, num_groups=compute_groups(out_channels))

        # Residual skip path: project channels via 1x1 conv if needed.
        if residual:
            self.skip_proj: nn.Module = (
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
                if in_channels != out_channels
                else nn.Identity()
            )
        else:
            self.skip_proj = None  # type: ignore[assignment]

        self.t_scale_proj_1 = nn.Linear(time_dim, int_channels) if time_dim is not None else None
        self.t_bias_proj_1 = nn.Linear(time_dim, int_channels) if time_dim is not None else None
        self.t_scale_proj_2 = nn.Linear(time_dim, out_channels) if time_dim is not None else None
        self.t_bias_proj_2 = nn.Linear(time_dim, out_channels) if time_dim is not None else None

        self.z_scale_proj_1 = nn.Linear(latent_vec_dim, int_channels) if latent_vec_dim is not None else None
        self.z_bias_proj_1 = nn.Linear(latent_vec_dim, int_channels) if latent_vec_dim is not None else None
        self.z_scale_proj_2 = nn.Linear(latent_vec_dim, out_channels) if latent_vec_dim is not None else None
        self.z_bias_proj_2 = nn.Linear(latent_vec_dim, out_channels) if latent_vec_dim is not None else None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_s1 = self.t_scale_proj_1(t) if self.t_scale_proj_1 is not None and t is not None else None
        t_b1 = self.t_bias_proj_1(t) if self.t_bias_proj_1 is not None and t is not None else None
        t_s2 = self.t_scale_proj_2(t) if self.t_scale_proj_2 is not None and t is not None else None
        t_b2 = self.t_bias_proj_2(t) if self.t_bias_proj_2 is not None and t is not None else None

        z_s1 = self.z_scale_proj_1(z) if self.z_scale_proj_1 is not None and z is not None else None
        z_b1 = self.z_bias_proj_1(z) if self.z_bias_proj_1 is not None and z is not None else None
        z_s2 = self.z_scale_proj_2(z) if self.z_scale_proj_2 is not None and z is not None else None
        z_b2 = self.z_bias_proj_2(z) if self.z_bias_proj_2 is not None and z is not None else None

        identity = x
        x = self.conv1(x)
        x = self.gn_1(x, t_s1, t_b1, z_s1, z_b1)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gn_2(x, t_s2, t_b2, z_s2, z_b2)
        if self.residual and self.skip_proj is not None:
            x = x + self.skip_proj(identity)
        return self.gelu(x)


class DownStep(nn.Module):
    """Downscaling: max-pool → double UNetConv → GDN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        int_channels: int | None = None,
        time_dim: int = 256,
    ):
        super().__init__()
        int_channels = int_channels if int_channels else in_channels

        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.conv1 = UNetConv(in_channels=in_channels, out_channels=int_channels, time_dim=time_dim, residual=True)
        self.conv2 = UNetConv(in_channels=int_channels, out_channels=out_channels, time_dim=time_dim, residual=True)
        self.gdn_layer = gdn.GDN(ch=out_channels, device="cuda")

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.gdn_layer(self.conv2(self.conv1(self.pooling(x), t, z), t, z))


class UpStep(nn.Module):
    """Upsample latent and incorporate skip-connection residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_channels: int,
        time_dim: int = 256,
    ):
        super().__init__()
        self.conv1 = UpsamplingUNetConv(
            in_channels=in_channels, int_channels=in_channels,
            out_channels=in_channels, residual=True, time_dim=time_dim,
        )
        self.conv2 = UNetConv(
            in_channels=in_channels + res_channels,
            int_channels=(in_channels + res_channels) // 2,
            out_channels=out_channels, time_dim=time_dim,
        )
        self.gdn_layer = gdn.GDN(ch=out_channels, device="cuda", inverse=True)

    def forward(
        self,
        x: torch.Tensor,
        res_x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.conv1(x, t, z)
        x = torch.cat([res_x, x], dim=1)
        x = self.conv2(x, t, z)
        return self.gdn_layer(x)


class UpStepWoutRes(nn.Module):
    """Upsample without skip connection."""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_dim: int = 256,
    ):
        super().__init__()
        self.conv1 = UNetConv(
            in_channels=in_channels, 
            out_channels=in_channels, 
            time_dim=None, 
            latent_vec_dim=None,
        )
        self.conv2 = UpsamplingUNetConv(
            in_channels=in_channels, int_channels=in_channels // 2,
            out_channels=out_channels, residual=True, time_dim=None, latent_vec_dim=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DepthwiseUpStep(nn.Module):
    """Upsample spatial resolution by 2x while strictly preserving channel independence."""

    def __init__(self, channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # Depthwise convolution: groups=channels
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
            padding_mode="circular",
        )
        # InstanceNorm equivalent
        self.norm = nn.GroupNorm(num_groups=channels, num_channels=channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class UNet(nn.Module):
    """Conditional UNet for velocity-field prediction in flow matching.

    Args:
        n_channels: Number of input/output image channels.
        time_dim: Dimensionality of the sinusoidal timestep embedding.
        latent_img_channels: Number of encoder output channels in the spatial latent.
        latent_vec_dim: Dimensionality of the pooled vector latent.
        use_temporal_masking: If True, apply time-dependent channel masking.
        channel_mults: Tuple of channel sizes for the 5 down-sampling stages.
    """

    def __init__(
        self,
        n_channels: int,
        time_dim: int = 256,
        latent_img_channels: int = 32,
        latent_vec_dim: int = 14 * 9,
        use_temporal_masking: bool = True,
        channel_mults: tuple[int, int, int, int, int] = (64, 128, 256, 512, 512),
        latent_upsample_steps: int = 4,
        conditioned_attention: bool = False,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.n_channels = n_channels
        self.num_latent_channels = latent_img_channels
        self.use_temporal_masking = use_temporal_masking
        self.conditioned_attention = conditioned_attention

        # Unpack configurable channel sequence
        c0, c1, c2, c3, c4 = channel_mults

        self.dropout = nn.Dropout2d(p=0.1)
        self.inc = UNetConv(
            in_channels=n_channels, 
            out_channels=c0, 
            time_dim=time_dim, 
            residual=True,
            latent_vec_dim=latent_vec_dim,
        )

        self.down1 = DownStep(in_channels=c0 + self.num_latent_channels, out_channels=c1, time_dim=time_dim)
        self.down2 = DownStep(in_channels=c1, out_channels=c2, time_dim=time_dim)
        self.sa2 = ConditionedSelfAttention(channels=c2, time_dim=time_dim) if conditioned_attention else SelfAttention(channels=c2)
        self.down3 = DownStep(in_channels=c2, out_channels=c3, time_dim=time_dim)
        self.sa3 = ConditionedSelfAttention(channels=c3, time_dim=time_dim) if conditioned_attention else SelfAttention(channels=c3)
        self.down4 = DownStep(in_channels=c3, out_channels=c4, time_dim=time_dim)

        self.up0 = UpStep(in_channels=c4, res_channels=c4, out_channels=c2, time_dim=time_dim)
        self.up1 = UpStep(in_channels=c2, res_channels=c2, out_channels=c2, time_dim=time_dim)
        self.sa1_inv = ConditionedSelfAttention(channels=c2, time_dim=time_dim) if conditioned_attention else SelfAttention(channels=c2)
        self.up2 = UpStep(in_channels=c2, res_channels=c1, out_channels=c1, time_dim=time_dim)
        self.up3 = UpStep(
            in_channels=c1, res_channels=c0 + self.num_latent_channels, out_channels=c0, time_dim=time_dim,
        )

        self.outc = nn.Conv2d(in_channels=c0, out_channels=n_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.latent_upsampler = nn.Sequential(*[
            DepthwiseUpStep(channels=self.num_latent_channels)
            for _ in range(latent_upsample_steps)
        ])

        self.latent_vec_dim = latent_vec_dim
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_latent_channels, latent_vec_dim)

    def pos_encoding(self, t: torch.Tensor, channels: int) -> torch.Tensor:
        """Generate sinusoidal timestep embedding."""
        device = t.device
        t = t * 1000.0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Noisy image tensor ``[B, C, H, W]``.
            t: Flow-matching timestep ``[B]`` or ``[B, 1]``.
            z: Spatial latent from encoder ``[B, C_latent, H_latent, W_latent]``.
        """
        assert isinstance(t, torch.Tensor)
        assert isinstance(z, torch.Tensor)
        spatial = z.clone()  # [B, C, H, W]
        B, C, _, _ = spatial.shape

        t = t.unsqueeze(-1) if t.dim() == 1 else t
        if t.shape[0] != B:
            t = t.expand(B, t.shape[0])

        # ── Time-dependent channel masking ──────────────────────────
        if self.use_temporal_masking:
            for b in range(B):
                t_val = float(t[b])
                num_mask = int(C * t_val)
                unmasked = C - num_mask
                if num_mask > 0:
                    spatial[b, unmasked :, ...] = 0

        # ── Vector latent from spatial ──────────────────────────────
        vec_latent = self.fc(self.pool(spatial).squeeze(-1).squeeze(-1))
        if vec_latent.dim() == 1:
            vec_latent = vec_latent.unsqueeze(0)

        t = self.pos_encoding(t, self.time_dim)

        # ── Encoder path ────────────────────────────────────────────
        x1 = self.inc(x, t, vec_latent)
        x1 = torch.cat([self.latent_upsampler(spatial), x1], dim=1)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3, t) if self.conditioned_attention else self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4, t) if self.conditioned_attention else self.sa3(x4)
        x5 = self.down4(x4, t)

        # ── Decoder path ────────────────────────────────────────────
        x = self.up0(x5, x4, t)
        x = self.up1(x, x3, t)
        x = self.sa1_inv(x, t) if self.conditioned_attention else self.sa1_inv(x)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)

        return self.outc(x)