"""Implements a UNet"""

import torch
import torch.nn as nn

from cosmo_compression.model import gdn


def compute_groups(channels: int) -> int:
    """Compute the number of groups for GroupNorm"""
    num_groups = 1
    while channels % 2 == 0:
        channels //= 2
        num_groups *= 2

    return min(num_groups, 8)


class AdaGN(nn.Module):
    """
    AdaGN allows model to modulate layer activations with conditioning latent
    """

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
        """Overloads forward method of nn.Module"""

        # Channelwise modulation of the latent by the timestep embedding
        if t_s is None or t_b is None:
            return self.gn(x)
        elif z_s is None or z_b is None:
            return t_s[:, :, None, None] * self.gn(x) + t_b[:, :, None, None]

        else:
            return (
                z_s[:, :, None, None]
                * (t_s[:, :, None, None] * self.gn(x) + t_b[:, :, None, None])
                + z_b[:, :, None, None]
            )


class SelfAttention(nn.Module):
    """Implementation of a self-attention module"""

    def __init__(self, channels: int):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 1, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overloads forward pass of nn.Module"""
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x, _ = self.mha(x, x, x)
        # attention_value = attention_value + x
        x = self.ff_self(x) + x
        return x.swapaxes(2, 1).view(-1, self.channels, size, size)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


class UpsamplingUNetConv(nn.Module):
    """2 sets of convolution plus batch norm. Basic UNet building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        int_channels: int | None = None,
        residual: bool = False,
        time_dim: int = 256,
        latent_vec_dim: int = 14,
    ):
        super().__init__()
        self.residual = residual
        if not int_channels:
            int_channels = out_channels

        self.shortcut = None
        if self.residual:
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                )
            else:
                self.shortcut = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=int_channels, kernel_size=3, padding=1
        )
        self.gn_1 = AdaGN(
            num_channels=int_channels,
            num_groups=compute_groups(int_channels),
        )
        self.gelu = nn.GELU()
        self.conv2 = subpel_conv3x3(in_ch=int_channels, out_ch=out_channels, r=2)
        self.gn_2 = AdaGN(
            num_channels=out_channels,
            num_groups=compute_groups(out_channels),
        )

        self.t_scale_proj_1 = nn.Linear(time_dim, int_channels)
        self.t_bias_proj_1 = nn.Linear(time_dim, int_channels)

        self.t_scale_proj_2 = nn.Linear(time_dim, out_channels)
        self.t_bias_proj_2 = nn.Linear(time_dim, out_channels)

        self.z_scale_proj_1 = nn.Linear(latent_vec_dim, int_channels)
        self.z_bias_proj_1 = nn.Linear(latent_vec_dim, int_channels)

        self.z_scale_proj_2 = nn.Linear(latent_vec_dim, out_channels)
        self.z_bias_proj_2 = nn.Linear(latent_vec_dim, out_channels)

    def forward(self, x: torch.Tensor, t=None, z=None) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        # t is shape [batch_size]

        identity = x

        t_s1 = self.t_scale_proj_1(t) if t is not None else None
        t_b1 = self.t_bias_proj_1(t) if t is not None else None

        t_s2 = self.t_scale_proj_2(t) if t is not None else None
        t_b2 = self.t_bias_proj_2(t) if t is not None else None

        z_s1 = self.z_scale_proj_1(z) if z is not None else None
        z_b1 = self.z_bias_proj_1(z) if z is not None else None

        z_s2 = self.z_scale_proj_2(z) if z is not None else None
        z_b2 = self.z_bias_proj_2(z) if z is not None else None

        out = self.conv1(x)
        out = self.gn_1(out, t_s1, t_b1, z_s1, z_b1)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.gn_2(out, t_s2, t_b2, z_s2, z_b2)
        
        if self.residual:
            out = out + self.shortcut(identity)

        return out


class UNetConv(nn.Module):
    """2 sets of convolution plus batch norm. Basic UNet building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        int_channels: int | None = None,
        residual: bool = False,
        latent_vec_dim: int = 14 * 9,
    ):
        super().__init__()
        self.residual = residual
        if not int_channels:
            int_channels = out_channels

        self.shortcut = None
        if self.residual:
            if in_channels != out_channels:
                self.shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    padding_mode="circular",
                )
            else:
                self.shortcut = nn.Identity()

        self.conv1 = nn.Conv2d(
            in_channels,
            int_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            padding_mode="circular",
        )
        self.gn_1 = AdaGN(
            num_channels=int_channels,
            num_groups=compute_groups(int_channels),
        )
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(
            int_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            padding_mode="circular",
        )
        self.gn_2 = AdaGN(
            num_channels=out_channels,
            num_groups=compute_groups(out_channels),
        )

        self.t_scale_proj_1 = nn.Linear(time_dim, int_channels)
        self.t_bias_proj_1 = nn.Linear(time_dim, int_channels)

        self.t_scale_proj_2 = nn.Linear(time_dim, out_channels)
        self.t_bias_proj_2 = nn.Linear(time_dim, out_channels)

        self.z_scale_proj_1 = nn.Linear(latent_vec_dim, int_channels)
        self.z_bias_proj_1 = nn.Linear(latent_vec_dim, int_channels)

        self.z_scale_proj_2 = nn.Linear(latent_vec_dim, out_channels)
        self.z_bias_proj_2 = nn.Linear(latent_vec_dim, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        # t is shape [batch_size]

        identity = x

        t_s1 = self.t_scale_proj_1(t) if t is not None else None
        t_b1 = self.t_bias_proj_1(t) if t is not None else None

        t_s2 = self.t_scale_proj_2(t) if t is not None else None
        t_b2 = self.t_bias_proj_2(t) if t is not None else None

        z_s1 = self.z_scale_proj_1(z) if z is not None else None
        z_b1 = self.z_bias_proj_1(z) if z is not None else None

        z_s2 = self.z_scale_proj_2(z) if z is not None else None
        z_b2 = self.z_bias_proj_2(z) if z is not None else None

        out = self.conv1(x)
        out = self.gn_1(out, t_s1, t_b1, z_s1, z_b1)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.gn_2(out, t_s2, t_b2, z_s2, z_b2)
        
        if self.residual:
            out = out + self.shortcut(identity)

        return out


class DownStep(nn.Module):
    """Downscaling input with max pool and double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        int_channels: int | None = None,
        time_dim: int = 256,
    ):
        super(
            DownStep,
            self,
        ).__init__()
        int_channels = int_channels if int_channels else in_channels

        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.conv1 = UNetConv(
            in_channels=in_channels,
            out_channels=int_channels,
            time_dim=time_dim,
            residual=True,
        )
        self.conv2 = UNetConv(
            in_channels=int_channels,
            out_channels=out_channels,
            time_dim=time_dim,
            residual=True,
        )
        self.gdn_layer = gdn.GDN(ch=out_channels, device="cuda")

    def forward(self, x: torch.Tensor, t, z=None) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        return self.gdn_layer(self.conv2(self.conv1(self.pooling(x), t, z), t, z))


class UpStep(nn.Module):
    """Upsample latent and incorporate residual"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_channels: int,
        time_dim: int = 256,
    ):
        super().__init__()
        self.conv1 = UpsamplingUNetConv(
            in_channels=in_channels,
            int_channels=in_channels,
            out_channels=in_channels,
            residual=True,
            time_dim=time_dim,
        )
        self.conv2 = UNetConv(
            in_channels=(in_channels + res_channels),
            int_channels=(in_channels + res_channels) // 2,
            out_channels=out_channels,
            time_dim=time_dim,
        )

        self.gdn_layer = gdn.GDN(ch=out_channels, device="cuda", inverse=True)

    def forward(self, x: torch.Tensor, res_x: torch.Tensor, t, z=None) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        x = self.conv1(x, t, z)
        x = torch.cat([res_x, x], dim=1)
        x = self.conv2(x, t, z)
        return self.gdn_layer(x)


class UpStepWoutRes(nn.Module):
    """Upsample latent and incorporate residual"""

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
            time_dim=time_dim,
        )

        self.conv2 = UpsamplingUNetConv(
            in_channels=in_channels,
            int_channels=in_channels // 2,
            out_channels=out_channels,
            residual=True,
            time_dim=time_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    """Creates a UNet using the building block modules in this file"""

    def __init__(
        self,
        n_channels: int,
        time_dim: int = 256,
        latent_img_channels: int = 32,
        latent_vec_dim: int = 14 * 9,
        use_temporal_masking: bool = True,
    ):
        super(UNet, self).__init__()
        self.time_dim = time_dim
        self.n_channels = n_channels
        self.num_latent_channels = latent_img_channels
        self.use_temporal_masking = use_temporal_masking

        self.dropout = nn.Dropout2d(p=0.1)
        self.inc = UNetConv(
            in_channels=n_channels,
            out_channels=64,
            time_dim=time_dim,
            residual=True,
        )
        # Downsampling stages
        down_channels = [64 + self.num_latent_channels, 128, 256, 512, 512]
        self.downs = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            self.downs.append(
                DownStep(
                    in_channels=down_channels[i],
                    out_channels=down_channels[i + 1],
                    time_dim=time_dim,
                )
            )
            # Add self-attention after out_channels=256 and out_channels=512 (first instance)
            if down_channels[i + 1] in [256, 512] and i < len(down_channels) - 2:
                self.downs.append(SelfAttention(channels=down_channels[i + 1]))

        # Upsampling stages
        up_in_ch = [512, 256, 256, 128]
        up_res_ch = [512, 256, 128, 64 + self.num_latent_channels]
        up_out_ch = [256, 256, 128, 64]
        
        self.ups = nn.ModuleList()
        for i in range(len(up_in_ch)):
            self.ups.append(
                UpStep(
                    in_channels=up_in_ch[i],
                    res_channels=up_res_ch[i],
                    out_channels=up_out_ch[i],
                    time_dim=time_dim,
                )
            )
            # Add self-attention after up0 and up1
            if i in [0, 1]:
                self.ups.append(SelfAttention(channels=up_out_ch[i]))

        self.outc = nn.Conv2d(
            in_channels=64,
            out_channels=n_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.latent_upsampler_0 = nn.Sequential(
            *[
                UpStepWoutRes(
                    in_channels=int(self.num_latent_channels),
                    out_channels=int(self.num_latent_channels),
                    time_dim=time_dim,
                )
                for _ in range(4)
            ]
        )

        self.latent_vec_dim = latent_vec_dim

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_latent_channels, 14 * 9)

    def pos_encoding(self, t: int, channels: int) -> torch.Tensor:
        """Generate sinusoidal timestep embedding"""
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module
        t is the full timestep embedding, with dimension time_dim
        z is the full latent, which will be split into latent_dim chunks
        """

        spatial = z  # [B, C, H, W]
        B, C, H, W = spatial.shape

        t = t.unsqueeze(-1)
        if t.shape[0] != B:
            t = t.expand(B, t.shape[0])

        if self.use_temporal_masking:
            for b in range(B):
                t_val = float(t[b])
                num_mask = int(C * t_val)
                unmasked = C - num_mask

                # index of the last channel we want to keep gradients on
                last_unmasked = unmasked - 1
                # zero-out everything *after* last_unmasked
                if num_mask > 0:
                    spatial[b, last_unmasked + 1 :, ...] = 0

        # --- End of masking ---

        vec_latent = self.fc(self.pool(spatial).squeeze())
        if vec_latent.dim() == 1:
            vec_latent = vec_latent.unsqueeze(0)

        t = self.pos_encoding(t, self.time_dim)

        # Downsampling stages
        x1 = self.inc(x, t, vec_latent)
        x1 = torch.cat(
            [
                self.latent_upsampler_0(spatial),
                x1,
            ],
            dim=1,
        )
        skips = [x1]
        x_stage = x1
        
        for layer in self.downs:
            if isinstance(layer, DownStep):
                x_stage = layer(x_stage, t)
                skips.append(x_stage)
            elif isinstance(layer, SelfAttention):
                x_stage = layer(x_stage)
                skips[-1] = x_stage

        # Upsampling stages
        x_stage = skips.pop()  # This gets x5
        for layer in self.ups:
            if isinstance(layer, UpStep):
                skip_x = skips.pop()
                x_stage = layer(x_stage, skip_x, t)
            elif isinstance(layer, SelfAttention):
                x_stage = layer(x_stage)

        return self.outc(x_stage)
