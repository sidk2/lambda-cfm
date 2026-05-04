"""ResNet encoder for compressing cosmological map data into spatial latents."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    """Basic building block for ResNet with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False, padding_mode="circular",
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False, padding_mode="circular",
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False, padding_mode="circular",
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResNet(nn.Module):
    """ResNet-18 style feature extractor (256×256 → 16×16 spatial latent)."""

    def __init__(self, in_channels: int, latent_img_channels: int = 32):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(
                in_channels, 64, kernel_size=3,
                stride=1, padding=1, bias=False, padding_mode="circular",
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.resnet_layers = nn.ModuleList([
            self._make_layer(64, 64, num_blocks=1, stride=1),
            self._make_layer(64, 64, num_blocks=1, stride=1),
            self._make_layer(64, 128, num_blocks=1, stride=2),
            self._make_layer(128, 128, num_blocks=1, stride=2),
            self._make_layer(128, 256, num_blocks=1, stride=1),
            self._make_layer(256, 256, num_blocks=1, stride=2),
        ])
        self.out_conv = nn.Conv2d(
            256, latent_img_channels, kernel_size=3,
            stride=1, padding=1, bias=False, padding_mode="circular",
        )

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, num_blocks: int, stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = [ResnetBlock(in_channels, out_channels, s) for s in strides]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_layer(x)
        for layer in self.resnet_layers:
            x = layer(x)
        return self.out_conv(x)


class ResNetEncoder(nn.Module):
    """Wrapper that stacks one or more ResNet feature extractors."""

    def __init__(self, in_channels: int, latent_img_channels: int = 32):
        super().__init__()
        self.resnet_list = nn.ModuleList([
            ResNet(in_channels=in_channels, latent_img_channels=latent_img_channels),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_list = [resnet_module(x) for resnet_module in self.resnet_list]
        return torch.cat(spatial_list, dim=1)