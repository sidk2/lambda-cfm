"""
Implements a ResNet, which will be used to encode CMD data for compression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T 

class ResnetBlock(nn.Module):
    """Basic building block for ResNet with residual connection"""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            padding_mode='circular'
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode='circular'
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    padding_mode='circular'
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return F.relu(out)

class ResNet(nn.Module):
    """Residual convolutional network (ResNet 18 architecture)"""

    def __init__(self, in_channels: int, latent_img_channels: int = 32, fc_out_dim: int = 256 * 2):
        super(ResNet, self).__init__()
        # CAMELS Multifield Dataset is 256x256
        self.in_layer = nn.Sequential(
            
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False, padding_mode='circular'
                ),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        self.resnet_layers = nn.ModuleList(
            [
                self._make_layer(in_channels=64, out_channels=64, num_blocks=1, stride=1),
                self._make_layer(in_channels=64, out_channels=64, num_blocks=1, stride=1),
                self._make_layer(in_channels=64, out_channels=128, num_blocks=1, stride=2),
                self._make_layer(in_channels=128, out_channels=128, num_blocks=1, stride=2),
                self._make_layer(in_channels=128, out_channels=256, num_blocks=1, stride=2),
                self._make_layer(in_channels=256, out_channels=256, num_blocks=1, stride=2),
            ]
        )
        
        self.out_conv = nn.Conv2d(
            in_channels=256,
            out_channels=latent_img_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode='circular'
        )
        

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResnetBlock(in_channels, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        x = self.in_layer(x)
        for i, layer in enumerate(self.resnet_layers):
            x = layer(x)
        return self.out_conv(x)
    
class ResNetEncoder(nn.Module):
    """Residual convolutional network (ResNet 18 architecture)"""

    def __init__(self, in_channels: int, latent_img_channels: int = 32, fc_out_dim: int = 14 * 9):
        super(ResNetEncoder, self).__init__()
        self.resnet_list = nn.ModuleList(
            [
                ResNet(in_channels=in_channels, latent_img_channels=latent_img_channels),
            ]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        spatial_list = []
        for resnet in self.resnet_list:
            spatial = resnet(x)
            spatial_list.append(spatial)
        
        spatial_latent = torch.cat(spatial_list, dim=1)
        return spatial_latent