"""CNN and MLP models for cosmological parameter estimation."""

import torch
import torch.nn as nn


def _make_block(in_ch: int, out_ch: int, downsample_padding: int = 0) -> nn.Sequential:
    """Helper to build a 3-layer convolutional downsampling block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="circular", bias=True),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="circular", bias=True),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2),
        nn.Conv2d(out_ch, out_ch, 2, stride=2, padding=downsample_padding, padding_mode="circular", bias=True),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2),
    )


class ParamEstimatorImg(nn.Module):
    """CNN for parameter estimation on raw 256×256 cosmological fields.

    Adapted from https://camels-multifield-dataset.readthedocs.io/en/latest/inference.html
    """

    def __init__(self, hidden: int, dr: float, channels: int, output_size: int = 2):
        super().__init__()
        
        # Build the dynamic sequential feature extractor
        self.features = nn.Sequential(
            # Block 0: 256→128
            _make_block(channels, 2 * hidden),
            # Block 1: 128→64
            _make_block(2 * hidden, 4 * hidden),
            # Block 2: 64→32
            _make_block(4 * hidden, 8 * hidden),
            # Block 3: 32→16
            _make_block(8 * hidden, 16 * hidden),
            # Block 4: 16→8 (Notice padding=1 here)
            _make_block(16 * hidden, 32 * hidden, downsample_padding=1),
            # Block 5: 8→4
            _make_block(32 * hidden, 64 * hidden),
            # Block 6: 4→1
            nn.Conv2d(64 * hidden, 128 * hidden, 4, padding_mode="circular", bias=True),
            nn.BatchNorm2d(128 * hidden),
            nn.LeakyReLU(0.2),
        )

        self.dropout = nn.Dropout(p=dr)
        self.classifier = nn.Sequential(
            nn.Linear(128 * hidden, 64 * hidden),
            nn.LeakyReLU(0.2),
            self.dropout,
            nn.Linear(64 * hidden, output_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        x = x.view(image.shape[0], -1)
        return self.classifier(x)
