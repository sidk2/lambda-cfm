"""Shared downstream model components."""

import torch
import torch.nn as nn


class SummaryStatisticMLP(nn.Module):
    """MLP for downstream prediction on summary-statistic vectors.

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: Width of each hidden layer.
        num_hiddens: Number of hidden layers before the final projection.
        output_size: Number of output predictions.
        use_layer_norm: If True, insert a LayerNorm after each hidden activation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_hiddens: int,
        in_dim: int,
        output_size: int,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.in_transform = nn.Linear(in_dim, hidden_dim)
        self.out_transform = nn.Linear(256, output_size)
        self.use_layer_norm = use_layer_norm

        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hiddens)]
        )
        self.hiddens.append(nn.Linear(hidden_dim, 256))
        self.LeakyReLU = nn.LeakyReLU(0.2)

        if use_layer_norm:
            # One LayerNorm per hidden linear (including the final projection to 256)
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(num_hiddens)]
                + [nn.LayerNorm(256)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.LeakyReLU(self.in_transform(x))
        for i, hidden in enumerate(self.hiddens):
            x = hidden(x)
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = self.LeakyReLU(x)
        return self.out_transform(x)
