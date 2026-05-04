"""Generalized Divisive Normalization (GDN) layer.

Adapted from https://github.com/jorge-pessoa/pytorch-gdn
"""

import torch
from torch import nn
from torch.autograd import Function


class LowerBound(Function):
    """Autograd function that clamps inputs to a lower bound with custom gradient."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b = torch.ones(inputs.size(), device=inputs.device) * bound.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0
        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.

    ``y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))``
    """

    def __init__(
        self,
        ch: int,
        device: str,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
        reparam_offset: float = 2**-18,
    ):
        super().__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.tensor([reparam_offset], device=device)
        self._build(ch, torch.device(device))

    def _build(self, ch: int, device: torch.device) -> None:
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset.to(device) ** 2) ** 0.5
        self.gamma_bound = self.reparam_offset.to(device)

        beta = torch.sqrt(torch.ones(ch, device=device) + self.pedestal.to(device))
        self.beta = nn.Parameter(beta)

        eye = torch.eye(ch, device=device)
        gamma = torch.sqrt(self.gamma_init * eye + self.pedestal)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        device = inputs.device
        _, ch, _, _ = inputs.size()

        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta.to(device) ** 2 - self.pedestal.to(device)

        gamma = LowerBound.apply(self.gamma, self.gamma_bound).to(device)
        gamma = gamma**2 - self.pedestal.to(device)
        gamma = gamma.view(ch, ch, 1, 1)

        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)  # pylint: disable=not-callable
        norm_ = torch.sqrt(norm_)

        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)  # type: ignore[possibly-undefined]
        return outputs
