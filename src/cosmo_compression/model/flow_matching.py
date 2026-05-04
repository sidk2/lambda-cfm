"""Flow-matching module with training loss and inference-time ODE integration."""

import torch
from torch import nn
from torchdyn.core import NeuralODE


class ConditionedVelocityModel(nn.Module):
    """Wraps a velocity model with optional conditioning and reverse flag."""

    def __init__(
        self,
        velocity_model: nn.Module,
        h: torch.Tensor | None,
        reverse: bool = False,
    ):
        super().__init__()
        self.reverse = reverse
        self.velocity_model = velocity_model
        self.h = h

    def forward(
        self,
        t: torch.Tensor | int,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        if h is None:
            h = self.h
        
        # Ensure t is fully batched to match x for the velocity model
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=x.dtype)
        elif t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.expand(x.shape[0], 1)

        velocity = self.velocity_model(x, t=t, z=h)
        return -velocity if self.reverse else velocity


class FlowMatching(nn.Module):
    """Flow-matching module with training loss and inference-time log-likelihood."""

    def __init__(
        self,
        velocity_model: nn.Module,
        sigma: float = 0.0,
        reverse: bool = False,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.sigma = sigma
        self.reverse = reverse

    def get_mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample distribution mean for rectified flow matching."""
        return t * x1 + (1 - t) * x0

    def get_gamma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Sample distribution variance for rectified flow matching."""
        return torch.sqrt(2 * t * (1 - t))

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None,
    ) -> torch.Tensor:
        """Sample from the interpolated distribution at time *t*."""
        t_broadcast = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        mu_t = self.get_mu_t(x0, x1, t_broadcast)
        if self.sigma != 0.0:
            sigma_t = self.get_gamma_t(t_broadcast)
            return mu_t + sigma_t * eps  # type: ignore[operator]
        return mu_t

    def compute_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        h: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the flow-matching MSE loss."""
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device).type_as(x0)

        eps = torch.randn_like(x0) if self.sigma != 0.0 else None
        xt = self.sample_xt(x0, x1, t, eps)
        ut = x1 - x0
        vt = self.velocity_model(xt, t=t, z=h)
        return torch.mean((vt - ut) ** 2)

    def predict(
        self,
        x0: torch.Tensor,
        h: torch.Tensor | None = None,
        n_sampling_steps: int = 100,
        solver: str = "dopri5",
        full_return: bool = False,
    ) -> torch.Tensor:
        """Run inference by solving the probability-flow ODE from *x0*."""
        conditional_velocity_model = ConditionedVelocityModel(
            velocity_model=self.velocity_model, h=h, reverse=self.reverse,
        )
        node = NeuralODE(
            conditional_velocity_model, solver=solver, sensitivity="adjoint",
        )
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, n_sampling_steps, device=x0.device),
            )
        return traj[-1] if not full_return else traj