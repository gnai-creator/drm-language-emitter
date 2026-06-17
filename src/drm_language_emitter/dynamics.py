from __future__ import annotations

import torch
from torch import nn

from .config import DRMConfig


class DRMFlow(nn.Module):
    """Velocity field constrained to the span of active DRM directions."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.coeff_net = nn.Sequential(
            nn.Linear(config.d_state + config.d_token, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, config.n_directions),
            nn.Tanh(),
        )

    def forward(
        self,
        z: torch.Tensor,
        token_embedding: torch.Tensor,
        directions: torch.Tensor,
        gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.coeff_net(torch.cat([z, token_embedding], dim=-1))
        active_coefficients = gates * coefficients
        dz = torch.einsum("bn,bnd->bd", active_coefficients, directions)
        return dz, coefficients


class StateUpdater(nn.Module):
    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config

    def forward(self, z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        z_next = z + self.config.dt * dz
        if self.config.bounded_state:
            norm = z_next.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            clip = torch.clamp(self.config.state_clip_norm / norm, max=1.0)
            z_next = z_next * clip
            z_next = self.config.state_clip_norm * torch.tanh(z_next / self.config.state_clip_norm)
        return z_next


def toroidal_pairs(theta: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
