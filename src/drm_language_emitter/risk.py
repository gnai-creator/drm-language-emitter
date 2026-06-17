from __future__ import annotations

import torch
from torch import nn

from .config import DRMConfig


class RiskField(nn.Module):
    """Experimental blindspot/dubiety scaffold that can thicken the metric."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.enabled = config.use_powerlaw_risk
        h = config.hidden_size
        self.net = nn.Sequential(
            nn.Linear(config.d_state, h),
            nn.GELU(),
            nn.Linear(h, 2),
            nn.Sigmoid(),
        )
        self.alpha_b = nn.Parameter(torch.tensor(0.1))
        self.alpha_d = nn.Parameter(torch.tensor(0.1))
        self.beta_b = nn.Parameter(torch.tensor(1.5))
        self.beta_d = nn.Parameter(torch.tensor(1.5))

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.enabled:
            zero = z.new_zeros(z.shape[0])
            return {"blindspot": zero, "dubiety": zero, "risk_mass": zero}
        values = self.net(z)
        blindspot = values[:, 0]
        dubiety = values[:, 1]
        risk_mass = self.alpha_b.abs() * blindspot.pow(self.beta_b.abs() + 1e-3)
        risk_mass = risk_mass + self.alpha_d.abs() * dubiety.pow(self.beta_d.abs() + 1e-3)
        return {"blindspot": blindspot, "dubiety": dubiety, "risk_mass": risk_mass}
