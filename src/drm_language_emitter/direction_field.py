from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import DRMConfig


class DirectionField(nn.Module):
    """Learns active, generally non-orthogonal directions D(z)."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.trunk = nn.Sequential(
            nn.Linear(config.d_state, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.direction_basis_size = config.direction_basis_size
        if self.direction_basis_size:
            self.direction_basis = nn.Parameter(torch.empty(self.direction_basis_size, config.d_state))
            nn.init.normal_(self.direction_basis, std=0.02)
            self.direction_head = nn.Linear(h, config.n_directions * self.direction_basis_size)
        else:
            self.direction_basis = None
            self.direction_head = nn.Linear(h, config.n_directions * config.d_state)
        self.gate_head = nn.Linear(h, config.n_directions)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        if self.direction_basis is not None:
            coefficients = self.direction_head(h).view(z.shape[0], self.config.n_directions, self.direction_basis_size)
            directions = torch.matmul(coefficients, self.direction_basis)
        else:
            directions = self.direction_head(h).view(
                z.shape[0], self.config.n_directions, self.config.d_state
            )
        if self.config.direction_norm:
            directions = F.normalize(directions, dim=-1)
        logits = self.gate_head(h) + self.config.gate_logit_bias
        gates = torch.sigmoid(logits / max(self.config.gate_temperature, 1e-6))
        if self.config.gate_top_k and 0 < self.config.gate_top_k < gates.shape[-1]:
            values, indices = torch.topk(gates, self.config.gate_top_k, dim=-1)
            sparse = torch.zeros_like(gates).scatter(-1, indices, values)
            if self.config.gate_top_k_renorm:
                sparse = sparse * (gates.sum(dim=-1, keepdim=True) / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8))
            gates = sparse
        return directions, gates

    @staticmethod
    def diagnostics(gates: torch.Tensor) -> dict[str, torch.Tensor]:
        dim_d = gates.sum(dim=-1)
        entropy = -(gates * torch.log(gates.clamp_min(1e-8)) + (1.0 - gates) * torch.log((1.0 - gates).clamp_min(1e-8)))
        return {
            "dimD_mean": dim_d.mean(),
            "dimD_std": dim_d.std(unbiased=False),
            "gate_entropy": entropy.mean(),
            "soft_active_fraction": gates.mean(),
            "hard_active_fraction_025": (gates > 0.25).float().mean(),
            "hard_active_fraction_050": (gates > 0.50).float().mean(),
            "hard_active_fraction_075": (gates > 0.75).float().mean(),
            "hard_active_fraction_090": (gates > 0.90).float().mean(),
            "gate_min": gates.min(),
            "gate_max": gates.max(),
        }
