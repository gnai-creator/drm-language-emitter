from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import DRMConfig


class RelationalMetric(nn.Module):
    """SPD metric G(z) = diag(softplus(d(z)) + eps) + U(z)U(z)^T."""

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
        self.diag_head = nn.Linear(h, config.d_state)
        self.u_basis_size = config.metric_u_basis_size
        if self.u_basis_size and config.metric_rank > 0:
            self.u_basis = nn.Parameter(torch.empty(self.u_basis_size, config.d_state))
            nn.init.normal_(self.u_basis, std=0.02)
            self.u_head = nn.Linear(h, config.metric_rank * self.u_basis_size)
        else:
            self.u_basis = None
            self.u_head = nn.Linear(h, config.d_state * config.metric_rank)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        diag = F.softplus(self.diag_head(h)) + self.config.metric_eps
        if self.config.metric_rank == 0:
            u = z.new_zeros(z.shape[0], self.config.d_state, 0)
        elif self.u_basis is not None:
            coefficients = self.u_head(h).view(z.shape[0], self.config.metric_rank, self.u_basis_size)
            u = torch.matmul(coefficients, self.u_basis).transpose(1, 2)
            u = u / max(self.config.metric_rank, 1) ** 0.5
        else:
            u = self.u_head(h).view(z.shape[0], self.config.d_state, self.config.metric_rank)
            u = u / max(self.config.metric_rank, 1) ** 0.5
        return diag, u

    def metric_energy(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
        metric_diag: torch.Tensor | None = None,
        metric_u: torch.Tensor | None = None,
        risk_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if metric_diag is None or metric_u is None:
            metric_diag, metric_u = self(z)
        diag = metric_diag
        if risk_mass is not None:
            diag = diag + risk_mass.view(-1, 1)
        diag_energy = (diag * v.pow(2)).sum(dim=-1)
        if metric_u.shape[-1] == 0:
            return diag_energy
        low_rank = torch.bmm(metric_u.transpose(1, 2), v.unsqueeze(-1)).squeeze(-1)
        return diag_energy + low_rank.pow(2).sum(dim=-1)

    def pairwise_coupling(
        self,
        z: torch.Tensor,
        directions: torch.Tensor,
        metric_diag: torch.Tensor | None = None,
        metric_u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if metric_diag is None or metric_u is None:
            metric_diag, metric_u = self(z)
        gv = metric_diag.unsqueeze(1) * directions
        if metric_u.shape[-1] > 0:
            projection = torch.bmm(directions, metric_u)
            gv = gv + torch.bmm(projection, metric_u.transpose(1, 2))
        return torch.bmm(directions, gv.transpose(1, 2))

    def naturalize(
        self,
        v: torch.Tensor,
        metric_diag: torch.Tensor,
        metric_u: torch.Tensor,
        strength: float = 1.0,
        damping: float = 0.0,
    ) -> torch.Tensor:
        """Apply a stable G^{-1} preconditioner using Woodbury."""
        if strength <= 0:
            return v
        damped_diag = metric_diag + damping
        inv_diag = damped_diag.reciprocal()
        diag_solution = inv_diag * v
        if metric_u.shape[-1] == 0:
            solved = diag_solution
        else:
            solve_dtype = torch.float32 if v.dtype in {torch.float16, torch.bfloat16} else v.dtype
            autocast_device = v.device.type if v.device.type in {"cuda", "cpu"} else "cpu"
            with torch.autocast(device_type=autocast_device, enabled=False):
                metric_u_solve = metric_u.to(solve_dtype)
                inv_diag_solve = inv_diag.to(solve_dtype)
                diag_solution_solve = diag_solution.to(solve_dtype)
                d_inv_u = inv_diag_solve.unsqueeze(-1) * metric_u_solve
                middle = torch.eye(metric_u.shape[-1], device=v.device, dtype=solve_dtype).unsqueeze(0)
                middle = middle + torch.bmm(metric_u_solve.transpose(1, 2), d_inv_u)
                rhs = torch.bmm(metric_u_solve.transpose(1, 2), diag_solution_solve.unsqueeze(-1))
                correction_coeff = torch.linalg.solve(middle, rhs)
                correction = torch.bmm(d_inv_u, correction_coeff).squeeze(-1).to(v.dtype)
            solved = diag_solution - correction
        return (1.0 - strength) * v + strength * solved

    @staticmethod
    def condition_proxy(metric_diag: torch.Tensor, metric_u: torch.Tensor) -> torch.Tensor:
        low_rank_scale = metric_u.pow(2).sum(dim=(1, 2)) if metric_u.numel() else metric_diag.new_zeros(metric_diag.shape[0])
        upper = metric_diag.max(dim=-1).values + low_rank_scale
        lower = metric_diag.min(dim=-1).values.clamp_min(1e-8)
        return upper / lower

    @staticmethod
    def metric_norms(metric_diag: torch.Tensor, metric_u: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "metric_diag_mean": metric_diag.mean(),
            "metric_diag_max": metric_diag.max(),
            "metric_U_norm_mean": metric_u.norm(dim=(1, 2)).mean() if metric_u.numel() else metric_diag.new_tensor(0.0),
            "metric_U_variance": metric_u.var(unbiased=False) if metric_u.numel() else metric_diag.new_tensor(0.0),
        }
