from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .config import DRMConfig
from .direction_field import DirectionField
from .dynamics import DRMFlow, StateUpdater
from .emitter import LanguageEmitter, TokenEmbedding
from .losses import (
    combine_losses,
    dimension_entropy,
    metric_diversity,
    next_token_cross_entropy,
    recurrence_proxy,
    stability_proxy,
)
from .metric import RelationalMetric
from .risk import RiskField


class DRMStateInitializer(nn.Module):
    def __init__(self, config: DRMConfig):
        super().__init__()
        self.z0 = nn.Parameter(torch.zeros(config.d_state))
        nn.init.normal_(self.z0, std=0.02)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.z0.unsqueeze(0).expand(batch_size, -1).to(device)


class DRMEmitterModel(nn.Module):
    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config)
        self.initializer = DRMStateInitializer(config)
        self.direction_field = DirectionField(config)
        self.metric = RelationalMetric(config)
        self.flow = DRMFlow(config)
        self.updater = StateUpdater(config)
        self.risk = RiskField(config)
        self.emitter = LanguageEmitter(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_states: bool = False,
        global_step: int | None = None,
    ) -> dict[str, Any]:
        batch, seq_len = input_ids.shape
        z = self.initializer(batch, input_ids.device)
        token_embeddings = self.token_embedding(input_ids)
        logits_steps = []
        states = []
        action_values = []
        dim_values = []
        entropy_values = []
        metric_regs = []
        metric_diag_steps = []
        condition_values = []
        active_025_values = []
        active_050_values = []
        active_075_values = []
        active_090_values = []
        gate_min_values = []
        gate_max_values = []
        gate_flat_values = []
        u_norm_values = []
        risk_values = []
        naturalization_strength = self._naturalization_strength(global_step)

        for t in range(seq_len):
            e_t = token_embeddings[:, t]
            for _ in range(self.config.n_flow_steps):
                directions, gates = self.direction_field(z)
                metric_diag, metric_u = self.metric(z)
                risk = self.risk(z)
                dz_raw, _coefficients = self.flow(z, e_t, directions, gates)
                dz = self.metric.naturalize(
                    dz_raw,
                    metric_diag,
                    metric_u,
                    strength=naturalization_strength,
                    damping=self.config.metric_damping,
                )
                energy = self.metric.metric_energy(
                    z, dz, metric_diag, metric_u, risk_mass=risk["risk_mass"]
                )
                action_values.append(energy)
                dim_values.append(gates.sum(dim=-1))
                entropy_values.append(dimension_entropy(gates))
                u_norm = metric_u.norm(dim=(1, 2)) if metric_u.numel() else metric_diag.new_zeros(batch)
                metric_regs.append(metric_diag.pow(2).mean() + metric_u.pow(2).mean())
                metric_diag_steps.append(metric_diag)
                condition_values.append(self.metric.condition_proxy(metric_diag, metric_u))
                active_025_values.append((gates > 0.25).float().mean(dim=-1))
                active_050_values.append((gates > 0.50).float().mean(dim=-1))
                active_075_values.append((gates > 0.75).float().mean(dim=-1))
                active_090_values.append((gates > 0.90).float().mean(dim=-1))
                gate_min_values.append(gates.min(dim=-1).values)
                gate_max_values.append(gates.max(dim=-1).values)
                gate_flat_values.append(gates.reshape(-1))
                u_norm_values.append(u_norm)
                risk_values.append(risk["risk_mass"])
                z = self.updater(z, dz)
            logits_steps.append(self.emitter(z))
            states.append(z)

        logits = torch.stack(logits_steps, dim=1)
        state_tensor = torch.stack(states, dim=1)
        metric_diag_tensor = torch.stack(metric_diag_steps, dim=1)
        action_loss = torch.stack(action_values, dim=1).mean()
        dim_tensor = torch.stack(dim_values, dim=1)
        dim_sparsity = dim_tensor.mean()
        dim_std_value = dim_tensor.std(unbiased=False)
        dim_entropy_value = torch.stack(entropy_values).mean()
        metric_reg = torch.stack(metric_regs).mean()
        metric_u_floor_loss = (
            (self.config.metric_u_min_norm - torch.stack(u_norm_values, dim=1))
            .clamp_min(0.0)
            .pow(2)
            .mean()
        )
        metric_div_value = metric_diversity(metric_diag_tensor)
        recurrence_value = recurrence_proxy(state_tensor)
        stability_value = stability_proxy(logits)
        blindspot_value = torch.stack(risk_values, dim=1).mean()
        hard_active_025_value = torch.stack(active_025_values, dim=1).mean()
        hard_active_050_value = torch.stack(active_050_values, dim=1).mean()
        hard_active_075_value = torch.stack(active_075_values, dim=1).mean()
        hard_active_090_value = torch.stack(active_090_values, dim=1).mean()
        soft_active_value = dim_sparsity / self.config.n_directions
        condition_value = torch.stack(condition_values, dim=1).mean()
        metric_u_norm_value = torch.stack(u_norm_values, dim=1).mean()
        all_gates = torch.cat(gate_flat_values)
        gate_quantiles = torch.quantile(
            all_gates,
            torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=all_gates.device, dtype=all_gates.dtype),
        )
        ce_loss = next_token_cross_entropy(logits, targets) if targets is not None else None
        total_loss, aux_losses = combine_losses(
            self.config,
            ce_loss,
            action_loss,
            dim_sparsity,
            dim_entropy_value,
            metric_reg,
            metric_div_value,
            recurrence_value,
            stability_value,
            blindspot_value,
            soft_active_value,
            dim_std_value,
            condition_value,
            metric_u_norm_value,
        )
        if self.config.lambda_metric_u_floor:
            total_loss = total_loss + self.config.lambda_metric_u_floor * metric_u_floor_loss
            aux_losses["metric_u_floor"] = metric_u_floor_loss

        diagnostics = {
            "dimD_mean": dim_sparsity,
            "dimD_std": dim_std_value,
            "soft_active_fraction": soft_active_value,
            "active_fraction": hard_active_050_value,
            "hard_active_fraction_025": hard_active_025_value,
            "hard_active_fraction_050": hard_active_050_value,
            "hard_active_fraction_075": hard_active_075_value,
            "hard_active_fraction_090": hard_active_090_value,
            "gate_min": torch.stack(gate_min_values, dim=1).min(),
            "gate_max": torch.stack(gate_max_values, dim=1).max(),
            "gate_q10": gate_quantiles[0],
            "gate_q25": gate_quantiles[1],
            "gate_q50": gate_quantiles[2],
            "gate_q75": gate_quantiles[3],
            "gate_q90": gate_quantiles[4],
            "gate_entropy": dim_entropy_value,
            "action_mean": action_loss,
            "metric_U_norm_mean": metric_u_norm_value,
            "metric_U_variance": torch.stack(u_norm_values, dim=1).var(unbiased=False),
            "condition_proxy": condition_value,
            "recurrence_proxy": recurrence_value,
            "stability_proxy": stability_value,
            "risk_mass_mean": blindspot_value,
            "metric_u_floor_loss": metric_u_floor_loss,
            "metric_naturalization_strength": input_ids.new_tensor(float(naturalization_strength), dtype=torch.float32),
        }
        out: dict[str, Any] = {
            "logits": logits,
            "loss": total_loss,
            "aux_losses": aux_losses,
            "diagnostics": diagnostics,
        }
        if return_states:
            out["states"] = state_tensor
        return out

    def state_dict_with_config(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "model": self.state_dict()}

    def _naturalization_strength(self, global_step: int | None) -> float:
        if not self.config.use_metric_naturalization:
            return 0.0
        max_strength = self.config.metric_naturalization_strength
        warmup = self.config.metric_naturalization_warmup_steps
        if global_step is None or warmup <= 0:
            return max_strength
        return max_strength * min(max(global_step, 0) / warmup, 1.0)
