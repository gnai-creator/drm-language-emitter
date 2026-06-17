from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import DRMConfig


def next_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def dimension_entropy(gates: torch.Tensor) -> torch.Tensor:
    entropy = -(gates * torch.log(gates.clamp_min(1e-8)) + (1.0 - gates) * torch.log((1.0 - gates).clamp_min(1e-8)))
    return entropy.mean()


def recurrence_proxy(states: torch.Tensor) -> torch.Tensor:
    if states.shape[1] < 2:
        return states.new_tensor(0.0)
    drift = states[:, -1].norm(dim=-1) - states[:, 0].norm(dim=-1)
    return drift.pow(2).mean()


def stability_proxy(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] < 2:
        return logits.new_tensor(0.0)
    return (logits[:, 1:] - logits[:, :-1]).pow(2).mean()


def active_fraction_loss(active_fraction: torch.Tensor, target: float) -> torch.Tensor:
    return (active_fraction - target).clamp_min(0.0).pow(2)


def dim_variance_loss(dim_std: torch.Tensor, target: float) -> torch.Tensor:
    return (target - dim_std).clamp_min(0.0).pow(2)


def condition_loss(condition_proxy: torch.Tensor, target: float) -> torch.Tensor:
    log_condition = torch.log(condition_proxy.clamp_min(1e-8))
    log_target = torch.log(condition_proxy.new_tensor(target))
    return (log_condition - log_target).clamp_min(0.0).pow(2)


def metric_diversity(metric_diag_steps: torch.Tensor) -> torch.Tensor:
    if metric_diag_steps.shape[1] < 2:
        return metric_diag_steps.new_tensor(0.0)
    return -metric_diag_steps.var(dim=1, unbiased=False).mean()


def combine_losses(
    config: DRMConfig,
    ce_loss: torch.Tensor | None,
    action_loss: torch.Tensor,
    dim_sparsity: torch.Tensor,
    dim_entropy_value: torch.Tensor,
    metric_reg: torch.Tensor,
    metric_diversity_value: torch.Tensor,
    recurrence_value: torch.Tensor,
    stability_value: torch.Tensor,
    blindspot_value: torch.Tensor,
    active_fraction_value: torch.Tensor | None = None,
    dim_std_value: torch.Tensor | None = None,
    condition_value: torch.Tensor | None = None,
    metric_u_norm_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = action_loss.new_tensor(0.0)
    losses: dict[str, torch.Tensor] = {}
    if ce_loss is not None:
        total = total + ce_loss
        losses["ce"] = ce_loss
    weighted = {
        "action": (config.lambda_action, action_loss),
        "dim_sparsity": (config.lambda_dim_sparsity, dim_sparsity),
        "dim_entropy": (-config.lambda_dim_entropy, dim_entropy_value),
        "metric_reg": (config.lambda_metric_reg, metric_reg),
        "metric_diversity": (config.lambda_metric_diversity, metric_diversity_value),
        "recurrence": (config.lambda_recurrence, recurrence_value),
        "stability": (config.lambda_stability, stability_value),
        "blindspot": (config.lambda_blindspot, blindspot_value),
    }
    for name, (weight, value) in weighted.items():
        losses[name] = value
        total = total + weight * value
    if active_fraction_value is not None:
        active_loss = active_fraction_loss(active_fraction_value, config.target_active_fraction)
        losses["active_fraction"] = active_loss
        total = total + config.lambda_active_fraction * active_loss
    if dim_std_value is not None:
        dim_var_loss = dim_variance_loss(dim_std_value, config.target_dim_std)
        losses["dim_variance"] = dim_var_loss
        total = total + config.lambda_dim_variance * dim_var_loss
    if condition_value is not None:
        cond_loss = condition_loss(condition_value, config.target_condition)
        losses["condition"] = cond_loss
        total = total + config.lambda_condition * cond_loss
    if metric_u_norm_value is not None:
        u_target = metric_u_norm_value.new_tensor(config.metric_u_target_norm)
        u_target_loss = (metric_u_norm_value - u_target).pow(2)
        losses["metric_u_target"] = u_target_loss
        total = total + config.lambda_metric_u_target * u_target_loss
    losses["total"] = total
    return total, losses
