from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import DRMEmitterModel


@torch.no_grad()
def generate(
    model: DRMEmitterModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float | None = None,
    top_k: int | None = None,
) -> torch.Tensor:
    model.eval()
    config = model.config
    temperature = config.generation_temperature if temperature is None else temperature
    top_k = config.top_k if top_k is None else top_k
    batch = input_ids.shape[0]
    z = model.initializer(batch, input_ids.device)

    for t in range(input_ids.shape[1]):
        z = _advance(model, z, model.token_embedding(input_ids[:, t]))

    generated = [input_ids]
    current = input_ids[:, -1]
    for _ in range(max_new_tokens):
        logits = model.emitter(z) / max(temperature, 1e-6)
        if top_k and top_k > 0 and top_k < logits.shape[-1]:
            values, indices = torch.topk(logits, top_k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            logits = filtered.scatter(-1, indices, values)
        probs = F.softmax(logits, dim=-1)
        current = torch.multinomial(probs, num_samples=1).squeeze(-1)
        generated.append(current[:, None])
        z = _advance(model, z, model.token_embedding(current))
    return torch.cat(generated, dim=1)


def _advance(model: DRMEmitterModel, z: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
    for _ in range(model.config.n_flow_steps):
        directions, gates = model.direction_field(z)
        metric_diag, metric_u = model.metric(z)
        dz_raw, _ = model.flow(z, token_embedding, directions, gates)
        dz = model.metric.naturalize(
            dz_raw,
            metric_diag,
            metric_u,
            strength=model.config.metric_naturalization_strength
            if model.config.use_metric_naturalization
            else 0.0,
            damping=model.config.metric_damping,
        )
        z = model.updater(z, dz)
    return z
