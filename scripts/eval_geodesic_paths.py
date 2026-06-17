from __future__ import annotations

import argparse
from pathlib import Path

import torch

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.generation import _advance
from drm_language_emitter.tokenizer import load_tokenizer
from drm_language_emitter.utils import save_json


@torch.no_grad()
def encode_state(model, ids: torch.Tensor) -> torch.Tensor:
    z = model.initializer(ids.shape[0], ids.device)
    for t in range(ids.shape[1]):
        z = _advance(model, z, model.token_embedding(ids[:, t]))
    return z


@torch.no_grad()
def rollout_action(model, z: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    actions = []
    decoded = []
    for t in range(ids.shape[1]):
        directions, gates = model.direction_field(z)
        metric_diag, metric_u = model.metric(z)
        e_t = model.token_embedding(ids[:, t])
        dz_raw, _ = model.flow(z, e_t, directions, gates)
        dz = model.metric.naturalize(
            dz_raw,
            metric_diag,
            metric_u,
            strength=model.config.metric_naturalization_strength
            if model.config.use_metric_naturalization
            else 0.0,
            damping=model.config.metric_damping,
        )
        actions.append(model.metric.metric_energy(z, dz, metric_diag, metric_u))
        z = model.updater(z, dz)
        decoded.append(int(model.emitter(z).argmax(dim=-1)[0]))
    return torch.stack(actions).mean(), decoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/tiny/drm_tiny.pt")
    parser.add_argument("--tokenizer", default="runs/tiny/tokenizer.json")
    parser.add_argument("--prompt-a", default="Directional ")
    parser.add_argument("--prompt-b", default="Generation ")
    parser.add_argument("--output", default="runs/tiny/geodesic_paths.json")
    args = parser.parse_args()
    model = load_model(args.checkpoint)
    tokenizer = load_tokenizer(args.tokenizer)
    ids_a = torch.tensor([tokenizer.encode(args.prompt_a)], dtype=torch.long)
    ids_b = torch.tensor([tokenizer.encode(args.prompt_b)], dtype=torch.long)
    z_a = encode_state(model, ids_a)
    z_b = encode_state(model, ids_b)
    linear_actions = []
    linear_tokens = []
    for alpha in torch.linspace(0, 1, 8):
        z = (1 - alpha) * z_a + alpha * z_b
        logits = model.emitter(z)
        linear_tokens.append(int(logits.argmax(dim=-1)[0]))
        v = z_b - z_a
        energy = model.metric.metric_energy(z, v)
        linear_actions.append(float(energy.mean().detach()))
    common = ids_b[:, : min(ids_a.shape[1], ids_b.shape[1])]
    drm_action, drm_tokens = rollout_action(model, z_a, common)
    linear_mean = sum(linear_actions) / len(linear_actions)
    drm_value = float(drm_action)
    payload = {
        "note": "This evaluates learned low-action trajectories, not an exact geodesic solver.",
        "euclidean_length": float((z_b - z_a).norm(dim=-1).mean()),
        "metric_action_length_linear_mean": linear_mean,
        "metric_action_length_drm_rollout": drm_value,
        "drm_to_linear_action_ratio": drm_value / max(linear_mean, 1e-8),
        "decoded_interpolation_linear": tokenizer.decode(linear_tokens),
        "decoded_drm_rollout": tokenizer.decode(drm_tokens),
    }
    save_json(args.output, payload)
    print(f"saved={Path(args.output)}")


if __name__ == "__main__":
    main()
