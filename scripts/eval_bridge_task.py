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
def rollout_bridge(model, z_a: torch.Tensor, ids_b: torch.Tensor) -> dict[str, float]:
    z = z_a
    actions = []
    decoded = []
    for t in range(ids_b.shape[1]):
        directions, gates = model.direction_field(z)
        metric_diag, metric_u = model.metric(z)
        dz_raw, _ = model.flow(z, model.token_embedding(ids_b[:, t]), directions, gates)
        dz = model.metric.naturalize(
            dz_raw,
            metric_diag,
            metric_u,
            strength=model.config.metric_naturalization_strength if model.config.use_metric_naturalization else 0.0,
            damping=model.config.metric_damping,
        )
        actions.append(model.metric.metric_energy(z, dz, metric_diag, metric_u))
        z = model.updater(z, dz)
        decoded.append(int(model.emitter(z).argmax(dim=-1)[0]))
    action_tensor = torch.stack(actions)
    return {
        "action_total": float(action_tensor.sum()),
        "action_per_step": float(action_tensor.mean()),
        "endpoint_distance": float((z - encode_state(model, ids_b)).norm(dim=-1).mean()),
        "decoded_ids": decoded,
    }


def save_bridge_svg(path: str | Path, payload: dict) -> None:
    path = Path(path)
    width, height = 760, 420
    values = [
        ("Initial endpoint", payload["euclidean_endpoint_distance_initial"], "#6b7280"),
        ("DRM bridge endpoint", payload["drm_bridge_endpoint_distance"], "#0f766e"),
        ("DRM action/step", payload["action_per_step"], "#2563eb"),
        ("Linear energy", payload["linear_endpoint_energy"], "#b91c1c"),
    ]
    max_value = max(v for _, v, _ in values) * 1.2 + 1e-8
    x0, y0 = 80, 330
    bar_w = 110
    gap = 45
    bars = []
    for i, (label, value, color) in enumerate(values):
        h = value / max_value * 240
        x = x0 + i * (bar_w + gap)
        y = y0 - h
        bars.append(f"<rect x='{x}' y='{y:.1f}' width='{bar_w}' height='{h:.1f}' fill='{color}'/>")
        bars.append(f"<text x='{x}' y='{y-8:.1f}' font-size='12' fill='#111827'>{value:.4f}</text>")
        bars.append(f"<text x='{x}' y='{y0+20}' font-size='11' fill='#374151'>{label}</text>")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="48" y="34" font-size="22" font-family="Arial" font-weight="700" fill="#111827">DRM Bridge Diagnostic</text>
  <text x="48" y="56" font-size="13" font-family="Arial" fill="#4b5563">Diagnostic only, not a formal geodesic solver. Ratio: {payload['bridge_to_linear_energy_ratio']:.2f}</text>
  <line x1="50" y1="{y0}" x2="{width-40}" y2="{y0}" stroke="#111827"/>
  {''.join(bars)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt-a", default="Directional relational ")
    parser.add_argument("--prompt-b", default="Generation through ")
    parser.add_argument("--output", default="runs/bridge_task.json")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    tokenizer = load_tokenizer(args.tokenizer)
    ids_a = torch.tensor([tokenizer.encode(args.prompt_a)], dtype=torch.long)
    ids_b = torch.tensor([tokenizer.encode(args.prompt_b)], dtype=torch.long)
    z_a = encode_state(model, ids_a)
    z_b = encode_state(model, ids_b)
    bridge = rollout_bridge(model, z_a, ids_b)
    linear_energy = model.metric.metric_energy(z_a, z_b - z_a).mean().detach()
    initial_distance = float((z_b - z_a).norm(dim=-1).mean())
    final_distance = bridge["endpoint_distance"]
    payload = {
        "note": "Low-action bridge diagnostic only; this is not a formal geodesic solver.",
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "endpoint_distance_initial": initial_distance,
        "endpoint_distance_final": final_distance,
        "euclidean_endpoint_distance_initial": initial_distance,
        "drm_bridge_action": bridge["action_per_step"],
        "drm_bridge_endpoint_distance": final_distance,
        "action_total": bridge["action_total"],
        "action_per_step": bridge["action_per_step"],
        "bridge_success_score": (initial_distance - final_distance) / max(initial_distance, 1e-8),
        "decoded_bridge_text": tokenizer.decode(bridge["decoded_ids"]),
        "linear_endpoint_energy": float(linear_energy),
        "bridge_to_linear_energy_ratio": bridge["action_per_step"] / max(float(linear_energy), 1e-8),
    }
    save_json(args.output, payload)
    save_bridge_svg(Path(args.output).with_suffix(".svg"), payload)
    print(f"saved={Path(args.output)}")
    print(f"saved={Path(args.output).with_suffix('.svg')}")


if __name__ == "__main__":
    main()
