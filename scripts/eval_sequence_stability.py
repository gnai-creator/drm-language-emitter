from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.generation import _advance
from drm_language_emitter.tokenizer import load_tokenizer
from drm_language_emitter.utils import save_json
from transformer.checkpoint import load_transformer


def synthetic_sequences() -> dict[str, str]:
    return {
        "repeated_pattern": ("abc123 " * 64),
        "alternating_pattern": ("AB" * 224),
        "delayed_copy": ("copy this later: alpha beta gamma. " * 8) + ("alpha beta gamma " * 12),
        "noisy_repeated_pattern": ("abc123 xbc123 abc923 abc123 " * 18),
    }


@torch.no_grad()
def drm_metrics(model, tokenizer, text: str) -> dict[str, float]:
    ids = torch.tensor([tokenizer.encode(text)[: model.config.max_seq_len]], dtype=torch.long)
    z = model.initializer(1, ids.device)
    states = []
    logits = []
    for t in range(ids.shape[1]):
        z = _advance(model, z, model.token_embedding(ids[:, t]))
        states.append(z)
        logits.append(model.emitter(z))
    state_tensor = torch.stack(states, dim=1)
    logits_tensor = torch.stack(logits, dim=1)
    norms = state_tensor.norm(dim=-1)
    return {
        "state_norm_mean": float(norms.mean()),
        "state_norm_std": float(norms.std(unbiased=False)),
        "state_drift": float((state_tensor[:, -1] - state_tensor[:, 0]).norm(dim=-1).mean()),
        "recurrence_proxy": float((norms[:, -1] - norms[:, 0]).pow(2).mean()),
        "stability_proxy": float((logits_tensor[:, 1:] - logits_tensor[:, :-1]).pow(2).mean()) if logits_tensor.shape[1] > 1 else 0.0,
    }


@torch.no_grad()
def transformer_metrics(model, tokenizer, text: str) -> dict[str, float]:
    ids = tokenizer.encode(text)
    seq_len = min(model.config.max_seq_len, max(len(ids) - 1, 1))
    x = torch.tensor([ids[:seq_len]], dtype=torch.long)
    y = torch.tensor([ids[1 : seq_len + 1]], dtype=torch.long)
    out = model(x, y)
    logits = out["logits"]
    ce_by_pos = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        reduction="none",
    ).view(1, -1)
    half = max(ce_by_pos.shape[1] // 2, 1)
    early = ce_by_pos[:, :half].mean()
    late = ce_by_pos[:, half:].mean()
    return {
        "ce_mean": float(ce_by_pos.mean()),
        "ce_early": float(early),
        "ce_late": float(late),
        "degradation_over_length": float(late - early),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drm-checkpoint", required=True)
    parser.add_argument("--drm-tokenizer", required=True)
    parser.add_argument("--transformer-checkpoint", default=None)
    parser.add_argument("--output", default="runs/sequence_stability.json")
    args = parser.parse_args()

    drm = load_model(args.drm_checkpoint)
    tokenizer = load_tokenizer(args.drm_tokenizer)
    transformer = load_transformer(args.transformer_checkpoint) if args.transformer_checkpoint else None
    payload = {"note": "Synthetic long-horizon stability diagnostics.", "sequences": {}}
    for name, text in synthetic_sequences().items():
        row = {"drm": drm_metrics(drm, tokenizer, text)}
        if transformer is not None:
            row["transformer"] = transformer_metrics(transformer, tokenizer, text)
        payload["sequences"][name] = row
    save_json(args.output, payload)
    print(f"saved={Path(args.output)}")


if __name__ == "__main__":
    main()
