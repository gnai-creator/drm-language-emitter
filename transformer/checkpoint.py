from __future__ import annotations

from pathlib import Path

import torch

from .tiny_transformer import TinyTransformerConfig, TinyTransformerLM


def load_transformer(checkpoint: str | Path) -> TinyTransformerLM:
    payload = torch.load(checkpoint, map_location="cpu")
    config = TinyTransformerConfig.from_dict(payload["config"])
    model = TinyTransformerLM(config)
    model.load_state_dict(payload["model"])
    model.eval()
    return model
