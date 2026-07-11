from __future__ import annotations

from pathlib import Path

import torch

from .config import DRMConfig
from .model import DRMEmitterModel


def load_model(checkpoint: str | Path) -> DRMEmitterModel:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    missing = {"config", "model"} - set(payload)
    if missing:
        raise ValueError(f"checkpoint missing required key(s): {', '.join(sorted(missing))}")
    if not isinstance(payload["config"], dict):
        raise ValueError("checkpoint 'config' must be a dictionary")
    if not isinstance(payload["model"], dict):
        raise ValueError("checkpoint 'model' must be a state_dict dictionary")
    config = DRMConfig.from_dict(payload["config"])
    model = DRMEmitterModel(config)
    model.load_state_dict(payload["model"])
    model.eval()
    return model
