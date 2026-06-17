from __future__ import annotations

from pathlib import Path

import torch

from .config import DRMConfig
from .model import DRMEmitterModel


def load_model(checkpoint: str | Path) -> DRMEmitterModel:
    payload = torch.load(checkpoint, map_location="cpu")
    config = DRMConfig.from_dict(payload["config"])
    model = DRMEmitterModel(config)
    model.load_state_dict(payload["model"])
    model.eval()
    return model
