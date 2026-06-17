from __future__ import annotations

from typing import Any

import torch

from .model import DRMEmitterModel
from .utils import to_jsonable


@torch.no_grad()
def geometry_report(
    model: DRMEmitterModel, input_ids: torch.Tensor, targets: torch.Tensor | None = None
) -> dict[str, Any]:
    if targets is None:
        targets = input_ids
    out = model(input_ids, targets, return_states=True)
    return {
        "diagnostics": to_jsonable(out["diagnostics"]),
        "aux_losses": to_jsonable(out["aux_losses"]),
    }


def recurrence_distance(states: torch.Tensor) -> torch.Tensor:
    if states.shape[1] < 2:
        return states.new_tensor(0.0)
    return (states[:, -1] - states[:, 0]).norm(dim=-1).mean()
