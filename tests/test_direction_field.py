import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.direction_field import DirectionField


def test_direction_field_shapes_and_dim_range():
    config = DRMConfig(d_state=10, n_directions=5, hidden_size=16)
    field = DirectionField(config)
    z = torch.randn(3, 10)
    directions, gates = field(z)
    assert directions.shape == (3, 5, 10)
    assert gates.shape == (3, 5)
    dim_d = gates.sum(dim=-1)
    assert torch.all(dim_d >= 0)
    assert torch.all(dim_d <= 5)
    diagnostics = field.diagnostics(gates)
    assert torch.isfinite(diagnostics["gate_entropy"])
