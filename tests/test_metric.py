import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.metric import RelationalMetric


def test_metric_energy_non_negative_and_coupling_shape():
    config = DRMConfig(d_state=9, n_directions=4, metric_rank=3, hidden_size=16)
    metric = RelationalMetric(config)
    z = torch.randn(2, 9)
    v = torch.randn(2, 9)
    diag, u = metric(z)
    energy = metric.metric_energy(z, v, diag, u)
    assert torch.all(energy >= 0)
    directions = torch.randn(2, 4, 9)
    coupling = metric.pairwise_coupling(z, directions, diag, u)
    assert coupling.shape == (2, 4, 4)
    assert torch.isfinite(metric.condition_proxy(diag, u)).all()
