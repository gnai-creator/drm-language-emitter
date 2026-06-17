import torch

from drm_language_emitter import DRMConfig, DRMEmitterModel
from drm_language_emitter.generation import generate


def test_generation_extends_sequence():
    config = DRMConfig(vocab_size=11, d_token=8, d_state=12, n_directions=4, metric_rank=2, hidden_size=16, top_k=5)
    model = DRMEmitterModel(config)
    x = torch.randint(0, 11, (1, 4))
    out = generate(model, x, max_new_tokens=3)
    assert out.shape == (1, 7)
    assert out.max() < 11
