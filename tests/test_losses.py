import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.losses import combine_losses, dimension_entropy, next_token_cross_entropy


def test_losses_finite():
    logits = torch.randn(2, 3, 7)
    targets = torch.randint(0, 7, (2, 3))
    ce = next_token_cross_entropy(logits, targets)
    gates = torch.rand(2, 3, 5)
    entropy = dimension_entropy(gates)
    total, losses = combine_losses(
        DRMConfig(),
        ce,
        torch.tensor(0.5),
        torch.tensor(2.0),
        entropy,
        torch.tensor(0.1),
        torch.tensor(-0.1),
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
    )
    assert torch.isfinite(total)
    assert set(["ce", "action", "total"]).issubset(losses)
