import torch

from drm_language_emitter import DRMConfig, DRMEmitterModel


def tiny_config() -> DRMConfig:
    return DRMConfig(vocab_size=17, d_token=8, d_state=12, n_directions=4, metric_rank=2, hidden_size=16, max_seq_len=8)


def test_forward_cpu_shapes_and_loss_finite():
    model = DRMEmitterModel(tiny_config())
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y)
    assert out["logits"].shape == (2, 6, 17)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["aux_losses"]["ce"])
