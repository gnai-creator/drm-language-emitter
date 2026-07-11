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


def test_metric_rank_zero_does_not_add_u_floor_loss():
    config = DRMConfig(
        vocab_size=17,
        d_token=8,
        d_state=12,
        n_directions=4,
        metric_rank=0,
        hidden_size=16,
        max_seq_len=8,
        lambda_metric_u_floor=1.0,
    )
    model = DRMEmitterModel(config)
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y)
    assert out["diagnostics"]["metric_u_floor_loss"].item() == 0.0
    assert "metric_u_floor" not in out["aux_losses"]


def test_compiled_forward_failure_falls_back_to_eager():
    model = DRMEmitterModel(tiny_config())

    def broken_compiled_forward(*args, **kwargs):
        raise RuntimeError("compile backend unavailable")

    model._compiled_forward = broken_compiled_forward
    x = torch.randint(0, 17, (2, 6))
    y = torch.randint(0, 17, (2, 6))
    out = model(x, y)
    assert out["logits"].shape == (2, 6, 17)
    assert model._compiled_forward is None
