from __future__ import annotations

import pytest
import torch

from world_model.symbolic_world_model import SymbolicWorldModel, SymbolicWorldModelConfig, encode_world_text


def run_forward(device: str) -> None:
    config = SymbolicWorldModelConfig(d_model=32, hidden_size=48, max_input_len=48, max_target_len=32)
    model = SymbolicWorldModel(config).to(device)
    x = torch.tensor(
        [encode_world_text("TASK=NEXT;S:N=5;A=1,1;G=4,4;W=.;T=0;ACT=R", config.max_input_len)],
        dtype=torch.long,
        device=device,
    )
    y = torch.tensor(
        [encode_world_text("NEXT:A=1,2;R=0;DONE=0", config.max_target_len, add_eos=True)],
        dtype=torch.long,
        device=device,
    )
    out = model(x, y)
    assert out["logits"].shape == (1, config.max_target_len, config.vocab_size)
    assert torch.isfinite(out["loss"])
    generated = model.generate(x, max_new_tokens=8)
    assert generated.shape == (1, 8)


def test_world_model_forward_cpu() -> None:
    run_forward("cpu")


def test_world_model_forward_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")
    run_forward("cuda")
