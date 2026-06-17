from pathlib import Path

import torch.nn as nn

from drm_language_emitter import DRMConfig, DRMEmitterModel


def test_model_does_not_use_multihead_attention():
    model = DRMEmitterModel(DRMConfig(vocab_size=13, d_token=8, d_state=12, n_directions=4, metric_rank=2, hidden_size=16))
    assert not any(isinstance(module, nn.MultiheadAttention) for module in model.modules())


def test_source_does_not_define_qkv_attention():
    root = Path(__file__).resolve().parents[1] / "src" / "drm_language_emitter"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    forbidden = ["MultiheadAttention", "q_proj", "k_proj", "v_proj", "query_proj", "key_proj", "value_proj"]
    for token in forbidden:
        assert token not in source
