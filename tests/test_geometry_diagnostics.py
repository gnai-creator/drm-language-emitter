import json

import torch

from drm_language_emitter import DRMConfig, DRMEmitterModel
from drm_language_emitter.utils import to_jsonable


def test_geometry_diagnostics_json_serializable():
    model = DRMEmitterModel(DRMConfig(vocab_size=19, d_token=8, d_state=12, n_directions=4, metric_rank=2, hidden_size=16))
    x = torch.randint(0, 19, (2, 5))
    out = model(x, x)
    payload = to_jsonable(out["diagnostics"])
    json.dumps(payload)
    assert "dimD_mean" in payload
    assert "action_mean" in payload
    assert "condition_proxy" in payload
