# Minimal Training Loop

This example shows the smallest useful in-repo training pattern. It uses the byte tokenizer and a tiny text string, so it is intended for smoke testing rather than quality.

```python
import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.data import build_tokenizer, make_lm_batch
from drm_language_emitter.model import DRMEmitterModel

text = "Directional relational manifolds guide language as trajectories.\n" * 16
config = DRMConfig(
    vocab_size=256,
    d_token=32,
    d_state=48,
    n_directions=8,
    metric_rank=4,
    hidden_size=64,
    max_seq_len=32,
)

tokenizer = build_tokenizer(text, config.tokenizer_type)
config.vocab_size = tokenizer.vocab_size
ids = tokenizer.encode(text)

device = torch.device("cpu")
model = DRMEmitterModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(10):
    x, y = make_lm_batch(ids, batch_size=4, seq_len=32, device=device)
    out = model(x, y, global_step=step + 1)
    optimizer.zero_grad(set_to_none=True)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    ce = out["aux_losses"]["ce"].detach().item()
    dim_d = out["diagnostics"]["dimD_mean"].detach().item()
    print(f"step={step + 1} ce={ce:.4f} dimD={dim_d:.2f}")
```

For reproducible project runs, prefer `scripts/train_tiny.py` and version the YAML config used for the run.
