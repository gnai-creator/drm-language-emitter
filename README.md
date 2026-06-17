# DRM Language Emitter

DRM Language Emitter is an experimental non-Transformer language model in which token generation is driven by trajectories on a Directional Relational Manifold. Instead of self-attention, the model evolves a latent state through active directional fields, a learned relational metric, and low-action dynamics, then emits tokens from the resulting state.

This repository is a research scaffold. It is not validated as a competitive language model.

## What It Is

The model is an autoregressive language emitter whose hidden computation is a geometric rollout:

```text
input token -> embedding e_t
latent state z_t in M
active directions D(z_t) with gates a_i(z_t)
relational metric g_z
velocity dz in span(D(z_t))
state update z_{t+1} = z_t + dt * dz
token logits p(token_{t+1} | z_{t+1})
```

There is no self-attention, no query/key/value mechanism, no Transformer block, and no KV cache.

## Difference From A Transformer

Transformers compute token-token interactions through attention over a sequence. DRM Language Emitter keeps a single evolving latent state and updates it through a learned directional field. The model can emit one token at a time, but its central operation is not attention over previous tokens. Memory enters through the trajectory of `z_t`.

## Architecture Diagram

```text
input_ids
  |
TokenEmbedding
  |
for each time step:
  z_t -----------------------+
   |                         |
DirectionField(z_t)          |
  -> directions V(z_t)       |
  -> gates a(z_t), dimD      |
   |                         |
RelationalMetric(z_t)        |
  -> diag + U U^T            |
   |                         |
DRMFlow(z_t, e_t, V, a)      |
  -> dz in span active D     |
   |                         |
metric action g_z(dz,dz)     |
   |                         |
StateUpdater                 |
  -> z_{t+1}                 |
   |                         |
LanguageEmitter(z_{t+1}) ----+
  -> logits
```

## Quickstart

```bash
pip install -e .
pytest -q
python scripts/train_tiny.py --config configs/tiny.yaml --text data/tiny.txt
python scripts/generate.py --checkpoint runs/tiny/drm_tiny.pt --prompt "DRM "
python scripts/eval_geometry.py --checkpoint runs/tiny/drm_tiny.pt
python scripts/eval_geodesic_paths.py --checkpoint runs/tiny/drm_tiny.pt
```

If `data/tiny.txt` is missing, the training script creates a tiny fallback corpus. The default tokenizer is byte-level with vocabulary size 256, so prompts such as `DRM`, punctuation, digits, and mixed case do not become unknown tokens.

## Tiny Training

`scripts/train_tiny.py` trains with next-token prediction on CPU by default. It logs train CE, fixed validation CE, approximate perplexity, action, effective active dimension, active fraction, metric norm, and a condition proxy.

## Generation

`scripts/generate.py` loads a checkpoint and tokenizer, warms the latent state with the prompt, then samples tokens from the emitter. Each sampled token updates the latent state through `DRMFlow`.

## Diagnostics

`scripts/eval_geometry.py` saves JSON metrics:

- `dimD_mean`, `dimD_std`, `active_fraction`
- `action_mean`
- `metric_U_norm_mean`, `metric_U_variance`
- `condition_proxy`
- `recurrence_proxy`, `stability_proxy`
- `risk_mass_mean` when the risk scaffold is active

`scripts/eval_geodesic_paths.py` compares linear interpolation against DRM rollout action. It evaluates learned low-action trajectories, not an exact geodesic solver.

`scripts/run_full_trainings.py` runs the standard sweep over full, risk, and fixed-dimension configs, then emits generations and geometry JSON reports for each run.

Recommended v3 stability run:

```bash
python scripts/run_full_trainings.py --steps 2000 --batch-size 16 --output-root runs/full_v3
```

Gate sparsity experiment:

```bash
python scripts/run_full_trainings.py --configs configs/tiny.yaml configs/tiny_gate_sparse.yaml configs/fixed_dim_ablation.yaml --steps 2000 --batch-size 16 --output-root runs/gate_sparsity_v3
```

Each training directory saves `drm_tiny_best.pt`, `drm_tiny_last.pt`, `drm_tiny.pt` as an alias of the best checkpoint, `tokenizer.json`, and `metrics.json`.

Summarize a completed sweep:

```bash
python scripts/summarize_runs.py --root runs/full_v3
```

## Tiny Transformer Baseline

Run a controlled DRM vs Transformer comparison:

```bash
python scripts/compare_drm_transformer.py --steps 400 --batch-size 8 --output-root runs/drm_vs_transformer_400
```

This saves:

```text
runs/drm_vs_transformer_400/comparison.json
runs/drm_vs_transformer_400/comparison.svg
```

Run a parameter/seed/step sweep:

```bash
python scripts/sweep_drm_transformer.py --steps 400 1000 2000 --seeds 1 2 3 --output-root runs/sweep_drm_transformer
```

## Competition Status

The current controlled 400-step tiny comparison shows the Tiny Transformer beating DRM in validation CE. DRM remains geometrically stable, but the project does not claim Transformer superiority or DRM superiority globally from this result.

Quick comparison:

```bash
python scripts/compare_drm_transformer.py --steps 50 --batch-size 4 --output-root runs/quick_compare
```

Competition sweep:

```bash
python scripts/sweep_drm_transformer.py --steps 400 1000 2000 --seeds 1 2 3 --output-root runs/sweep_drm_transformer
python scripts/summarize_competition.py --root runs/sweep_drm_transformer
```

Robustness:

```bash
python scripts/eval_robustness.py --drm-checkpoint runs/quick_compare/drm/drm_tiny.pt --drm-tokenizer runs/quick_compare/drm/tokenizer.json --transformer-checkpoint runs/quick_compare/transformer/tiny_transformer.pt
```

Bridge:

```bash
python scripts/eval_bridge_task.py --checkpoint runs/quick_compare/drm/drm_tiny.pt --tokenizer runs/quick_compare/drm/tokenizer.json
```

Sequence stability:

```bash
python scripts/eval_sequence_stability.py --drm-checkpoint runs/quick_compare/drm/drm_tiny.pt --drm-tokenizer runs/quick_compare/drm/tokenizer.json --transformer-checkpoint runs/quick_compare/transformer/tiny_transformer.pt
```

Profile:

```bash
python scripts/profile_drm.py --checkpoint runs/quick_compare/drm/drm_tiny.pt
```

Interpretation: the arena is designed to discover where DRM can win honestly: internal ablation, robustness, low-action bridge, recurrence/stability, parameter matching, or long-horizon learning.

## Scientific Status

The code implements a functional hypothesis: language generation can be driven by state trajectories over active relational directions and a learned metric. The present version is only a minimal CPU-runnable prototype.

## Limitations

- The temporal loop is slow compared with modern Transformer kernels.
- The tokenizer is only a simple character-level fallback.
- The learned low-action diagnostic is not a formal geodesic solver.
- No large benchmark, safety evaluation, RLHF, alignment claim, or production claim is included.
- Toroidal convergence is not guaranteed; it is only a possible diagnostic under boundedness, recurrence, and stability assumptions.

## Roadmap

- Add stronger trajectory integrators and explicit variational path objectives.
- Add richer tokenization and batching.
- Compare variable active dimension against fixed-dimension baselines.
- Study metric diversity, recurrence, and stability across training.
- Investigate pullback/Fisher-style metrics as future work.
