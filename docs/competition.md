# DRM Competition Phase 1

## Current Result

In the controlled 400-step tiny byte-level comparison, the Tiny Transformer baseline outperformed DRM Language Emitter in validation CE:

- DRM best validation CE: `2.8752`
- Tiny Transformer best validation CE: `2.1369`

This does not support a claim that DRM beats Transformers in language modeling CE.

At the same time, DRM maintained stable geometry:

- active relational metric;
- controlled condition proxy;
- non-collapsed metric low-rank term;
- measurable low-action trajectory diagnostics.

## Permitted Claims

- DRM Language Emitter is a functional non-Transformer language model.
- DRM geometry is active and measurable in the current implementation.
- The project now has a controlled Transformer comparison harness.
- Competition Phase 1 can test whether DRM wins in specific arenas.

## Non-Permitted Claims

- DRM beats Transformers globally.
- DRM is production ready.
- DRM has proven emergent geodesics.
- DRM has proven toroidal topology.
- DRM has safety, alignment, or robustness guarantees.

## Arenas

### Step-Matched

Train every model for the same optimizer-step budgets. Current long sweep:

- `1000` steps;
- `2000` steps;
- `3000` steps.

This answers: which model gets lower validation CE for the same number of updates?

### Parameter-Matched

Compare explicit parameter bands:

- `drm_tiny` vs `transformer_tiny_93k`;
- `drm_tiny_104k` vs `transformer_tiny`;
- `drm_stronger` vs `transformer_tiny_220k`.

The dashboard reports `param_ratio`, `gap_abs`, `gap_rel`, and `speed_ratio` for these pairs.

### Time-Matched

Train each model for the same wall-clock budget, such as:

- `60` seconds;
- `300` seconds;
- `900` seconds.

This answers whether DRM's lower CE per step compensates for slower throughput.

### Target-CE

Measure steps and seconds until validation CE crosses:

- CE `< 1.0`;
- CE `< 0.75`;
- CE `< 0.5`.

This separates final quality from time-to-threshold.

### CE / PPL

Next-token validation CE and approximate perplexity. This is the Transformer's strongest arena.

### Internal Ablation

Compare full DRM against variants such as no metric, fixed dimension, no gates, and no action loss. A win here tests whether geometry has causal value.

### Robustness

Corrupt input context with random bytes, zero bytes, adjacent swaps, or deleted context bytes. Measure CE degradation and recovery score.

### Bridge

Measure whether a rollout from one prompt state toward another prompt state reduces endpoint distance with controlled action. This is a low-action bridge diagnostic, not a formal geodesic solver.

### Sequence Stability

Test repeated, alternating, delayed-copy, and noisy repeated patterns. Measure DRM state norm, drift, recurrence proxy, stability proxy, and Transformer CE degradation over sequence length.

### Efficiency

Measure tokens/sec and module-level DRM profile. Current DRM is much slower than Transformer.

## Decision Gate

The project moves to a stronger claim only after at least one controlled win:

- full DRM beats its internal ablations;
- DRM 104k approaches or beats Transformer with comparable parameters;
- DRM degrades less under noise;
- DRM wins bridge/recovery/stability diagnostics;
- DRM improves more with longer training horizon.

Until then, the honest claim is that DRM is a stable research architecture with an active competitive arena.

## Commands

Quick comparison:

```bash
python scripts/compare_drm_transformer.py --steps 50 --batch-size 4 --output-root runs/quick_compare
```

Competition sweep:

```bash
python scripts/sweep_drm_transformer.py --steps 1000 2000 3000 --seeds 1 2 3 --output-root runs/sweep_drm_transformer
```

Dashboard:

```bash
python scripts/make_competition_dashboard.py --root runs/sweep_drm_transformer --title "DRM vs Transformer Sweep"
```

Time-matched:

```bash
python scripts/time_matched_competition.py --durations-sec 60 300 900 --output-root runs/time_matched_competition
```

Summarize:

```bash
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
