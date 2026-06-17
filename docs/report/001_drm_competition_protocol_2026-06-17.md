# DRM Competition Protocol Report

Date: 2026-06-17

## Summary

This report freezes the current DRM Language Emitter work as a measurable research baseline and adds the next experimental layer required to test whether the geometry matters.

The current evidence says:

- DRM Language Emitter trains and generates.
- The model is non-Transformer and does not use attention in `src/drm_language_emitter`.
- The relational metric is active and numerically controlled.
- The tiny Transformer baseline currently wins validation CE in the 400-step tiny comparison.
- DRM shows useful geometric diagnostics, including stable condition proxy and improved low-action trajectory ratio.

The next claim to test is not "DRM beats Transformer" globally. The next claim is:

> DRM geometry provides causal value relative to internal ablations and may win on tasks where compact state, trajectory continuity, perturbation robustness, or low-action bridging matter.

## Implemented Features

### 1. Stronger DRM Configs

Added:

- `configs/tiny_drm_stronger.yaml`
- `configs/tiny_drm_topk_gates.yaml`
- `configs/tiny_104k.yaml`

These configs test:

- larger latent state;
- more directions;
- higher metric rank;
- two flow steps per token;
- residual SwiGLU emitter;
- top-k directional gates;
- parameter matching against the current Transformer baseline.

### 2. Transformer Baselines

Added:

- `transformer/tiny_transformer.py`
- `transformer/train_tiny_transformer.py`
- `transformer/run_train.py`
- `transformer/checkpoint.py`
- `transformer/tiny_transformer.yaml`
- `transformer/tiny_transformer_93k.yaml`

The Transformer lives outside `src/drm_language_emitter` so it does not contaminate the DRM package or the no-attention tests.

### 3. DRM vs Transformer Comparison

Added:

- `scripts/compare_drm_transformer.py`
- `scripts/sweep_drm_transformer.py`

The comparison records:

- train CE;
- validation CE;
- parameter count;
- elapsed seconds;
- tokens seen;
- tokens/sec;
- DRM geometry diagnostics;
- DRM low-action trajectory diagnostics;
- SVG plot of CE curves.

### 4. Robustness Evaluation

Added:

- `scripts/eval_robustness.py`

This corrupts input context bytes while keeping next-token targets clean. It measures CE degradation under perturbation for DRM and, optionally, Transformer.

This is a better target for DRM than tiny CE alone because it tests whether the latent trajectory can absorb context noise.

### 5. Bridge / Interpolation Evaluation

Added:

- `scripts/eval_bridge_task.py`

This measures whether a rollout from prompt A using prompt B tokens approaches the prompt B state with low metric action.

This does not prove geodesic emergence. It is a diagnostic for the kind of low-action trajectory behavior DRM should eventually improve.

## Current 400-Step Baseline

Command:

```bash
python scripts/compare_drm_transformer.py --steps 400 --batch-size 8 --output-root runs/drm_vs_transformer_400
```

Observed result:

- DRM best validation CE: `2.8752`
- Tiny Transformer best validation CE: `2.1369`
- DRM parameters: `92,710`
- Tiny Transformer parameters: `104,192`
- DRM condition proxy: `91.86`
- DRM metric U norm mean: `1.11`
- DRM rollout/linear action ratio: `12.78`

Interpretation:

The Transformer wins next-token CE in this tiny run. DRM remains stable and measurable, but this is not evidence of Transformer superiority in every regime or DRM superiority in this regime.

## Required Experiments

### Internal Ablation

Run:

```bash
python scripts/run_full_trainings.py --steps 2000 --batch-size 16 --output-root runs/full_v3
python scripts/summarize_runs.py --root runs/full_v3
```

The decisive question:

- Does full DRM beat `no_metric_U`, `fixed_dimension`, `no_direction_gates`, and `no_action_loss`?

### Parameter-Matched Transformer Sweep

Run:

```bash
python scripts/sweep_drm_transformer.py --steps 400 1000 2000 --seeds 1 2 3 --output-root runs/sweep_drm_transformer
```

This compares:

- current DRM vs current Transformer;
- DRM near 104k params vs Transformer near 93k params;
- multiple seeds;
- multiple training horizons;
- elapsed time and tokens/sec.

### Stronger DRM Candidate

Run:

```bash
python scripts/compare_drm_transformer.py --drm-config configs/tiny_drm_stronger.yaml --steps 1000 --batch-size 8 --output-root runs/drm_stronger_vs_transformer_1000
```

Track whether the stronger DRM improves CE without losing metric stability.

### Robustness

After a comparison run:

```bash
python scripts/eval_robustness.py \
  --drm-checkpoint runs/drm_vs_transformer_400/drm/drm_tiny.pt \
  --drm-tokenizer runs/drm_vs_transformer_400/drm/tokenizer.json \
  --transformer-checkpoint runs/drm_vs_transformer_400/transformer/tiny_transformer.pt \
  --output runs/drm_vs_transformer_400/robustness.json
```

Look for lower CE degradation under noise.

### Bridge

After a DRM run:

```bash
python scripts/eval_bridge_task.py \
  --checkpoint runs/drm_vs_transformer_400/drm/drm_tiny.pt \
  --tokenizer runs/drm_vs_transformer_400/drm/tokenizer.json \
  --output runs/drm_vs_transformer_400/bridge_task.json
```

Look for lower bridge action and smaller endpoint distance over training.

## Claim Discipline

Allowed current claim:

> DRM Language Emitter is a stable, measurable, non-Transformer tiny language model with active relational geometry and a working Transformer comparison harness.

Not yet allowed:

- DRM beats Transformer.
- DRM geodesics have emerged strongly.
- DRM topology is toroidal.
- DRM is safer or aligned.
- DRM generalizes on real corpora.

## Next Decision Gate

The next milestone is one controlled win:

1. full DRM beats its internal ablations; or
2. DRM beats a parameter-matched Transformer in validation CE; or
3. DRM beats Transformer on robustness, bridge, recurrence, or perturbation stability.

Without at least one of these, the project remains promising architecture research rather than a validated alternative.
