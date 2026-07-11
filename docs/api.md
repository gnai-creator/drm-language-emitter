# API Reference

This page documents the stable public surface used by scripts and tests. The project is still a research scaffold, so internal module details may change between experiments.

## Configuration

`drm_language_emitter.config.DRMConfig`

Main fields:

- `vocab_size`, `d_token`, `d_state`, `n_directions`, `metric_rank`, `hidden_size`: model shape.
- `n_flow_steps`, `dt`, `bounded_state`: recurrent state update controls.
- `use_powerlaw_risk`, `risk_mass_max`, `risk_exponent_min`, `risk_exponent_max`, `risk_alpha_max`: blindspot/dubiety risk controls.
- `use_metric_naturalization`, `metric_naturalization_strength`, `metric_damping`: metric preconditioning controls.
- `use_torch_compile`: opt-in compilation of the DRM forward path with fallback to eager execution.

`DRMConfig.from_dict(data)` rejects unknown keys. This is intentional: experiment config typos should fail before training starts.

## Model

`drm_language_emitter.model.DRMEmitterModel`

```python
out = model(input_ids, targets=None, return_states=False, global_step=None)
```

Inputs:

- `input_ids`: `LongTensor` with shape `[batch, seq_len]`.
- `targets`: optional `LongTensor` with shape `[batch, seq_len]`.
- `return_states`: when true, includes latent states with shape `[batch, seq_len, d_state]`.
- `global_step`: optional integer used for metric naturalization warmup.

Output keys:

- `logits`: token logits `[batch, seq_len, vocab_size]`.
- `loss`: total scalar loss.
- `aux_losses`: component losses.
- `diagnostics`: scalar tensors for geometry, gates, action, metric condition, and risk.
- `states`: present only when `return_states=True`.

## Generation

`drm_language_emitter.generation.generate`

```python
tokens = generate(model, input_ids, max_new_tokens=32, temperature=0.9, top_k=20)
```

Generation replays the prompt into the latent state, samples from the emitter, and advances the state with each generated token. It does not use attention or a KV cache.

## Tokenizers

- `ByteTokenizer`: fixed UTF-8 byte vocabulary of size 256.
- `CharTokenizer`: character vocabulary trained from supplied text.

Use `drm_language_emitter.tokenizer.load_tokenizer(path)` to reload saved tokenizer metadata.

## Core Modules

- `DirectionField`: maps latent state `z` to directions `[B, n_directions, d_state]` and gates `[B, n_directions]`.
- `RelationalMetric`: returns a positive diagonal metric and optional low-rank factor `U`.
- `DRMFlow`: computes velocity constrained to active directions.
- `StateUpdater`: applies the recurrent state update and bounded-state clipping.
- `RiskField`: optional bounded risk signal that thickens metric energy.
- `LanguageEmitter`: decodes latent state to token logits.
