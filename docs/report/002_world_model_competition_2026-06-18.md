# Report 002: Tiny Symbolic World Model Competition

Date: 2026-06-18

## Objective

Compare three small model families on a deterministic symbolic gridworld serialized as text:

- DRM Language Emitter;
- Tiny Transformer;
- tiny supervised symbolic world model in `world_model/`.

The question is narrow:

> In a tiny symbolic world serialized as language, which family performs best under CE, next-state prediction, rollout prediction, invalid-state rate, and efficiency metrics?

This report does not compare against large multimodal world models.

## Commands

Dataset:

```powershell
.\.venv\Scripts\python.exe scripts\make_tiny_world_dataset.py --output-root data\tiny_world --seed 1 --grid-size 5 --num-train 20000 --num-val 2000 --max-rollout-len 8
```

Sweep:

```powershell
.\.venv\Scripts\python.exe scripts\sweep_world_model_competition.py --steps 1000 2000 3000 --seeds 1 2 3 --dataset-root data\tiny_world --output-root runs\world_model_competition
```

Dashboard:

```powershell
.\.venv\Scripts\python.exe scripts\make_world_model_dashboard.py --root runs\world_model_competition --title "DRM vs Transformer vs Tiny Symbolic World Model"
```

Output:

```text
runs/world_model_competition/dashboard.html
runs/world_model_competition/summary.json
runs/world_model_competition/aggregate.csv
runs/world_model_competition/competition_table.md
runs/world_model_competition/*.svg
```

Versioned copy:

```text
docs/benchmarks/world_model_competition/dashboard.html
docs/benchmarks/world_model_competition/summary.json
docs/benchmarks/world_model_competition/aggregate.csv
docs/benchmarks/world_model_competition/competition_table.md
docs/benchmarks/world_model_competition/*.svg
```

## CUDA Status

CUDA was requested as a follow-up check, but this local environment is CPU-only:

```text
torch=2.12.0+cpu
cuda_available=False
cuda_device_count=0
```

The benchmark therefore ran on CPU. A CUDA run requires installing a CUDA-enabled PyTorch build and exposing a CUDA device to the environment.

## Run Size

The completed benchmark produced:

```text
runs: 72
aggregate rows: 24
```

The full dataset is generated and intentionally ignored by git:

```text
data/tiny_world/
data/tiny_world_smoke/
```

These folders are reproducible via `scripts/make_tiny_world_dataset.py`.

## Top Results By Next-State Exact Match

| model | steps | family | next_state_exact_match | rollout_exact_match | best_val_ce | invalid_state_rate | params |
|---|---:|---|---:|---:|---:|---:|---:|
| `drm_tiny` | 2000 | DRM | 0.0751 | 0.0058 | 0.5511 | 0.1328 | 92710 |
| `transformer_tiny_220k` | 3000 | Transformer | 0.0563 | 0.0000 | 0.4008 | 0.0026 | 220208 |
| `transformer_tiny_93k` | 2000 | Transformer | 0.0516 | 0.0000 | 0.4594 | 0.2969 | 93872 |
| `world_model_tiny` | 2000 | World Model | 0.0476 | 0.0000 | 0.2573 | 0.4668 | 102051 |
| `transformer_tiny_220k` | 2000 | Transformer | 0.0469 | 0.0000 | 0.4201 | 0.0365 | 220208 |
| `transformer_tiny_93k` | 3000 | Transformer | 0.0469 | 0.0058 | 0.4232 | 0.1875 | 93872 |
| `world_model_tiny` | 3000 | World Model | 0.0415 | 0.0000 | 0.2497 | 0.4668 | 102051 |
| `transformer_tiny` | 2000 | Transformer | 0.0376 | 0.0000 | 0.4539 | 0.1875 | 104192 |

## Interpretation

The strongest next-state exact-match result came from `drm_tiny @ 2000`.

However, absolute exact-match values are low. This means the benchmark is currently more diagnostic than decisive. It shows a signal that DRM can compete on a symbolic transition metric, but it does not yet demonstrate robust symbolic world modeling.

The Transformer 220k model produced the lowest invalid-state rate among the top next-state rows, which means its outputs were syntactically/structurally better constrained in that regime.

The tiny supervised world model reached low supervised CE, especially `world_model_tiny @ 3000`, but this did not translate into strong exact-match or rollout performance. This suggests the supervised decoder is learning token-level regularities without reliably producing exact symbolic transitions.

## Claims Allowed

- The benchmark infrastructure now compares DRM, Transformer, and a tiny top-level `world_model/` family on the same serialized symbolic dataset.
- In this run, `drm_tiny @ 2000` had the best next-state exact-match score.
- The Transformer 220k model had stronger invalid-state behavior among the top rows.
- The tiny world model achieved low supervised CE but weak exact-match metrics.

## Claims Not Allowed

- DRM is broadly better than Transformers.
- DRM is broadly better than world models.
- This benchmark says anything about large multimodal world models.
- Low CE alone proves symbolic transition understanding.

## Next Steps

- Run the same benchmark under CUDA after installing a CUDA-enabled PyTorch build.
- Improve symbolic decoding/evaluation so rollout exact match is not near zero across families.
- Add constrained decoding for all families, or explicitly report unconstrained vs constrained decoding separately.
- Add easier curriculum variants before the full wall/rollout task.
- Add time-matched runs for this benchmark once CUDA is available.
