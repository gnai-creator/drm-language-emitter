# Tiny Symbolic World Model Competition

## Objective

This benchmark compares three small model families on the same serialized symbolic gridworld:

- DRM Language Emitter;
- Tiny Transformer baseline;
- Tiny supervised symbolic world model.

The question is narrow: in a tiny symbolic world serialized as text, which family performs best under CE, next-state prediction, rollout prediction, and efficiency metrics?

This benchmark does not compare against large multimodal world models and does not support claims about general world-model superiority.

## Tiny Symbolic World

The environment is a deterministic 2D grid:

- agent position;
- goal position;
- optional walls;
- actions: `U`, `D`, `L`, `R`;
- reward is `1` when the agent reaches the goal;
- done is true when reward is `1` or max rollout length is reached.

Examples are serialized as text:

```text
TASK=NEXT;S:N=5;A=1,2;G=4,4;W=2,2|3,2;T=0;ACT=R => NEXT:A=1,3;R=0;DONE=0
```

Rollout examples serialize an initial state plus action list and target state sequence.

## Models

DRM and Transformer are trained as byte-level autoregressive language models on the same text corpus generated from the JSONL dataset.

The tiny world model lives in the top-level `world_model/` package. It is a supervised seq2seq GRU model:

- token encoder over the serialized input;
- recurrent latent state;
- recurrent decoder over target bytes;
- supervised target-token CE.

## Metrics

Common metrics:

- `best_val_ce`;
- `final_val_ce`;
- `next_state_exact_match`;
- `reward_accuracy`;
- `done_accuracy`;
- `rollout_exact_match`;
- `rollout_token_accuracy`;
- `invalid_state_rate`;
- `parameter_count`;
- `elapsed_sec`;
- `tokens_seen`;
- `tokens_per_sec`.

CE is not interpreted as identical across all families when the world model uses a supervised decoder head. The symbolic exact-match and rollout metrics are the primary world-modeling comparison.

## Commands

Generate the dataset:

```bash
python scripts/make_tiny_world_dataset.py --output-root data/tiny_world --seed 1 --grid-size 5 --num-train 20000 --num-val 2000 --max-rollout-len 8
```

Run the competition:

```bash
python scripts/sweep_world_model_competition.py --steps 1000 2000 3000 --seeds 1 2 3 --dataset-root data/tiny_world --output-root runs/world_model_competition
```

Build the dashboard:

```bash
python scripts/make_world_model_dashboard.py --root runs/world_model_competition --title "DRM vs Transformer vs Tiny Symbolic World Model"
```

Smoke run:

```bash
python scripts/make_tiny_world_dataset.py --output-root data/tiny_world_smoke --seed 1 --grid-size 5 --num-train 512 --num-val 128 --max-rollout-len 6
python scripts/sweep_world_model_competition.py --steps 20 --seeds 1 --dataset-root data/tiny_world_smoke --output-root runs/world_model_competition_smoke
python scripts/make_world_model_dashboard.py --root runs/world_model_competition_smoke --title "Smoke: DRM vs Transformer vs Tiny World Model"
```

## Interpretation

Report the winning model per metric. If DRM wins CE but loses rollout, say that. If the world model wins rollout but loses CE, say that. If Transformer is much faster, report throughput plainly.

Safe phrasing:

> In this tiny symbolic text-world benchmark, a model family performs best under a specific metric. This does not imply superiority over general multimodal world models.

## Limitations

- The world is tiny and deterministic.
- Text serialization may favor language models differently from structured models.
- DRM and Transformer use autoregressive decoding for symbolic targets, while the world model uses supervised seq2seq decoding.
- CUDA is optional; the benchmark must run on CPU.
- Large world-model claims are out of scope.

## Next Steps

- Add harder layouts and longer rollouts.
- Add out-of-distribution wall patterns.
- Add time-matched comparison.
- Add explicit sample-efficiency curves.
