# 006 - DRM 500M / 5B Tokens Multi-GPU Training Roadmap

Report date: 2026-07-13

## Scope

This report defines the source-code preparation roadmap for training a larger DRM Language Emitter model suitable for a public demo, with the immediate target:

```text
Model scale: ~500M trainable parameters
Training budget: ~5B tokens
Hardware target: multi-GPU cloud node or cluster
Forward path: reusable for ~1B parameter DRM training
```

The goal is not just to make a larger model fit in memory. The goal is to produce a checkpoint that can generate legible language, survive interactive CLI testing, and support a credible technical presentation.

## Executive Summary

The current repository can train DRM models on a single device and has already produced a real 125M-parameter benchmark. That is enough for architecture validation, but not enough for a convincing conversational demo at larger scale.

Before training a 500M model for 5B tokens, the codebase needs four upgrades:

- multi-GPU training with DDP or FSDP;
- scalable streaming/tokenized data pipeline;
- robust checkpoint/resume and run metadata;
- evaluation and demo loops that measure generation quality, not only validation CE.

For the startup presentation, the recommended sequence is:

1. Build and validate a 500M DRM config.
2. Add single-node multi-GPU support first.
3. Run a short 500M smoke train.
4. Run a 500M long train for 5B tokens.
5. Fine-tune conversationally.
6. Preserve benchmark artifacts and demo checkpoints.
7. Reuse the same machinery for a later 1B run.

## Current Baseline

Current known real benchmark:

```text
drm_125m_real: 125,161,862 parameters
tokens seen per seed: 2,048,000
dataset file: data/wikipedia_en_20231101_500m.txt
dataset chars: 500,000,002
docs written: 106,450
```

Benchmark result from `docs/benchmarks/bench_125M/aggregate.csv`:

```text
DRM 125M train throughput: ~848 tokens/sec
DRM 125M peak CUDA memory: ~3.8 GB
```

That memory number means 500M parameters should fit on high-memory cloud GPUs, but throughput and training quality require additional engineering.

## Target Training Budget

For batch size `B`, sequence length `L`, gradient accumulation `A`, and steps `S`:

```text
tokens = S * A * B * L * world_size
```

The immediate 500M target is:

```text
tokens = 5,000,000,000
```

Example step counts:

| GPUs | batch/GPU | seq_len | grad_accum | tokens/step | steps for 5B |
|---:|---:|---:|---:|---:|---:|
| 8 | 4 | 512 | 1 | 16,384 | 305,176 |
| 8 | 8 | 512 | 1 | 32,768 | 152,588 |
| 8 | 8 | 1024 | 1 | 65,536 | 76,294 |
| 8 | 4 | 1024 | 2 | 65,536 | 76,294 |

Recommendation for first long run:

```text
seq_len: 512 initially
global batch tokens: 32K to 65K if stable
target tokens: 5B
eval interval: every 25M-50M tokens
checkpoint interval: every 100M-250M tokens
```

## Hardware Assumptions

The intended RunPod class discussed is:

```text
8 x H200 SXM
141 GB VRAM per GPU
approx price: US$ 4.39/GPU-hour
total hourly price: 8 * 4.39 = US$ 35.12/hour
```

Cost examples:

| duration | cost |
|---:|---:|
| 24 hours | US$ 842.88 |
| 3 days | US$ 2,528.64 |
| 7 days | US$ 5,900.16 |
| 14 days | US$ 11,800.32 |

These are infrastructure estimates only. Actual cost depends on availability, storage, network, provisioning overhead, failed runs, and restart time.

## Expected Runtime Range

The current single-GPU 125M DRM run is too small to predict 500M/1B multi-GPU throughput accurately. The correct plan is to measure tokens/sec after implementing DDP/FSDP.

Planning ranges for 5B tokens:

| effective throughput | time for 5B tokens | compute cost at US$35.12/h |
|---:|---:|---:|
| 10K tokens/sec | 5.8 days | ~US$4,880 |
| 20K tokens/sec | 2.9 days | ~US$2,440 |
| 40K tokens/sec | 1.45 days | ~US$1,220 |

The first objective is to make the 500M run reliable, not to assume the best-case throughput.

## Phase 1 - Model Scale Preparation

### Goals

- Create a real ~500M DRM config.
- Create a parameter-count utility to verify configs before training.
- Keep `drm_125m_real` intact for regression comparison.
- Prepare a future `drm_1b` config but do not train it before validating 500M.

### Required changes

- Add `configs/drm_500m.yaml`.
- Add `configs/drm_1b.yaml` as an experimental draft.
- Add a script such as `scripts/count_model_params.py`.
- Extend `MODEL_SPECS` in `scripts/run_scale_lm_comparison.py` or create a new large-run script registry.
- Add a dry-run command that reports parameter count, estimated activation settings, and config.

### Acceptance criteria

- `python scripts/count_model_params.py --config configs/drm_500m.yaml` reports 450M-550M parameters.
- `drm_500m` can instantiate on CPU for parameter counting.
- A tiny forward pass can run on one GPU with small `seq_len`.

## Phase 2 - Multi-GPU Training Backend

### Goals

- Use all GPUs in a single node.
- Preserve deterministic run metadata.
- Keep single-GPU training path working.

### Recommended first implementation

Use PyTorch DDP first, because it is simpler than FSDP and the current model likely fits per H200 at 500M.

DDP scope:

- launch with `torchrun`;
- one process per GPU;
- `DistributedDataParallel`;
- rank-aware logging;
- rank-zero-only checkpoint and artifact writing;
- distributed token accounting;
- `DistributedSampler` equivalent for token windows or rank-specific RNG streams.

### Later implementation

Add FSDP if:

- 1B model memory is inefficient;
- optimizer state dominates;
- activation memory blocks useful batch sizes;
- checkpoint size and load time become operational problems.

### Required changes

- Create `scripts/train_drm_distributed.py` or refactor `scripts/run_scale_lm_comparison.py` into reusable training functions.
- Add distributed initialization:

```text
torch.distributed.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

- Add global rank/world size handling.
- Replace naive random batch sampling with rank-safe sampling.
- Add rank-zero-only saves.
- Add barrier around checkpoint saves.
- Add `--resume` support.

### Acceptance criteria

- `torchrun --nproc_per_node=2 ... --dry-run-forward` works.
- Per-rank GPU utilization is nonzero.
- Only rank 0 writes summaries/checkpoints.
- Token counts include `world_size`.

## Phase 3 - Mixed Precision and Memory Controls

### Goals

- Train in `bf16` on H100/H200.
- Reduce memory pressure without destabilizing DRM.
- Support future 1B run.

### Required changes

- Add `--precision {fp32,bf16}`.
- Use `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` for forward pass.
- Keep loss scaling simple; bf16 usually does not need GradScaler.
- Add optional activation checkpointing for the geometry blocks if needed.
- Add `--compile` option only after correctness is stable.

### Acceptance criteria

- 125M and 500M dry-run losses are finite in bf16.
- No NaNs in `action_mean`, `condition_proxy`, or CE.
- Checkpoint reload works from bf16-trained state.

## Phase 4 - Data Pipeline for 5B Tokens

### Problem

The current runner reads the entire text file into memory and samples random windows from a Python list of token IDs. This is acceptable for 500M characters but not ideal for multi-billion-token training.

### Goals

- Use a dataset large enough for 5B token training.
- Avoid loading huge corpora as one Python string.
- Make sampling deterministic and distributed.

### Recommended stages

Stage A: use the existing 500M-char Wikipedia file for smoke tests.

Stage B: prepare a larger corpus:

```text
Wikipedia EN/PT
FineWeb/OpenWebText-style corpus if licensing allows
project documentation
technical Q&A
conversation/instruction data for fine-tuning
```

Stage C: tokenize into binary shards:

```text
data/tokens/train_000000.bin
data/tokens/train_000001.bin
...
data/tokens/val_000000.bin
```

For byte-level tokenization, each byte can be stored as `uint8`.

### Required changes

- Add `scripts/tokenize_corpus_to_uint8.py`.
- Add a memory-mapped token dataset.
- Add train/validation shard manifests.
- Add rank-aware random window sampling.
- Record exact dataset manifest hash in every run.

### Acceptance criteria

- Training can run without loading the whole corpus into RAM.
- Validation set is fixed and never sampled for training.
- Each run writes dataset manifest metadata.

## Phase 5 - Robust Checkpointing and Resume

### Goals

Long cloud runs must survive interruption.

### Required changes

- Save checkpoints every N tokens, not just every N steps.
- Include:
  - model state;
  - optimizer state;
  - scheduler state if added;
  - global step;
  - tokens seen;
  - RNG states;
  - config;
  - dataset manifest;
  - git commit hash;
  - world size;
  - precision.
- Add `--resume latest`.
- Keep a small number of rolling checkpoints.
- Save best validation checkpoint separately.

### Acceptance criteria

- Kill and resume a dry-run training job.
- Resumed job continues token count and logs correctly.
- Final summary marks whether the run was resumed.

## Phase 6 - Evaluation and Demo Readiness

### Goals

Validation CE is necessary but not enough for the presentation. The model must be tested through generation.

### Required changes

- Add `scripts/eval_generation_prompts.py`.
- Add prompt suites:
  - Portuguese conversational prompts;
  - English conversational prompts;
  - DRM technical questions;
  - project pitch questions;
  - factual sanity prompts;
  - refusal/safety prompts.
- Save generated samples at fixed intervals.
- Track repetition, invalid UTF-8 replacement rate, average length, and stop-marker behavior.

### Acceptance criteria

- Every checkpoint interval produces a sample report.
- Demo prompts are versioned.
- The final checkpoint can be loaded by `scripts/chat_drm_125m_real.py` or a generalized chat script.

## Phase 7 - Conversational Fine-Tuning

### Why this is required

A base language model trained on Wikipedia or web text will complete text. It will not automatically behave like an assistant. For the presentation, a short supervised fine-tune is likely more valuable than adding another few hundred million pretraining tokens.

### Dataset

Create a small high-quality instruction/dialogue set:

- project Q&A;
- technical FAQ;
- pitch explanations;
- questions about DRM vs Transformer;
- limitations and safety answers;
- Portuguese and English examples;
- expected demo prompts.

### Required changes

- Add a supervised fine-tuning script using the same byte tokenizer.
- Use a prompt format, for example:

```text
User: ...
Assistant: ...
```

- Mask loss optionally so only assistant tokens contribute.
- Save a separate fine-tuned checkpoint.

### Acceptance criteria

- Fine-tuned model gives stable answers to 20-50 prepared demo prompts.
- Base checkpoint remains preserved.
- Fine-tuned checkpoint is clearly labeled as instruction/dialogue tuned.

## Phase 8 - 1B Readiness

The 1B model should not be trained before the 500M run proves:

- multi-GPU training works;
- checkpoint resume works;
- dataset pipeline scales;
- generation improves with tokens;
- cost/time estimates are grounded in measured throughput.

### 1B-specific additions

- Add FSDP if DDP memory becomes inefficient.
- Add optimizer state sharding.
- Increase dataset target to 10B-20B tokens.
- Use measured 500M scaling to estimate budget.

### Gate to launch 1B

Do not launch 1B until:

```text
500M smoke run passes
500M 100M-token run is stable
500M checkpoint resumes cleanly
500M generation samples improve over time
500M fine-tune produces usable demo answers
```

## Proposed Commands

### Count parameters

```powershell
.\.venv\Scripts\python.exe scripts\count_model_params.py --config configs\drm_500m.yaml
```

### Distributed smoke run

```powershell
torchrun --nproc_per_node=8 scripts\train_drm_distributed.py `
  --config configs\drm_500m.yaml `
  --dataset-manifest data\tokens\manifest.json `
  --output-root runs\drm_500m_smoke `
  --steps 100 `
  --seq-len 512 `
  --batch-size 1 `
  --precision bf16
```

### Long 5B-token run

```powershell
torchrun --nproc_per_node=8 scripts\train_drm_distributed.py `
  --config configs\drm_500m.yaml `
  --dataset-manifest data\tokens\manifest.json `
  --output-root runs\drm_500m_5b `
  --target-tokens 5000000000 `
  --seq-len 512 `
  --batch-size 4 `
  --grad-accum-steps 1 `
  --precision bf16 `
  --eval-tokens-interval 50000000 `
  --checkpoint-tokens-interval 250000000 `
  --resume latest
```

These commands are targets for the roadmap. They require the implementation work above before they can be expected to run.

## Cost Control Rules

- Always run a 100-step smoke test before launching a long cloud job.
- Always validate checkpoint resume before spending more than a few hours.
- Start with one seed for long 500M training.
- Save enough checkpoints to recover, but not enough to exhaust storage.
- Keep checkpoints out of git.
- Copy only lightweight artifacts into `docs/benchmarks/`.
- Track tokens/sec early and stop if throughput is far below expectation.

## Deliverables Checklist

### Required before 500M long run

- [ ] `configs/drm_500m.yaml`
- [ ] parameter-count script
- [ ] DDP or FSDP training script
- [ ] bf16 support
- [ ] memory-mapped token dataset
- [ ] checkpoint/resume
- [ ] run metadata with git hash and hardware
- [ ] generation evaluation prompt suite

### Required before presentation

- [ ] final base checkpoint
- [ ] final conversation-tuned checkpoint
- [ ] generated demo samples
- [ ] technical benchmark dashboard
- [ ] public limitations section
- [ ] fallback 125M checkpoint demo if 500M underperforms

### Required before 1B run

- [ ] 500M stability confirmed
- [ ] 500M generation quality acceptable
- [ ] measured throughput and cost model
- [ ] FSDP/optimizer sharding if needed
- [ ] 10B-20B token dataset plan

## Final Recommendation

For Startup Summit 2026, the strongest engineering path is:

```text
500M DRM + 5B tokens + conversational fine-tune
```

This is ambitious but more defensible than jumping directly to 1B. The 1B model should remain the next scaling milestone once the 500M pipeline proves stable and cost-effective.
