# 007 - RTX 4090 Local Conversational Training Plan

Report date: 2026-07-15

## Scope

This report defines a practical local training plan for producing a DRM checkpoint that can be tested through an interactive chat loop on a single RTX 4090.

The previous 500M / 5B-token plan remains the right cloud-scale target, but it is not a practical local run:

```text
500M DRM on RTX 4090 measured throughput: ~220 tokens/sec
5B tokens at that rate: ~263 days
local time budget: 2 to 7 days
```

The local plan therefore optimizes for a checkpoint that can be trained, resumed, evaluated, and chatted with in a useful time window.

## Recommendation

Use a 125M-parameter DRM model for local base training, then add a short conversational fine-tuning stage later.

Immediate local target:

```text
model: DRM 125M
config: configs/drm_125m_4090.yaml
dataset: data/tokens_5b/manifest.json
hardware: 1 x RTX 4090
precision: bf16
sequence length: 512
batch/GPU: 2 initially
gradient accumulation: 8
```

The 125M model is the best local balance because it is large enough to learn byte-level language patterns, but small enough to train for hundreds of millions of tokens on a single 4090.

## Why Not 500M Locally

The 500M model is useful for cloud multi-GPU training, but it is too slow locally:

```text
500M local tokens/sec: ~220
tokens needed for 5B: 5,000,000,000
estimated runtime: ~263 days
```

Even a reduced 500M run of 10M tokens takes roughly half a day at the measured speed. That is useful as an engineering smoke test, not as a path to a conversational model.

## Local Training Budgets

Recommended local budgets:

| Time budget | Target tokens | Purpose |
|---:|---:|---|
| smoke | 1M-10M | verify stability, throughput, checkpointing |
| 2 days | 150M | first useful base checkpoint |
| 7 days | 500M | stronger local base checkpoint |

Expected behavior:

- A base model trained on Wikipedia-style data will complete text.
- It will not automatically behave like an assistant.
- Conversational quality requires a later supervised fine-tune on `User:` / `Assistant:` examples.

## Execution Plan

### Phase 1 - Base Training

Run the 125M model on the 5B-token shard manifest.

Default 2-day command target:

```text
target_tokens: 150,000,000
tokens/step: batch_size * grad_accum * seq_len = 2 * 8 * 512 = 8,192
steps: ~18,311
```

7-day target:

```text
target_tokens: 500,000,000
steps: ~61,036
```

### Phase 2 - Chat Smoke Test

After a checkpoint exists, run the chat wrapper against:

```text
runs/drm_125m_4090_base/checkpoint_last.pt
```

The first checkpoint may still sound like text completion. That is expected before instruction/dialogue tuning.

### Phase 3 - Conversational Fine-Tuning

Next implementation milestone:

- create a small high-quality dialogue dataset;
- train on examples formatted as:

```text
User: ...
Assistant: ...
```

- optionally mask loss so only assistant tokens count;
- save a separate dialogue-tuned checkpoint.

This fine-tune is expected to improve interactive behavior more than simply adding a small number of additional Wikipedia tokens.

## Commands

Dry-run:

```powershell
.\scripts\run_drm_125m_4090_base.ps1 `
  -DryRun `
  -DryRunForward `
  -OutputRoot runs\drm_125m_4090_base_dryrun
```

2-day base run:

```powershell
.\scripts\run_drm_125m_4090_base.ps1 `
  -OutputRoot runs\drm_125m_4090_base `
  -TargetTokens 150000000 `
  -BatchSize 2 `
  -GradAccumSteps 8 `
  -SeqLen 512 `
  -Precision bf16 `
  -Resume latest
```

7-day base run:

```powershell
.\scripts\run_drm_125m_4090_base.ps1 `
  -OutputRoot runs\drm_125m_4090_base_500m `
  -TargetTokens 500000000 `
  -BatchSize 2 `
  -GradAccumSteps 8 `
  -SeqLen 512 `
  -Precision bf16 `
  -Resume latest
```

If VRAM is tight:

```powershell
-BatchSize 1 -GradAccumSteps 16
```

Chat after training:

```powershell
.\scripts\chat_drm_125m_4090.ps1 `
  -Checkpoint runs\drm_125m_4090_base\checkpoint_last.pt `
  -Device cuda
```

## Acceptance Criteria

- Dry-run forward completes on RTX 4090 in bf16.
- Training writes `checkpoint_latest.pt` and `checkpoint_last.pt`.
- `metrics_latest.json` shows stable CE and nonzero tokens/sec.
- Checkpoint loads in the chat script.
- If base chat is not assistant-like, proceed to supervised dialogue fine-tuning rather than increasing model size locally.
