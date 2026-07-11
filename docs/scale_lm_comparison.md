# Scale LM Comparison: DRM vs GPT-2 vs OPT

This benchmark trains DRM Language Emitter, GPT-2-style, and OPT-style causal language models under one local protocol:

- same Wikipedia EN text snapshot/sample;
- same byte tokenizer by default;
- same sequence length;
- same batch size and gradient accumulation;
- same steps and seeds;
- same validation schedule;
- same CSV/JSON/SVG/HTML reporting.

The Hugging Face GPT-2 and OPT models are initialized from scratch by default. This avoids comparing a locally trained DRM model against externally pretrained Transformer weights.

## Install Optional Dependency

```bash
pip install -e ".[hf]"
```

## Validate The Setup Without Training

This instantiates all models, counts parameters, and writes comparison artifacts without running optimization. It does not need Wikipedia download unless `--dataset wikipedia-en` is also passed:

```bash
python scripts/run_scale_lm_comparison.py --dry-run --output-root runs/scale_lm_dry
```

Prepare the Wikipedia EN text sample explicitly:

```bash
python scripts/prepare_wikipedia_en.py \
  --output data/wikipedia_en_20231101_sample.txt \
  --dataset-name wikimedia/wikipedia \
  --dataset-config 20231101.en \
  --split train \
  --max-chars 50000000 \
  --streaming
```

To also run a tiny forward pass during validation:

```bash
python scripts/run_scale_lm_comparison.py --dry-run --dry-run-forward --models drm_125m gpt2_125m opt_125m --seq-len 16 --output-root runs/scale_lm_dry_forward
```

## Run The Full Default Comparison

```bash
python scripts/run_scale_lm_comparison.py \
  --dataset wikipedia-en \
  --wikipedia-output data/wikipedia_en_20231101_sample.txt \
  --wikipedia-dataset-name wikimedia/wikipedia \
  --wikipedia-dataset-config 20231101.en \
  --wikipedia-split train \
  --wikipedia-max-chars 50000000 \
  --output-root runs/scale_lm_comparison \
  --steps 1000 \
  --seeds 1 2 3 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --seq-len 512 \
  --device cuda
```

Default models:

- `drm_125m`
- `drm_350m`
- `gpt2_125m`
- `gpt2_350m`
- `opt_125m`
- `opt_350m`

Run only the 125M group:

```bash
python scripts/run_scale_lm_comparison.py --dataset wikipedia-en --models drm_125m gpt2_125m opt_125m --steps 1000 --device cuda
```

Run only the 350M group:

```bash
python scripts/run_scale_lm_comparison.py --dataset wikipedia-en --models drm_350m gpt2_350m opt_350m --steps 1000 --device cuda
```

## Outputs

The script writes:

- `summary.json`: run summaries, aggregate summaries, and model specs.
- `dataset.json`: exact Wikipedia dataset/source/sample metadata.
- `runs.csv`: one row per run.
- `aggregate.csv`: mean/std by model.
- `dashboard.html`: visual dashboard.
- `*.svg`: plots for CE, perplexity, throughput, parameter count, memory, and DRM-only diagnostics.

DRM-specific plots include:

- action mean;
- active dimension `dimD`;
- metric condition proxy.

## Vocabulary Note

By default, all models use the project byte tokenizer and `--hf-vocab-size 256` for equal-token CE/perplexity comparisons. If you set `--hf-vocab-size 50257`, GPT-2/OPT parameter counts move closer to their published sizes, but CE includes many unused output classes for this byte-tokenized dataset. Keep this tradeoff explicit when reporting results.

## Wikipedia EN Note

The default Wikipedia source is Hugging Face `wikimedia/wikipedia` with config `20231101.en`, split `train`. The script writes a deterministic prefix sample to `data/wikipedia_en_20231101_sample.txt` and reuses it unless `--wikipedia-overwrite` is passed. Increase `--wikipedia-max-chars` for longer runs; set it to `0` only if you intentionally want to materialize the full split locally.

## Hardware Note

The 350M group is intended for CUDA or other accelerator runs. CPU runs are useful for dry-run validation but will be slow for full training.
