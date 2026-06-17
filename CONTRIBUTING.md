# Contributing

Thank you for considering a contribution to DRM Language Emitter.

## Scope

Useful contributions include:

- CPU-runnable tests;
- geometric diagnostics;
- training and ablation scripts;
- documentation that clarifies limitations;
- bug fixes that preserve the non-Transformer design.

Do not add Transformer blocks, self-attention, Q/K/V attention, or `nn.MultiheadAttention`.

## Development Setup

```bash
pip install -e .
python -m pytest -q
```

## Before Opening A PR

Run:

```bash
python -m pytest -q
python scripts/train_tiny.py --config configs/tiny.yaml --text data/tiny.txt --output-dir runs/contrib_smoke --steps 3 --batch-size 2
```

`runs/` is ignored and should not be committed.

## Documentation Standard

Documentation must be honest about experimental status. Do not claim:

- production readiness;
- safety certification;
- AGI;
- alignment;
- superiority over Transformers without evidence;
- exact geodesic solving unless implemented and tested.

## Licensing

By contributing, you agree that your contribution is provided under the repository license, AGPL-3.0-only, unless a separate written commercial agreement is signed with the copyright holder.

Commercial licensing inquiries: felupe@truthagi.ai
