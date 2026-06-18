# Benchmark Artifacts

This directory stores lightweight, versionable benchmark outputs.

Generated training directories under `runs/` remain local working artifacts and are ignored by git. The selected dashboards, summaries, CSVs, and SVG plots are copied here when a result should be preserved in the repository.

## Available Benchmarks

- `drm_transformer_full_1k_3k/`
  - Step-matched and parameter-matched DRM vs Tiny Transformer sweep.
  - Main entry: `drm_transformer_full_1k_3k/dashboard.html`

- `world_model_competition/`
  - DRM vs Tiny Transformer vs tiny symbolic `world_model/` benchmark.
  - Main entry: `world_model_competition/dashboard.html`

## Reproducibility

The corresponding commands are documented in:

- `docs/competition.md`
- `docs/world_model_competition.md`
- `docs/report/001_drm_competition_protocol_2026-06-17.md`
- `docs/report/002_world_model_competition_2026-06-18.md`

