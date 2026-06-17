# Experiments

## 1. Tiny Language Modeling

Train on `data/tiny.txt` and verify that next-token loss decreases over a few CPU steps.

## 2. Fixed Dimension vs Variable Dimension

Compare normal gated directions against fixed gates. Track `dimD_mean`, `dimD_std`, action, and CE.

Initial target: avoid `active_fraction = 1.0`; look for `dimD_std > 0.1` before treating variable dimension as dynamically meaningful.

## 3. Metric On/Off

Compare `metric_rank > 0` with `metric_rank = 0`. Track pairwise coupling, metric norm, condition proxy, and action.

The strongest test is whether validation CE, action, or stability degrades when metric naturalization and the low-rank term are removed.

For tiny runs, use temporary stability targets:

- `condition_proxy < 200`
- `metric_U_norm_mean` roughly between `0.5` and `2.5`
- `drm_to_linear_action_ratio < 20`
- geometry CE close to validation CE from `metrics.json`

For gate selectivity, inspect:

- `soft_active_fraction`
- `hard_active_fraction_050`
- `hard_active_fraction_075`
- `hard_active_fraction_090`
- `gate_q10`, `gate_q50`, `gate_q90`

The baseline may keep all gates above `0.5` while still varying effective dimension. Use `configs/tiny_gate_sparse.yaml` to test whether stronger gate sparsity improves or hurts validation CE and action.

## 4. Action Loss On/Off

Run with `lambda_action > 0` and `lambda_action = 0`. Compare generated trajectories by metric action and diversity.

## 5. Geodesic Interpolation

Use `scripts/eval_geodesic_paths.py` to compare decoded linear interpolation and DRM rollout. Interpret this as a low-action diagnostic only.

## 6. Recurrence And Stability

Track norm drift, local logit changes, and bounded-state behavior. Toroidal diagnostics should only be discussed if bounded recurrent stable trajectories are observed.

## 7. Tiny Transformer Baseline

Use:

```bash
python scripts/compare_drm_transformer.py --steps 400 --batch-size 8 --output-root runs/drm_vs_transformer_400
```

Current controlled 400-step result:

- DRM: `best_val_ce = 2.8752`, `parameter_count = 92,710`
- Tiny Transformer: `best_val_ce = 2.1369`, `parameter_count = 104,192`
- DRM geometry stayed stable: `condition_proxy = 91.86`, `metric_U_norm_mean = 1.11`
- DRM low-action diagnostic improved: `drm_to_linear_action_ratio = 12.78`

Interpretation: the Transformer wins validation CE in this tiny run. DRM remains a stable, measurable research architecture, but this result is not evidence of superiority over Transformers.

## 8. Parameter-Matched And Seed Sweeps

Use parameter-paired configs:

- `configs/tiny_104k.yaml` for DRM near the current Transformer size.
- `transformer/tiny_transformer_93k.yaml` for a Transformer near current DRM size.

Run multi-step, multi-seed sweeps:

```bash
python scripts/sweep_drm_transformer.py --steps 400 1000 2000 --seeds 1 2 3 --output-root runs/sweep_drm_transformer
```

Compare `best_val_ce_mean`, `best_val_ce_std`, `elapsed_sec_mean`, and `final_tokens_per_sec_mean`.
