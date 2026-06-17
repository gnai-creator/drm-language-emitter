from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_competition_dashboard import collect_rows, save_csv, save_dashboard, save_gap_svg, save_scatter_svg
from summarize_competition import main as summarize_main
from sweep_drm_transformer import aggregate


def write_metrics(root: Path, model: str, steps: int, seed: int, best: float, params: int, tokens_sec: float) -> None:
    run_dir = root / model / f"steps_{steps}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "history": [
            {"step": 1, "val_ce": 2.0, "elapsed_sec": 0.1, "tokens_per_sec": tokens_sec},
            {"step": steps, "val_ce": best, "elapsed_sec": 1.0, "tokens_per_sec": tokens_sec},
        ],
        "best_val_ce": best,
        "parameter_count": params,
        "elapsed_sec": 1.0,
        "tokens_seen": steps * 16,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    if model.startswith("drm"):
        geometry = {"diagnostics": {"condition_proxy": 42.0, "metric_U_norm_mean": 1.0, "dimD_std": 0.2}}
        (run_dir / "geometry.json").write_text(json.dumps(geometry), encoding="utf-8")
        (run_dir / "geodesic_paths.json").write_text(json.dumps({"drm_to_linear_action_ratio": 10.0}), encoding="utf-8")


def test_competition_dashboard_smoke(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "sweep"
    for steps in [100, 200]:
        for seed in [1, 2]:
            write_metrics(root, "drm_tiny", steps, seed, best=1.2 - steps / 1000 - seed * 0.01, params=92710, tokens_sec=2500.0)
            write_metrics(root, "transformer_tiny_93k", steps, seed, best=1.4 - steps / 1000 - seed * 0.01, params=93872, tokens_sec=80000.0)

    rows = collect_rows(root)
    aggregate_rows = aggregate(rows)
    save_csv(root / "summary.csv", rows)
    save_csv(root / "sweep_aggregate.csv", aggregate_rows)
    (root / "sweep_summary.json").write_text(json.dumps({"runs": rows, "aggregate": aggregate_rows}), encoding="utf-8")
    save_gap_svg(root / "drm_vs_transformer_gap.svg", aggregate_rows)
    save_scatter_svg(root / "pareto_ce_vs_params.svg", aggregate_rows, "parameter_count", "best_val_ce_mean", "Pareto", "params")
    save_dashboard(root, "Smoke Dashboard", aggregate_rows)

    monkeypatch.setattr(sys, "argv", ["summarize_competition.py", "--root", str(root)])
    summarize_main()

    html = (root / "dashboard.html").read_text(encoding="utf-8")
    assert "DRM vs Transformer Summary" in html
    assert "Parameter-Matched Pairs" in html
    assert "gap_abs" in html
    assert "transformer_tiny_93k" in html
    assert (root / "competition_summary.json").exists()
    assert (root / "competition_table.md").exists()
    assert (root / "sweep_aggregate.csv").exists()
