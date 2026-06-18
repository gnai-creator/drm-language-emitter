from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_world_model_dashboard import main as dashboard_main


def write_run(root: Path, model: str, kind: str, steps: int, seed: int, next_exact: float) -> None:
    run_dir = root / model / f"steps_{steps}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "history": [{"step": steps, "val_ce": 1.0, "tokens_per_sec": 100.0}],
        "best_val_ce": 1.0,
        "final_val_ce": 1.1,
        "next_state_exact_match": next_exact if kind == "world_model" else None,
        "reward_accuracy": next_exact if kind == "world_model" else None,
        "done_accuracy": next_exact if kind == "world_model" else None,
        "rollout_exact_match": next_exact / 2,
        "rollout_token_accuracy": 0.5,
        "invalid_state_rate": 0.1,
        "parameter_count": 100000,
        "elapsed_sec": 2.0,
        "tokens_seen": 1000,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    if kind != "world_model":
        (run_dir / "world_metrics.json").write_text(
            json.dumps(
                {
                    "next_state_exact_match": next_exact,
                    "reward_accuracy": next_exact,
                    "done_accuracy": next_exact,
                    "rollout_exact_match": next_exact / 2,
                    "rollout_token_accuracy": 0.5,
                    "invalid_state_rate": 0.1,
                }
            ),
            encoding="utf-8",
        )


def test_world_model_dashboard_smoke(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    for model, kind, score in [
        ("drm_tiny", "drm", 0.2),
        ("transformer_tiny_93k", "transformer", 0.3),
        ("world_model_tiny", "world_model", 0.8),
    ]:
        write_run(root, model, kind, 20, 1, score)
    monkeypatch.setattr(sys, "argv", ["make_world_model_dashboard.py", "--root", str(root), "--title", "Smoke"])
    dashboard_main()
    assert (root / "dashboard.html").exists()
    assert (root / "summary.json").exists()
    assert (root / "competition_table.md").exists()
    assert (root / "aggregate.csv").exists()
    assert (root / "next_state_exact_match_by_model.svg").exists()
    html = (root / "dashboard.html").read_text(encoding="utf-8")
    assert "Executive Summary" in html
    assert "Tiny World Model" not in html or "world_model_tiny" in html
