from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from drm_language_emitter.utils import save_json


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def rank(rows: list[dict[str, Any]], key: str, reverse: bool = False) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get(key) is not None]
    return sorted(valid, key=lambda row: float(row[key]), reverse=reverse)


def collect_aux(root: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        run_dir = Path(row.get("run_dir", ""))
        metrics = load_json(run_dir / "metrics.json")
        history = metrics.get("history", [])
        for target in [1.0, 0.75, 0.5]:
            hit = next((item for item in history if item.get("val_ce") is not None and float(item["val_ce"]) < target), None)
            suffix = str(target).replace(".", "_")
            if hit:
                row[f"steps_to_ce_lt_{suffix}"] = hit.get("step")
                row[f"seconds_to_ce_lt_{suffix}"] = hit.get("elapsed_sec")
        robustness = load_json(run_dir / "robustness.json")
        if robustness:
            deltas = []
            recoveries = []
            for group in robustness.get("corruptions", {}).values():
                for item in group.values():
                    if row["kind"] == "drm" and "drm_relative_degradation" in item:
                        deltas.append(float(item["drm_relative_degradation"]))
                        recoveries.append(float(item["drm_recovery_score"]))
                    if row["kind"] == "transformer" and "transformer_relative_degradation" in item:
                        deltas.append(float(item["transformer_relative_degradation"]))
                        recoveries.append(float(item["transformer_recovery_score"]))
            if deltas:
                row["robustness_degradation"] = sum(deltas) / len(deltas)
                row["robustness_recovery_score"] = sum(recoveries) / len(recoveries)
        bridge = load_json(run_dir / "bridge_task.json")
        if bridge and row["kind"] == "drm":
            row["bridge_success_score"] = bridge.get("bridge_success_score")
            row["bridge_to_linear_energy_ratio"] = bridge.get("bridge_to_linear_energy_ratio")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["rank", "model", "steps", "seed", "best_val_ce", "tokens/sec", "robust_deg", "bridge_score"]
    lines = ["| " + " | ".join(headers) + " |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for i, row in enumerate(rows, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    str(row.get("model")),
                    str(row.get("steps")),
                    str(row.get("seed")),
                    fmt(row.get("best_val_ce")),
                    fmt(row.get("tokens_per_sec")),
                    fmt(row.get("robustness_degradation")),
                    fmt(row.get("bridge_success_score")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/sweep_drm_transformer")
    args = parser.parse_args()
    root = Path(args.root)
    summary = load_json(root / "sweep_summary.json")
    rows = summary.get("runs", [])
    collect_aux(root, rows)
    rankings = {
        "best_val_ce": rank(rows, "best_val_ce"),
        "tokens_per_sec": rank(rows, "tokens_per_sec", reverse=True),
        "robustness_degradation": rank(rows, "robustness_degradation"),
        "bridge_success_score": rank(rows, "bridge_success_score", reverse=True),
    }
    md = "## Competition Ranking By Validation CE\n\n" + markdown_table(rankings["best_val_ce"][:20]) + "\n"
    payload = {"rankings": rankings, "markdown_table": md}
    save_json(root / "competition_summary.json", payload)
    (root / "competition_table.md").write_text(md, encoding="utf-8")
    print(f"saved={root / 'competition_summary.json'}")
    print(f"saved={root / 'competition_table.md'}")
    print(md)


if __name__ == "__main__":
    main()
