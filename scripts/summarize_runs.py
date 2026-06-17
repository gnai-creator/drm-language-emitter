from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from drm_language_emitter.utils import save_json


KEYS = [
    "run",
    "best_val_ce",
    "final_val_ce",
    "final_train_ce",
    "dimD_mean",
    "dimD_std",
    "soft_active_fraction",
    "hard_active_fraction_050",
    "hard_active_fraction_075",
    "hard_active_fraction_090",
    "gate_q10",
    "gate_q50",
    "gate_q90",
    "metric_U_norm_mean",
    "condition_proxy",
    "geometry_ce",
    "drm_to_linear_action_ratio",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    geometry = load_json(run_dir / "geometry.json")
    geodesic = load_json(run_dir / "geodesic_paths.json")
    history = metrics.get("history") or []
    final = history[-1] if history else {}
    diagnostics = geometry.get("diagnostics") or {}
    aux = geometry.get("aux_losses") or {}
    return {
        "run": run_dir.name,
        "best_val_ce": metrics.get("best_val_ce"),
        "final_val_ce": final.get("val_ce"),
        "final_train_ce": final.get("train_ce"),
        "dimD_mean": diagnostics.get("dimD_mean", final.get("dimD_mean")),
        "dimD_std": diagnostics.get("dimD_std", final.get("dimD_std")),
        "soft_active_fraction": diagnostics.get("soft_active_fraction", final.get("soft_active_fraction")),
        "hard_active_fraction_050": diagnostics.get("hard_active_fraction_050", final.get("active_fraction")),
        "hard_active_fraction_075": diagnostics.get("hard_active_fraction_075", final.get("hard_active_fraction_075")),
        "hard_active_fraction_090": diagnostics.get("hard_active_fraction_090", final.get("hard_active_fraction_090")),
        "gate_q10": diagnostics.get("gate_q10", final.get("gate_q10")),
        "gate_q50": diagnostics.get("gate_q50", final.get("gate_q50")),
        "gate_q90": diagnostics.get("gate_q90", final.get("gate_q90")),
        "metric_U_norm_mean": diagnostics.get("metric_U_norm_mean", final.get("metric_U_norm_mean")),
        "condition_proxy": diagnostics.get("condition_proxy", final.get("condition_proxy")),
        "geometry_ce": aux.get("ce"),
        "drm_to_linear_action_ratio": geodesic.get("drm_to_linear_action_ratio"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/full_v3")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"run root does not exist: {root}")
    rows = [summarize_run(path) for path in sorted(root.iterdir()) if path.is_dir()]
    output_json = Path(args.output_json) if args.output_json else root / "summary.json"
    output_csv = Path(args.output_csv) if args.output_csv else root / "summary.csv"
    save_json(output_json, {"runs": rows})
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEYS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={output_json}")
    print(f"saved={output_csv}")
    for row in rows:
        print(
            f"{row['run']}: best_val={row['best_val_ce']} "
            f"dimD_std={row['dimD_std']} metricU={row['metric_U_norm_mean']} "
            f"cond={row['condition_proxy']} ratio={row['drm_to_linear_action_ratio']}"
        )


if __name__ == "__main__":
    main()
