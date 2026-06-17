from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from drm_language_emitter.utils import load_yaml_or_json, save_json


DRM_MODELS = {
    "drm_tiny": "configs/tiny.yaml",
    "drm_tiny_104k": "configs/tiny_104k.yaml",
    "drm_stronger": "configs/tiny_drm_stronger.yaml",
    "drm_topk_gates": "configs/tiny_drm_topk_gates.yaml",
}

TRANSFORMER_MODELS = {
    "transformer_tiny": "transformer/tiny_transformer.yaml",
    "transformer_tiny_93k": "transformer/tiny_transformer_93k.yaml",
    "transformer_tiny_220k": "transformer/tiny_transformer_220k.yaml",
}


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_seed_config(base_path: str, seed: int, out_path: Path) -> None:
    config = load_yaml_or_json(base_path)
    config["seed"] = seed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(f"{k}: {v}" for k, v in config.items()) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def final(history: list[dict[str, Any]], key: str) -> Any:
    return history[-1].get(key) if history else None


def train_model(
    root: Path,
    model_name: str,
    kind: str,
    config_path: str,
    steps: int,
    seed: int,
    text: str,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    run_dir = root / model_name / f"steps_{steps}" / f"seed_{seed}"
    seed_config = run_dir / "config.yaml"
    write_seed_config(config_path, seed, seed_config)
    if kind == "drm":
        run(
            [
                sys.executable,
                "scripts/train_tiny.py",
                "--config",
                str(seed_config),
                "--text",
                text,
                "--output-dir",
                str(run_dir),
                "--steps",
                str(steps),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(lr),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/eval_geometry.py",
                "--checkpoint",
                str(run_dir / "drm_tiny.pt"),
                "--tokenizer",
                str(run_dir / "tokenizer.json"),
                "--text",
                text,
                "--output",
                str(run_dir / "geometry.json"),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/eval_geodesic_paths.py",
                "--checkpoint",
                str(run_dir / "drm_tiny.pt"),
                "--tokenizer",
                str(run_dir / "tokenizer.json"),
                "--output",
                str(run_dir / "geodesic_paths.json"),
            ]
        )
    else:
        run(
            [
                sys.executable,
                "-m",
                "transformer.run_train",
                "--config",
                str(seed_config),
                "--text",
                text,
                "--output-dir",
                str(run_dir),
                "--steps",
                str(steps),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(lr),
            ]
        )

    metrics = load_json(run_dir / "metrics.json")
    geometry = load_json(run_dir / "geometry.json").get("diagnostics", {})
    geodesic = load_json(run_dir / "geodesic_paths.json")
    history = metrics.get("history", [])
    row = {
        "model": model_name,
        "kind": kind,
        "steps": steps,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_val_ce": metrics.get("best_val_ce"),
        "final_val_ce": final(history, "val_ce"),
        "final_train_ce": final(history, "train_ce"),
        "parameter_count": metrics.get("parameter_count"),
        "elapsed_sec": metrics.get("elapsed_sec"),
        "tokens_seen": metrics.get("tokens_seen"),
        "tokens_per_sec": final(history, "tokens_per_sec"),
        "condition_proxy": geometry.get("condition_proxy"),
        "metric_U_norm_mean": geometry.get("metric_U_norm_mean"),
        "dimD_std": geometry.get("dimD_std"),
        "soft_active_fraction": geometry.get("soft_active_fraction"),
        "recurrence_proxy": geometry.get("recurrence_proxy"),
        "stability_proxy": geometry.get("stability_proxy"),
        "drm_to_linear_action_ratio": geodesic.get("drm_to_linear_action_ratio"),
    }
    save_json(run_dir / "run_summary.json", row)
    return row


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], int(row["steps"]), row["kind"]), []).append(row)
    out = []
    metrics = [
        "best_val_ce",
        "final_val_ce",
        "tokens_per_sec",
        "elapsed_sec",
        "tokens_seen",
        "condition_proxy",
        "metric_U_norm_mean",
        "dimD_std",
        "drm_to_linear_action_ratio",
        "recurrence_proxy",
        "stability_proxy",
    ]
    for (model, steps, kind), items in sorted(grouped.items()):
        summary: dict[str, Any] = {"model": model, "kind": kind, "steps": steps, "n": len(items)}
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            if values:
                summary[f"{metric}_mean"] = mean(values)
                summary[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        params = [item.get("parameter_count") for item in items if item.get("parameter_count") is not None]
        if params:
            summary["parameter_count"] = int(params[0])
        out.append(summary)
    return out


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_line_svg(path: Path, rows: list[dict[str, Any]], metric: str, title: str, ylabel: str) -> None:
    points = [(r["model"], int(r["steps"]), float(r[metric])) for r in rows if r.get(metric) is not None]
    width, height, margin = 1080, 620, 72
    if not points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    max_step = max(p[1] for p in points)
    min_v, max_v = min(p[2] for p in points), max(p[2] for p in points)
    pad = max((max_v - min_v) * 0.12, 0.1)
    min_v, max_v = min_v - pad, max_v + pad
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    palette = ["#0f766e", "#2563eb", "#9333ea", "#0891b2", "#b91c1c", "#ea580c", "#4b5563"]
    models = sorted({p[0] for p in points})
    colors = {m: palette[i % len(palette)] for i, m in enumerate(models)}

    def xy(step: int, value: float) -> tuple[float, float]:
        return (
            margin + step / max(max_step, 1) * plot_w,
            margin + (max_v - value) / max(max_v - min_v, 1e-8) * plot_h,
        )

    lines = []
    for model in models:
        pts = sorted((s, v) for m, s, v in points if m == model)
        coords = " ".join(f"{xy(s, v)[0]:.1f},{xy(s, v)[1]:.1f}" for s, v in pts)
        lines.append(f"<polyline points='{coords}' fill='none' stroke='{colors[model]}' stroke-width='3'/>")
        for s, v in pts:
            x, y = xy(s, v)
            lines.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{colors[model]}'/>")
    legend = []
    for i, model in enumerate(models):
        y = 86 + i * 22
        legend.append(f"<line x1='{width-300}' y1='{y}' x2='{width-260}' y2='{y}' stroke='{colors[model]}' stroke-width='3'/>")
        legend.append(f"<text x='{width-250}' y='{y+4}' font-size='12' fill='#111827'>{model}</text>")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-size="22" font-family="Arial" font-weight="700" fill="#111827">{title}</text>
<text x="{margin}" y="56" font-size="13" font-family="Arial" fill="#4b5563">{ylabel}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>
{''.join(lines)}
{''.join(legend)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def save_bar_svg(path: Path, rows: list[dict[str, Any]], metric: str, title: str, ylabel: str) -> None:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get(metric) is None:
            continue
        if row["model"] not in latest or int(row["steps"]) > int(latest[row["model"]]["steps"]):
            latest[row["model"]] = row
    items = sorted((m, float(r[metric])) for m, r in latest.items())
    width, height = 1080, 560
    if not items:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    max_v = max(v for _, v in items) * 1.2 + 1e-8
    bars = []
    x0, y0, bar_w, gap = 70, 450, 115, 25
    for i, (model, value) in enumerate(items):
        h = value / max_v * 320
        x, y = x0 + i * (bar_w + gap), y0 - h
        bars.append(f"<rect x='{x}' y='{y:.1f}' width='{bar_w}' height='{h:.1f}' fill='#0f766e'/>")
        bars.append(f"<text x='{x}' y='{y-8:.1f}' font-size='11'>{value:.3f}</text>")
        bars.append(f"<text x='{x}' y='{y0+18}' font-size='10'>{model}</text>")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="60" y="34" font-size="22" font-family="Arial" font-weight="700" fill="#111827">{title}</text>
<text x="60" y="56" font-size="13" font-family="Arial" fill="#4b5563">{ylabel}</text>
<line x1="50" y1="{y0}" x2="{width-40}" y2="{y0}" stroke="#111827"/>
{''.join(bars)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", type=int, default=[400, 1000, 2000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--text", default="data/tiny.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-root", default="runs/sweep_drm_transformer")
    parser.add_argument("--models", nargs="+", default=list(DRM_MODELS) + list(TRANSFORMER_MODELS))
    args = parser.parse_args()

    model_specs: dict[str, tuple[str, str]] = {}
    for name, config in DRM_MODELS.items():
        model_specs[name] = ("drm", config)
    for name, config in TRANSFORMER_MODELS.items():
        model_specs[name] = ("transformer", config)

    root = Path(args.output_root)
    rows = []
    for model_name in args.models:
        if model_name not in model_specs:
            raise SystemExit(f"unknown model {model_name}; available={sorted(model_specs)}")
        kind, config_path = model_specs[model_name]
        for steps in args.steps:
            for seed in args.seeds:
                rows.append(train_model(root, model_name, kind, config_path, steps, seed, args.text, args.batch_size, args.lr))

    aggregate_rows = aggregate(rows)
    save_json(root / "sweep_summary.json", {"runs": rows, "aggregate": aggregate_rows})
    save_csv(root / "summary.csv", rows)
    save_csv(root / "sweep_summary.csv", rows)
    save_csv(root / "sweep_aggregate.csv", aggregate_rows)
    save_line_svg(root / "val_ce_by_step.svg", aggregate_rows, "best_val_ce_mean", "Best Validation CE By Step", "Mean over seeds. Lower is better.")
    save_bar_svg(root / "best_val_ce_by_model.svg", aggregate_rows, "best_val_ce_mean", "Best Validation CE By Model", "Latest configured step per model. Lower is better.")
    save_bar_svg(root / "tokens_sec_by_model.svg", aggregate_rows, "tokens_per_sec_mean", "Tokens Per Second By Model", "Latest configured step per model. Higher is better.")
    drm_rows = [r for r in aggregate_rows if r["kind"] == "drm"]
    save_line_svg(root / "drm_geometry_by_step.svg", drm_rows, "condition_proxy_mean", "DRM Geometry Condition By Step", "Mean condition proxy over seeds. Lower is more stable.")
    save_line_svg(root / "sweep_aggregate.svg", aggregate_rows, "best_val_ce_mean", "Sweep Best Validation CE", "Mean over seeds. Lower is better.")
    print(f"saved={root / 'sweep_summary.json'}")
    print(f"saved={root / 'summary.csv'}")
    print(f"saved={root / 'val_ce_by_step.svg'}")
    print(f"saved={root / 'best_val_ce_by_model.svg'}")
    print(f"saved={root / 'tokens_sec_by_model.svg'}")
    print(f"saved={root / 'drm_geometry_by_step.svg'}")


if __name__ == "__main__":
    main()
