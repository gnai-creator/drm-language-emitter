from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from drm_language_emitter.utils import save_json


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def series(metrics: dict[str, Any], key: str) -> list[tuple[int, float]]:
    return [(int(row["step"]), float(row[key])) for row in metrics.get("history", []) if key in row]


def last(metrics: dict[str, Any], key: str) -> float | None:
    history = metrics.get("history", [])
    if not history:
        return None
    value = history[-1].get(key)
    return None if value is None else float(value)


def make_svg(path: Path, drm_metrics: dict[str, Any], transformer_metrics: dict[str, Any]) -> None:
    width, height = 920, 520
    margin = 64
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    drm_val = series(drm_metrics, "val_ce")
    drm_train = series(drm_metrics, "train_ce")
    tr_val = series(transformer_metrics, "val_ce")
    tr_train = series(transformer_metrics, "train_ce")
    all_points = drm_val + drm_train + tr_val + tr_train
    if not all_points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    max_step = max(step for step, _ in all_points)
    min_ce = min(value for _, value in all_points)
    max_ce = max(value for _, value in all_points)
    pad = max((max_ce - min_ce) * 0.08, 0.1)
    min_ce -= pad
    max_ce += pad

    def xy(point: tuple[int, float]) -> tuple[float, float]:
        step, value = point
        x = margin + (step / max(max_step, 1)) * plot_w
        y = margin + ((max_ce - value) / max(max_ce - min_ce, 1e-8)) * plot_h
        return x, y

    def poly(points: list[tuple[int, float]], color: str, dash: str = "") -> str:
        if not points:
            return ""
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, points))
        dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
        return f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='3'{dash_attr}/>"

    grid = []
    for i in range(6):
        y = margin + i * plot_h / 5
        value = max_ce - i * (max_ce - min_ce) / 5
        grid.append(f"<line x1='{margin}' y1='{y:.1f}' x2='{width-margin}' y2='{y:.1f}' stroke='#e5e7eb'/>")
        grid.append(f"<text x='16' y='{y+4:.1f}' font-size='12' fill='#374151'>{value:.2f}</text>")
    for i in range(6):
        x = margin + i * plot_w / 5
        step = int(i * max_step / 5)
        grid.append(f"<line x1='{x:.1f}' y1='{margin}' x2='{x:.1f}' y2='{height-margin}' stroke='#f3f4f6'/>")
        grid.append(f"<text x='{x-10:.1f}' y='{height-24}' font-size='12' fill='#374151'>{step}</text>")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{margin}" y="30" font-size="22" font-family="Arial" font-weight="700" fill="#111827">DRM vs Tiny Transformer CE</text>
  <text x="{margin}" y="52" font-size="13" font-family="Arial" fill="#4b5563">Lower is better. Solid = validation CE, dashed = train CE.</text>
  {''.join(grid)}
  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>
  {poly(drm_val, "#0f766e")}
  {poly(drm_train, "#0f766e", "7 5")}
  {poly(tr_val, "#b91c1c")}
  {poly(tr_train, "#b91c1c", "7 5")}
  <rect x="{width-260}" y="70" width="210" height="96" fill="white" stroke="#d1d5db"/>
  <line x1="{width-242}" y1="94" x2="{width-202}" y2="94" stroke="#0f766e" stroke-width="3"/>
  <text x="{width-192}" y="98" font-size="13" font-family="Arial" fill="#111827">DRM val CE</text>
  <line x1="{width-242}" y1="118" x2="{width-202}" y2="118" stroke="#0f766e" stroke-width="3" stroke-dasharray="7 5"/>
  <text x="{width-192}" y="122" font-size="13" font-family="Arial" fill="#111827">DRM train CE</text>
  <line x1="{width-242}" y1="142" x2="{width-202}" y2="142" stroke="#b91c1c" stroke-width="3"/>
  <text x="{width-192}" y="146" font-size="13" font-family="Arial" fill="#111827">Transformer val CE</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="data/tiny.txt")
    parser.add_argument("--drm-config", default="configs/tiny.yaml")
    parser.add_argument("--transformer-config", default="transformer/tiny_transformer.yaml")
    parser.add_argument("--output-root", default="runs/drm_vs_transformer")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    drm_dir = root / "drm"
    transformer_dir = root / "transformer"
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        run(
            [
                sys.executable,
                "scripts/train_tiny.py",
                "--config",
                args.drm_config,
                "--text",
                args.text,
                "--output-dir",
                str(drm_dir),
                "--steps",
                str(args.steps),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "transformer.run_train",
                "--config",
                args.transformer_config,
                "--text",
                args.text,
                "--output-dir",
                str(transformer_dir),
                "--steps",
                str(args.steps),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/eval_geometry.py",
                "--checkpoint",
                str(drm_dir / "drm_tiny.pt"),
                "--tokenizer",
                str(drm_dir / "tokenizer.json"),
                "--text",
                args.text,
                "--output",
                str(drm_dir / "geometry.json"),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/eval_geodesic_paths.py",
                "--checkpoint",
                str(drm_dir / "drm_tiny.pt"),
                "--tokenizer",
                str(drm_dir / "tokenizer.json"),
                "--output",
                str(drm_dir / "geodesic_paths.json"),
            ]
        )

    drm_metrics = load_json(drm_dir / "metrics.json")
    transformer_metrics = load_json(transformer_dir / "metrics.json")
    drm_geometry = load_json(drm_dir / "geometry.json") if (drm_dir / "geometry.json").exists() else {}
    drm_geodesic = load_json(drm_dir / "geodesic_paths.json") if (drm_dir / "geodesic_paths.json").exists() else {}
    summary = {
        "drm": {
            "best_val_ce": drm_metrics.get("best_val_ce"),
            "final_val_ce": last(drm_metrics, "val_ce"),
            "final_train_ce": last(drm_metrics, "train_ce"),
            "parameter_count": drm_metrics.get("parameter_count"),
            "elapsed_sec": drm_metrics.get("elapsed_sec"),
            "tokens_seen": drm_metrics.get("tokens_seen"),
            "final_tokens_per_sec": last(drm_metrics, "tokens_per_sec"),
            "geometry": drm_geometry.get("diagnostics", {}),
            "geodesic": drm_geodesic,
        },
        "transformer": {
            "best_val_ce": transformer_metrics.get("best_val_ce"),
            "final_val_ce": last(transformer_metrics, "val_ce"),
            "final_train_ce": last(transformer_metrics, "train_ce"),
            "parameter_count": transformer_metrics.get("parameter_count"),
            "elapsed_sec": transformer_metrics.get("elapsed_sec"),
            "tokens_seen": transformer_metrics.get("tokens_seen"),
            "final_tokens_per_sec": last(transformer_metrics, "tokens_per_sec"),
        },
    }
    save_json(root / "comparison.json", summary)
    make_svg(root / "comparison.svg", drm_metrics, transformer_metrics)
    print(f"saved={root / 'comparison.json'}")
    print(f"saved={root / 'comparison.svg'}")


if __name__ == "__main__":
    main()
