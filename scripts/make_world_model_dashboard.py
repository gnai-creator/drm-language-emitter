from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

from drm_language_emitter.utils import save_json
from sweep_world_model_competition import aggregate


METRICS = [
    "best_val_ce_mean",
    "next_state_exact_match_mean",
    "reward_accuracy_mean",
    "done_accuracy_mean",
    "rollout_exact_match_mean",
    "rollout_token_accuracy_mean",
    "invalid_state_rate_mean",
    "elapsed_sec_mean",
    "tokens_seen_mean",
    "tokens_per_sec_mean",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        run_dir = path.parent
        rel = run_dir.relative_to(root).parts
        if len(rel) < 3:
            continue
        model = rel[0]
        kind = "world_model" if model.startswith("world_model") else "transformer" if model.startswith("transformer") else "drm"
        steps = int(rel[1].replace("steps_", ""))
        seed = int(rel[2].replace("seed_", ""))
        metrics = load_json(path)
        wm = load_json(run_dir / "world_metrics.json")
        history = metrics.get("history", [])
        row = {
            "model": model,
            "kind": kind,
            "steps": steps,
            "seed": seed,
            "run_dir": str(run_dir),
            "best_val_ce": metrics.get("best_val_ce"),
            "final_val_ce": metrics.get("final_val_ce", history[-1].get("val_ce") if history else None),
            "parameter_count": metrics.get("parameter_count"),
            "elapsed_sec": metrics.get("elapsed_sec"),
            "tokens_seen": metrics.get("tokens_seen"),
            "tokens_per_sec": history[-1].get("tokens_per_sec") if history else None,
        }
        for key in ["next_state_exact_match", "reward_accuracy", "done_accuracy", "rollout_exact_match", "rollout_token_accuracy", "invalid_state_rate"]:
            row[key] = wm.get(key, metrics.get(key))
        rows.append(row)
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def best_row(rows: list[dict[str, Any]], metric: str, higher: bool) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(metric) is not None]
    return sorted(valid, key=lambda row: float(row[metric]), reverse=higher)[0] if valid else None


def executive_summary(rows: list[dict[str, Any]]) -> str:
    best_ce = best_row(rows, "best_val_ce_mean", higher=False)
    best_next = best_row(rows, "next_state_exact_match_mean", higher=True)
    best_rollout = best_row(rows, "rollout_exact_match_mean", higher=True)
    best_speed = best_row(rows, "tokens_per_sec_mean", higher=True)
    lines = ["<ul>"]
    for label, row, metric in [
        ("Best CE", best_ce, "best_val_ce_mean"),
        ("Best next-state exact match", best_next, "next_state_exact_match_mean"),
        ("Best rollout exact match", best_rollout, "rollout_exact_match_mean"),
        ("Best throughput", best_speed, "tokens_per_sec_mean"),
    ]:
        if row:
            lines.append(f"<li>{label}: <strong>{html.escape(row['model'])}</strong> at {row['steps']} steps ({fmt(row.get(metric))}).</li>")
    lines.append("</ul>")
    lines.append(
        "<p>This tiny symbolic text-world benchmark compares only this serialized gridworld setup. "
        "It does not imply superiority over general multimodal world models.</p>"
    )
    return "".join(lines)


def table_html(rows: list[dict[str, Any]]) -> str:
    cols = ["model", "kind", "steps", "n", "parameter_count"] + METRICS
    body = []
    for row in sorted(rows, key=lambda r: (int(r["steps"]), r["kind"], r["model"])):
        body.append("<tr>" + "".join(f"<td>{html.escape(fmt(row.get(col)))}</td>" for col in cols) + "</tr>")
    return "<table><thead><tr>" + "".join(f"<th>{col}</th>" for col in cols) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def parameter_pairs_html(rows: list[dict[str, Any]]) -> str:
    pairs = [
        ("drm_tiny", "transformer_tiny_93k", "world_model_tiny"),
        ("drm_tiny_104k", "transformer_tiny", "world_model_tiny"),
        ("drm_stronger", "transformer_tiny_220k", "world_model_stronger"),
    ]
    by_key = {(row["model"], int(row["steps"])): row for row in rows}
    body = []
    for pair in pairs:
        for step in sorted({int(row["steps"]) for row in rows}):
            items = [by_key.get((model, step)) for model in pair]
            if any(item is None for item in items):
                continue
            body.append(
                "<tr>"
                f"<td>{step}</td>"
                + "".join(
                    f"<td>{html.escape(item['model'])}</td><td>{item.get('parameter_count')}</td><td>{fmt(item.get('next_state_exact_match_mean'))}</td><td>{fmt(item.get('rollout_exact_match_mean'))}</td><td>{fmt(item.get('best_val_ce_mean'))}</td>"
                    for item in items
                    if item is not None
                )
                + "</tr>"
            )
    return (
        "<table><thead><tr><th>steps</th>"
        "<th>DRM</th><th>params</th><th>next</th><th>rollout</th><th>CE</th>"
        "<th>Transformer</th><th>params</th><th>next</th><th>rollout</th><th>CE</th>"
        "<th>World Model</th><th>params</th><th>next</th><th>rollout</th><th>CE</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def save_bar_svg(path: Path, rows: list[dict[str, Any]], metric: str, title: str, higher: bool = True) -> None:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get(metric) is None:
            continue
        if row["model"] not in latest or int(row["steps"]) > int(latest[row["model"]]["steps"]):
            latest[row["model"]] = row
    items = sorted((name, float(row[metric]), row["kind"]) for name, row in latest.items())
    width, height, margin = 1080, 520, 70
    if not items:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    max_v = max(v for _, v, _ in items) * 1.15 + 1e-8
    colors = {"drm": "#0f766e", "transformer": "#b91c1c", "world_model": "#2563eb"}
    bars = []
    bar_w, gap, y0 = 95, 24, 430
    for i, (model, value, kind) in enumerate(items):
        h = value / max_v * 300
        x = margin + i * (bar_w + gap)
        y = y0 - h
        bars.append(f"<rect x='{x}' y='{y:.1f}' width='{bar_w}' height='{h:.1f}' fill='{colors.get(kind, '#4b5563')}'/>")
        bars.append(f"<text x='{x}' y='{y-8:.1f}' font-size='11'>{value:.3f}</text><text x='{x}' y='{y0+18}' font-size='10'>{html.escape(model)}</text>")
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-size="22" font-family="Arial" font-weight="700">{html.escape(title)}</text>
<text x="{margin}" y="56" font-size="13" font-family="Arial">Latest step per model. {'Higher' if higher else 'Lower'} is better.</text>
<line x1="{margin}" y1="{y0}" x2="{width-margin}" y2="{y0}" stroke="#111827"/>
{''.join(bars)}
</svg>""",
        encoding="utf-8",
    )


def save_scatter_svg(path: Path, rows: list[dict[str, Any]], x_metric: str, y_metric: str, title: str) -> None:
    points = [(row["model"], row["kind"], float(row[x_metric]), float(row[y_metric])) for row in rows if row.get(x_metric) is not None and row.get(y_metric) is not None]
    width, height, margin = 920, 560, 70
    if not points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    xs, ys = [p[2] for p in points], [p[3] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min_x - max((max_x - min_x) * 0.1, 1e-8), max_x + max((max_x - min_x) * 0.1, 1e-8)
    min_y, max_y = min_y - max((max_y - min_y) * 0.1, 0.05), max_y + max((max_y - min_y) * 0.1, 0.05)
    colors = {"drm": "#0f766e", "transformer": "#b91c1c", "world_model": "#2563eb"}

    def xy(x: float, y: float) -> tuple[float, float]:
        return (margin + (x - min_x) / max(max_x - min_x, 1e-8) * (width - 2 * margin), margin + (max_y - y) / max(max_y - min_y, 1e-8) * (height - 2 * margin))

    elems = []
    for model, kind, x, y in points:
        px, py = xy(x, y)
        elems.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='5' fill='{colors.get(kind, '#4b5563')}'/><text x='{px+7:.1f}' y='{py-6:.1f}' font-size='11'>{html.escape(model)}</text>")
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-size="22" font-family="Arial" font-weight="700">{html.escape(title)}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>
{''.join(elems)}
</svg>""",
        encoding="utf-8",
    )


def svg_inline(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else f"<p>Missing {html.escape(path.name)}</p>"


def honest_interpretation(rows: list[dict[str, Any]]) -> str:
    best_ce = best_row(rows, "best_val_ce_mean", higher=False)
    best_rollout = best_row(rows, "rollout_exact_match_mean", higher=True)
    best_speed = best_row(rows, "tokens_per_sec_mean", higher=True)
    parts = []
    if best_ce:
        parts.append(f"Best CE in this run belongs to {best_ce['model']}.")
    if best_rollout:
        parts.append(f"Best rollout exact match belongs to {best_rollout['model']}.")
    if best_speed:
        parts.append(f"Highest throughput belongs to {best_speed['model']}.")
    parts.append("Do not generalize this result to large, multimodal, or production world models.")
    return "<p>" + " ".join(html.escape(part) for part in parts) + "</p>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/world_model_competition")
    parser.add_argument("--title", default="DRM vs Transformer vs Tiny World Model")
    args = parser.parse_args()
    root = Path(args.root)
    rows = collect_rows(root)
    if not rows:
        raise SystemExit(f"no run rows found under {root}")
    aggregate_rows = aggregate(rows)
    save_json(root / "summary.json", {"runs": rows, "aggregate": aggregate_rows})
    save_csv(root / "aggregate.csv", aggregate_rows)
    save_bar_svg(root / "best_val_ce_by_model.svg", aggregate_rows, "best_val_ce_mean", "Best Validation CE", higher=False)
    save_bar_svg(root / "next_state_exact_match_by_model.svg", aggregate_rows, "next_state_exact_match_mean", "Next-State Exact Match")
    save_bar_svg(root / "rollout_exact_match_by_model.svg", aggregate_rows, "rollout_exact_match_mean", "Rollout Exact Match")
    save_bar_svg(root / "invalid_state_rate_by_model.svg", aggregate_rows, "invalid_state_rate_mean", "Invalid State Rate", higher=False)
    save_scatter_svg(root / "pareto_exact_match_vs_params.svg", aggregate_rows, "parameter_count", "next_state_exact_match_mean", "Pareto: Next-State Exact Match vs Params")
    save_scatter_svg(root / "pareto_exact_match_vs_elapsed_sec.svg", aggregate_rows, "elapsed_sec_mean", "next_state_exact_match_mean", "Pareto: Next-State Exact Match vs Elapsed Sec")
    save_scatter_svg(root / "pareto_exact_match_vs_tokens_seen.svg", aggregate_rows, "tokens_seen_mean", "next_state_exact_match_mean", "Pareto: Next-State Exact Match vs Tokens Seen")
    save_scatter_svg(root / "pareto_ce_vs_params.svg", aggregate_rows, "parameter_count", "best_val_ce_mean", "Pareto: CE vs Params")
    save_bar_svg(root / "drm_vs_transformer_vs_world_model_gap.svg", aggregate_rows, "next_state_exact_match_mean", "Tri-Family Next-State Accuracy")
    svg_names = [
        "best_val_ce_by_model.svg",
        "next_state_exact_match_by_model.svg",
        "rollout_exact_match_by_model.svg",
        "invalid_state_rate_by_model.svg",
        "pareto_exact_match_vs_params.svg",
        "pareto_exact_match_vs_elapsed_sec.svg",
        "pareto_exact_match_vs_tokens_seen.svg",
        "pareto_ce_vs_params.svg",
        "drm_vs_transformer_vs_world_model_gap.svg",
    ]
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>{html.escape(args.title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f9fafb; color: #111827; }}
section {{ background: white; border: 1px solid #d1d5db; margin: 18px 0; padding: 16px; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th,td {{ border: 1px solid #d1d5db; padding: 5px 8px; text-align: right; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align: left; }}
svg {{ max-width: 100%; height: auto; display: block; }}
</style></head><body>
<h1>{html.escape(args.title)}</h1>
<section><h2>Executive Summary</h2>{executive_summary(aggregate_rows)}</section>
<section><h2>Step-Matched Results</h2>{table_html(aggregate_rows)}</section>
<section><h2>Parameter-Matched Pairs</h2>{parameter_pairs_html(aggregate_rows)}</section>
<section><h2>Rollout Metrics</h2>{table_html([row for row in aggregate_rows if row.get('rollout_exact_match_mean') is not None])}</section>
<section><h2>Honest Interpretation</h2>{honest_interpretation(aggregate_rows)}</section>
{''.join(f'<section><h2>{html.escape(name)}</h2>{svg_inline(root / name)}</section>' for name in svg_names)}
</body></html>"""
    (root / "dashboard.html").write_text(page, encoding="utf-8")
    md = "# " + args.title + "\n\n" + honest_interpretation(aggregate_rows).replace("<p>", "").replace("</p>", "") + "\n"
    (root / "competition_table.md").write_text(md, encoding="utf-8")
    print(f"saved={root / 'dashboard.html'}")
    print(f"saved={root / 'summary.json'}")


if __name__ == "__main__":
    main()
