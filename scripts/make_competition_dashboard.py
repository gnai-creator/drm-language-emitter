from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drm_language_emitter.utils import save_json
from sweep_drm_transformer import aggregate, save_bar_svg, save_line_svg


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def final(history: list[dict[str, Any]], key: str) -> Any:
    return history[-1].get(key) if history else None


def infer_run(metrics_path: Path, root: Path) -> tuple[str, str, int, int, Path]:
    run_dir = metrics_path.parent
    rel = run_dir.relative_to(root).parts
    if len(rel) < 3:
        raise ValueError(f"cannot infer model/steps/seed from {run_dir}")
    model = rel[0]
    steps = int(rel[1].replace("steps_", ""))
    seed = int(rel[2].replace("seed_", ""))
    kind = "transformer" if model.startswith("transformer") else "drm"
    if len(rel) >= 4 and rel[3] in {"drm", "transformer"}:
        kind = rel[3]
        model = f"{model}_{kind}"
    return model, kind, steps, seed, run_dir


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        model, kind, steps, seed, run_dir = infer_run(metrics_path, root)
        metrics = load_json(metrics_path)
        geometry = load_json(run_dir / "geometry.json").get("diagnostics", {})
        geodesic = load_json(run_dir / "geodesic_paths.json")
        history = metrics.get("history", [])
        rows.append(
            {
                "model": model,
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
        )
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_inline(path: Path) -> str:
    if not path.exists():
        return f"<p>Missing: {html.escape(str(path.name))}</p>"
    return path.read_text(encoding="utf-8")


def best_by_kind_step(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = sorted({int(row["steps"]) for row in aggregate_rows})
    out = []
    for step in steps:
        step_rows = [row for row in aggregate_rows if int(row["steps"]) == step and row.get("best_val_ce_mean") is not None]
        drm_rows = [row for row in step_rows if row["kind"] == "drm"]
        transformer_rows = [row for row in step_rows if row["kind"] == "transformer"]
        best_drm = min(drm_rows, key=lambda row: float(row["best_val_ce_mean"])) if drm_rows else None
        best_transformer = min(transformer_rows, key=lambda row: float(row["best_val_ce_mean"])) if transformer_rows else None
        row = {"steps": step, "best_drm": best_drm, "best_transformer": best_transformer}
        if best_drm and best_transformer:
            drm_ce = float(best_drm["best_val_ce_mean"])
            transformer_ce = float(best_transformer["best_val_ce_mean"])
            row["ce_gap_drm_minus_transformer"] = drm_ce - transformer_ce
            row["gap_abs"] = transformer_ce - drm_ce
            row["gap_rel"] = (transformer_ce - drm_ce) / max(transformer_ce, 1e-8)
            if best_drm.get("tokens_per_sec_mean") and best_transformer.get("tokens_per_sec_mean"):
                row["speed_ratio"] = float(best_transformer["tokens_per_sec_mean"]) / max(float(best_drm["tokens_per_sec_mean"]), 1e-8)
            if best_drm.get("parameter_count") and best_transformer.get("parameter_count"):
                row["param_ratio"] = float(best_drm["parameter_count"]) / max(float(best_transformer["parameter_count"]), 1e-8)
        out.append(row)
    return out


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def find_aggregate(aggregate_rows: list[dict[str, Any]], model: str, steps: int) -> dict[str, Any] | None:
    for row in aggregate_rows:
        if row["model"] == model and int(row["steps"]) == int(steps):
            return row
    return None


def parameter_matched_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = [
        ("drm_tiny", "transformer_tiny_93k"),
        ("drm_tiny_104k", "transformer_tiny"),
        ("drm_stronger", "transformer_tiny_220k"),
    ]
    steps = sorted({int(row["steps"]) for row in aggregate_rows})
    out = []
    for drm_model, transformer_model in pairs:
        for step in steps:
            drm = find_aggregate(aggregate_rows, drm_model, step)
            tr = find_aggregate(aggregate_rows, transformer_model, step)
            if not drm or not tr:
                continue
            drm_ce = float(drm["best_val_ce_mean"])
            tr_ce = float(tr["best_val_ce_mean"])
            out.append(
                {
                    "steps": step,
                    "drm": drm,
                    "transformer": tr,
                    "gap_abs": tr_ce - drm_ce,
                    "gap_rel": (tr_ce - drm_ce) / max(tr_ce, 1e-8),
                    "speed_ratio": float(tr.get("tokens_per_sec_mean", 0.0)) / max(float(drm.get("tokens_per_sec_mean", 0.0)), 1e-8),
                    "param_ratio": float(drm.get("parameter_count", 0.0)) / max(float(tr.get("parameter_count", 0.0)), 1e-8),
                }
            )
    return out


def save_gap_svg(path: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in best_by_kind_step(aggregate_rows) if row.get("ce_gap_drm_minus_transformer") is not None]
    width, height, margin = 900, 460, 70
    if not rows:
        path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='180'>"
            "<rect width='100%' height='100%' fill='white'/>"
            "<text x='32' y='52' font-size='18' font-family='Arial'>No DRM vs Transformer pair is available in this sweep root.</text>"
            "</svg>",
            encoding="utf-8",
        )
        return
    max_step = max(int(row["steps"]) for row in rows)
    values = [float(row["ce_gap_drm_minus_transformer"]) for row in rows]
    min_v, max_v = min(values + [0.0]), max(values + [0.0])
    pad = max((max_v - min_v) * 0.15, 0.1)
    min_v, max_v = min_v - pad, max_v + pad
    plot_w, plot_h = width - 2 * margin, height - 2 * margin

    def xy(step: int, value: float) -> tuple[float, float]:
        return (
            margin + step / max(max_step, 1) * plot_w,
            margin + (max_v - value) / max(max_v - min_v, 1e-8) * plot_h,
        )

    coords = " ".join(f"{xy(int(r['steps']), float(r['ce_gap_drm_minus_transformer']))[0]:.1f},{xy(int(r['steps']), float(r['ce_gap_drm_minus_transformer']))[1]:.1f}" for r in rows)
    zero_y = xy(0, 0.0)[1]
    points = []
    for row in rows:
        x, y = xy(int(row["steps"]), float(row["ce_gap_drm_minus_transformer"]))
        points.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='#0f766e'/>")
        points.append(f"<text x='{x+7:.1f}' y='{y-7:.1f}' font-size='12'>{float(row['ce_gap_drm_minus_transformer']):.3f}</text>")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-size="22" font-family="Arial" font-weight="700" fill="#111827">DRM vs Transformer CE Gap</text>
<text x="{margin}" y="56" font-size="13" font-family="Arial" fill="#4b5563">Gap = best DRM CE - best Transformer CE. Below zero means DRM wins CE.</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>
<line x1="{margin}" y1="{zero_y:.1f}" x2="{width-margin}" y2="{zero_y:.1f}" stroke="#9ca3af" stroke-dasharray="6 4"/>
<polyline points="{coords}" fill="none" stroke="#0f766e" stroke-width="3"/>
{''.join(points)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def comparison_table_html(aggregate_rows: list[dict[str, Any]]) -> str:
    rows = best_by_kind_step(aggregate_rows)
    if not rows:
        return "<p>No aggregate rows available.</p>"
    trs = []
    has_transformer = False
    for row in rows:
        drm = row.get("best_drm")
        tr = row.get("best_transformer")
        has_transformer = has_transformer or tr is not None
        gap = row.get("ce_gap_drm_minus_transformer")
        drm_model = html.escape(drm["model"]) if drm else ""
        drm_ce = f"{float(drm['best_val_ce_mean']):.4f}" if drm else ""
        tr_model = html.escape(tr["model"]) if tr else ""
        tr_ce = f"{float(tr['best_val_ce_mean']):.4f}" if tr else ""
        gap_text = f"{float(gap):.4f}" if gap is not None else ""
        gap_abs = fmt(row.get("gap_abs"))
        gap_rel = fmt(row.get("gap_rel"))
        speed_ratio = fmt(row.get("speed_ratio"), 2)
        param_ratio = fmt(row.get("param_ratio"), 3)
        trs.append(
            f"<tr><td>{row['steps']}</td><td>{drm_model}</td><td>{drm_ce}</td>"
            f"<td>{tr_model}</td><td>{tr_ce}</td><td>{gap_text}</td><td>{gap_abs}</td>"
            f"<td>{gap_rel}</td><td>{speed_ratio}</td><td>{param_ratio}</td></tr>"
        )
    warning = "" if has_transformer else "<p><strong>Warning:</strong> this sweep root has no Transformer runs, so it cannot compare DRM against Transformer.</p>"
    return (
        warning
        + "<table><thead><tr><th>steps</th><th>best DRM</th><th>DRM CE</th><th>best Transformer</th><th>Transformer CE</th><th>drm-transformer</th><th>gap_abs</th><th>gap_rel</th><th>speed_ratio</th><th>param_ratio</th></tr></thead><tbody>"
        + "".join(trs)
        + "</tbody></table>"
    )


def parameter_matched_table_html(aggregate_rows: list[dict[str, Any]]) -> str:
    rows = parameter_matched_rows(aggregate_rows)
    if not rows:
        return "<p>No parameter-matched pairs are complete in this sweep root.</p>"
    trs = []
    for row in rows:
        drm = row["drm"]
        tr = row["transformer"]
        trs.append(
            f"<tr><td>{row['steps']}</td><td>{html.escape(drm['model'])}</td><td>{drm.get('parameter_count', '')}</td><td>{fmt(drm.get('best_val_ce_mean'))}</td>"
            f"<td>{html.escape(tr['model'])}</td><td>{tr.get('parameter_count', '')}</td><td>{fmt(tr.get('best_val_ce_mean'))}</td>"
            f"<td>{fmt(row['gap_abs'])}</td><td>{fmt(row['gap_rel'])}</td><td>{fmt(row['speed_ratio'], 2)}</td><td>{fmt(row['param_ratio'], 3)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>steps</th><th>DRM</th><th>DRM params</th><th>DRM CE</th>"
        "<th>Transformer</th><th>Transformer params</th><th>Transformer CE</th><th>gap_abs</th><th>gap_rel</th><th>speed_ratio</th><th>param_ratio</th></tr></thead><tbody>"
        + "".join(trs)
        + "</tbody></table>"
    )


def save_scatter_svg(path: Path, rows: list[dict[str, Any]], x_metric: str, y_metric: str, title: str, xlabel: str) -> None:
    points = [
        (row["model"], row["kind"], float(row[x_metric]), float(row[y_metric]))
        for row in rows
        if row.get(x_metric) is not None and row.get(y_metric) is not None
    ]
    width, height, margin = 920, 560, 70
    if not points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    xs = [p[2] for p in points]
    ys = [p[3] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_pad = max((max_x - min_x) * 0.1, 1e-6)
    y_pad = max((max_y - min_y) * 0.1, 0.05)
    min_x, max_x = min_x - x_pad, max_x + x_pad
    min_y, max_y = min_y - y_pad, max_y + y_pad
    plot_w, plot_h = width - 2 * margin, height - 2 * margin

    def xy(x: float, y: float) -> tuple[float, float]:
        return (
            margin + (x - min_x) / max(max_x - min_x, 1e-8) * plot_w,
            margin + (max_y - y) / max(max_y - min_y, 1e-8) * plot_h,
        )

    elems = []
    for model, kind, x, y in points:
        px, py = xy(x, y)
        color = "#0f766e" if kind == "drm" else "#b91c1c"
        elems.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='5' fill='{color}'/>")
        elems.append(f"<text x='{px+7:.1f}' y='{py-6:.1f}' font-size='11'>{html.escape(model)}</text>")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-size="22" font-family="Arial" font-weight="700" fill="#111827">{title}</text>
<text x="{margin}" y="56" font-size="13" font-family="Arial" fill="#4b5563">{xlabel} vs best validation CE. Lower CE is better.</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>
{''.join(elems)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def save_dashboard(root: Path, title: str, aggregate_rows: list[dict[str, Any]]) -> None:
    svg_names = [
        "val_ce_by_step.svg",
        "best_val_ce_by_model.svg",
        "tokens_sec_by_model.svg",
        "drm_geometry_by_step.svg",
        "drm_vs_transformer_gap.svg",
        "pareto_ce_vs_params.svg",
        "pareto_ce_vs_tokens_sec.svg",
        "pareto_ce_vs_elapsed_sec.svg",
        "pareto_ce_vs_tokens_seen.svg",
        "sweep_aggregate.svg",
    ]
    sections = []
    for name in svg_names:
        sections.append(f"<section><h2>{html.escape(name)}</h2>{svg_inline(root / name)}</section>")
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f9fafb; color: #111827; }}
    section {{ background: white; border: 1px solid #d1d5db; margin: 20px 0; padding: 16px; }}
    svg {{ max-width: 100%; height: auto; display: block; }}
    code {{ background: #f3f4f6; padding: 2px 4px; }}
    table {{ border-collapse: collapse; background: white; margin: 16px 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 10px; text-align: right; }}
    th:nth-child(2), td:nth-child(2), th:nth-child(4), td:nth-child(4) {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Generated from <code>{html.escape(str(root))}</code>. This dashboard can include partial sweep results.</p>
  <section><h2>DRM vs Transformer Summary</h2>{comparison_table_html(aggregate_rows)}</section>
  <section><h2>Parameter-Matched Pairs</h2>{parameter_matched_table_html(aggregate_rows)}</section>
  {''.join(sections)}
</body>
</html>
"""
    (root / "dashboard.html").write_text(page, encoding="utf-8")
    md = ["# " + title, "", f"Root: `{root}`", ""]
    for name in svg_names:
        md.append(f"- [{name}]({name})")
    (root / "dashboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/sweep_drm_transformer")
    parser.add_argument("--title", default="DRM Competition Dashboard")
    args = parser.parse_args()
    root = Path(args.root)
    rows = collect_rows(root)
    if not rows:
        raise SystemExit(f"no metrics.json found under {root}")
    aggregate_rows = aggregate(rows)
    save_json(root / "sweep_summary.json", {"runs": rows, "aggregate": aggregate_rows})
    save_csv(root / "summary.csv", rows)
    save_csv(root / "sweep_summary.csv", rows)
    save_csv(root / "sweep_aggregate.csv", aggregate_rows)
    save_line_svg(root / "val_ce_by_step.svg", aggregate_rows, "best_val_ce_mean", "Best Validation CE By Step", "Mean over seeds. Lower is better.")
    save_bar_svg(root / "best_val_ce_by_model.svg", aggregate_rows, "best_val_ce_mean", "Best Validation CE By Model", "Latest configured step per model. Lower is better.")
    save_bar_svg(root / "tokens_sec_by_model.svg", aggregate_rows, "tokens_per_sec_mean", "Tokens Per Second By Model", "Latest configured step per model. Higher is better.")
    save_line_svg(root / "drm_geometry_by_step.svg", [r for r in aggregate_rows if r["kind"] == "drm"], "condition_proxy_mean", "DRM Geometry Condition By Step", "Mean condition proxy over seeds. Lower is more stable.")
    save_gap_svg(root / "drm_vs_transformer_gap.svg", aggregate_rows)
    save_scatter_svg(root / "pareto_ce_vs_params.svg", aggregate_rows, "parameter_count", "best_val_ce_mean", "Pareto: CE vs Parameters", "parameter count")
    save_scatter_svg(root / "pareto_ce_vs_tokens_sec.svg", aggregate_rows, "tokens_per_sec_mean", "best_val_ce_mean", "Pareto: CE vs Tokens/Sec", "tokens/sec")
    save_scatter_svg(root / "pareto_ce_vs_elapsed_sec.svg", aggregate_rows, "elapsed_sec_mean", "best_val_ce_mean", "Pareto: CE vs Wall-Clock", "wall-clock training seconds")
    save_scatter_svg(root / "pareto_ce_vs_tokens_seen.svg", aggregate_rows, "tokens_seen_mean", "best_val_ce_mean", "Pareto: CE vs Tokens Seen", "tokens processed")
    save_line_svg(root / "sweep_aggregate.svg", aggregate_rows, "best_val_ce_mean", "Sweep Best Validation CE", "Mean over seeds. Lower is better.")
    save_dashboard(root, args.title, aggregate_rows)
    print(f"saved={root / 'dashboard.html'}")
    print(f"saved={root / 'dashboard.md'}")
    print(f"saved={root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
