from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.data import build_tokenizer, ensure_text, make_lm_batch
from drm_language_emitter.model import DRMEmitterModel
from drm_language_emitter.utils import load_yaml_or_json, save_json
from prepare_wikipedia_en import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_NAME,
    DEFAULT_SPLIT,
    prepare_wikipedia_en,
)


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "drm_125m": {"family": "drm", "config": "configs/drm_125m.yaml", "scale": "125m"},
    "drm_350m": {"family": "drm", "config": "configs/drm_350m.yaml", "scale": "350m"},
    "gpt2_125m": {"family": "gpt2", "scale": "125m", "n_layer": 12, "n_head": 12, "n_embd": 504},
    "gpt2_350m": {"family": "gpt2", "scale": "350m", "n_layer": 24, "n_head": 16, "n_embd": 1024},
    "opt_125m": {"family": "opt", "scale": "125m", "hidden_size": 504, "layers": 12, "heads": 12, "ffn_dim": 2016},
    "opt_350m": {"family": "opt", "scale": "350m", "hidden_size": 1024, "layers": 24, "heads": 16, "ffn_dim": 4096},
}


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_hf_model(
    model_name: str,
    vocab_size: int,
    max_seq_len: int,
    dropout: float,
) -> torch.nn.Module:
    try:
        from transformers import GPT2Config, GPT2LMHeadModel, OPTConfig, OPTForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency 'transformers'. Install with: pip install -e \".[hf]\""
        ) from exc

    spec = MODEL_SPECS[model_name]
    if spec["family"] == "gpt2":
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_seq_len,
            n_ctx=max_seq_len,
            n_embd=spec["n_embd"],
            n_layer=spec["n_layer"],
            n_head=spec["n_head"],
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            bos_token_id=0,
            eos_token_id=0,
        )
        return GPT2LMHeadModel(config)
    if spec["family"] == "opt":
        config = OPTConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len,
            hidden_size=spec["hidden_size"],
            num_hidden_layers=spec["layers"],
            num_attention_heads=spec["heads"],
            ffn_dim=spec["ffn_dim"],
            dropout=dropout,
            attention_dropout=dropout,
            activation_function="gelu",
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
        )
        return OPTForCausalLM(config)
    raise ValueError(f"unsupported HF model family for {model_name}")


def make_model(
    model_name: str,
    vocab_size: int,
    max_seq_len: int,
    dropout: float,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    spec = MODEL_SPECS[model_name]
    if spec["family"] == "drm":
        config = DRMConfig.from_dict(load_yaml_or_json(spec["config"]))
        config.vocab_size = vocab_size
        config.max_seq_len = max_seq_len
        config.dropout = dropout
        model = DRMEmitterModel(config)
        metadata = config.to_dict()
    else:
        model = make_hf_model(model_name, vocab_size, max_seq_len, dropout)
        metadata = {"model_name": model_name, "vocab_size": vocab_size, "max_seq_len": max_seq_len, **spec}
    return model.to(device), metadata


def forward_loss(
    model: torch.nn.Module,
    family: str,
    x: torch.Tensor,
    y: torch.Tensor,
    global_step: int,
    collect_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if family == "drm":
        out = model(x, y, global_step=global_step, collect_diagnostics=collect_diagnostics)
        return out["loss"], out.get("diagnostics", {})
    out = model(input_ids=x, labels=y)
    loss = out.loss
    return loss, {}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    family: str,
    ids: list[int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
    batches: int,
    global_step: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = []
    diagnostics_accum: dict[str, list[float]] = {}
    for _ in range(max(batches, 1)):
        x, y = make_lm_batch(ids, batch_size, seq_len, device)
        loss, diagnostics = forward_loss(model, family, x, y, global_step, collect_diagnostics=True)
        losses.append(float(loss.detach()))
        for key, value in diagnostics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                diagnostics_accum.setdefault(key, []).append(float(value.detach().cpu()))
    model.train()
    diag = {key: mean(values) for key, values in diagnostics_accum.items() if values}
    return mean(losses), diag


def memory_allocated_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    try:
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    except (AttributeError, RuntimeError):
        return None


def synchronize_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
    except (AttributeError, RuntimeError):
        pass


def initialize_device_seed(device: torch.device, seed: int, track_cuda_memory: bool) -> None:
    if device.type != "cuda":
        return
    try:
        torch.cuda.manual_seed_all(seed)
    except (AttributeError, RuntimeError):
        pass
    if track_cuda_memory:
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except (AttributeError, RuntimeError):
            pass


def train_one(
    model_name: str,
    run_dir: Path,
    text: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    seed: int,
    device: torch.device,
    eval_interval: int,
    eval_batches: int,
    eval_first: bool,
    log_interval: int,
    grad_accum_steps: int,
    dropout: float,
    hf_vocab_size: int,
    dry_run: bool,
    dry_run_forward: bool,
    track_cuda_memory: bool,
    save_best_checkpoint: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    initialize_device_seed(device, seed, track_cuda_memory)
    tokenizer = build_tokenizer(text, "byte")
    data_vocab_size = tokenizer.vocab_size
    model_vocab_size = hf_vocab_size if MODEL_SPECS[model_name]["family"] != "drm" else data_vocab_size
    ids = tokenizer.encode(text)
    split = max(int(len(ids) * 0.9), seq_len + 2)
    train_ids = ids[:split]
    val_ids = ids[max(0, split - seq_len - 1) :]
    model, model_config = make_model(model_name, model_vocab_size, seq_len, dropout, device)
    family = MODEL_SPECS[model_name]["family"]
    parameter_count = count_parameters(model)
    run_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        if dry_run_forward:
            probe_seq_len = min(seq_len, 8)
            x, y = make_lm_batch(train_ids, 1, probe_seq_len, device)
            loss, diagnostics = forward_loss(model, family, x, y, global_step=1, collect_diagnostics=True)
            probe_loss = float(loss.detach().cpu())
            probe_diag = {key: float(value.detach().cpu()) for key, value in diagnostics.items() if isinstance(value, torch.Tensor) and value.numel() == 1}
        else:
            probe_loss = None
            probe_diag = {}
        row = {
            "model": model_name,
            "family": family,
            "scale": MODEL_SPECS[model_name]["scale"],
            "seed": seed,
            "parameter_count": parameter_count,
            "dry_run": True,
            "probe_loss": probe_loss,
            **{f"probe_{key}": value for key, value in probe_diag.items()},
        }
        save_json(run_dir / "metrics.json", {"history": [], "summary": row, "config": model_config})
        return row

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    best_val_ce = float("inf")
    best_step = None
    started = time.perf_counter()
    train_elapsed = 0.0
    eval_elapsed = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, steps + 1):
        synchronize_device(device)
        train_step_started = time.perf_counter()
        step_loss_total = 0.0
        step_diag: dict[str, float] = {}
        for accum_index in range(grad_accum_steps):
            x, y = make_lm_batch(train_ids, batch_size, seq_len, device)
            loss, diagnostics = forward_loss(model, family, x, y, step, collect_diagnostics=False)
            (loss / grad_accum_steps).backward()
            step_loss_total += float(loss.detach().cpu())
            if diagnostics:
                step_diag = {
                    key: float(value.detach().cpu())
                    for key, value in diagnostics.items()
                    if isinstance(value, torch.Tensor) and value.numel() == 1
                }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize_device(device)
        train_elapsed += time.perf_counter() - train_step_started

        should_eval = (eval_first and step == 1) or step % eval_interval == 0 or step == steps
        should_log_train = log_interval > 0 and step % log_interval == 0 and not should_eval
        if should_log_train:
            tokens_seen = step * grad_accum_steps * batch_size * seq_len
            train_ce = step_loss_total / grad_accum_steps
            print(
                f"model={model_name} step={step} train_ce={train_ce:.4f} "
                f"train_tokens_sec={tokens_seen / max(train_elapsed, 1e-8):.1f}",
                flush=True,
            )
        if should_eval:
            train_ce = step_loss_total / grad_accum_steps
            synchronize_device(device)
            eval_started = time.perf_counter()
            val_ce, val_diag = evaluate(model, family, val_ids, seq_len, batch_size, device, eval_batches, step)
            synchronize_device(device)
            eval_elapsed += time.perf_counter() - eval_started
            best_val_ce = min(best_val_ce, val_ce)
            elapsed = time.perf_counter() - started
            tokens_seen = step * grad_accum_steps * batch_size * seq_len
            row = {
                "step": step,
                "train_ce": train_ce,
                "val_ce": val_ce,
                "train_ppl": float(math.exp(min(train_ce, 20.0))),
                "val_ppl": float(math.exp(min(val_ce, 20.0))),
                "best_val_ce": best_val_ce,
                "elapsed_sec": elapsed,
                "train_elapsed_sec": train_elapsed,
                "eval_elapsed_sec": eval_elapsed,
                "tokens_seen": tokens_seen,
                "tokens_per_sec": tokens_seen / max(train_elapsed, 1e-8),
                "wall_tokens_per_sec": tokens_seen / max(elapsed, 1e-8),
                "parameter_count": parameter_count,
                "max_memory_mb": memory_allocated_mb(device) if track_cuda_memory else None,
            }
            for key, value in {**step_diag, **{f"val_{k}": v for k, v in val_diag.items()}}.items():
                row[key] = value
            history.append(row)
            if val_ce <= best_val_ce:
                best_step = step
                if save_best_checkpoint:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": model_config,
                            "summary": {
                                "model": model_name,
                                "family": family,
                                "scale": MODEL_SPECS[model_name]["scale"],
                                "seed": seed,
                                "parameter_count": parameter_count,
                                "best_val_ce": best_val_ce,
                                "best_step": best_step,
                                "elapsed_sec": elapsed,
                                "train_elapsed_sec": train_elapsed,
                                "eval_elapsed_sec": eval_elapsed,
                                "tokens_seen": tokens_seen,
                            },
                        },
                        run_dir / "checkpoint_best.pt",
                    )
            print(
                f"model={model_name} step={step} train_ce={train_ce:.4f} "
                f"val_ce={val_ce:.4f} val_ppl={row['val_ppl']:.2f} "
                f"train_tokens_sec={row['tokens_per_sec']:.1f} "
                f"wall_tokens_sec={row['wall_tokens_per_sec']:.1f} "
                f"eval_sec={eval_elapsed:.1f}",
                flush=True,
            )

    wall_elapsed = time.perf_counter() - started
    summary = {
        "model": model_name,
        "family": family,
        "scale": MODEL_SPECS[model_name]["scale"],
        "seed": seed,
        "parameter_count": parameter_count,
        "best_val_ce": best_val_ce,
        "best_step": best_step,
        "final_train_ce": history[-1]["train_ce"] if history else None,
        "final_val_ce": history[-1]["val_ce"] if history else None,
        "final_val_ppl": history[-1]["val_ppl"] if history else None,
        "elapsed_sec": wall_elapsed,
        "train_elapsed_sec": train_elapsed,
        "eval_elapsed_sec": eval_elapsed,
        "tokens_seen": steps * grad_accum_steps * batch_size * seq_len,
        "tokens_per_sec": history[-1]["tokens_per_sec"] if history else None,
        "wall_tokens_per_sec": (steps * grad_accum_steps * batch_size * seq_len) / max(wall_elapsed, 1e-8),
        "max_memory_mb": memory_allocated_mb(device) if track_cuda_memory else None,
    }
    save_json(run_dir / "metrics.json", {"history": history, "summary": summary, "config": model_config})
    torch.save({"model": model.state_dict(), "config": model_config, "summary": summary}, run_dir / "checkpoint_last.pt")
    return summary


def profile_drm_run(run_dir: Path, batch_size: int, seq_len: int, repeats: int) -> None:
    checkpoint = run_dir / "checkpoint_best.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "checkpoint_last.pt"
    if not checkpoint.exists():
        return
    output_json = run_dir / "profile.json"
    output_md = run_dir / "profile.md"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("profile_drm.py")),
            "--checkpoint",
            str(checkpoint),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--batch-size",
            str(batch_size),
            "--seq-len",
            str(seq_len),
            "--repeats",
            str(repeats),
        ],
        check=True,
    )


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], str(row["scale"])), []).append(row)
    metrics = [
        "parameter_count",
        "best_val_ce",
        "best_step",
        "final_train_ce",
        "final_val_ce",
        "final_val_ppl",
        "elapsed_sec",
        "train_elapsed_sec",
        "eval_elapsed_sec",
        "tokens_seen",
        "tokens_per_sec",
        "wall_tokens_per_sec",
        "max_memory_mb",
        "probe_loss",
    ]
    out = []
    for (model, scale), items in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "model": model,
            "family": items[0]["family"],
            "scale": scale,
            "n": len(items),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            if values:
                summary[f"{metric}_mean"] = mean(values)
                summary[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(summary)
    return out


def svg_line(path: Path, rows: list[dict[str, Any]], metric: str, title: str) -> None:
    series: dict[str, list[tuple[int, float]]] = {}
    for metrics_path in sorted(path.parent.rglob("metrics.json")):
        payload = load_json(metrics_path)
        model = payload.get("summary", {}).get("model")
        if not model:
            continue
        points = [(int(row["step"]), float(row[metric])) for row in payload.get("history", []) if metric in row and row.get(metric) is not None]
        if points:
            series.setdefault(model, []).extend(points)
    width, height, margin = 1100, 620, 72
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    max_step = max(step for step, _ in all_points)
    min_v, max_v = min(value for _, value in all_points), max(value for _, value in all_points)
    pad = max((max_v - min_v) * 0.12, 0.1)
    min_v, max_v = min_v - pad, max_v + pad
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    palette = ["#0f766e", "#b91c1c", "#2563eb", "#9333ea", "#ea580c", "#4b5563"]
    elems = []
    for idx, model in enumerate(sorted(series)):
        color = palette[idx % len(palette)]
        points = sorted(series[model])
        coords = []
        for step, value in points:
            x = margin + step / max(max_step, 1) * plot_w
            y = margin + (max_v - value) / max(max_v - min_v, 1e-8) * plot_h
            coords.append(f"{x:.1f},{y:.1f}")
        elems.append(f"<polyline points='{' '.join(coords)}' fill='none' stroke='{color}' stroke-width='3'/>")
        y_legend = 88 + idx * 22
        elems.append(f"<line x1='{width-300}' y1='{y_legend}' x2='{width-260}' y2='{y_legend}' stroke='{color}' stroke-width='3'/>")
        elems.append(f"<text x='{width-250}' y='{y_legend+4}' font-size='12'>{model}</text>")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-size="22" font-family="Arial" font-weight="700" fill="#111827">{title}</text>
<text x="{margin}" y="56" font-size="13" font-family="Arial" fill="#4b5563">Lower is better for CE/perplexity. Lines show logged checkpoints.</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>
{''.join(elems)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def svg_bar(path: Path, rows: list[dict[str, Any]], metric: str, title: str, higher_is_better: bool = False) -> None:
    items = [(row["model"], float(row[metric])) for row in rows if row.get(metric) is not None]
    width, height = 1100, 580
    if not items:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    max_v = max(value for _, value in items) * 1.15 + 1e-8
    x0, y0, bar_w, gap = 70, 470, 110, 24
    bars = []
    for idx, (model, value) in enumerate(items):
        h = value / max_v * 340
        x, y = x0 + idx * (bar_w + gap), y0 - h
        color = "#0f766e" if model.startswith("drm") else "#b91c1c" if model.startswith("gpt2") else "#2563eb"
        bars.append(f"<rect x='{x}' y='{y:.1f}' width='{bar_w}' height='{h:.1f}' fill='{color}'/>")
        bars.append(f"<text x='{x}' y='{y-8:.1f}' font-size='11'>{value:.3g}</text>")
        bars.append(f"<text x='{x}' y='{y0+18}' font-size='10'>{model}</text>")
    direction = "Higher is better." if higher_is_better else "Lower is better."
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="60" y="34" font-size="22" font-family="Arial" font-weight="700" fill="#111827">{title}</text>
<text x="60" y="56" font-size="13" font-family="Arial" fill="#4b5563">{direction}</text>
<line x1="50" y1="{y0}" x2="{width-40}" y2="{y0}" stroke="#111827"/>
{''.join(bars)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_dashboard(root: Path, rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    graph_names = [
        "val_ce_by_step.svg",
        "val_ppl_by_step.svg",
        "best_val_ce_by_model.svg",
        "tokens_sec_by_model.svg",
        "params_by_model.svg",
        "memory_by_model.svg",
        "drm_action_by_step.svg",
        "drm_dimD_by_step.svg",
        "drm_condition_by_step.svg",
    ]
    rows_html = []
    for row in aggregate_rows:
        rows_html.append(
            "<tr>"
            f"<td>{row['model']}</td><td>{row['family']}</td><td>{row['scale']}</td>"
            f"<td>{row.get('parameter_count_mean', ''):.0f}</td>"
            f"<td>{row.get('best_val_ce_mean', float('nan')):.4f}</td>"
            f"<td>{row.get('final_val_ppl_mean', float('nan')):.3f}</td>"
            f"<td>{row.get('tokens_per_sec_mean', float('nan')):.1f}</td>"
            f"<td>{row.get('max_memory_mb_mean', float('nan')):.1f}</td>"
            "</tr>"
        )
    graph_html = []
    for name in graph_names:
        graph_path = root / name
        if graph_path.exists():
            graph_html.append(f"<section><h2>{name}</h2>{graph_path.read_text(encoding='utf-8')}</section>")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>DRM vs GPT-2 vs OPT Scale Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #f9fafb; }}
    table {{ border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 10px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
    section {{ background: white; border: 1px solid #d1d5db; padding: 16px; margin: 20px 0; }}
    svg {{ max-width: 100%; height: auto; }}
    code {{ background: #f3f4f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>DRM vs GPT-2 vs OPT Scale Comparison</h1>
  <p>All default runs use the same project byte tokenizer and train from scratch. See <code>summary.json</code> and <code>aggregate.csv</code> for machine-readable results.</p>
  <section>
    <h2>Aggregate</h2>
    <table>
      <thead><tr><th>model</th><th>family</th><th>scale</th><th>params</th><th>best val CE</th><th>final val PPL</th><th>tokens/sec</th><th>max memory MB</th></tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
  </section>
  {''.join(graph_html)}
</body>
</html>
"""
    (root / "dashboard.html").write_text(html, encoding="utf-8")


def build_outputs(root: Path, rows: list[dict[str, Any]]) -> None:
    aggregate_rows = aggregate(rows)
    save_json(root / "summary.json", {"runs": rows, "aggregate": aggregate_rows, "model_specs": MODEL_SPECS})
    save_csv(root / "runs.csv", rows)
    save_csv(root / "aggregate.csv", aggregate_rows)
    svg_line(root / "val_ce_by_step.svg", rows, "val_ce", "Validation CE By Step")
    svg_line(root / "val_ppl_by_step.svg", rows, "val_ppl", "Validation Perplexity By Step")
    svg_line(root / "drm_action_by_step.svg", rows, "action_mean", "DRM Action Mean By Step")
    svg_line(root / "drm_dimD_by_step.svg", rows, "dimD_mean", "DRM Active Dimension By Step")
    svg_line(root / "drm_condition_by_step.svg", rows, "condition_proxy", "DRM Condition Proxy By Step")
    svg_bar(root / "best_val_ce_by_model.svg", aggregate_rows, "best_val_ce_mean", "Best Validation CE By Model")
    svg_bar(root / "tokens_sec_by_model.svg", aggregate_rows, "tokens_per_sec_mean", "Tokens Per Second By Model", higher_is_better=True)
    svg_bar(root / "params_by_model.svg", aggregate_rows, "parameter_count_mean", "Parameter Count By Model")
    svg_bar(root / "memory_by_model.svg", aggregate_rows, "max_memory_mb_mean", "Peak CUDA Memory By Model")
    write_dashboard(root, rows, aggregate_rows)


def resolve_training_text(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.dataset == "text":
        text = ensure_text(args.text)
        return text, {"dataset": "text", "text_path": args.text}
    wikipedia_text_path = Path(args.wikipedia_output)
    metadata = prepare_wikipedia_en(
        output=wikipedia_text_path,
        dataset_name=args.wikipedia_dataset_name,
        dataset_config=args.wikipedia_dataset_config,
        split=args.wikipedia_split,
        max_chars=args.wikipedia_max_chars,
        max_docs=args.wikipedia_max_docs,
        min_doc_chars=args.wikipedia_min_doc_chars,
        streaming=args.wikipedia_streaming,
        overwrite=args.wikipedia_overwrite,
    )
    text = wikipedia_text_path.read_text(encoding="utf-8")
    return text, {"dataset": "wikipedia-en", **metadata}


def resolve_device(requested: str, strict: bool) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        message = (
            "CUDA was requested, but this PyTorch build has no available CUDA backend "
            f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda})."
        )
        if strict:
            raise SystemExit(message)
        print(f"warning: {message} Falling back to CPU. Pass --strict-device to fail instead.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--dataset", choices=["text", "wikipedia-en"], default="text")
    parser.add_argument("--text", default="data/tiny.txt")
    parser.add_argument("--wikipedia-output", default="data/wikipedia_en_20231101_sample.txt")
    parser.add_argument("--wikipedia-dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--wikipedia-dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--wikipedia-split", default=DEFAULT_SPLIT)
    parser.add_argument("--wikipedia-max-chars", type=int, default=50_000_000)
    parser.add_argument("--wikipedia-max-docs", type=int, default=0)
    parser.add_argument("--wikipedia-min-doc-chars", type=int, default=200)
    parser.add_argument("--wikipedia-streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wikipedia-overwrite", action="store_true")
    parser.add_argument("--output-root", default="runs/scale_lm_comparison")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--eval-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hf-vocab-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-forward", action="store_true")
    parser.add_argument("--save-best-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-drm", action="store_true")
    parser.add_argument("--profile-batch-size", type=int, default=1)
    parser.add_argument("--profile-seq-len", type=int, default=32)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--no-cuda-memory-stats", action="store_true")
    parser.add_argument("--strict-device", action="store_true")
    args = parser.parse_args()

    unknown = sorted(set(args.models) - set(MODEL_SPECS))
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}; available={sorted(MODEL_SPECS)}")
    device = resolve_device(args.device, args.strict_device)
    text, dataset_metadata = resolve_training_text(args)
    root = Path(args.output_root)
    rows = []
    for model_name in args.models:
        for seed in args.seeds:
            run_dir = root / model_name / f"seed_{seed}"
            row = train_one(
                model_name=model_name,
                run_dir=run_dir,
                text=text,
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                lr=args.lr,
                seed=seed,
                device=device,
                eval_interval=args.eval_interval,
                eval_batches=args.eval_batches,
                eval_first=args.eval_first,
                log_interval=args.log_interval,
                grad_accum_steps=args.grad_accum_steps,
                dropout=args.dropout,
                hf_vocab_size=args.hf_vocab_size,
                dry_run=args.dry_run,
                dry_run_forward=args.dry_run_forward,
                track_cuda_memory=not args.no_cuda_memory_stats,
                save_best_checkpoint=args.save_best_checkpoint,
            )
            if args.profile_drm and MODEL_SPECS[model_name]["family"] == "drm" and not args.dry_run:
                profile_drm_run(
                    run_dir=run_dir,
                    batch_size=args.profile_batch_size,
                    seq_len=args.profile_seq_len,
                    repeats=args.profile_repeats,
                )
            rows.append(row)
    build_outputs(root, rows)
    save_json(root / "dataset.json", dataset_metadata)
    print(f"saved={root / 'summary.json'}")
    print(f"saved={root / 'aggregate.csv'}")
    print(f"saved={root / 'dashboard.html'}")
    print(f"saved={root / 'dataset.json'}")


if __name__ == "__main__":
    main()
