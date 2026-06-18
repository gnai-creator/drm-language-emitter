from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.data import build_tokenizer, ensure_text, make_lm_batch
from drm_language_emitter.model import DRMEmitterModel
from drm_language_emitter.training import count_parameters, evaluate_ce
from drm_language_emitter.utils import load_yaml_or_json, save_json
from sweep_drm_transformer import DRM_MODELS, TRANSFORMER_MODELS, save_csv
from transformer.tiny_transformer import TinyTransformerConfig, TinyTransformerLM, count_parameters as count_transformer_parameters
from transformer.train_tiny_transformer import evaluate_transformer_ce


TARGETS = [1.0, 0.75, 0.5]


def target_hits(history: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target in TARGETS:
        hit = next((item for item in history if item.get("val_ce") is not None and float(item["val_ce"]) < target), None)
        suffix = str(target).replace(".", "_")
        if hit:
            out[f"steps_to_ce_lt_{suffix}"] = hit["step"]
            out[f"seconds_to_ce_lt_{suffix}"] = hit["elapsed_sec"]
    return out


def run_drm(config_path: str, text_path: str, duration_sec: float, batch_size: int, lr: float) -> dict[str, Any]:
    device = torch.device("cpu")
    config = DRMConfig.from_dict(load_yaml_or_json(config_path))
    torch.manual_seed(config.seed)
    text = ensure_text(text_path)
    tokenizer = build_tokenizer(text, config.tokenizer_type)
    config.vocab_size = tokenizer.vocab_size
    model = DRMEmitterModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ids = tokenizer.encode(text)
    split = max(int(len(ids) * 0.9), 2)
    train_ids = ids[:split]
    val_ids = ids[max(0, split - config.max_seq_len - 1) :]
    seq_len = min(config.max_seq_len, 64)
    start = perf_counter()
    step = 0
    best_val_ce = float("inf")
    history = []
    while perf_counter() - start < duration_sec:
        step += 1
        x, y = make_lm_batch(train_ids, batch_size, seq_len, device)
        out = model(x, y, global_step=step)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            elapsed = perf_counter() - start
            val_ce, diag = evaluate_ce(model, val_ids, seq_len, device, global_step=step)
            best_val_ce = min(best_val_ce, val_ce)
            tokens_seen = step * batch_size * seq_len
            history.append(
                {
                    "step": step,
                    "elapsed_sec": elapsed,
                    "val_ce": val_ce,
                    "best_val_ce": best_val_ce,
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": tokens_seen / max(elapsed, 1e-8),
                    "condition_proxy": diag.get("condition_proxy"),
                    "dimD_std": diag.get("dimD_std"),
                }
            )
    return {
        "kind": "drm",
        "best_val_ce": best_val_ce,
        "steps": step,
        "elapsed_sec": perf_counter() - start,
        "tokens_seen": step * batch_size * seq_len,
        "tokens_per_sec": step * batch_size * seq_len / max(perf_counter() - start, 1e-8),
        "parameter_count": count_parameters(model),
        "history": history,
        **target_hits(history),
    }


def run_transformer(config_path: str, text_path: str, duration_sec: float, batch_size: int, lr: float) -> dict[str, Any]:
    device = torch.device("cpu")
    config = TinyTransformerConfig.from_dict(load_yaml_or_json(config_path))
    torch.manual_seed(config.seed)
    text = ensure_text(text_path)
    tokenizer = build_tokenizer(text, "byte")
    config.vocab_size = tokenizer.vocab_size
    model = TinyTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ids = tokenizer.encode(text)
    split = max(int(len(ids) * 0.9), 2)
    train_ids = ids[:split]
    val_ids = ids[max(0, split - config.max_seq_len - 1) :]
    seq_len = min(config.max_seq_len, 64)
    start = perf_counter()
    step = 0
    best_val_ce = float("inf")
    history = []
    while perf_counter() - start < duration_sec:
        step += 1
        x, y = make_lm_batch(train_ids, batch_size, seq_len, device)
        out = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            elapsed = perf_counter() - start
            val_ce = evaluate_transformer_ce(model, val_ids, seq_len, device)
            best_val_ce = min(best_val_ce, val_ce)
            tokens_seen = step * batch_size * seq_len
            history.append(
                {
                    "step": step,
                    "elapsed_sec": elapsed,
                    "val_ce": val_ce,
                    "best_val_ce": best_val_ce,
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": tokens_seen / max(elapsed, 1e-8),
                }
            )
    return {
        "kind": "transformer",
        "best_val_ce": best_val_ce,
        "steps": step,
        "elapsed_sec": perf_counter() - start,
        "tokens_seen": step * batch_size * seq_len,
        "tokens_per_sec": step * batch_size * seq_len / max(perf_counter() - start, 1e-8),
        "parameter_count": count_transformer_parameters(model),
        "history": history,
        **target_hits(history),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--durations-sec", nargs="+", type=float, default=[60.0, 300.0, 900.0])
    parser.add_argument("--models", nargs="+", default=["drm_tiny", "drm_tiny_104k", "drm_stronger", "transformer_tiny", "transformer_tiny_93k", "transformer_tiny_220k"])
    parser.add_argument("--text", default="data/tiny.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-root", default="runs/time_matched_competition")
    args = parser.parse_args()

    specs: dict[str, tuple[str, str]] = {name: ("drm", path) for name, path in DRM_MODELS.items()}
    specs.update({name: ("transformer", path) for name, path in TRANSFORMER_MODELS.items()})
    root = Path(args.output_root)
    rows = []
    for duration in args.durations_sec:
        for model_name in args.models:
            kind, config_path = specs[model_name]
            print(f"running model={model_name} duration_sec={duration}", flush=True)
            result = run_drm(config_path, args.text, duration, args.batch_size, args.lr) if kind == "drm" else run_transformer(config_path, args.text, duration, args.batch_size, args.lr)
            result.update({"model": model_name, "duration_sec": duration})
            out_dir = root / model_name / f"seconds_{int(duration)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json(out_dir / "metrics.json", result)
            rows.append({k: v for k, v in result.items() if k != "history"})
    save_json(root / "time_matched_summary.json", {"runs": rows})
    save_csv(root / "time_matched_summary.csv", rows)
    print(f"saved={root / 'time_matched_summary.json'}")


if __name__ == "__main__":
    main()
