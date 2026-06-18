from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.generation import _advance
from drm_language_emitter.tokenizer import load_tokenizer
from drm_language_emitter.utils import load_yaml_or_json, save_json
from sweep_drm_transformer import DRM_MODELS, TRANSFORMER_MODELS, save_csv, write_seed_config
from transformer.checkpoint import load_transformer
from transformer.tiny_transformer import generate_transformer
from world_model.symbolic_world_model import SymbolicWorldModelConfig, count_parameters, decode_world_ids, encode_world_text
from world_model.tiny_world import invalid_prediction, parse_next_target, read_jsonl


WORLD_MODELS = {
    "world_model_tiny": "configs/world_model_tiny.yaml",
    "world_model_stronger": "configs/world_model_stronger.yaml",
}


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def final(history: list[dict[str, Any]], key: str) -> Any:
    return history[-1].get(key) if history else None


@torch.no_grad()
def greedy_drm(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    model.eval()
    z = model.initializer(input_ids.shape[0], input_ids.device)
    for t in range(input_ids.shape[1]):
        z = _advance(model, z, model.token_embedding(input_ids[:, t]))
    out = [input_ids]
    current = input_ids[:, -1]
    for _ in range(max_new_tokens):
        logits = model.emitter(z)
        current = logits.argmax(dim=-1)
        out.append(current[:, None])
        z = _advance(model, z, model.token_embedding(current))
    return torch.cat(out, dim=1)


@torch.no_grad()
def greedy_transformer(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    model.eval()
    out = input_ids
    for _ in range(max_new_tokens):
        context = out[:, -model.config.max_seq_len :]
        logits = model(context)["logits"][:, -1]
        out = torch.cat([out, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return out


def score_prediction(pred: str, target: str, task: str, grid_size: int) -> dict[str, float]:
    exact = float(pred == target)
    next_exact = exact if task == "next_state" else 0.0
    rollout_exact = exact if task == "rollout" else 0.0
    token_total = max(len(target), 1)
    token_correct = sum(int(a == b) for a, b in zip(pred, target)) / token_total
    reward_acc = 0.0
    done_acc = 0.0
    reward_n = 0
    done_n = 0
    if task == "next_state":
        p_next = parse_next_target(pred)
        t_next = parse_next_target(target)
        if p_next and t_next:
            reward_acc = float(p_next["reward"] == t_next["reward"])
            done_acc = float(p_next["done"] == t_next["done"])
            reward_n = 1
            done_n = 1
    return {
        "exact": exact,
        "next_exact": next_exact,
        "rollout_exact": rollout_exact,
        "rollout_token_accuracy": token_correct,
        "reward_accuracy": reward_acc,
        "done_accuracy": done_acc,
        "reward_n": reward_n,
        "done_n": done_n,
        "invalid": float(invalid_prediction(pred, grid_size)),
    }


def evaluate_lm_world_metrics(kind: str, run_dir: Path, dataset_root: Path, max_items: int = 128) -> dict[str, float]:
    records = read_jsonl(dataset_root / "val.jsonl")[:max_items]
    dataset_config = load_yaml_or_json(dataset_root / "dataset_config.json")
    grid_size = int(dataset_config.get("grid_size", 5))
    if kind == "drm":
        model = load_model(run_dir / "drm_tiny.pt")
        tokenizer = load_tokenizer(run_dir / "tokenizer.json")
        decode = lambda prompt, n: tokenizer.decode(greedy_drm(model, torch.tensor([tokenizer.encode(prompt)], dtype=torch.long), n)[0].tolist()[len(tokenizer.encode(prompt)) :])
    else:
        model = load_transformer(run_dir / "tiny_transformer.pt")
        tokenizer = load_tokenizer(run_dir / "tokenizer.json")
        decode = lambda prompt, n: tokenizer.decode(greedy_transformer(model, torch.tensor([tokenizer.encode(prompt)], dtype=torch.long), n)[0].tolist()[len(tokenizer.encode(prompt)) :])
    totals = {
        "next_state_exact_match": 0.0,
        "reward_accuracy": 0.0,
        "done_accuracy": 0.0,
        "rollout_exact_match": 0.0,
        "rollout_token_accuracy": 0.0,
        "invalid_state_rate": 0.0,
    }
    next_n = 0
    rollout_n = 0
    reward_n = 0
    done_n = 0
    for item in records:
        prompt = item["input"] + " => "
        pred = decode(prompt, len(item["target"])).strip()
        target = item["target"].strip()
        scored = score_prediction(pred, target, item["task"], grid_size)
        if item["task"] == "next_state":
            next_n += 1
            totals["next_state_exact_match"] += scored["next_exact"]
        else:
            rollout_n += 1
            totals["rollout_exact_match"] += scored["rollout_exact"]
        totals["rollout_token_accuracy"] += scored["rollout_token_accuracy"]
        totals["invalid_state_rate"] += scored["invalid"]
        totals["reward_accuracy"] += scored["reward_accuracy"]
        totals["done_accuracy"] += scored["done_accuracy"]
        reward_n += scored["reward_n"]
        done_n += scored["done_n"]
    n = max(len(records), 1)
    return {
        "next_state_exact_match": totals["next_state_exact_match"] / max(next_n, 1),
        "reward_accuracy": totals["reward_accuracy"] / max(reward_n, 1),
        "done_accuracy": totals["done_accuracy"] / max(done_n, 1),
        "rollout_exact_match": totals["rollout_exact_match"] / max(rollout_n, 1),
        "rollout_token_accuracy": totals["rollout_token_accuracy"] / n,
        "invalid_state_rate": totals["invalid_state_rate"] / n,
    }


def row_from_run(model_name: str, kind: str, steps: int, seed: int, run_dir: Path, dataset_root: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    history = metrics.get("history", [])
    row = {
        "model": model_name,
        "kind": kind,
        "steps": steps,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_val_ce": metrics.get("best_val_ce"),
        "final_val_ce": metrics.get("final_val_ce", final(history, "val_ce")),
        "parameter_count": metrics.get("parameter_count"),
        "elapsed_sec": metrics.get("elapsed_sec"),
        "tokens_seen": metrics.get("tokens_seen"),
        "tokens_per_sec": final(history, "tokens_per_sec"),
        "next_state_exact_match": metrics.get("next_state_exact_match"),
        "reward_accuracy": metrics.get("reward_accuracy"),
        "done_accuracy": metrics.get("done_accuracy"),
        "rollout_exact_match": metrics.get("rollout_exact_match"),
        "rollout_token_accuracy": metrics.get("rollout_token_accuracy"),
        "invalid_state_rate": metrics.get("invalid_state_rate"),
    }
    if kind in {"drm", "transformer"}:
        row.update(evaluate_lm_world_metrics(kind, run_dir, dataset_root))
        save_json(run_dir / "world_metrics.json", row)
    return row


def train_one(root: Path, model_name: str, kind: str, config_path: str, steps: int, seed: int, dataset_root: Path, batch_size: int, lr: float) -> dict[str, Any]:
    run_dir = root / model_name / f"steps_{steps}" / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and (kind == "world_model" or (run_dir / "world_metrics.json").exists()):
        return row_from_run(model_name, kind, steps, seed, run_dir, dataset_root)
    seed_config = run_dir / "config.yaml"
    write_seed_config(config_path, seed, seed_config)
    if kind == "drm":
        run([sys.executable, "scripts/train_tiny.py", "--config", str(seed_config), "--text", str(dataset_root / "train.txt"), "--output-dir", str(run_dir), "--steps", str(steps), "--batch-size", str(batch_size), "--lr", str(lr)])
    elif kind == "transformer":
        run([sys.executable, "-m", "transformer.run_train", "--config", str(seed_config), "--text", str(dataset_root / "train.txt"), "--output-dir", str(run_dir), "--steps", str(steps), "--batch-size", str(batch_size), "--lr", str(lr)])
    else:
        run([sys.executable, "scripts/train_tiny_world_model.py", "--config", str(seed_config), "--dataset-root", str(dataset_root), "--output-dir", str(run_dir), "--steps", str(steps), "--batch-size", str(batch_size), "--lr", str(lr)])
    return row_from_run(model_name, kind, steps, seed, run_dir, dataset_root)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], int(row["steps"]), row["kind"]), []).append(row)
    metrics = [
        "best_val_ce",
        "final_val_ce",
        "next_state_exact_match",
        "reward_accuracy",
        "done_accuracy",
        "rollout_exact_match",
        "rollout_token_accuracy",
        "invalid_state_rate",
        "elapsed_sec",
        "tokens_seen",
        "tokens_per_sec",
    ]
    out = []
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", type=int, default=[1000, 2000, 3000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--dataset-root", default="data/tiny_world")
    parser.add_argument("--output-root", default="runs/world_model_competition")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--models", nargs="+", default=["drm_tiny", "drm_tiny_104k", "drm_stronger", "transformer_tiny", "transformer_tiny_93k", "transformer_tiny_220k", "world_model_tiny", "world_model_stronger"])
    args = parser.parse_args()
    specs: dict[str, tuple[str, str]] = {name: ("drm", path) for name, path in DRM_MODELS.items()}
    specs.update({name: ("transformer", path) for name, path in TRANSFORMER_MODELS.items()})
    specs.update({name: ("world_model", path) for name, path in WORLD_MODELS.items()})
    root = Path(args.output_root)
    dataset_root = Path(args.dataset_root)
    rows = []
    for model_name in args.models:
        kind, config_path = specs[model_name]
        for steps in args.steps:
            for seed in args.seeds:
                rows.append(train_one(root, model_name, kind, config_path, steps, seed, dataset_root, args.batch_size, args.lr))
    aggregate_rows = aggregate(rows)
    save_json(root / "summary.json", {"runs": rows, "aggregate": aggregate_rows})
    save_csv(root / "summary.csv", rows)
    save_csv(root / "aggregate.csv", aggregate_rows)
    print(f"saved={root / 'summary.json'}")


if __name__ == "__main__":
    main()
