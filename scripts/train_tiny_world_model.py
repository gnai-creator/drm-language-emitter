from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from drm_language_emitter.utils import load_yaml_or_json, save_json
from world_model.symbolic_world_model import (
    EOS_ID,
    PAD_ID,
    SymbolicWorldModel,
    SymbolicWorldModelConfig,
    count_parameters,
    decode_world_ids,
    encode_world_text,
)
from world_model.tiny_world import invalid_prediction, parse_next_target, read_jsonl


def make_batch(records: list[dict[str, Any]], config: SymbolicWorldModelConfig, batch_size: int, rng: random.Random, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    items = [records[rng.randrange(len(records))] for _ in range(batch_size)]
    x = [encode_world_text(item["input"], config.max_input_len) for item in items]
    y = [encode_world_text(item["target"], config.max_target_len, add_eos=True) for item in items]
    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_world_model(
    model: SymbolicWorldModel,
    records: list[dict[str, Any]],
    max_items: int = 256,
    grid_size: int = 5,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    config = model.config
    total_ce = 0.0
    total_tokens = 0
    token_correct = 0
    exact = 0
    next_exact = 0
    next_total = 0
    rollout_exact = 0
    rollout_total = 0
    reward_correct = 0
    reward_total = 0
    done_correct = 0
    done_total = 0
    invalid = 0
    items = records[:max_items]
    for item in items:
        x = torch.tensor([encode_world_text(item["input"], config.max_input_len)], dtype=torch.long, device=device)
        y = torch.tensor([encode_world_text(item["target"], config.max_target_len, add_eos=True)], dtype=torch.long, device=device)
        out = model(x, y)
        logits = out["logits"]
        pred_ids = logits.argmax(dim=-1)[0].tolist()
        target_ids = y[0].tolist()
        mask = [idx != PAD_ID for idx in target_ids]
        total_ce += float(out["loss"]) * sum(mask)
        total_tokens += sum(mask)
        token_correct += sum(int(p == t) for p, t, m in zip(pred_ids, target_ids, mask) if m)
        pred = decode_world_ids(pred_ids).strip()
        target = item["target"].strip()
        is_exact = int(pred == target)
        exact += is_exact
        invalid += int(invalid_prediction(pred, grid_size))
        if item["task"] == "next_state":
            next_total += 1
            next_exact += is_exact
            pred_next = parse_next_target(pred)
            target_next = parse_next_target(target)
            if pred_next is not None and target_next is not None:
                reward_total += 1
                done_total += 1
                reward_correct += int(pred_next["reward"] == target_next["reward"])
                done_correct += int(pred_next["done"] == target_next["done"])
        else:
            rollout_total += 1
            rollout_exact += is_exact
    denom = max(len(items), 1)
    return {
        "best_val_ce": total_ce / max(total_tokens, 1),
        "final_val_ce": total_ce / max(total_tokens, 1),
        "next_state_exact_match": next_exact / max(next_total, 1),
        "reward_accuracy": reward_correct / max(reward_total, 1),
        "done_accuracy": done_correct / max(done_total, 1),
        "rollout_exact_match": rollout_exact / max(rollout_total, 1),
        "rollout_token_accuracy": token_correct / max(total_tokens, 1),
        "invalid_state_rate": invalid / denom,
    }


def train_world_model(
    config: SymbolicWorldModelConfig,
    dataset_root: str | Path,
    output_dir: str | Path,
    steps: int = 1000,
    batch_size: int = 16,
    lr: float = 3e-4,
    device_name: str = "cpu",
) -> Path:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = read_jsonl(dataset_root / "train.jsonl")
    val_records = read_jsonl(dataset_root / "val.jsonl")
    dataset_config = load_yaml_or_json(dataset_root / "dataset_config.json")
    grid_size = int(dataset_config.get("grid_size", 5))
    model = SymbolicWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_ce = float("inf")
    best_checkpoint = output_dir / "world_model_best.pt"
    last_checkpoint = output_dir / "world_model_last.pt"
    history = []
    train_start = perf_counter()
    for step in range(steps):
        x, y = make_batch(train_records, config, batch_size, rng, device)
        out = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            metrics = evaluate_world_model(model, val_records, max_items=256, grid_size=grid_size)
            elapsed = perf_counter() - train_start
            row = {
                "step": step + 1,
                "train_ce": float(out["loss"].detach()),
                "val_ce": metrics["final_val_ce"],
                "best_val_ce": min(best_val_ce, metrics["final_val_ce"]),
                "next_state_exact_match": metrics["next_state_exact_match"],
                "rollout_exact_match": metrics["rollout_exact_match"],
                "rollout_token_accuracy": metrics["rollout_token_accuracy"],
                "invalid_state_rate": metrics["invalid_state_rate"],
                "parameter_count": count_parameters(model),
                "elapsed_sec": elapsed,
                "tokens_seen": (step + 1) * batch_size * (config.max_input_len + config.max_target_len),
            }
            row["tokens_per_sec"] = row["tokens_seen"] / max(elapsed, 1e-8)
            history.append(row)
            print(
                f"step={step+1} world_train_ce={row['train_ce']:.4f} val_ce={row['val_ce']:.4f} "
                f"next_exact={row['next_state_exact_match']:.3f} rollout_exact={row['rollout_exact_match']:.3f} "
                f"invalid={row['invalid_state_rate']:.3f} params={row['parameter_count']}"
            )
            if metrics["final_val_ce"] < best_val_ce:
                best_val_ce = metrics["final_val_ce"]
                torch.save(model.state_dict_with_config(), best_checkpoint)
    torch.save(model.state_dict_with_config(), last_checkpoint)
    if not best_checkpoint.exists():
        torch.save(model.state_dict_with_config(), best_checkpoint)
    alias = output_dir / "world_model.pt"
    alias.write_bytes(best_checkpoint.read_bytes())
    final_metrics = evaluate_world_model(model, val_records, max_items=512, grid_size=grid_size)
    save_json(
        output_dir / "metrics.json",
        {
            "history": history,
            **final_metrics,
            "best_val_ce": best_val_ce,
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "parameter_count": count_parameters(model),
            "elapsed_sec": perf_counter() - train_start,
            "tokens_seen": steps * batch_size * (config.max_input_len + config.max_target_len),
            "config": config.to_dict(),
            "ce_note": "CE is supervised target-token CE for the symbolic world model decoder.",
            "device": str(device),
        },
    )
    save_json(output_dir / "config.json", config.to_dict())
    return alias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-root", default="data/tiny_world")
    parser.add_argument("--output-dir", default="runs/world_model_tiny")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    config = SymbolicWorldModelConfig.from_dict(load_yaml_or_json(args.config)) if args.config else SymbolicWorldModelConfig()
    checkpoint = train_world_model(config, args.dataset_root, args.output_dir, args.steps, args.batch_size, args.lr, args.device)
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
