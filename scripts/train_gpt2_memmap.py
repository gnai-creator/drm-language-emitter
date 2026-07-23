from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from drm_language_emitter.data import MemmapTokenDataset
from drm_language_emitter.training import count_parameters
from drm_language_emitter.utils import save_json


GPT2_SPECS: dict[str, dict[str, int]] = {
    "gpt2_125m": {"n_layer": 12, "n_head": 12, "n_embd": 504},
    "gpt2_125m_real": {"n_layer": 13, "n_head": 14, "n_embd": 896},
}


def distributed_state() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        torch.distributed.init_process_group(backend="nccl")
    return enabled, rank, local_rank, world_size


def resolve_device(requested: str, local_rank: int) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device(requested)


def count_tokens_per_step(batch_size: int, seq_len: int, grad_accum_steps: int, world_size: int) -> int:
    return batch_size * seq_len * grad_accum_steps * world_size


def autocast_context(device: torch.device, precision: str):
    enabled = precision == "bf16" and device.type == "cuda"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def make_gpt2_model(model_size: str, seq_len: int, vocab_size: int, dropout: float) -> torch.nn.Module:
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as exc:
        raise SystemExit('Missing dependency "transformers". Install with: pip install -e ".[hf]"') from exc

    if model_size not in GPT2_SPECS:
        choices = ", ".join(sorted(GPT2_SPECS))
        raise SystemExit(f"unknown model size {model_size!r}. Choices: {choices}")
    spec = GPT2_SPECS[model_size]
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
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


def model_config_dict(model_size: str, seq_len: int, vocab_size: int, dropout: float) -> dict[str, Any]:
    return {
        "model_name": model_size,
        "family": "gpt2",
        "scale": "125m_real" if model_size == "gpt2_125m_real" else "125m",
        "vocab_size": vocab_size,
        "max_seq_len": seq_len,
        "dropout": dropout,
        **GPT2_SPECS[model_size],
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: dict[str, Any],
    args: argparse.Namespace,
    step: int,
    tokens_seen: int,
    parameter_count: int,
    best_val_ce: float,
    world_size: int,
) -> dict[str, Any]:
    module = model.module if hasattr(model, "module") else model
    return {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "step": step,
        "tokens_seen": tokens_seen,
        "parameter_count": parameter_count,
        "best_val_ce": best_val_ce,
        "world_size": world_size,
        "precision": args.precision,
        "dataset_manifest": str(args.dataset_manifest),
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> tuple[int, int, float]:
    payload = torch.load(path, map_location=device, weights_only=False)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return int(payload["step"]), int(payload["tokens_seen"]), float(payload.get("best_val_ce", math.inf))


def resolve_resume_path(output_root: Path, resume: str) -> Path | None:
    if not resume:
        return None
    if resume == "latest":
        path = output_root / "checkpoint_latest.pt"
        return path if path.exists() else None
    return Path(resume)


@torch.no_grad()
def evaluate_ce(
    model: torch.nn.Module,
    dataset: MemmapTokenDataset,
    batch_size: int,
    seq_len: int,
    batches: int,
    device: torch.device,
    precision: str,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(100_000)
    losses: list[float] = []
    for _ in range(max(batches, 1)):
        x, y = dataset.make_batch(batch_size, seq_len, device, generator=generator)
        with autocast_context(device, precision):
            out = model(input_ids=x, labels=y)
        losses.append(float(out.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPT-2 on uint8 token shards from a manifest.")
    parser.add_argument("--model-size", default="gpt2_125m_real", choices=sorted(GPT2_SPECS))
    parser.add_argument("--dataset-manifest", default="data/tokens_5b/manifest.json")
    parser.add_argument("--output-root", default="runs/gpt2_125m_4090_base")
    parser.add_argument("--target-tokens", type=int, default=150_000_000)
    parser.add_argument("--steps", type=int, default=0, help="Override target-tokens with a fixed step count when > 0.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-process batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-tokens-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-tokens-interval", type=int, default=50_000_000)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", default="", help="Path to checkpoint, or 'latest'.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-forward", action="store_true")
    parser.add_argument("--save-best-checkpoint", action="store_true")
    args = parser.parse_args()

    ddp, rank, local_rank, world_size = distributed_state()
    rank_zero = rank == 0
    device = resolve_device(args.device, local_rank)
    output_root = Path(args.output_root)
    dataset_manifest = Path(args.dataset_manifest)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    config = model_config_dict(args.model_size, args.seq_len, 256, args.dropout)
    model = make_gpt2_model(args.model_size, args.seq_len, 256, args.dropout).to(device)
    parameter_count = count_parameters(model)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if rank_zero:
        output_root.mkdir(parents=True, exist_ok=True)
        save_json(
            output_root / "run_config.json",
            {
                "config": config,
                "parameter_count": parameter_count,
                "dataset_manifest": str(dataset_manifest),
                "target_tokens": args.target_tokens,
                "tokens_per_step": count_tokens_per_step(args.batch_size, args.seq_len, args.grad_accum_steps, world_size),
                "world_size": world_size,
                "args": vars(args),
            },
        )
        print(f"parameter_count={parameter_count}", flush=True)
        print(f"dataset_manifest={dataset_manifest}", flush=True)
        print(f"world_size={world_size}", flush=True)

    train_dataset = MemmapTokenDataset(dataset_manifest, split="train")
    val_dataset = MemmapTokenDataset(dataset_manifest, split="val")
    if rank_zero:
        print(f"train_tokens_available={len(train_dataset)}", flush=True)
        print(f"val_tokens_available={len(val_dataset)}", flush=True)

    if args.dry_run:
        if args.dry_run_forward:
            x, y = train_dataset.make_batch(1, min(args.seq_len, 16), device, generator=torch.Generator().manual_seed(args.seed))
            with autocast_context(device, args.precision):
                out = model(input_ids=x, labels=y)
            if rank_zero:
                print(f"dry_run_loss={float(out.loss.detach().cpu()):.6f}", flush=True)
        train_dataset.close()
        val_dataset.close()
        if ddp:
            torch.distributed.destroy_process_group()
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    start_step = 0
    tokens_seen = 0
    best_val_ce = math.inf
    resume_path = resolve_resume_path(output_root, args.resume)
    if resume_path is not None:
        start_step, tokens_seen, best_val_ce = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        if rank_zero:
            print(f"resumed={resume_path} step={start_step} tokens_seen={tokens_seen}", flush=True)

    tokens_per_step = count_tokens_per_step(args.batch_size, args.seq_len, args.grad_accum_steps, world_size)
    total_steps = args.steps if args.steps > 0 else math.ceil(max(args.target_tokens - tokens_seen, 0) / tokens_per_step)
    final_step = start_step + total_steps
    next_eval_tokens = ((tokens_seen // args.eval_tokens_interval) + 1) * args.eval_tokens_interval
    next_checkpoint_tokens = ((tokens_seen // args.checkpoint_tokens_interval) + 1) * args.checkpoint_tokens_interval
    generator = torch.Generator().manual_seed(args.seed + rank * 9973 + start_step)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step + 1, final_step + 1):
        step_loss = 0.0
        for _accum_index in range(args.grad_accum_steps):
            x, y = train_dataset.make_batch(
                args.batch_size,
                args.seq_len,
                device,
                generator=generator,
                rank=rank,
                world_size=world_size,
            )
            with autocast_context(device, args.precision):
                out = model(input_ids=x, labels=y)
                loss = out.loss / args.grad_accum_steps
            loss.backward()
            step_loss += float(out.loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step

        if ddp:
            loss_tensor = torch.tensor(step_loss / args.grad_accum_steps, device=device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
            train_ce = float(loss_tensor.detach().cpu())
        else:
            train_ce = step_loss / args.grad_accum_steps

        eval_due = tokens_seen >= next_eval_tokens
        checkpoint_due = tokens_seen >= next_checkpoint_tokens
        should_log = rank_zero and args.log_interval > 0 and (step == 1 or step % args.log_interval == 0)

        if eval_due and rank_zero:
            val_ce = evaluate_ce(model, val_dataset, args.batch_size, args.seq_len, args.eval_batches, device, args.precision)
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                if args.save_best_checkpoint:
                    payload = checkpoint_payload(model, optimizer, scheduler, config, args, step, tokens_seen, parameter_count, best_val_ce, world_size)
                    save_checkpoint(output_root / "checkpoint_best.pt", payload)
        else:
            val_ce = None
        if eval_due:
            next_eval_tokens += args.eval_tokens_interval
            if ddp:
                torch.distributed.barrier()

        if should_log or (eval_due and rank_zero):
            elapsed = time.perf_counter() - started
            row = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_ce": train_ce,
                "val_ce": val_ce,
                "best_val_ce": best_val_ce if math.isfinite(best_val_ce) else None,
                "tokens_per_sec": (tokens_seen - (start_step * tokens_per_step)) / max(elapsed, 1e-8),
                "elapsed_sec": elapsed,
            }
            history.append(row)
            save_json(output_root / "metrics_latest.json", {"history": history, "latest": row})
            print(json.dumps(row), flush=True)

        if checkpoint_due and rank_zero:
            payload = checkpoint_payload(model, optimizer, scheduler, config, args, step, tokens_seen, parameter_count, best_val_ce, world_size)
            save_checkpoint(output_root / "checkpoint_latest.pt", payload)
            save_checkpoint(output_root / f"checkpoint_tokens_{tokens_seen}.pt", payload)
        if checkpoint_due:
            next_checkpoint_tokens += args.checkpoint_tokens_interval
            if ddp:
                torch.distributed.barrier()

        if tokens_seen >= args.target_tokens:
            break

    if rank_zero:
        payload = checkpoint_payload(model, optimizer, scheduler, config, args, step, tokens_seen, parameter_count, best_val_ce, world_size)
        save_checkpoint(output_root / "checkpoint_last.pt", payload)
        save_json(
            output_root / "summary.json",
            {
                "final_step": step,
                "tokens_seen": tokens_seen,
                "target_tokens": args.target_tokens,
                "parameter_count": parameter_count,
                "best_val_ce": best_val_ce if math.isfinite(best_val_ce) else None,
                "world_size": world_size,
                "dataset_manifest": str(dataset_manifest),
            },
        )

    train_dataset.close()
    val_dataset.close()
    if ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
