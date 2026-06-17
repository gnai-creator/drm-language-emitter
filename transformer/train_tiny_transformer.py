from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from drm_language_emitter.data import build_tokenizer, ensure_text, make_lm_batch
from drm_language_emitter.training import evaluate_ce
from drm_language_emitter.utils import save_json
from .tiny_transformer import TinyTransformerConfig, TinyTransformerLM, count_parameters


@torch.no_grad()
def evaluate_transformer_ce(
    model: TinyTransformerLM, ids: list[int], seq_len: int, device: torch.device
) -> float:
    model.eval()
    if len(ids) < seq_len + 1:
        ids = ids * ((seq_len + 1) // max(len(ids), 1) + 1)
    x = torch.tensor(ids[:seq_len], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1 : seq_len + 1], dtype=torch.long, device=device).unsqueeze(0)
    out = model(x, y)
    model.train()
    return float(out["aux_losses"]["ce"].detach())


def train_tiny_transformer(
    config: TinyTransformerConfig,
    text_path: str | Path,
    output_dir: str | Path,
    steps: int = 400,
    batch_size: int = 8,
    lr: float = 3e-4,
    val_fraction: float = 0.1,
) -> Path:
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    text = ensure_text(text_path)
    tokenizer = build_tokenizer(text, "byte")
    config.vocab_size = tokenizer.vocab_size
    model = TinyTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ids = tokenizer.encode(text)
    split = max(int(len(ids) * (1.0 - val_fraction)), 2)
    train_ids = ids[:split]
    val_ids = ids[max(0, split - config.max_seq_len - 1) :]
    seq_len = min(config.max_seq_len, 64)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    train_start = perf_counter()
    best_val_ce = float("inf")
    best_checkpoint = output_dir / "tiny_transformer_best.pt"

    for step in range(steps):
        x, y = make_lm_batch(train_ids, batch_size, seq_len, device)
        out = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            train_ce = float(out["aux_losses"]["ce"].detach())
            val_ce = evaluate_transformer_ce(model, val_ids, seq_len, device)
            row = {
                "step": step + 1,
                "train_ce": train_ce,
                "val_ce": val_ce,
                "train_ppl": float(torch.exp(torch.tensor(min(train_ce, 20.0)))),
                "parameter_count": count_parameters(model),
                "elapsed_sec": perf_counter() - train_start,
                "tokens_seen": (step + 1) * batch_size * seq_len,
            }
            row["tokens_per_sec"] = row["tokens_seen"] / max(row["elapsed_sec"], 1e-8)
            history.append(row)
            print(
                f"step={step+1} transformer_train_ce={train_ce:.4f} "
                f"transformer_val_ce={val_ce:.4f} params={row['parameter_count']}"
            )
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                torch.save(model.state_dict_with_config(), best_checkpoint)

    last_checkpoint = output_dir / "tiny_transformer_last.pt"
    torch.save(model.state_dict_with_config(), last_checkpoint)
    if not best_checkpoint.exists():
        torch.save(model.state_dict_with_config(), best_checkpoint)
    alias = output_dir / "tiny_transformer.pt"
    alias.write_bytes(best_checkpoint.read_bytes())
    tokenizer.save(output_dir / "tokenizer.json")
    save_json(
        output_dir / "metrics.json",
        {
            "history": history,
            "best_val_ce": best_val_ce,
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "parameter_count": count_parameters(model),
            "elapsed_sec": perf_counter() - train_start,
            "tokens_seen": steps * batch_size * seq_len,
            "config": config.to_dict(),
        },
    )
    return alias
