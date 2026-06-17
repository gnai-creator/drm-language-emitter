from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import ByteTokenizer, CharTokenizer, make_tokenizer


DEFAULT_TINY_TEXT = (
    "Directional relational manifolds guide language as trajectories. "
    "The emitter learns low action motion through active directions. "
    "This tiny corpus is only a smoke test for geometry and generation.\n"
)


def ensure_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_TINY_TEXT * 8, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def make_lm_batch(ids: list[int], batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(ids) < seq_len + 2:
        ids = ids * ((seq_len + 2) // max(len(ids), 1) + 1)
    starts = torch.randint(0, len(ids) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(ids[s : s + seq_len], dtype=torch.long) for s in starts]).to(device)
    y = torch.stack([torch.tensor(ids[s + 1 : s + seq_len + 1], dtype=torch.long) for s in starts]).to(device)
    return x, y


def build_tokenizer(text: str, tokenizer_type: str = "byte") -> ByteTokenizer | CharTokenizer:
    return make_tokenizer(text, tokenizer_type)
