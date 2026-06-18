from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


PAD_ID = 256
EOS_ID = 257
BOS_ID = 258
WORLD_VOCAB_SIZE = 259


@dataclass
class SymbolicWorldModelConfig:
    vocab_size: int = WORLD_VOCAB_SIZE
    d_model: int = 128
    hidden_size: int = 160
    n_layers: int = 1
    dropout: float = 0.0
    max_input_len: int = 192
    max_target_len: int = 192
    seed: int = 1337

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SymbolicWorldModelConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


class SymbolicWorldModel(nn.Module):
    def __init__(self, config: SymbolicWorldModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.encoder = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.hidden_size,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.hidden_size,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.out = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None) -> dict[str, Any]:
        enc_emb = self.embedding(input_ids)
        _enc_out, hidden = self.encoder(enc_emb)
        out: dict[str, Any] = {}
        if target_ids is not None:
            bos = torch.full((target_ids.shape[0], 1), BOS_ID, dtype=torch.long, device=target_ids.device)
            decoder_input = torch.cat([bos, target_ids[:, :-1]], dim=1)
            dec_out, _ = self.decoder(self.embedding(decoder_input), hidden)
            logits = self.out(dec_out)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), ignore_index=PAD_ID)
            out.update({"logits": logits, "loss": loss, "aux_losses": {"ce": loss, "total": loss}})
        else:
            out["hidden"] = hidden
        return out

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int | None = None) -> torch.Tensor:
        self.eval()
        max_new_tokens = max_new_tokens or self.config.max_target_len
        _enc_out, hidden = self.encoder(self.embedding(input_ids))
        token = torch.full((input_ids.shape[0], 1), BOS_ID, dtype=torch.long, device=input_ids.device)
        generated = []
        for _ in range(max_new_tokens):
            dec_out, hidden = self.decoder(self.embedding(token), hidden)
            logits = self.out(dec_out[:, -1])
            token = logits.argmax(dim=-1, keepdim=True)
            generated.append(token)
        return torch.cat(generated, dim=1) if generated else input_ids.new_zeros((input_ids.shape[0], 0))

    def state_dict_with_config(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "model": self.state_dict()}


def encode_world_text(text: str, max_len: int, add_eos: bool = False) -> list[int]:
    ids = list(text.encode("utf-8", errors="replace"))
    if add_eos:
        ids.append(EOS_ID)
    ids = ids[:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


def decode_world_ids(ids: list[int]) -> str:
    clean = []
    for idx in ids:
        idx = int(idx)
        if idx == EOS_ID:
            break
        if idx in {PAD_ID, BOS_ID}:
            continue
        clean.append(idx % 256)
    return bytes(clean).decode("utf-8", errors="replace")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
