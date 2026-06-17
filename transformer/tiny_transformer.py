from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TinyTransformerConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    hidden_size: int = 128
    max_seq_len: int = 64
    dropout: float = 0.0
    generation_temperature: float = 0.9
    top_k: int = 20
    seed: int = 1337

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TinyTransformerConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


class TinyTransformerLM(nn.Module):
    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> dict[str, Any]:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.encoder(x, mask=mask)
        logits = self.lm_head(self.norm(x))
        out: dict[str, Any] = {"logits": logits}
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            out["loss"] = ce
            out["aux_losses"] = {"ce": ce, "total": ce}
        return out

    def state_dict_with_config(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "model": self.state_dict()}


@torch.no_grad()
def generate_transformer(
    model: TinyTransformerLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float | None = None,
    top_k: int | None = None,
) -> torch.Tensor:
    model.eval()
    temperature = model.config.generation_temperature if temperature is None else temperature
    top_k = model.config.top_k if top_k is None else top_k
    out = input_ids
    for _ in range(max_new_tokens):
        context = out[:, -model.config.max_seq_len :]
        logits = model(context)["logits"][:, -1] / max(temperature, 1e-6)
        if top_k and 0 < top_k < logits.shape[-1]:
            values, indices = torch.topk(logits, top_k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            logits = filtered.scatter(-1, indices, values)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_token], dim=1)
    return out


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
