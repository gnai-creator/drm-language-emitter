from __future__ import annotations

import torch
from torch import nn

from .config import DRMConfig
from .utils import rms_norm


class TokenEmbedding(nn.Module):
    def __init__(self, config: DRMConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_token)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * rms_norm(x, self.eps)


class LanguageEmitter(nn.Module):
    def __init__(self, config: DRMConfig):
        super().__init__()
        self.legacy = config.emitter_layers == 1 and not config.emitter_swiglu and not config.emitter_residual
        if self.legacy:
            self.net = nn.Sequential(
                RMSNorm(config.d_state),
                nn.Linear(config.d_state, config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.vocab_size),
            )
            return
        self.norm = RMSNorm(config.d_state)
        self.blocks = nn.ModuleList(
            EmitterBlock(config.d_state, config.hidden_size, config.dropout, config.emitter_swiglu, config.emitter_residual)
            for _ in range(max(config.emitter_layers, 1))
        )
        self.lm_head = nn.Linear(config.d_state, config.vocab_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.legacy:
            return self.net(z)
        x = self.norm(z)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


class EmitterBlock(nn.Module):
    def __init__(self, d_state: int, hidden_size: int, dropout: float, swiglu: bool, residual: bool):
        super().__init__()
        self.swiglu = swiglu
        self.residual = residual
        if swiglu:
            self.up = nn.Linear(d_state, hidden_size * 2)
            self.down = nn.Linear(hidden_size, d_state)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_state, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, d_state),
            )
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        if self.swiglu:
            value, gate = self.up(y).chunk(2, dim=-1)
            y = self.down(value * torch.nn.functional.silu(gate))
            y = self.dropout(y)
        else:
            y = self.net(y)
        return x + y if self.residual else y
