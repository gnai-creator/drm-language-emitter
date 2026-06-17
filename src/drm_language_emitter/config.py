from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DRMConfig:
    vocab_size: int = 256
    d_token: int = 64
    d_state: int = 96
    n_directions: int = 12
    metric_rank: int = 8
    hidden_size: int = 128
    n_flow_steps: int = 1
    dt: float = 0.1
    max_seq_len: int = 256
    dropout: float = 0.0
    bounded_state: bool = True
    use_toroidal_state: bool = False
    use_powerlaw_risk: bool = False
    lambda_action: float = 0.01
    lambda_dim_sparsity: float = 0.001
    lambda_dim_entropy: float = 0.001
    lambda_dim_variance: float = 0.01
    target_dim_std: float = 0.15
    lambda_metric_reg: float = 0.001
    lambda_metric_diversity: float = 0.001
    lambda_recurrence: float = 0.0
    lambda_stability: float = 0.0
    lambda_blindspot: float = 0.0
    generation_temperature: float = 1.0
    top_k: int = 40
    metric_eps: float = 1e-4
    state_clip_norm: float = 8.0
    direction_norm: bool = True
    tie_embeddings: bool = False
    tokenizer_type: str = "byte"
    seed: int = 1337
    gate_temperature: float = 1.5
    gate_logit_bias: float = -1.0
    gate_top_k: int = 0
    gate_top_k_renorm: bool = False
    lambda_active_fraction: float = 0.01
    target_active_fraction: float = 0.65
    use_metric_naturalization: bool = True
    metric_naturalization_strength: float = 0.5
    metric_naturalization_warmup_steps: int = 500
    metric_damping: float = 0.3
    metric_u_min_norm: float = 0.05
    lambda_metric_u_floor: float = 0.001
    metric_u_target_norm: float = 1.0
    lambda_metric_u_target: float = 0.001
    target_condition: float = 100.0
    lambda_condition: float = 0.001
    emitter_layers: int = 1
    emitter_swiglu: bool = False
    emitter_residual: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DRMConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})
