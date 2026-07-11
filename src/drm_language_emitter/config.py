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
    risk_mass_max: float = 10.0
    risk_exponent_min: float = 0.25
    risk_exponent_max: float = 4.0
    risk_alpha_max: float = 10.0
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
    use_torch_compile: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        """Validate configuration values and raise ValueError if any check fails."""

        # Integer / positive checks
        int_fields = [
            ("vocab_size", 1, None),
            ("d_token", 1, None),
            ("d_state", 1, None),
            ("n_directions", 1, None),
            ("metric_rank", 0, None),  # rank can be zero
            ("hidden_size", 1, None),
            ("n_flow_steps", 1, None),
            ("max_seq_len", 1, None),
            ("gate_top_k", 0, None),
        ]
        for name, min_val, max_val in int_fields:
            val = getattr(self, name)
            if not isinstance(val, int):
                raise ValueError(f"'{name}' must be an integer, got {type(val).__name__}")
            if val < min_val or (max_val is not None and val > max_val):
                raise ValueError(
                    f"'{name}' must be between {min_val}"
                    + (f" and {max_val}" if max_val is not None else "")
                    + f", got {val}"
                )

        # Float / non-negative checks
        float_fields = [
            ("dt", 0.0, None),
            ("dropout", 0.0, 1.0),
            ("lambda_action", 0.0, None),
            ("lambda_dim_sparsity", 0.0, None),
            ("lambda_dim_entropy", 0.0, None),
            ("lambda_dim_variance", 0.0, None),
            ("target_dim_std", 0.0, None),
            ("lambda_metric_reg", 0.0, None),
            ("lambda_metric_diversity", 0.0, None),
            ("lambda_recurrence", 0.0, None),
            ("lambda_stability", 0.0, None),
            ("lambda_blindspot", 0.0, None),
            ("risk_mass_max", 0.0, None),
            ("risk_exponent_min", 0.0, None),
            ("risk_exponent_max", 0.0, None),
            ("risk_alpha_max", 0.0, None),
            ("generation_temperature", 0.1, None),   # temperature > 0
            ("metric_eps", 0.0, None),
            ("state_clip_norm", 0.0, None),
            ("gate_temperature", 0.01, None),       # temperature > 0
            ("gate_logit_bias", -10.0, 10.0),
            ("lambda_active_fraction", 0.0, 1.0),
            ("target_active_fraction", 0.0, 1.0),
            ("metric_naturalization_strength", 0.0, None),
            ("metric_naturalization_warmup_steps", 0, None),
            ("metric_damping", 0.0, None),
            ("metric_u_min_norm", 0.0, None),
            ("lambda_metric_u_floor", 0.0, None),
            ("metric_u_target_norm", 0.0, None),
            ("lambda_metric_u_target", 0.0, None),
            ("target_condition", 1.0, None),
            ("lambda_condition", 0.0, None),
        ]
        for name, min_val, max_val in float_fields:
            val = getattr(self, name)
            if not isinstance(val, (float, int)):
                raise ValueError(f"'{name}' must be a number, got {type(val).__name__}")
            if val < min_val or (max_val is not None and val > max_val):
                raise ValueError(
                    f"'{name}' must be between {min_val}"
                    + (f" and {max_val}" if max_val is not None else "")
                    + f", got {val}"
                )

        # Boolean checks
        bool_fields = [
            "bounded_state",
            "use_toroidal_state",
            "use_powerlaw_risk",
            "direction_norm",
            "tie_embeddings",
            "use_metric_naturalization",
            "gate_top_k_renorm",
            "emitter_swiglu",
            "emitter_residual",
            "use_torch_compile",
        ]
        for name in bool_fields:
            val = getattr(self, name)
            if not isinstance(val, bool):
                raise ValueError(f"'{name}' must be a boolean, got {type(val).__name__}")
        if self.risk_exponent_min > self.risk_exponent_max:
            raise ValueError("'risk_exponent_min' must be <= 'risk_exponent_max'")

    def __post_init__(self) -> None:  # pragma: no cover
        """Automatically validate configuration on instantiation."""
        self._validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DRMConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown DRMConfig field(s): {', '.join(unknown)}")
        return cls(**data)
