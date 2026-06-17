from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.data import build_tokenizer, ensure_text, make_lm_batch
from drm_language_emitter.model import DRMEmitterModel
from drm_language_emitter.utils import load_yaml_or_json, save_json


def run_variant(base: DRMConfig, name: str, ids: list[int], vocab_size: int) -> dict[str, float]:
    config = deepcopy(base)
    torch.manual_seed(config.seed)
    config.vocab_size = vocab_size
    if name == "no_metric_U":
        config.metric_rank = 0
        config.use_metric_naturalization = False
    elif name == "no_action_loss":
        config.lambda_action = 0.0
    elif name == "no_risk_field":
        config.use_powerlaw_risk = False
    elif name == "fixed_dimension":
        config.lambda_dim_sparsity = 0.0
        config.lambda_dim_entropy = 0.0
        config.lambda_active_fraction = 0.0
        config.gate_logit_bias = 4.0
    elif name == "no_metric_reg":
        config.lambda_metric_reg = 0.0
        config.lambda_metric_u_floor = 0.0
    elif name == "strong_gate_sparsity":
        config.lambda_dim_sparsity = max(config.lambda_dim_sparsity, 0.02)
        config.lambda_active_fraction = max(config.lambda_active_fraction, 0.05)
        config.target_active_fraction = 0.45
        config.gate_logit_bias = -1.8
    model = DRMEmitterModel(config)
    if name == "no_direction_gates":
        model.direction_field.gate_head.weight.data.zero_()
        model.direction_field.gate_head.bias.data.fill_(8.0)
    split = max(int(len(ids) * 0.9), 2)
    train_ids = ids[:split]
    val_ids = ids[max(0, split - config.max_seq_len - 1) :]
    seq_len = min(32, config.max_seq_len)
    x, y = make_lm_batch(train_ids, 2, seq_len, torch.device("cpu"))
    xv, yv = make_lm_batch(val_ids, 2, seq_len, torch.device("cpu"))
    with torch.no_grad():
        out = model(x, y)
        val_out = model(xv, yv)
    result = {k: float(v) for k, v in out["diagnostics"].items()}
    result["train_ce"] = float(out["aux_losses"]["ce"])
    result["val_ce"] = float(val_out["aux_losses"]["ce"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--text", default="data/tiny.txt")
    parser.add_argument("--output", default="runs/tiny/ablations.json")
    args = parser.parse_args()
    config = DRMConfig.from_dict(load_yaml_or_json(args.config))
    text = ensure_text(args.text)
    tokenizer = build_tokenizer(text, config.tokenizer_type)
    ids = tokenizer.encode(text)
    variants = [
        "full",
        "no_metric_U",
        "no_direction_gates",
        "no_action_loss",
        "no_risk_field",
        "fixed_dimension",
        "no_metric_reg",
        "strong_gate_sparsity",
    ]
    results = {name: run_variant(config, name, ids, tokenizer.vocab_size) for name in variants}
    save_json(args.output, results)
    print(f"saved={Path(args.output)}")


if __name__ == "__main__":
    main()
