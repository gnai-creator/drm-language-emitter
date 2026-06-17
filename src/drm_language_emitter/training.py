from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from .config import DRMConfig
from .data import build_tokenizer, ensure_text, make_lm_batch
from .model import DRMEmitterModel
from .utils import save_json


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_tiny(
    config: DRMConfig,
    text_path: str | Path,
    output_dir: str | Path,
    steps: int = 40,
    batch_size: int = 8,
    lr: float = 3e-4,
    val_fraction: float = 0.1,
) -> Path:
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    text = ensure_text(text_path)
    tokenizer = build_tokenizer(text, config.tokenizer_type)
    config.vocab_size = tokenizer.vocab_size
    model = DRMEmitterModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ids = tokenizer.encode(text)
    split = max(int(len(ids) * (1.0 - val_fraction)), 2)
    train_ids = ids[:split]
    val_ids = ids[max(0, split - config.max_seq_len - 1) :]
    seq_len = min(config.max_seq_len, 64)
    history = []
    train_start = perf_counter()
    best_val_ce = float("inf")
    best_checkpoint = output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "drm_tiny_best.pt"
    for step in range(steps):
        x, y = make_lm_batch(train_ids, batch_size, seq_len, device)
        out = model(x, y, global_step=step + 1)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            ce = out["aux_losses"].get("ce", out["loss"]).detach().item()
            val_ce, val_diag = evaluate_ce(model, val_ids, seq_len, device, global_step=step + 1)
            ppl = float(torch.exp(torch.tensor(min(ce, 20.0))))
            diag = out["diagnostics"]
            row = {
                "step": step + 1,
                "train_ce": ce,
                "val_ce": val_ce,
                "train_ppl": ppl,
                "dimD_mean": float(diag["dimD_mean"].detach()),
                "dimD_std": float(diag["dimD_std"].detach()),
                "active_fraction": float(diag["active_fraction"].detach()),
                "soft_active_fraction": float(diag["soft_active_fraction"].detach()),
                "hard_active_fraction_075": float(diag["hard_active_fraction_075"].detach()),
                "hard_active_fraction_090": float(diag["hard_active_fraction_090"].detach()),
                "gate_q10": float(diag["gate_q10"].detach()),
                "gate_q50": float(diag["gate_q50"].detach()),
                "gate_q90": float(diag["gate_q90"].detach()),
                "action_mean": float(diag["action_mean"].detach()),
                "metric_U_norm_mean": float(diag["metric_U_norm_mean"].detach()),
                "condition_proxy": float(diag["condition_proxy"].detach()),
                "val_action_mean": float(val_diag["action_mean"]),
                "metric_naturalization_strength": float(diag["metric_naturalization_strength"].detach()),
                "parameter_count": count_parameters(model),
                "elapsed_sec": perf_counter() - train_start,
                "tokens_seen": (step + 1) * batch_size * seq_len,
            }
            row["tokens_per_sec"] = row["tokens_seen"] / max(row["elapsed_sec"], 1e-8)
            history.append(row)
            print(
                f"step={step+1} train_ce={ce:.4f} val_ce={val_ce:.4f} ppl={ppl:.2f} "
                f"action={float(diag['action_mean'].detach()):.4f} dimD={float(diag['dimD_mean'].detach()):.2f} "
                f"dimD_std={float(diag['dimD_std'].detach()):.3f} "
                f"active={float(diag['active_fraction'].detach()):.2f} soft_active={float(diag['soft_active_fraction'].detach()):.2f} "
                f"h075={float(diag['hard_active_fraction_075'].detach()):.2f} "
                f"metricU={float(diag['metric_U_norm_mean'].detach()):.4f} cond={float(diag['condition_proxy'].detach()):.2f}"
            )
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                torch.save(model.state_dict_with_config(), best_checkpoint)
    last_checkpoint = output_dir / "drm_tiny_last.pt"
    torch.save(model.state_dict_with_config(), last_checkpoint)
    if not best_checkpoint.exists():
        torch.save(model.state_dict_with_config(), best_checkpoint)
    checkpoint = output_dir / "drm_tiny.pt"
    checkpoint.write_bytes(best_checkpoint.read_bytes())
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
    return checkpoint


@torch.no_grad()
def evaluate_ce(
    model: DRMEmitterModel,
    ids: list[int],
    seq_len: int,
    device: torch.device,
    global_step: int | None = None,
) -> tuple[float, dict[str, float]]:
    model.eval()
    if len(ids) < seq_len + 1:
        ids = ids * ((seq_len + 1) // max(len(ids), 1) + 1)
    x = torch.tensor(ids[:seq_len], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1 : seq_len + 1], dtype=torch.long, device=device).unsqueeze(0)
    out = model(x, y, global_step=global_step)
    ce = float(out["aux_losses"].get("ce", out["loss"]).detach())
    diag = {k: float(v.detach()) for k, v in out["diagnostics"].items()}
    model.train()
    return ce, diag
