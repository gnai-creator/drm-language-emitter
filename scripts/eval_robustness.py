from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.data import ensure_text
from drm_language_emitter.tokenizer import load_tokenizer
from drm_language_emitter.utils import save_json
from transformer.checkpoint import load_transformer


def corrupt(input_ids: torch.Tensor, vocab_size: int, probability: float, seed: int, corruption_type: str) -> torch.Tensor:
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    mask = torch.rand(input_ids.shape, generator=generator, device=input_ids.device) < probability
    if corruption_type == "random_byte":
        noise = torch.randint(0, vocab_size, input_ids.shape, generator=generator, device=input_ids.device)
        return torch.where(mask, noise, input_ids)
    if corruption_type == "zero_byte":
        return torch.where(mask, torch.zeros_like(input_ids), input_ids)
    if corruption_type == "swap_adjacent":
        out = input_ids.clone()
        swap_mask = mask[:, :-1]
        left = out[:, :-1].clone()
        right = out[:, 1:].clone()
        out[:, :-1] = torch.where(swap_mask, right, out[:, :-1])
        out[:, 1:] = torch.where(swap_mask, left, out[:, 1:])
        return out
    if corruption_type == "delete_context_byte":
        out = input_ids.clone()
        for b in range(out.shape[0]):
            keep = [int(tok) for tok, use_delete in zip(out[b].tolist(), mask[b].tolist()) if not use_delete]
            if not keep:
                keep = [0]
            keep = keep + [0] * (out.shape[1] - len(keep))
            out[b] = torch.tensor(keep[: out.shape[1]], dtype=out.dtype, device=out.device)
        return out
    raise ValueError(f"unknown corruption_type={corruption_type}")


@torch.no_grad()
def ce(model, x: torch.Tensor, y: torch.Tensor) -> float:
    out = model(x, y)
    return float(out["aux_losses"]["ce"].detach())


def save_robustness_svg(path: str | Path, payload: dict) -> None:
    path = Path(path)
    width, height = 820, 460
    margin = 58
    points = []
    for key, row in payload["noise"].items():
        noise = float(key)
        points.append(("DRM", noise, float(row["drm_delta_ce"])))
        if "transformer_delta_ce" in row:
            points.append(("Transformer", noise, float(row["transformer_delta_ce"])))
    if not points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    max_noise = max(p[1] for p in points)
    max_delta = max(p[2] for p in points) * 1.15 + 1e-8
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    def xy(noise: float, value: float) -> tuple[float, float]:
        x = margin + noise / max(max_noise, 1e-8) * plot_w
        y = height - margin - value / max_delta * plot_h
        return x, y

    def poly(model: str, color: str) -> str:
        model_points = [(noise, value) for name, noise, value in points if name == model]
        model_points.sort()
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in [xy(n, v) for n, v in model_points])
        circles = "".join(
            f"<circle cx='{xy(n, v)[0]:.1f}' cy='{xy(n, v)[1]:.1f}' r='4' fill='{color}'/>"
            for n, v in model_points
        )
        return f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='3'/>{circles}"

    grid = []
    for i in range(6):
        y = margin + i * plot_h / 5
        value = max_delta - i * max_delta / 5
        grid.append(f"<line x1='{margin}' y1='{y:.1f}' x2='{width-margin}' y2='{y:.1f}' stroke='#e5e7eb'/>")
        grid.append(f"<text x='12' y='{y+4:.1f}' font-size='12' fill='#374151'>{value:.2f}</text>")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{margin}" y="30" font-size="22" font-family="Arial" font-weight="700" fill="#111827">Robustness Under Input Corruption</text>
  <text x="{margin}" y="52" font-size="13" font-family="Arial" fill="#4b5563">Y-axis: CE increase over clean input. Lower is better.</text>
  {''.join(grid)}
  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>
  {poly("DRM", "#0f766e")}
  {poly("Transformer", "#b91c1c") if any(p[0] == "Transformer" for p in points) else ""}
  <text x="{width-210}" y="86" font-size="13" fill="#0f766e">DRM delta CE</text>
  <text x="{width-210}" y="110" font-size="13" fill="#b91c1c">Transformer delta CE</text>
  <text x="{width//2-35}" y="{height-18}" font-size="13" fill="#374151">noise probability</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drm-checkpoint", required=True)
    parser.add_argument("--drm-tokenizer", required=True)
    parser.add_argument("--transformer-checkpoint", default=None)
    parser.add_argument("--text", default="data/tiny.txt")
    parser.add_argument("--output", default="runs/robustness.json")
    parser.add_argument("--noise", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20])
    parser.add_argument("--corruption-types", nargs="+", default=["random_byte", "zero_byte", "swap_adjacent", "delete_context_byte"])
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    drm = load_model(args.drm_checkpoint)
    tokenizer = load_tokenizer(args.drm_tokenizer)
    ids = tokenizer.encode(ensure_text(args.text))
    seq_len = min(drm.config.max_seq_len, max(len(ids) - 1, 1))
    if len(ids) < seq_len + 1:
        ids = ids * ((seq_len + 1) // max(len(ids), 1) + 1)
    x = torch.tensor([ids[:seq_len]], dtype=torch.long)
    y = torch.tensor([ids[1 : seq_len + 1]], dtype=torch.long)

    payload = {
        "note": "Input context is corrupted while next-token targets remain clean.",
        "corruptions": {},
        "clean": {"drm_ce": ce(drm, x, y)},
    }
    transformer = load_transformer(args.transformer_checkpoint) if args.transformer_checkpoint else None
    if transformer is not None:
        payload["clean"]["transformer_ce"] = ce(transformer, x[:, -transformer.config.max_seq_len :], y[:, -transformer.config.max_seq_len :])

    for corruption_type in args.corruption_types:
        payload["corruptions"][corruption_type] = {}
        for p in args.noise:
            noisy = corrupt(x, drm.config.vocab_size, p, args.seed + int(p * 1000), corruption_type)
            row = {"drm_clean_ce": payload["clean"]["drm_ce"], "drm_corrupted_ce": ce(drm, noisy, y)}
            row["drm_ce_delta"] = row["drm_corrupted_ce"] - row["drm_clean_ce"]
            row["drm_relative_degradation"] = row["drm_ce_delta"] / max(row["drm_clean_ce"], 1e-8)
            row["drm_recovery_score"] = 1.0 / (1.0 + max(row["drm_relative_degradation"], 0.0))
            if transformer is not None:
                tx = noisy[:, -transformer.config.max_seq_len :]
                ty = y[:, -transformer.config.max_seq_len :]
                row["transformer_clean_ce"] = payload["clean"]["transformer_ce"]
                row["transformer_corrupted_ce"] = ce(transformer, tx, ty)
                row["transformer_ce_delta"] = row["transformer_corrupted_ce"] - row["transformer_clean_ce"]
                row["transformer_relative_degradation"] = row["transformer_ce_delta"] / max(row["transformer_clean_ce"], 1e-8)
                row["transformer_recovery_score"] = 1.0 / (1.0 + max(row["transformer_relative_degradation"], 0.0))
            payload["corruptions"][corruption_type][str(p)] = row
    # Backward-compatible flat view for the SVG: random_byte only.
    payload["noise"] = {
        p: {
            "drm_delta_ce": row["drm_ce_delta"],
            **({"transformer_delta_ce": row["transformer_ce_delta"]} if "transformer_ce_delta" in row else {}),
        }
        for p, row in payload["corruptions"].get("random_byte", {}).items()
    }

    save_json(args.output, payload)
    save_robustness_svg(Path(args.output).with_suffix(".svg"), payload)
    print(f"saved={Path(args.output)}")
    print(f"saved={Path(args.output).with_suffix('.svg')}")


if __name__ == "__main__":
    main()
