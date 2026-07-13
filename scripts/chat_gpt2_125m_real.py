from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from drm_language_emitter.tokenizer import ByteTokenizer


DEFAULT_CHECKPOINTS = [
    Path("runs/wiki_en_125m_real_matched/gpt2_125m_real/seed_1/checkpoint_best.pt"),
    Path("runs/wiki_en_125m_real_matched/gpt2_125m_real/seed_1/checkpoint_last.pt"),
]


def resolve_checkpoint(path: str | None) -> Path:
    if path:
        checkpoint = Path(path)
        if checkpoint.exists():
            return checkpoint
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    for checkpoint in DEFAULT_CHECKPOINTS:
        if checkpoint.exists():
            return checkpoint
    choices = "\n".join(f"  - {checkpoint}" for checkpoint in DEFAULT_CHECKPOINTS)
    raise SystemExit(f"no default checkpoint found. Expected one of:\n{choices}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def load_gpt2_checkpoint(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("checkpoint must contain dictionary keys: config, model")

    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as exc:
        raise SystemExit('Missing dependency "transformers". Install with: pip install -e ".[hf]"') from exc

    config_data: dict[str, Any] = payload["config"]
    config = GPT2Config(
        vocab_size=int(config_data.get("vocab_size", 256)),
        n_positions=int(config_data.get("max_seq_len", 512)),
        n_ctx=int(config_data.get("max_seq_len", 512)),
        n_embd=int(config_data["n_embd"]),
        n_layer=int(config_data["n_layer"]),
        n_head=int(config_data["n_head"]),
        resid_pdrop=float(config_data.get("dropout", 0.0)),
        embd_pdrop=float(config_data.get("dropout", 0.0)),
        attn_pdrop=float(config_data.get("dropout", 0.0)),
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


def build_prompt(history: list[tuple[str, str]], user_text: str, max_turns: int) -> str:
    turns = history[-max_turns:] if max_turns > 0 else []
    pieces: list[str] = []
    for user, assistant in turns:
        pieces.append(f"User: {user}\nGPT2: {assistant}\n")
    pieces.append(f"User: {user_text}\nGPT2:")
    return "\n".join(pieces)


def trim_reply(text: str) -> str:
    for marker in ("\nUser:", "\nGPT2:", "\nDRM:"):
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos]
    return text.strip()


@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    if top_k > 0 and top_k < logits.shape[-1]:
        values, indices = torch.topk(logits, top_k, dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        logits = filtered.scatter(-1, indices, values)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    generated = input_ids
    for _ in range(max_new_tokens):
        max_ctx = getattr(model.config, "n_positions", generated.shape[1])
        window = generated[:, -max_ctx:]
        logits = model(input_ids=window).logits[:, -1, :]
        next_token = sample_next_token(logits, temperature, top_k)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Minimal interactive CLI for a trained gpt2_125m_real checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint_best.pt or checkpoint_last.pt.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-turns", type=int, default=3, help="How many previous turns to include in the prompt.")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    tokenizer = ByteTokenizer()
    model = load_gpt2_checkpoint(checkpoint, device)

    print(f"checkpoint: {checkpoint}")
    print(f"device: {device}")
    print("Type /exit to quit, /clear to reset history.\n")

    history: list[tuple[str, str]] = []
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/clear":
            history.clear()
            print("history cleared")
            continue

        prompt = build_prompt(history, user_text, args.max_turns)
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        output = generate(model, input_ids, args.max_new_tokens, args.temperature, args.top_k)
        decoded = tokenizer.decode(output[0].detach().cpu().tolist())
        reply = trim_reply(decoded[len(prompt) :])
        print(f"gpt2> {reply}\n")
        history.append((user_text, reply))


if __name__ == "__main__":
    main()
