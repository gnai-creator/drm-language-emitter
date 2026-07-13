from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.generation import generate
from drm_language_emitter.tokenizer import ByteTokenizer


DEFAULT_CHECKPOINTS = [
    Path("runs/wiki_en_125m_real_matched/drm_125m_real/seed_1/checkpoint_best.pt"),
    Path("runs/wiki_en_125m_real_matched/drm_125m_real/seed_1/checkpoint_last.pt"),
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


def build_prompt(history: list[tuple[str, str]], user_text: str, max_turns: int) -> str:
    turns = history[-max_turns:] if max_turns > 0 else []
    pieces: list[str] = []
    for user, assistant in turns:
        pieces.append(f"User: {user}\nDRM: {assistant}\n")
    pieces.append(f"User: {user_text}\nDRM:")
    return "\n".join(pieces)


def trim_reply(text: str) -> str:
    # This model is a plain language model, not instruction tuned. Stop at the
    # next apparent dialogue marker to keep the CLI readable.
    for marker in ("\nUser:", "\nDRM:"):
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos]
    return text.strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Minimal interactive CLI for a trained drm_125m_real checkpoint.")
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
    model = load_model(checkpoint).to(device)
    model.eval()

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
        with torch.no_grad():
            output = generate(
                model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        decoded = tokenizer.decode(output[0].detach().cpu().tolist())
        reply = trim_reply(decoded[len(prompt) :])
        print(f"drm> {reply}\n")
        history.append((user_text, reply))


if __name__ == "__main__":
    main()
