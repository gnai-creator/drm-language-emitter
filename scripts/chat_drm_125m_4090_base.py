from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.generation import generate
from drm_language_emitter.tokenizer import ByteTokenizer


DEFAULT_RUN_DIR = Path("runs/drm_125m_4090_base")
CHECKPOINT_PREFERENCE = (
    "checkpoint_last.pt",
    "checkpoint_latest.pt",
    "checkpoint_best.pt",
)


def resolve_checkpoint(run_dir: str, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if path.exists():
            return path
        raise SystemExit(f"checkpoint not found: {path}")

    root = Path(run_dir)
    for name in CHECKPOINT_PREFERENCE:
        path = root / name
        if path.exists():
            return path

    token_checkpoints = sorted(
        root.glob("checkpoint_tokens_*.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if token_checkpoints:
        return token_checkpoints[0]

    expected = "\n".join(f"  - {root / name}" for name in CHECKPOINT_PREFERENCE)
    raise SystemExit(f"no checkpoint found. Expected one of:\n{expected}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def apply_dtype(model: torch.nn.Module, device: torch.device, dtype_name: str) -> str:
    if dtype_name == "auto":
        dtype_name = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if dtype_name == "bf16":
        if device.type != "cuda":
            raise SystemExit("bf16 inference is only supported by this script on CUDA")
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("bf16 requested, but this CUDA device does not report bf16 support")
        model.to(dtype=torch.bfloat16)
    elif dtype_name != "fp32":
        raise SystemExit(f"unknown dtype: {dtype_name}")
    return dtype_name


def checkpoint_summary(checkpoint: Path) -> dict[str, object]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        return {}
    return {
        "step": payload.get("step"),
        "tokens_seen": payload.get("tokens_seen"),
        "best_val_ce": payload.get("best_val_ce"),
        "parameter_count": payload.get("parameter_count"),
        "precision": payload.get("precision"),
    }


def run_summary(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_prompt(history: list[tuple[str, str]], user_text: str, max_turns: int) -> str:
    turns = history[-max_turns:] if max_turns > 0 else []
    pieces: list[str] = []
    for user, assistant in turns:
        pieces.append(f"User: {user}\nDRM: {assistant}\n")
    pieces.append(f"User: {user_text}\nDRM:")
    return "\n".join(pieces)


def trim_prompt_tokens(token_ids: list[int], max_prompt_tokens: int) -> list[int]:
    if max_prompt_tokens <= 0 or len(token_ids) <= max_prompt_tokens:
        return token_ids
    return token_ids[-max_prompt_tokens:]


def trim_reply(text: str) -> str:
    for marker in ("\nUser:", "\nDRM:", "\nGPT2:", "\r\nUser:", "\r\nDRM:", "\r\nGPT2:"):
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos]
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CLI for runs/drm_125m_4090_base.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Directory containing DRM checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path. Overrides --run-dir.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-turns", type=int, default=3, help="Previous turns to include in the prompt.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    checkpoint = resolve_checkpoint(args.run_dir, args.checkpoint)
    device = resolve_device(args.device)
    tokenizer = ByteTokenizer()

    metadata = checkpoint_summary(checkpoint)
    model = load_model(checkpoint).to(device)
    dtype = apply_dtype(model, device, args.dtype)
    model.eval()

    print(f"checkpoint: {checkpoint}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    for key, value in metadata.items():
        if value is not None:
            print(f"{key}: {value}")
    final_summary = run_summary(Path(args.run_dir))
    if final_summary:
        best_val = final_summary.get("best_val_ce")
        val_ce = final_summary.get("val_ce")
        if best_val is not None or val_ce is not None:
            print(f"summary_val_ce: {val_ce} best_val_ce: {best_val}")
    print("Commands: /exit, /quit, /clear, /settings\n")

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
        if user_text == "/settings":
            print(
                "max_new_tokens="
                f"{args.max_new_tokens} temperature={args.temperature} top_k={args.top_k} "
                f"max_turns={args.max_turns} max_prompt_tokens={args.max_prompt_tokens}"
            )
            continue

        prompt = build_prompt(history, user_text, args.max_turns)
        prompt_ids = trim_prompt_tokens(tokenizer.encode(prompt), args.max_prompt_tokens)
        if args.show_prompt:
            print(f"prompt_tokens: {len(prompt_ids)}")
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = generate(
                model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        reply_ids = output[0, input_ids.shape[1] :].detach().cpu().tolist()
        reply = trim_reply(tokenizer.decode(reply_ids))
        print(f"drm> {reply}\n")
        history.append((user_text, reply))


if __name__ == "__main__":
    main()
