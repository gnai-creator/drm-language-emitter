from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.generation import generate
from drm_language_emitter.tokenizer import load_tokenizer


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/tiny/drm_tiny.pt")
    parser.add_argument("--tokenizer", default="runs/tiny/tokenizer.json")
    parser.add_argument("--prompt", default="DRM ")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.checkpoint)
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    out = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
