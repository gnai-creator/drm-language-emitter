from __future__ import annotations

import argparse
from pathlib import Path

import torch

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.data import ensure_text
from drm_language_emitter.diagnostics import geometry_report
from drm_language_emitter.tokenizer import load_tokenizer
from drm_language_emitter.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/tiny/drm_tiny.pt")
    parser.add_argument("--tokenizer", default="runs/tiny/tokenizer.json")
    parser.add_argument("--text", default="data/tiny.txt")
    parser.add_argument("--output", default="runs/tiny/geometry.json")
    args = parser.parse_args()
    model = load_model(args.checkpoint)
    tokenizer = load_tokenizer(args.tokenizer)
    ids_list = tokenizer.encode(ensure_text(args.text))
    seq_len = min(model.config.max_seq_len, max(len(ids_list) - 1, 1))
    if len(ids_list) < seq_len + 1:
        ids_list = ids_list * ((seq_len + 1) // max(len(ids_list), 1) + 1)
    input_ids = torch.tensor([ids_list[:seq_len]], dtype=torch.long)
    targets = torch.tensor([ids_list[1 : seq_len + 1]], dtype=torch.long)
    save_json(args.output, geometry_report(model, input_ids, targets))
    print(f"saved={Path(args.output)}")


if __name__ == "__main__":
    main()
