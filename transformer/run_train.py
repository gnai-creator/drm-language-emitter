from __future__ import annotations

import argparse

from drm_language_emitter.utils import load_yaml_or_json
from transformer.tiny_transformer import TinyTransformerConfig
from transformer.train_tiny_transformer import train_tiny_transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--text", default="data/tiny.txt")
    parser.add_argument("--output-dir", default="runs/tiny_transformer")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    config = TinyTransformerConfig.from_dict(load_yaml_or_json(args.config)) if args.config else TinyTransformerConfig()
    checkpoint = train_tiny_transformer(
        config=config,
        text_path=args.text,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
    )
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
