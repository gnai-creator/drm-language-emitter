from __future__ import annotations

import argparse
from pathlib import Path

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.training import train_tiny
from drm_language_emitter.utils import load_yaml_or_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--text", default="data/tiny.txt")
    parser.add_argument("--output-dir", default="runs/tiny")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    config = DRMConfig.from_dict(load_yaml_or_json(args.config))
    ckpt = train_tiny(config, args.text, args.output_dir, args.steps, args.batch_size, args.lr, args.val_fraction)
    print(f"checkpoint={Path(ckpt)}")


if __name__ == "__main__":
    main()
