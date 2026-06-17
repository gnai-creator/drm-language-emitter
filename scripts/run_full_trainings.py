from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.training import train_tiny
from drm_language_emitter.utils import load_yaml_or_json


DEFAULT_CONFIGS = [
    "configs/tiny.yaml",
    "configs/tiny_risk.yaml",
    "configs/fixed_dim_ablation.yaml",
]


def safe_name(config_path: str | Path) -> str:
    return Path(config_path).stem.replace("-", "_")


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full DRM Language Emitter training/eval sweeps on CPU."
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--text", default="data/tiny.txt")
    parser.add_argument("--output-root", default="runs/full")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--prompt", default="DRM ")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for config_path in args.configs:
        name = safe_name(config_path)
        output_dir = output_root / name
        print(f"\n=== training {name} ===", flush=True)
        config = DRMConfig.from_dict(load_yaml_or_json(config_path))
        checkpoint = train_tiny(
            config=config,
            text_path=args.text,
            output_dir=output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            val_fraction=args.val_fraction,
        )

        if args.skip_eval:
            continue

        tokenizer = output_dir / "tokenizer.json"
        run_command(
            [
                sys.executable,
                "scripts/generate.py",
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                str(tokenizer),
                "--prompt",
                args.prompt,
                "--max-new-tokens",
                str(args.max_new_tokens),
            ]
        )
        run_command(
            [
                sys.executable,
                "scripts/eval_geometry.py",
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                str(tokenizer),
                "--text",
                args.text,
                "--output",
                str(output_dir / "geometry.json"),
            ]
        )
        run_command(
            [
                sys.executable,
                "scripts/eval_geodesic_paths.py",
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                str(tokenizer),
                "--output",
                str(output_dir / "geodesic_paths.json"),
            ]
        )

    if not args.skip_eval:
        run_command(
            [
                sys.executable,
                "scripts/eval_ablations.py",
                "--config",
                args.configs[0],
                "--text",
                args.text,
                "--output",
                str(output_root / "runtime_ablations.json"),
            ]
        )

    print(f"\ncompleted output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
