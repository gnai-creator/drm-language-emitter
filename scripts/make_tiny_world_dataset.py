from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from drm_language_emitter.utils import save_json
from world_model.tiny_world import make_records, write_jsonl, write_text_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="data/tiny_world")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--num-train", type=int, default=20_000)
    parser.add_argument("--num-val", type=int, default=2_000)
    parser.add_argument("--max-rollout-len", type=int, default=8)
    parser.add_argument("--no-walls", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    train = make_records(args.num_train, args.seed, args.grid_size, args.max_rollout_len, walls=not args.no_walls)
    val = make_records(args.num_val, args.seed + 10_000, args.grid_size, args.max_rollout_len, walls=not args.no_walls)
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "val.jsonl", val)
    write_text_corpus(root / "train.txt", train)
    write_text_corpus(root / "val.txt", val)
    save_json(
        root / "dataset_config.json",
        {
            "seed": args.seed,
            "grid_size": args.grid_size,
            "num_train": args.num_train,
            "num_val": args.num_val,
            "max_rollout_len": args.max_rollout_len,
            "walls": not args.no_walls,
            "format": "jsonl + autoregressive text corpus",
        },
    )
    print(f"saved={root}")


if __name__ == "__main__":
    main()
