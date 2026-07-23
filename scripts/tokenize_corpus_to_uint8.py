from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO


CHUNK_SIZE = 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_shard(
    output_dir: Path,
    split: str,
    shard_index: int,
) -> tuple[Path, BinaryIO, Any]:
    path = output_dir / f"{split}_{shard_index:06d}.bin"
    return path, path.open("wb"), hashlib.sha256()


def tokenize_corpus_to_uint8(
    inputs: list[str | Path],
    output_dir: str | Path,
    shard_bytes: int = 100_000_000,
    val_fraction: float = 0.001,
    val_bytes: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    if shard_bytes <= 0:
        raise ValueError("shard_bytes must be positive")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0.0, 1.0)")

    input_paths = [Path(path) for path in inputs]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing input file(s): {missing}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8")) | {"reused_existing": True}

    existing_shards = list(output_dir.glob("train_*.bin")) + list(output_dir.glob("val_*.bin"))
    if existing_shards and not overwrite:
        raise FileExistsError(f"{output_dir} already contains token shards; pass --overwrite")
    for shard in existing_shards:
        shard.unlink()

    source_sizes = [path.stat().st_size for path in input_paths]
    total_bytes = sum(source_sizes)
    target_val_bytes = val_bytes if val_bytes > 0 else int(total_bytes * val_fraction)
    if total_bytes > 0 and target_val_bytes <= 0:
        target_val_bytes = min(total_bytes, 1)
    target_val_bytes = min(target_val_bytes, total_bytes)
    train_limit = total_bytes - target_val_bytes

    shards: list[dict[str, Any]] = []
    corpus_digest = hashlib.sha256()
    split = "train"
    split_index = 0
    shard_size = 0
    shard_path, shard_handle, shard_digest = _new_shard(output_dir, split, split_index)
    absolute_offset = 0

    def close_current() -> None:
        nonlocal shard_size
        if shard_handle.closed:
            return
        shard_handle.close()
        if shard_size == 0:
            shard_path.unlink(missing_ok=True)
            return
        shards.append(
            {
                "split": split,
                "path": shard_path.name,
                "bytes": shard_size,
                "sha256": shard_digest.hexdigest(),
            }
        )

    def rotate(next_split: str | None = None) -> None:
        nonlocal split, split_index, shard_size, shard_path, shard_handle, shard_digest
        close_current()
        if next_split is not None and next_split != split:
            split = next_split
            split_index = 0
        else:
            split_index += 1
        shard_size = 0
        shard_path, shard_handle, shard_digest = _new_shard(output_dir, split, split_index)

    for input_path in input_paths:
        with input_path.open("rb") as source:
            for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                cursor = 0
                corpus_digest.update(chunk)
                while cursor < len(chunk):
                    if absolute_offset >= train_limit and split != "val":
                        rotate("val")
                    room = shard_bytes - shard_size
                    split_remaining = train_limit - absolute_offset if split == "train" else len(chunk) - cursor
                    take = min(len(chunk) - cursor, room, split_remaining if split == "train" else room)
                    if take <= 0:
                        rotate("val" if absolute_offset >= train_limit else split)
                        continue
                    piece = chunk[cursor : cursor + take]
                    shard_handle.write(piece)
                    shard_digest.update(piece)
                    shard_size += take
                    cursor += take
                    absolute_offset += take
                    if shard_size >= shard_bytes:
                        rotate()

    close_current()

    manifest = {
        "format": "drm-language-emitter-token-shards",
        "version": 1,
        "tokenizer_type": "byte",
        "dtype": "uint8",
        "total_tokens": total_bytes,
        "train_tokens": train_limit,
        "val_tokens": target_val_bytes,
        "shard_bytes": shard_bytes,
        "corpus_sha256": corpus_digest.hexdigest(),
        "sources": [
            {"path": str(path), "bytes": size, "sha256": _sha256_file(path)}
            for path, size in zip(input_paths, source_sizes)
        ],
        "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize UTF-8 corpus bytes into uint8 training shards.")
    parser.add_argument("--input", nargs="+", required=True, help="Input text files. UTF-8 bytes are byte-level tokens.")
    parser.add_argument("--output-dir", default="data/tokens")
    parser.add_argument("--shard-bytes", type=int, default=100_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.001)
    parser.add_argument("--val-bytes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = tokenize_corpus_to_uint8(
        inputs=args.input,
        output_dir=args.output_dir,
        shard_bytes=args.shard_bytes,
        val_fraction=args.val_fraction,
        val_bytes=args.val_bytes,
        overwrite=args.overwrite,
    )
    print(f"manifest={Path(args.output_dir) / 'manifest.json'}")
    print(f"total_tokens={manifest['total_tokens']}")
    print(f"train_tokens={manifest['train_tokens']}")
    print(f"val_tokens={manifest['val_tokens']}")
    print(f"shards={len(manifest['shards'])}")


if __name__ == "__main__":
    main()
