from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from drm_language_emitter.utils import save_json


DEFAULT_DATASET_NAME = "wikimedia/wikipedia"
DEFAULT_DATASET_CONFIG = "20231101.en"
DEFAULT_SPLIT = "train"


def iter_wikipedia_texts(
    dataset_name: str,
    dataset_config: str,
    split: str,
    streaming: bool,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing optional dependency 'datasets'. Install with: pip install -e \".[hf]\"") from exc

    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    for row in dataset:
        text = row.get("text") if isinstance(row, dict) else None
        if isinstance(text, str) and text.strip():
            yield text


def prepare_wikipedia_en(
    output: str | Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    split: str = DEFAULT_SPLIT,
    max_chars: int = 50_000_000,
    max_docs: int = 0,
    min_doc_chars: int = 200,
    streaming: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output)
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if output.exists() and not overwrite:
        return {
            "output": str(output),
            "metadata": str(metadata_path),
            "reused_existing": True,
            **({"existing_bytes": output.stat().st_size} if output.exists() else {}),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    chars_written = 0
    docs_written = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for text in iter_wikipedia_texts(dataset_name, dataset_config, split, streaming):
            text = " ".join(text.split())
            if len(text) < min_doc_chars:
                continue
            remaining = max_chars - chars_written if max_chars > 0 else None
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and len(text) > remaining:
                text = text[:remaining]
            handle.write(text)
            handle.write("\n\n")
            chars_written += len(text) + 2
            docs_written += 1
            if max_docs > 0 and docs_written >= max_docs:
                break
            if max_chars > 0 and chars_written >= max_chars:
                break

    metadata = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "streaming": streaming,
        "max_chars": max_chars,
        "max_docs": max_docs,
        "min_doc_chars": min_doc_chars,
        "chars_written": chars_written,
        "docs_written": docs_written,
        "output": str(output),
    }
    save_json(metadata_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/wikipedia_en_20231101_sample.txt")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--max-chars", type=int, default=50_000_000)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = prepare_wikipedia_en(
        output=args.output,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        max_chars=args.max_chars,
        max_docs=args.max_docs,
        min_doc_chars=args.min_doc_chars,
        streaming=args.streaming,
        overwrite=args.overwrite,
    )
    print(f"output={metadata['output']}")
    if "chars_written" in metadata:
        print(f"chars_written={metadata['chars_written']}")
        print(f"docs_written={metadata['docs_written']}")


if __name__ == "__main__":
    main()
