from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class ByteTokenizer:
    kind = "byte"
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(idx) % 256 for idx in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": self.kind}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        return cls()


@dataclass
class CharTokenizer:
    kind = "char"
    stoi: dict[str, int]
    itos: dict[int, str]
    unk_token: str = "\uFFFD"

    @classmethod
    def train(cls, text: str, min_vocab: int = 0) -> "CharTokenizer":
        chars = sorted(set(text))
        if "\uFFFD" not in chars:
            chars.insert(0, "\uFFFD")
        stoi = {ch: i for i, ch in enumerate(chars)}
        while len(stoi) < min_vocab:
            token = f"<extra_{len(stoi)}>"
            stoi[token] = len(stoi)
        return cls(stoi=stoi, itos={i: ch for ch, i in stoi.items()})

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[self.unk_token]
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: list[int]) -> str:
        pieces = []
        for idx in ids:
            token = self.itos.get(int(idx), self.unk_token)
            if not token.startswith("<extra_"):
                pieces.append(token)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"kind": self.kind, "stoi": self.stoi, "unk_token": self.unk_token}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stoi = {str(k): int(v) for k, v in payload["stoi"].items()}
        return cls(stoi=stoi, itos={i: ch for ch, i in stoi.items()}, unk_token=payload.get("unk_token", "\uFFFD"))


def make_tokenizer(text: str, tokenizer_type: str = "byte") -> ByteTokenizer | CharTokenizer:
    if tokenizer_type == "byte":
        return ByteTokenizer()
    if tokenizer_type == "char":
        return CharTokenizer.train(text)
    raise ValueError(f"unknown tokenizer_type={tokenizer_type!r}")


def load_tokenizer(path: str | Path) -> ByteTokenizer | CharTokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") == "byte":
        return ByteTokenizer.load(path)
    return CharTokenizer.load(path)
