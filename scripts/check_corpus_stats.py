"""Audit split sizes, UTF-8 validity, document counts, and vocabulary totals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import regex


ROOT = Path(__file__).resolve().parents[1]
WORD = regex.compile(r"[\p{L}\p{M}]+(?:\u200c[\p{L}\p{M}]+)*")
END_OF_TEXT = "<|endoftext|>"


def inspect(path: Path, unique_words: set[str]) -> dict[str, object]:
    digest = hashlib.sha256()
    documents = 0
    words = 0
    characters = 0
    with path.open("rb") as binary:
        while block := binary.read(8 * 1024 * 1024):
            digest.update(block)
    with path.open("r", encoding="utf-8", errors="strict", newline="") as text:
        for line in text:
            characters += len(line)
            documents += line.count(END_OF_TEXT)
            found = WORD.findall(line.replace(END_OF_TEXT, ""))
            words += len(found)
            unique_words.update(found)
    return {
        "bytes": path.stat().st_size,
        "mebibytes": path.stat().st_size / 1024 / 1024,
        "documents": documents,
        "characters": characters,
        "words": words,
        "sha256": digest.hexdigest(),
        "valid_utf8": True,
    }


def main() -> None:
    unique_words: set[str] = set()
    results: dict[str, object] = {}
    total_bytes = 0
    total_documents = 0
    total_words = 0
    for split in ("train", "val", "test"):
        path = ROOT / f"corpus/{split}.txt"
        result = inspect(path, unique_words)
        results[split] = result
        total_bytes += int(result["bytes"])
        total_documents += int(result["documents"])
        total_words += int(result["words"])
    results["total"] = {
        "bytes": total_bytes,
        "mebibytes": total_bytes / 1024 / 1024,
        "documents": total_documents,
        "words": total_words,
        "unique_words": len(unique_words),
    }
    output = ROOT / "corpus/corpus_stats.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
