"""Stream assignment text files to flat NumPy uint16 token arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from bpe import BPETokenizer, END_OF_TEXT_ID, iter_documents


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def encode_file(tokenizer: BPETokenizer, input_path: Path, output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    document_count = 0
    minimum = 65_535
    maximum = 0
    started = time.perf_counter()
    with output_path.open("wb") as destination:
        for document in iter_documents(input_path):
            ids = tokenizer.encode(document, allow_special=False)
            ids.append(END_OF_TEXT_ID)
            array = np.asarray(ids, dtype=np.uint16)
            array.tofile(destination)
            token_count += len(ids)
            document_count += 1
            if ids:
                minimum = min(minimum, min(ids))
                maximum = max(maximum, max(ids))
            if document_count % 10_000 == 0:
                print(f"file={input_path.name} documents={document_count:,} tokens={token_count:,}", flush=True)

    elapsed = time.perf_counter() - started
    expected_bytes = token_count * np.dtype(np.uint16).itemsize
    if output_path.stat().st_size != expected_bytes:
        raise RuntimeError("uint16 output size does not match token count")
    return {
        "input": str(input_path).replace("\\", "/"),
        "output": str(output_path).replace("\\", "/"),
        "documents": document_count,
        "tokens": token_count,
        "minimum_token_id": minimum if token_count else None,
        "maximum_token_id": maximum if token_count else None,
        "output_bytes": expected_bytes,
        "bytes_per_token": input_path.stat().st_size / token_count if token_count else None,
        "elapsed_seconds": elapsed,
        "tokens_per_second": token_count / elapsed if elapsed else None,
        "input_sha256": sha256(input_path),
        "output_sha256": sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode train and validation corpora as uint16")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--train-in", type=Path, default=Path("corpus/train.txt"))
    parser.add_argument("--val-in", type=Path, default=Path("corpus/val.txt"))
    parser.add_argument("--train-out", type=Path, default=Path("data/train.bin"))
    parser.add_argument("--val-out", type=Path, default=Path("data/val.bin"))
    parser.add_argument("--stats", type=Path, default=Path("data/encoding_stats.json"))
    arguments = parser.parse_args()

    tokenizer = BPETokenizer.from_file(arguments.tokenizer)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocabulary does not fit uint16")
    results = {
        "dtype": "uint16",
        "train": encode_file(tokenizer, arguments.train_in, arguments.train_out),
        "validation": encode_file(tokenizer, arguments.val_in, arguments.val_out),
    }
    arguments.stats.parent.mkdir(parents=True, exist_ok=True)
    arguments.stats.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
