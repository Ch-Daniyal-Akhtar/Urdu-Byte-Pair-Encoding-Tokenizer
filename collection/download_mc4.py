"""Download open Urdu mC4 shards and extract a size-bounded raw corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


END_OF_TEXT = "<|endoftext|>"
REPOSITORY = "allenai/c4"
SOURCE_PAGE = "https://huggingface.co/datasets/allenai/c4"
SHARD_TEMPLATE = "c4-ur.tfrecord-{index:05d}-of-00128.json.gz"
USER_AGENT = "UrduBPEAssignment/1.0 (NUST SEECS CS417; educational research)"


def download_with_resume(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    response = requests.get(url, headers=headers, stream=True, timeout=120)
    response.raise_for_status()
    if existing and response.status_code == 206:
        mode = "ab"
    else:
        mode = "wb"
        existing = 0
    expected_length = response.headers.get("Content-Length")
    expected_total = existing + int(expected_length) if expected_length else None
    downloaded = existing
    with destination.open(mode) as stream:
        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
            if not chunk:
                continue
            stream.write(chunk)
            downloaded += len(chunk)
            suffix = f"/{expected_total / 1024 / 1024:.1f}" if expected_total else ""
            print(f"download {destination.name}: {downloaded / 1024 / 1024:.1f}{suffix} MB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract Urdu mC4")
    parser.add_argument("--download-dir", type=Path, default=Path("corpus/downloads/mc4"))
    parser.add_argument("--output", type=Path, default=Path("corpus/raw/mc4/mc4_raw.txt"))
    parser.add_argument("--metadata", type=Path, default=Path("corpus/raw/mc4/metadata.jsonl"))
    parser.add_argument("--target-mb", type=float, default=560.0)
    parser.add_argument("--first-shard", type=int, default=0)
    parser.add_argument("--last-shard", type=int, default=127)
    arguments = parser.parse_args()

    if arguments.output.exists() or arguments.metadata.exists():
        raise FileExistsError("mC4 extraction outputs already exist; move them before a fresh deterministic run")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.metadata.parent.mkdir(parents=True, exist_ok=True)
    target_bytes = int(arguments.target_mb * 1024 * 1024)
    written_bytes = 0
    documents = 0
    collected_at = datetime.now(timezone.utc).isoformat()

    with arguments.output.open("w", encoding="utf-8", newline="\n") as corpus, arguments.metadata.open(
        "w", encoding="utf-8", newline="\n"
    ) as evidence:
        for index in range(arguments.first_shard, arguments.last_shard + 1):
            shard_name = SHARD_TEMPLATE.format(index=index)
            shard = arguments.download_dir / shard_name
            url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/main/multilingual/{shard_name}?download=true"
            download_with_resume(url, shard)
            with gzip.open(shard, "rt", encoding="utf-8", errors="strict") as rows:
                for row_index, line in enumerate(rows):
                    record = json.loads(line)
                    text = record.get("text", "")
                    if not isinstance(text, str) or len(text.strip()) < 200:
                        continue
                    document = text.strip()
                    serialized = f"{document}\n\n{END_OF_TEXT}\n\n"
                    byte_count = len(serialized.encode("utf-8"))
                    corpus.write(serialized)
                    evidence.write(
                        json.dumps(
                            {
                                "dataset": REPOSITORY,
                                "source_page": SOURCE_PAGE,
                                "shard": shard_name,
                                "row_index": row_index,
                                "url": record.get("url"),
                                "source_timestamp": record.get("timestamp"),
                                "collected_at_utc": collected_at,
                                "utf8_bytes": byte_count,
                                "sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written_bytes += byte_count
                    documents += 1
                    if written_bytes >= target_bytes:
                        print(f"documents={documents:,} raw_mb={written_bytes / 1024 / 1024:.2f}")
                        return
            corpus.flush()
            evidence.flush()
            print(f"documents={documents:,} raw_mb={written_bytes / 1024 / 1024:.2f}", flush=True)
    raise RuntimeError(f"available shards ended before target {arguments.target_mb:.2f} MB")


if __name__ == "__main__":
    main()
