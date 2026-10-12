"""Make deterministic, document-level, source-stratified 98/1/1 splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
END_OF_TEXT_BYTES = b"\n\n<|endoftext|>\n\n"
DEFAULT_SEED = 4_172_026


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    hashes = [record["sha256"] for record in records]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("cleaned manifest contains duplicate document hashes")
    return records


def stratified_split(records: list[dict], seed: int) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["source"]].append(record)
    result = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for source in sorted(grouped):
        group = grouped[source]
        rng.shuffle(group)
        train_end = int(len(group) * 0.98)
        val_end = train_end + int(len(group) * 0.01)
        result["train"].extend(group[:train_end])
        result["val"].extend(group[train_end:val_end])
        result["test"].extend(group[val_end:])
    for records_for_split in result.values():
        rng.shuffle(records_for_split)
    return result


def write_split(name: str, records: list[dict], output: Path, audit_stream) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    source_counts: dict[str, int] = defaultdict(int)
    with output.open("wb") as destination:
        for record in records:
            clean_file = ROOT / record["cleaned_file"]
            with clean_file.open("rb") as source:
                source.seek(record["offset"])
                document = source.read(record["length"])
            destination.write(document)
            destination.write(END_OF_TEXT_BYTES)
            source_counts[record["source"]] += 1
            audit_stream.write(
                json.dumps(
                    {"sha256": record["sha256"], "source": record["source"], "split": name},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "documents": len(records),
        "bytes": output.stat().st_size,
        "source_documents": dict(sorted(source_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split cleaned documents into 98/1/1 files")
    parser.add_argument("--manifest", type=Path, default=ROOT / "corpus/cleaned/manifest.jsonl")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()

    records = load_manifest(arguments.manifest)
    splits = stratified_split(records, arguments.seed)
    split_manifest = ROOT / "corpus/split_manifest.jsonl"
    summary: dict[str, object] = {"seed": arguments.seed, "strategy": "source-stratified document shuffle"}
    with split_manifest.open("w", encoding="utf-8", newline="\n") as audit:
        for name in ("train", "val", "test"):
            summary[name] = write_split(name, splits[name], ROOT / f"corpus/{name}.txt", audit)

    sets = {name: {record["sha256"] for record in values} for name, values in splits.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise RuntimeError("document leakage detected between splits")
    summary["leakage_check"] = "passed: document SHA-256 sets are pairwise disjoint"
    summary["unique_document_hashes"] = len(set().union(*sets.values()))
    (ROOT / "corpus/split_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
