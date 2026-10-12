"""Write DATA_CARD.md from measured collection and cleaning artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def mib(value: int | float) -> str:
    return f"{value / 1024 / 1024:.2f}"


def main() -> None:
    cleaning = read("corpus/cleaned/cleaning_stats.json")
    corpus = read("corpus/corpus_stats.json")
    sources = {item["source"]: item for item in cleaning["sources"]}
    self_collected = sources["wikipedia_api"]["raw_file_bytes"] + sources["wikisource_api"]["raw_file_bytes"]
    total = corpus["total"]
    rows = [
        (
            "Urdu Wikipedia API",
            "Encyclopedia",
            "https://ur.wikipedia.org/w/api.php",
            "CC BY-SA 4.0 / GFDL",
            sources["wikipedia_api"],
        ),
        (
            "Urdu Wikisource API",
            "Literature",
            "https://ur.wikisource.org/w/api.php",
            "CC BY-SA 4.0; per-page/public-domain notices may apply",
            sources["wikisource_api"],
        ),
        (
            "Urdu mC4 (AllenAI C4)",
            "Diverse web",
            "https://huggingface.co/datasets/allenai/c4",
            "ODC-BY packaging; underlying page rights vary",
            sources["mc4_ur"],
        ),
    ]
    table = "\n".join(
        f"| {name} | {genre} | {url} | {licence} | 2026-10-11 | {mib(stats['raw_file_bytes'])} | "
        f"{mib(stats['retained_utf8_bytes'])} |"
        for name, genre, url, licence, stats in rows
    )
    removal_rows = "\n".join(
        f"| {name} | {mib(stats['raw_file_bytes'] - stats['extracted_utf8_bytes'])} | "
        f"{mib(stats['extracted_utf8_bytes'] - stats['normalized_utf8_bytes'])} | "
        f"{stats['filtered_lines']:,} | {stats['boilerplate_lines']:,} | {stats['exact_duplicates']:,} | "
        f"{mib(stats['retained_utf8_bytes'])} |"
        for name, stats in ((row[0], row[4]) for row in rows)
    )
    card = f"""# DATA CARD — independently collected Urdu corpus

## Summary

| Clean UTF-8 | Documents | Words | Unique words | Self-collected raw | Split |
|---:|---:|---:|---:|---:|---|
| **{total['mebibytes']:.2f} MiB** | **{total['documents']:,}** | **{total['words']:,}** | **{total['unique_words']:,}** | **{mib(self_collected)} MiB** | 98/1/1 by document, seed 4172026 |

The corpus is for CS 417 coursework and is not a copy of the supplied solution data. It excludes Express, Jang, BBC Urdu, Makhzan and CC-100. The MediaWiki API JSONL logs provide page URL, revision, collection time, byte count and SHA-256 evidence for the ≥50 MB self-collection requirement.

## Sources

| Source | Genre | URL | Licence / terms | Collected | Raw MiB | Clean MiB |
|---|---|---|---|---|---:|---:|
{table}

## Cleaning and removals

Plain text extraction removes wiki templates/tables/references, HTML and URLs. The hand-written NFC normalizer maps Arabic yeh/kaf/heh lookalikes, preserves the two Urdu heh letters, removes tatweel/direction controls, keeps only internal ZWNJ, converts every digit family to ASCII, and canonicalizes whitespace. We redact email/phone/IP patterns, require Urdu-script density while retaining natural code mixing, remove lines appearing in at least {cleaning['boilerplate_document_frequency_threshold']} documents, and SHA-256-deduplicate normalized documents globally.

| Source | Markup removed MiB | Normalization/filter delta MiB | Non-Urdu lines | Boilerplate lines | Duplicate docs | Retained MiB |
|---|---:|---:|---:|---:|---:|---:|
{removal_rows}

## Known gaps, biases, and responsible use

- mC4 is the largest component, so formal web prose and repeated web conventions dominate; conversational Urdu, regional dialects and Roman Urdu are underrepresented.
- Wikipedia adds encyclopedic/name vocabulary; Wikisource adds older literary spelling and highly vocalized Arabic passages. Source-stratified splitting keeps all three domains in each split but cannot remove source-family similarity.
- Automatic language and PII filters reduce risk but are imperfect. Public biographies are not treated as private messages; emails, phone-like strings and IP addresses are removed. The corpus stays within the course.
- mC4's dataset packaging is ODC-BY, but the creators do not own every underlying webpage. Wikimedia reuse must preserve attribution/share-alike obligations and page-specific notices. Do not publish the raw corpus without a fresh rights review.
"""
    (ROOT / "DATA_CARD.md").write_text(card, encoding="utf-8", newline="\n")
    print("data_card=DATA_CARD.md")


if __name__ == "__main__":
    main()
