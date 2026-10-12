"""Extract, normalize, filter, de-boilerplate, and deduplicate the new corpus."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import regex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from normalize import normalize  # noqa: E402


END_OF_TEXT = "<|endoftext|>"
SOURCE_SPECS = (
    ("wikipedia_api", ROOT / "corpus/raw/wikipedia/wikipedia_raw.txt", "wikitext"),
    ("wikisource_api", ROOT / "corpus/raw/wikisource/wikisource_raw.txt", "wikitext"),
    ("mc4_ur", ROOT / "corpus/raw/mc4/mc4_raw.txt", "plain"),
)

_COMMENT = regex.compile(r"<!--.*?-->", regex.DOTALL)
_REF_BLOCK = regex.compile(r"<ref\b[^>]*>.*?</ref\s*>", regex.IGNORECASE | regex.DOTALL)
_REF_SINGLE = regex.compile(r"<ref\b[^>]*/\s*>", regex.IGNORECASE)
_TABLE = regex.compile(r"\{\|.*?\|\}", regex.DOTALL)
_TEMPLATE = regex.compile(r"\{\{[^{}]*\}\}")
_FILE_LINK = regex.compile(r"\[\[(?:File|Image|فائل|تصویر):[^\]]+\]\]", regex.IGNORECASE)
_CATEGORY_LINK = regex.compile(r"\[\[(?:Category|زمرہ):[^\]]+\]\]", regex.IGNORECASE)
_WIKI_LINK_WITH_LABEL = regex.compile(r"\[\[[^\]|]+\|([^\]]+)\]\]")
_WIKI_LINK = regex.compile(r"\[\[([^\]]+)\]\]")
_EXTERNAL_LINK = regex.compile(r"\[https?://\S+\s+([^\]]+)\]")
_URL = regex.compile(r"(?:https?://|www\.)\S+", regex.IGNORECASE)
_HTML_TAG = regex.compile(r"<[^>]+>")
_WIKI_HEADING = regex.compile(r"(?m)^\s*=+\s*(.*?)\s*=+\s*$")
_WIKI_QUOTES = regex.compile(r"'{2,5}")
_EMAIL = regex.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_IPV4 = regex.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_PHONE = regex.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)")
_ARABIC_LETTER = regex.compile(r"[\p{Script=Arabic}&&\p{L}]", regex.V1)
_ANY_LETTER = regex.compile(r"\p{L}")


@dataclass
class SourceStats:
    source: str
    raw_file_bytes: int = 0
    raw_documents: int = 0
    extracted_utf8_bytes: int = 0
    normalized_utf8_bytes: int = 0
    retained_utf8_bytes: int = 0
    retained_documents: int = 0
    exact_duplicates: int = 0
    low_quality_documents: int = 0
    filtered_lines: int = 0
    boilerplate_lines: int = 0
    pii_matches_removed: int = 0


def iter_documents(path: Path, chunk_chars: int = 4 * 1024 * 1024) -> Iterator[str]:
    carry = ""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        while block := stream.read(chunk_chars):
            carry += block
            while END_OF_TEXT in carry:
                document, carry = carry.split(END_OF_TEXT, 1)
                if document.strip():
                    yield document.strip()
        if carry.strip():
            yield carry.strip()


def strip_wikitext(text: str) -> str:
    """Conservative, dependency-free extraction of readable wiki source text."""

    text = _COMMENT.sub(" ", text)
    text = _REF_BLOCK.sub(" ", text)
    text = _REF_SINGLE.sub(" ", text)
    text = _TABLE.sub(" ", text)
    # Iteration removes nested templates from the inside out.
    for _ in range(12):
        updated = _TEMPLATE.sub(" ", text)
        if updated == text:
            break
        text = updated
    text = _FILE_LINK.sub(" ", text)
    text = _CATEGORY_LINK.sub(" ", text)
    text = _WIKI_LINK_WITH_LABEL.sub(r"\1", text)
    text = _WIKI_LINK.sub(r"\1", text)
    text = _EXTERNAL_LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _WIKI_HEADING.sub(r"\1", text)
    text = _WIKI_QUOTES.sub("", text)
    return html.unescape(text)


def redact_personal_patterns(text: str) -> tuple[str, int]:
    removed = 0
    for pattern in (_EMAIL, _IPV4, _PHONE):
        text, count = pattern.subn(" ", text)
        removed += count
    return text, removed


def line_is_urdu(line: str) -> bool:
    letters = len(_ANY_LETTER.findall(line))
    urdu_letters = len(_ARABIC_LETTER.findall(line))
    return urdu_letters >= 8 and (letters == 0 or urdu_letters / letters >= 0.35)


def prepare_for_counting(document: str, kind: str) -> str:
    extracted = strip_wikitext(document) if kind == "wikitext" else _URL.sub(" ", _HTML_TAG.sub(" ", document))
    redacted, _ = redact_personal_patterns(extracted)
    return normalize(redacted)


def line_fingerprint(line: str) -> bytes:
    return hashlib.blake2b(line.encode("utf-8"), digest_size=12).digest()


def find_boilerplate(sources: list[tuple[str, Path, str]], threshold: int) -> set[bytes]:
    counts: Counter[bytes] = Counter()
    for name, path, kind in sources:
        print(f"boilerplate_pass source={name}", flush=True)
        for document in iter_documents(path):
            normalized = prepare_for_counting(document, kind)
            per_document = {
                line_fingerprint(line)
                for line in normalized.splitlines()
                if 15 <= len(line) <= 300
            }
            counts.update(per_document)
    boilerplate = {fingerprint for fingerprint, count in counts.items() if count >= threshold}
    print(f"boilerplate_candidates={len(boilerplate):,} threshold={threshold}")
    return boilerplate


def clean_source(
    source: str,
    input_path: Path,
    kind: str,
    output_path: Path,
    manifest_stream,
    boilerplate: set[bytes],
    seen_documents: set[bytes],
) -> SourceStats:
    stats = SourceStats(source=source, raw_file_bytes=input_path.stat().st_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        for raw_document in iter_documents(input_path):
            stats.raw_documents += 1
            extracted = strip_wikitext(raw_document) if kind == "wikitext" else _URL.sub(
                " ", _HTML_TAG.sub(" ", raw_document)
            )
            stats.extracted_utf8_bytes += len(extracted.encode("utf-8"))
            extracted, removed = redact_personal_patterns(extracted)
            stats.pii_matches_removed += removed
            normalized = normalize(extracted)
            stats.normalized_utf8_bytes += len(normalized.encode("utf-8"))

            kept_lines: list[str] = []
            for line in normalized.splitlines():
                line = line.strip()
                if not line:
                    if kept_lines and kept_lines[-1] != "":
                        kept_lines.append("")
                    continue
                if 15 <= len(line) <= 300 and line_fingerprint(line) in boilerplate:
                    stats.boilerplate_lines += 1
                    continue
                if not line_is_urdu(line):
                    stats.filtered_lines += 1
                    continue
                kept_lines.append(line)
            document = "\n".join(kept_lines).strip()
            if len(document) < 100 or len(_ARABIC_LETTER.findall(document)) < 40:
                stats.low_quality_documents += 1
                continue

            document_bytes = document.encode("utf-8")
            digest = hashlib.sha256(document_bytes).digest()
            if digest in seen_documents:
                stats.exact_duplicates += 1
                continue
            seen_documents.add(digest)

            offset = output.tell()
            trailer = f"\n\n{END_OF_TEXT}\n\n".encode("utf-8")
            output.write(document_bytes)
            output.write(trailer)
            manifest_stream.write(
                json.dumps(
                    {
                        "source": source,
                        "cleaned_file": str(output_path.relative_to(ROOT)).replace("\\", "/"),
                        "offset": offset,
                        "length": len(document_bytes),
                        "sha256": digest.hex(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats.retained_documents += 1
            stats.retained_utf8_bytes += len(document_bytes) + len(trailer)
            if stats.raw_documents % 10_000 == 0:
                print(
                    f"clean source={source} seen={stats.raw_documents:,} kept={stats.retained_documents:,} "
                    f"mb={stats.retained_utf8_bytes / 1024 / 1024:.2f}",
                    flush=True,
                )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the independently collected Urdu corpus")
    parser.add_argument("--boilerplate-threshold", type=int, default=50)
    parser.add_argument("--skip-boilerplate-pass", action="store_true")
    arguments = parser.parse_args()

    sources = [(name, path, kind) for name, path, kind in SOURCE_SPECS if path.exists()]
    if len(sources) < 3:
        missing = [str(path) for _, path, _ in SOURCE_SPECS if not path.exists()]
        raise FileNotFoundError(f"all three source files are required; missing: {missing}")

    boilerplate = set() if arguments.skip_boilerplate_pass else find_boilerplate(sources, arguments.boilerplate_threshold)
    manifest = ROOT / "corpus/cleaned/manifest.jsonl"
    stats_path = ROOT / "corpus/cleaned/cleaning_stats.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    seen_documents: set[bytes] = set()
    all_stats: list[SourceStats] = []
    with manifest.open("w", encoding="utf-8", newline="\n") as manifest_stream:
        for source, input_path, kind in sources:
            output_path = ROOT / f"corpus/cleaned/{source}.txt"
            stats = clean_source(
                source,
                input_path,
                kind,
                output_path,
                manifest_stream,
                boilerplate,
                seen_documents,
            )
            all_stats.append(stats)
            print(asdict(stats), flush=True)
    stats_path.write_text(
        json.dumps(
            {
                "boilerplate_document_frequency_threshold": arguments.boilerplate_threshold,
                "boilerplate_fingerprints": len(boilerplate),
                "sources": [asdict(stats) for stats in all_stats],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
