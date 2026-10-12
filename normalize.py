"""Deterministic Unicode normalization for the Urdu corpus.

Normalization is deliberately separate from tokenization: ``BPETokenizer`` must
round-trip the caller's original string, while corpus preparation may choose a
canonical spelling. This module uses only Python's standard library plus the
``regex`` package already required for Urdu-aware character properties.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import regex


_LETTER_TRANSLATION = str.maketrans(
    {
        "\u064a": "\u06cc",  # Arabic yeh -> Farsi/Urdu yeh
        "\u0649": "\u06cc",  # alef maksura -> Farsi/Urdu yeh
        "\u0643": "\u06a9",  # Arabic kaf -> keheh
        "\u0647": "\u06c1",  # Arabic heh -> Urdu gol heh
    }
)

# U+200C ZWNJ is handled separately. Do-chashmi heh U+06BE and gol heh U+06C1
# are intentionally absent: they are different Urdu letters.
_DROP_CODEPOINTS = {
    0x00AD,  # soft hyphen
    0x061C,  # Arabic letter mark
    0x0640,  # tatweel
    0x200B,  # zero-width space
    0x200D,  # zero-width joiner
    0x200E,  # LRM
    0x200F,  # RLM
    0x202A,
    0x202B,
    0x202C,
    0x202D,
    0x202E,
    0x2060,  # word joiner
    0x2066,
    0x2067,
    0x2068,
    0x2069,
    0xFEFF,  # BOM
}
_DROP_TRANSLATION = {value: None for value in _DROP_CODEPOINTS}

# One documented digit convention: everything becomes ASCII 0-9. This keeps
# code-mixed dates natural and prevents three parallel sets of digit tokens.
_DIGIT_TRANSLATION = str.maketrans(
    {
        **{chr(0x0660 + index): str(index) for index in range(10)},
        **{chr(0x06F0 + index): str(index) for index in range(10)},
    }
)

_EXOTIC_SPACES = regex.compile(r"[\p{Zs}\t\v\f]+")
_EXCESS_NEWLINES = regex.compile(r"\n{3,}")
_ZWNJ_RUN = regex.compile(r"\u200c+")


def _canonicalize_zwnj(text: str, preserve_zwnj: bool) -> str:
    """Keep one ZWNJ only when it separates two letter/mark sequences."""

    if not preserve_zwnj:
        return text.replace("\u200c", "")
    if "\u200c" not in text:
        return text

    text = _ZWNJ_RUN.sub("\u200c", text)
    output: list[str] = []
    for index, character in enumerate(text):
        if character != "\u200c":
            output.append(character)
            continue
        left = index - 1
        while left >= 0 and unicodedata.category(text[left]).startswith("M"):
            left -= 1
        right = index + 1
        while right < len(text) and unicodedata.category(text[right]).startswith("M"):
            right += 1
        left_is_letter = left >= 0 and unicodedata.category(text[left]).startswith("L")
        right_is_letter = right < len(text) and unicodedata.category(text[right]).startswith("L")
        if left_is_letter and right_is_letter:
            output.append(character)
    return "".join(output)


def _normalize_spacing(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _EXOTIC_SPACES.sub(" ", text)
    lines = [line.strip(" ") for line in text.split("\n")]
    return _EXCESS_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def normalize(text: str, *, preserve_zwnj: bool = True) -> str:
    """Return the assignment corpus's canonical Urdu representation.

    The operation is idempotent. NFC is used instead of NFKC so compatibility
    characters and intentional presentation distinctions are not silently
    rewritten. Arabic/Farsi digit variants are mapped to ASCII digits.
    """

    if not isinstance(text, str):
        raise TypeError("normalize() expects str input")
    if not text:
        return ""

    result = unicodedata.normalize("NFC", text)
    result = result.translate(_LETTER_TRANSLATION)
    result = result.translate(_DROP_TRANSLATION)
    result = result.translate(_DIGIT_TRANSLATION)
    result = _canonicalize_zwnj(result, preserve_zwnj)
    return _normalize_spacing(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize an Urdu UTF-8 text file")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--remove-zwnj", action="store_true")
    arguments = parser.parse_args()

    source = arguments.input.read_text(encoding="utf-8", errors="replace")
    cleaned = normalize(source, preserve_zwnj=not arguments.remove_zwnj)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(cleaned, encoding="utf-8", newline="\n")
    print(f"normalized_chars={len(cleaned):,}")


if __name__ == "__main__":
    main()
