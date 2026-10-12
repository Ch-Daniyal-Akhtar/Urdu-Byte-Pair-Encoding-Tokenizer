"""Generate every measured result required by Part C of the assignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regex
import tiktoken


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bpe import BPETokenizer, END_OF_TEXT, iter_documents  # noqa: E402


WORD_PATTERN = regex.compile(r"[\p{L}\p{M}]+(?:\u200c[\p{L}\p{M}]+)*|\p{N}+")
ARABIC_WORD = regex.compile(r"[\p{Script=Arabic}\p{M}]+(?:\u200c[\p{Script=Arabic}\p{M}]+)*", regex.V1)

HANDPICKED = {
    "news": "وفاقی کابینہ نے توانائی منصوبوں کی بروقت تکمیل کے لیے نئی نگرانی کمیٹی قائم کر دی ہے۔",
    "poetry": "دل ہی تو ہے نہ سنگ و خشت، درد سے بھر نہ آئے کیوں\nروئیں گے ہم ہزار بار، کوئی ہمیں ستائے کیوں",
    "urdu_english": "جامعہ کی AI Lab نے open-source language model کے نتائج آج شائع کیے۔",
    "roman_urdu": "Kal class ke baad library mein milna, phir assignment mil kar complete karte hain.",
    "numbers_dates": "اجلاس 11 اکتوبر 2026 کو صبح 9:30 بجے ہوگا اور فیس 4,750 روپے ہے۔",
    "diacritics": "اِنَّ مَعَ الْعُسْرِ یُسْرًا، فَإِنَّ مَعَ الْعُسْرِ یُسْرًا۔",
    "personal_names": "فاطمہ جناح، عبدالستار ایدھی، فیض احمد فیض اور پروین شاکر معروف پاکستانی شخصیات ہیں۔",
}

FAILURE_INPUTS = {
    "roman_chat": "yr plzz mjhy bta do assignment kb submit krni hy???",
    "rare_diacritics": "قُرْآنِ حَکِیْمْ کِی تِلَاوَتْ بِالتَّجْوِیْدِ",
    "technical_identifier": "ماڈل ID XQ-97_v2.4 کا SHA256=9f86d081884c7d65 ہے۔",
}


def safe_decode(raw: bytes) -> tuple[str, bool]:
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        return "".join(f"\\x{byte:02x}" for byte in raw), False


def merge_analysis(tokenizer: BPETokenizer, train_path: Path) -> dict[str, object]:
    selected_indices = list(range(1, 51)) + list(range(5_000, len(tokenizer.merges) + 1, 5_000))
    selected: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    letter_merges: list[dict[str, object]] = []
    word_candidates: dict[str, dict[str, object]] = {}
    token_rows: list[dict[str, object]] = []

    for merge_index, (pair, token_id) in enumerate(tokenizer.merges, start=1):
        raw = tokenizer.token_bytes[token_id]
        decoded, valid = safe_decode(raw)
        row = {
            "merge_index": merge_index,
            "token_id": token_id,
            "pair": list(pair),
            "decoded": decoded,
            "valid_utf8": valid,
            "hex": raw.hex(" "),
            "byte_length": len(raw),
            "character_length": len(decoded) if valid else None,
        }
        token_rows.append(row)
        if merge_index in selected_indices:
            selected.append(row)
        if not valid:
            invalid.append(row)
        if valid and len(raw) == 2 and len(decoded) == 1 and ARABIC_WORD.fullmatch(decoded):
            letter_merges.append(row)
        candidate = decoded.removeprefix(" ") if valid else ""
        if (
            valid
            and decoded.startswith(" ")
            and len(candidate) >= 2
            and len(regex.findall(r"\p{L}", candidate)) >= 2
            and ARABIC_WORD.fullmatch(candidate)
        ):
            word_candidates.setdefault(decoded, row)

    attested: set[str] = set()
    for document in iter_documents(train_path):
        for piece in tokenizer.pre_tokenize(document):
            if piece in word_candidates:
                attested.add(piece)
    first_word = min((word_candidates[piece] for piece in attested), key=lambda row: row["merge_index"], default=None)

    longest = sorted(token_rows, key=lambda row: (row["byte_length"], row["token_id"]), reverse=True)[:20]
    return {
        "selected_merges": selected,
        "early_letter_merges": [row for row in letter_merges if row["merge_index"] <= 200],
        "invalid_utf8_count": len(invalid),
        "invalid_utf8_examples": invalid[:15],
        "first_whole_urdu_word": first_word,
        "longest_tokens": longest,
        "all_token_byte_lengths": [row["byte_length"] for row in token_rows],
    }


def metrics(text: str, tokenizer: BPETokenizer, gpt2, multilingual) -> dict[str, object]:
    natural_text = text.replace(END_OF_TEXT, "")
    words = len(WORD_PATTERN.findall(natural_text))
    utf8_bytes = len(natural_text.encode("utf-8"))
    counts = {
        "ours": len(tokenizer.encode(natural_text, allow_special=False)),
        "gpt2": len(gpt2.encode(natural_text)),
        "o200k_base": len(multilingual.encode(natural_text)),
    }
    return {
        "words": words,
        "utf8_bytes": utf8_bytes,
        "tokenizers": {
            name: {
                "tokens": count,
                "tokens_per_word": count / words if words else 0,
                "bytes_per_token": utf8_bytes / count if count else 0,
            }
            for name, count in counts.items()
        },
    }


def comparison(tokenizer: BPETokenizer, test_path: Path, gpt2, multilingual) -> dict[str, object]:
    test_text = test_path.read_text(encoding="utf-8")
    handpicked = {
        name: {"text": text, "metrics": metrics(text, tokenizer, gpt2, multilingual)}
        for name, text in HANDPICKED.items()
    }
    failures: dict[str, object] = {}
    for name, text in FAILURE_INPUTS.items():
        measured = metrics(text, tokenizer, gpt2, multilingual)
        ids = tokenizer.encode(text, allow_special=False)
        pieces = [safe_decode(tokenizer.token_bytes[token_id])[0] for token_id in ids]
        failures[name] = {"text": text, "metrics": measured, "our_token_pieces": pieces}
    return {
        "word_definition": "Unicode letter/mark sequences (including internal ZWNJ) or number sequences",
        "test_set": metrics(test_text, tokenizer, gpt2, multilingual),
        "handpicked": handpicked,
        "failure_cases": failures,
    }


def make_plots(merges: dict[str, object], comparisons: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lengths = merges["all_token_byte_lengths"]
    fig, axis = plt.subplots(figsize=(8, 3.4))
    axis.scatter(range(1, len(lengths) + 1), lengths, s=2, alpha=0.22, color="#4C8EDA")
    window = 1_000
    moving = [sum(lengths[index : index + window]) / len(lengths[index : index + window]) for index in range(0, len(lengths), window)]
    axis.plot([index + window / 2 for index in range(0, len(lengths), window)], moving, color="#D97921", linewidth=2)
    axis.set(xlabel="Merge index", ylabel="Token length (UTF-8 bytes)", title="Learned token length over training")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "merge_lengths.png", dpi=180)
    plt.close(fig)

    labels = ["test"] + list(comparisons["handpicked"])
    series = {name: [] for name in ("ours", "o200k_base", "gpt2")}
    for label in labels:
        measured = comparisons["test_set"] if label == "test" else comparisons["handpicked"][label]["metrics"]
        for name in series:
            series[name].append(measured["tokenizers"][name]["tokens_per_word"])
    positions = list(range(len(labels)))
    width = 0.25
    fig, axis = plt.subplots(figsize=(9, 3.8))
    colors = {"ours": "#2F6655", "o200k_base": "#6B8FB3", "gpt2": "#B8B8B8"}
    for offset, name in enumerate(series):
        axis.bar([position + (offset - 1) * width for position in positions], series[name], width, label=name, color=colors[name])
    axis.set_xticks(positions, labels, rotation=22, ha="right")
    axis.set(ylabel="Tokens per word (lower is better)", title="Tokenizer comparison")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "tokens_per_word.png", dpi=180)
    plt.close(fig)


def markdown_summary(results: dict[str, object]) -> str:
    merges = results["merges"]
    comparison_results = results["comparison"]
    lines = ["# Part C measured results", "", "## Test-set comparison", ""]
    lines += ["| Tokenizer | Tokens | Tokens/word | Bytes/token |", "|---|---:|---:|---:|"]
    for name, values in comparison_results["test_set"]["tokenizers"].items():
        lines.append(f"| {name} | {values['tokens']:,} | {values['tokens_per_word']:.3f} | {values['bytes_per_token']:.3f} |")
    first = merges["first_whole_urdu_word"]
    lines += [
        "",
        "## Merge findings",
        "",
        f"- Invalid standalone UTF-8 merge tokens: {merges['invalid_utf8_count']:,}",
        f"- First space-prefixed whole Urdu word: merge {first['merge_index']} / token {first['token_id']} / `{first['decoded']}`" if first else "- No whole Urdu word detected.",
        "- The JSON file contains merges 1–50, each 5,000th merge, early letter reconstruction, and the 20 longest tokens.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the assignment tokenizer analysis")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer.json")
    parser.add_argument("--test", type=Path, default=ROOT / "corpus/test.txt")
    parser.add_argument("--train", type=Path, default=ROOT / "corpus/train.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/analysis_results.json")
    arguments = parser.parse_args()

    tokenizer = BPETokenizer.from_file(arguments.tokenizer)
    # Resolve external baseline assets before the expensive full-corpus scan,
    # so a missing tiktoken cache fails quickly rather than wasting that pass.
    gpt2 = tiktoken.get_encoding("gpt2")
    multilingual = tiktoken.get_encoding("o200k_base")
    merges = merge_analysis(tokenizer, arguments.train)
    comparisons = comparison(tokenizer, arguments.test, gpt2, multilingual)
    results = {"merges": merges, "comparison": comparisons}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (arguments.output.parent / "analysis_summary.md").write_text(markdown_summary(results), encoding="utf-8")
    make_plots(merges, comparisons, arguments.output.parent)
    print(f"analysis={arguments.output}")


if __name__ == "__main__":
    main()
