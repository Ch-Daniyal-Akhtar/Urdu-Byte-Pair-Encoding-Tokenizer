"""Run deterministic local stages after the three raw source files exist."""

from __future__ import annotations

import argparse
import subprocess
import sys


STAGES = [
    ("clean", [sys.executable, "scripts/clean_corpus.py", "--boilerplate-threshold", "50"]),
    ("split", [sys.executable, "scripts/split_corpus.py", "--seed", "4172026"]),
    ("stats", [sys.executable, "scripts/check_corpus_stats.py"]),
    ("data-card", [sys.executable, "scripts/generate_data_card.py"]),
    ("test", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]),
    (
        "train",
        [
            sys.executable,
            "train_tokenizer.py",
            "--input",
            "corpus/train.txt",
            "--vocab-size",
            "50000",
            "--output",
            "tokenizer.json",
        ],
    ),
    ("encode", [sys.executable, "encode_corpus.py"]),
    ("analyze", [sys.executable, "scripts/analyze_tokenizer.py"]),
    ("report", [sys.executable, "scripts/generate_report_pdf.py"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible post-collection pipeline")
    parser.add_argument("--from-stage", choices=[name for name, _ in STAGES], default="clean")
    parser.add_argument("--through-stage", choices=[name for name, _ in STAGES], default="report")
    arguments = parser.parse_args()
    names = [name for name, _ in STAGES]
    start = names.index(arguments.from_stage)
    end = names.index(arguments.through_stage)
    if end < start:
        raise ValueError("--through-stage must not precede --from-stage")
    for name, command in STAGES[start : end + 1]:
        print(f"\n=== {name}: {' '.join(command)} ===", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
