# Urdu byte-level BPE tokenizer from scratch

This is an independently built CS 417 Assignment 1 submission. It creates a
new multi-genre Urdu corpus, learns exactly 49,743 merges for a 50,000-token
byte vocabulary, writes `uint16` training artifacts, and measures the result
against GPT-2 and `o200k_base`.

## Corpus design

The data is intentionally different from the supplied reference solution:

| Source | Genre | Acquisition | Role |
|---|---|---|---|
| Urdu Wikipedia | Encyclopedia | Official MediaWiki API | Self-collected evidence |
| Urdu Wikisource | Literature | Official MediaWiki API | Self-collected third source/genre |
| Urdu mC4 (`allenai/c4`) | Diverse web prose | Published ODC-BY corpus | Volume and domain breadth |

API requests are sequential, throttled, resumable, identify the project, obey
`maxlag`/`Retry-After`, and write one JSONL provenance record per page. Corpus
text is kept inside the course because upstream web-page rights may vary.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Exact rebuild commands

Run from this repository's root.

```powershell
# 1. Self-collect encyclopedia text (70 MB exceeds the mandatory 50 MB)
python collection/collect_wikimedia.py --project wikipedia `
  --output corpus/raw/wikipedia/wikipedia_raw.txt `
  --metadata corpus/raw/wikipedia/metadata.jsonl `
  --state corpus/raw/wikipedia/state.json --target-mb 70 --delay 0.25

# 2. Collect the complete main namespace of Urdu Wikisource
python collection/collect_wikimedia.py --project wikisource `
  --output corpus/raw/wikisource/wikisource_raw.txt `
  --metadata corpus/raw/wikisource/metadata.jsonl `
  --state corpus/raw/wikisource/state.json --target-mb 0 --delay 0.25

# 3. Download open Urdu mC4 shards and retain 560 MB of raw documents
python collection/download_mc4.py --target-mb 560

# 4. Extract wiki markup, normalize, filter, redact, de-boilerplate, deduplicate
python scripts/clean_corpus.py --boilerplate-threshold 50

# 5. Fixed-seed, source-stratified, document-level 98/1/1 split
python scripts/split_corpus.py --seed 4172026
python scripts/check_corpus_stats.py
python scripts/generate_data_card.py

# 6. Unit tests and randomized-equivalence-covered implementation
python -m unittest discover -s tests -v

# 7. Verify reference/fast trainers on 1 MB, then learn exactly 50,000 tokens
python train_tokenizer.py --input corpus/train.txt --vocab-size 50000 `
  --output tokenizer.json --verify-sample-mb 1 --verify-merges 500

# 8. Flat NumPy-compatible uint16 arrays
python encode_corpus.py --tokenizer tokenizer.json `
  --train-in corpus/train.txt --val-in corpus/val.txt `
  --train-out data/train.bin --val-out data/val.bin

# 9. Part C measurements and plots
python scripts/analyze_tokenizer.py

# 10. Cover page plus four-page PDF report
python scripts/generate_report_pdf.py
```

`normalize.py` is separate from tokenization. It maps Arabic yeh/kaf/heh
lookalikes to Urdu forms, preserves gol heh versus do-chashmi heh, removes
formatting controls and tatweel, retains only internal ZWNJ, maps all digit
families to ASCII, and is idempotent.

`bpe.py` uses byte IDs 0–255, `<|endoftext|>` at 256, lexicographically greater
pair tie-breaking, an Urdu-aware regex that keeps marks and internal ZWNJ in a
word, a recount-all reference trainer, and a weighted/incremental fast trainer.

Measured corpus, training, encoding, and comparison results are recorded in
`DATA_CARD.md`, `corpus/*.json`, `data/*.json`, and `report.pdf`.
See `COMPLIANCE_CHECKLIST.md` for a requirement-by-requirement evidence map and
the two user-specific steps that remain before submission.
