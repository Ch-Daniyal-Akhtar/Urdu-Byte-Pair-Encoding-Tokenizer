# Assignment compliance checklist

This file maps each assessed requirement to a concrete artifact. All measured
values are from this independently collected corpus, not the supplied solution.

## Part A — corpus

- [x] More than 500 MB of clean Urdu UTF-8: `corpus/corpus_stats.json` records
  609.05 MiB and validates every split as UTF-8.
- [x] At least three sources and two genres: Urdu Wikipedia (encyclopedia), Urdu
  Wikisource (literature), and Urdu mC4 (diverse web prose).
- [x] More than 50 MB self-collected: official Wikimedia API collectors produced
  118.34 MiB raw, with request/page evidence in
  `corpus/raw/wikipedia/metadata.jsonl` and
  `corpus/raw/wikisource/metadata.jsonl`.
- [x] Licences/terms and responsible-use limits: `DATA_CARD.md`.
- [x] Markup/URL removal, Urdu-density filtering, PII-pattern redaction,
  document SHA-256 deduplication, and document-frequency boilerplate removal:
  `scripts/clean_corpus.py` and `corpus/cleaned/cleaning_stats.json`.
- [x] Hand-written idempotent Urdu normalization with the required letter,
  digit, tatweel, control, whitespace, and ZWNJ conventions: `normalize.py` and
  `tests/test_normalize.py`.
- [x] Fixed-seed document-level 98/1/1 source-stratified split with explicit EOT
  markers: `scripts/split_corpus.py`, `corpus/split_stats.json`, and
  `corpus/split_manifest.jsonl`.
- [x] No exact-document leakage: split SHA-256 sets are pairwise disjoint.

## Part B — tokenizer

- [x] Hand-written lossless byte BPE; no tokenizer library is used for training
  or encoding: `bpe.py`.
- [x] IDs 0–255 are bytes, ID 256 is `<|endoftext|>`, and exactly 49,743 ordered
  merges occupy IDs 257–49,999: `tokenizer.json`.
- [x] Urdu-aware regex keeps marks and internal ZWNJ, with leading-space
  attachment: `URDU_PRETOKEN_PATTERN` in `bpe.py`.
- [x] Deterministic maximum-frequency choice with lexicographically greater pair
  on ties: implementation plus `test_lexicographically_greater_pair_wins_tie`.
- [x] Slow recounting reference trainer and weighted incremental fast trainer:
  `train_reference` and `train_fast` in `bpe.py`.
- [x] Reference/fast proof: the same first 500 merges on a 1 MiB training sample;
  428.91 s versus 6.96 s, recorded in `tokenizer.json` metadata.
- [x] Encoding cache and streaming corpus reader: `bpe.py`.
- [x] Flat NumPy-compatible `uint16` outputs: `data/train.bin` has 88,404,583
  tokens and `data/val.bin` has 876,941; IDs are within 0–49,999. Checksums and
  timings are in `data/encoding_stats.json`.
- [x] Automated coverage: 23 focused unit tests plus 40 randomized equivalence
  checks; all passed.

## Part C and submission

- [x] Merges 1–50 and every 5,000th merge, early letter reconstruction,
  invalid-UTF-8 intermediate tokens, first attested space-prefixed whole Urdu
  word, and 20 longest tokens: `data/analysis_results.json` and report pages 2–3.
- [x] Held-out test plus seven required input categories compared with GPT-2 and
  `o200k_base`: report page 4 and `data/analysis_results.json`.
- [x] Three original weak inputs with token-piece explanations: report page 4.
- [x] Two generated plots: `data/merge_lengths.png` and
  `data/tokens_per_word.png`.
- [x] Report has one formal cover page followed by exactly four numbered A4
  content pages: `report.pdf` (duplicate under `report/report.pdf`).
- [x] Exact rebuild commands: `README.md`; corpus documentation: `DATA_CARD.md`;
  reference isolation: `ORIGINALITY.md`.

## Before submitting

- [x] Cover page identifies Ch Daniyal (453835), Malik Muhammad Hassaan
  (460233), and Syed Danish Abbas (463657).
- [ ] Upload `corpus/train.txt`, `corpus/val.txt`, `corpus/test.txt`,
  `data/train.bin`, and `data/val.bin` to the student's own shared folder and
  place its accessible link in the final submission.
