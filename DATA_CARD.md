# DATA CARD — independently collected Urdu corpus

## Summary

| Clean UTF-8 | Documents | Words | Unique words | Self-collected raw | Split |
|---:|---:|---:|---:|---:|---|
| **609.05 MiB** | **129,770** | **74,627,570** | **857,782** | **118.34 MiB** | 98/1/1 by document, seed 4172026 |

The corpus is for CS 417 coursework and is not a copy of the supplied solution data. It excludes Express, Jang, BBC Urdu, Makhzan and CC-100. The MediaWiki API JSONL logs provide page URL, revision, collection time, byte count and SHA-256 evidence for the ≥50 MB self-collection requirement.

## Sources

| Source | Genre | URL | Licence / terms | Collected | Raw MiB | Clean MiB |
|---|---|---|---|---|---:|---:|
| Urdu Wikipedia API | Encyclopedia | https://ur.wikipedia.org/w/api.php | CC BY-SA 4.0 / GFDL | 2026-10-11 | 70.54 | 22.72 |
| Urdu Wikisource API | Literature | https://ur.wikisource.org/w/api.php | CC BY-SA 4.0; per-page/public-domain notices may apply | 2026-10-11 | 47.80 | 43.41 |
| Urdu mC4 (AllenAI C4) | Diverse web | https://huggingface.co/datasets/allenai/c4 | ODC-BY packaging; underlying page rights vary | 2026-10-11 | 560.00 | 542.91 |

## Cleaning and removals

Plain text extraction removes wiki templates/tables/references, HTML and URLs. The hand-written NFC normalizer maps Arabic yeh/kaf/heh lookalikes, preserves the two Urdu heh letters, removes tatweel/direction controls, keeps only internal ZWNJ, converts every digit family to ASCII, and canonicalizes whitespace. We redact email/phone/IP patterns, require Urdu-script density while retaining natural code mixing, remove lines appearing in at least 50 documents, and SHA-256-deduplicate normalized documents globally.

| Source | Markup removed MiB | Normalization/filter delta MiB | Non-Urdu lines | Boilerplate lines | Duplicate docs | Retained MiB |
|---|---:|---:|---:|---:|---:|---:|
| Urdu Wikipedia API | 44.55 | 0.45 | 120,696 | 27,744 | 4 | 22.72 |
| Urdu Wikisource API | 4.25 | 0.23 | 5,367 | 330 | 2 | 43.41 |
| Urdu mC4 (AllenAI C4) | 2.87 | 1.54 | 165,545 | 327 | 2 | 542.91 |

## Known gaps, biases, and responsible use

- mC4 is the largest component, so formal web prose and repeated web conventions dominate; conversational Urdu, regional dialects and Roman Urdu are underrepresented.
- Wikipedia adds encyclopedic/name vocabulary; Wikisource adds older literary spelling and highly vocalized Arabic passages. Source-stratified splitting keeps all three domains in each split but cannot remove source-family similarity.
- Automatic language and PII filters reduce risk but are imperfect. Public biographies are not treated as private messages; emails, phone-like strings and IP addresses are removed. The corpus stays within the course.
- mC4's dataset packaging is ODC-BY, but the creators do not own every underlying webpage. Wikimedia reuse must preserve attribution/share-alike obligations and page-specific notices. Do not publish the raw corpus without a fresh rights review.
