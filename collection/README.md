# New corpus sources

This project intentionally does **not** use the reference solution's Express,
Jang, BBC Urdu, Makhzan, or CC-100 text. Its sources are:

1. Urdu Wikipedia through the official MediaWiki API (encyclopedia; self-collected).
2. Urdu Wikisource through the official MediaWiki API (literature; self-collected).
3. The Urdu portion of multilingual C4 / mC4 (diverse web prose; published corpus).

Every API page and extracted mC4 row receives a JSONL provenance record.
The Wikimedia API collection is sequential and resumable and honors `maxlag` and
`Retry-After`. The exact commands are documented in the project README.
