"""Render a cover page and an exact four-page academic assignment report."""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def mb(value: int | float) -> str:
    return f"{value / 1024 / 1024:.2f}"


def display_token(value: str) -> str:
    return html.escape(value.replace(" ", "␠").replace("\n", "↵").replace("\t", "⇥")) or "∅"


def browser_path() -> Path:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    located = shutil.which("msedge") or shutil.which("chrome") or shutil.which("chromium")
    if located:
        return Path(located)
    raise FileNotFoundError("Edge/Chrome/Chromium is required to render report.pdf")


def build_html() -> str:
    corpus = load("corpus/corpus_stats.json")
    cleaning = load("corpus/cleaned/cleaning_stats.json")
    tokenizer_file = load("tokenizer.json")
    encoding = load("data/encoding_stats.json")
    analysis = load("data/analysis_results.json")
    merge_data = analysis["merges"]
    comparison = analysis["comparison"]
    metadata = tokenizer_file["metadata"]

    source_details = {
        "wikipedia_api": ("Urdu Wikipedia API", "Encyclopedia", "CC BY-SA 4.0 / GFDL"),
        "wikisource_api": ("Urdu Wikisource API", "Literature", "CC BY-SA 4.0 + page notices"),
        "mc4_ur": ("Urdu mC4 (AllenAI C4)", "Diverse web", "ODC-BY; underlying rights vary"),
    }
    source_rows = "".join(
        f"<tr><td>{source_details[row['source']][0]}</td><td>{source_details[row['source']][1]}</td>"
        f"<td>{mb(row['raw_file_bytes'])}</td><td>{mb(row['retained_utf8_bytes'])}</td>"
        f"<td>{row['retained_documents']:,}</td><td>{source_details[row['source']][2]}</td></tr>"
        for row in cleaning["sources"]
    )

    first_50 = [row for row in merge_data["selected_merges"] if row["merge_index"] <= 50]
    merge_cells = []
    for start in range(0, 50, 10):
        merge_cells.append(
            "<tr>"
            + "".join(
                f"<td><b>#{row['merge_index']}</b> {display_token(row['decoded'])}<br>"
                f"<span class='muted'>{row['hex']}</span></td>"
                for row in first_50[start : start + 10]
            )
            + "</tr>"
        )
    every_5k = [row for row in merge_data["selected_merges"] if row["merge_index"] >= 5_000]
    every_5k_rows = "".join(
        f"<tr><td>{row['merge_index']:,}</td><td>{row['token_id']:,}</td>"
        f"<td class='rtl'>{display_token(row['decoded'])}</td><td>{row['byte_length']}</td></tr>"
        for row in every_5k
    )
    letters = merge_data["early_letter_merges"][:24]
    letter_text = " · ".join(
        f"<span class='rtl'>{display_token(row['decoded'])}</span> #{row['merge_index']}" for row in letters
    )
    first_word = merge_data["first_whole_urdu_word"]

    longest_rows = "".join(
        f"<tr><td>{rank}</td><td>{row['token_id']}</td><td class='rtl token'>{display_token(row['decoded'])}</td>"
        f"<td>{row['byte_length']}</td><td>{row['character_length'] or '-'}</td></tr>"
        for rank, row in enumerate(merge_data["longest_tokens"], start=1)
    )
    invalid_examples = " · ".join(
        f"#{row['merge_index']} <code>{row['hex']}</code>" for row in merge_data["invalid_utf8_examples"][:6]
    )

    test = comparison["test_set"]
    comparison_rows = "".join(
        f"<tr><td>{name}</td><td>{values['tokens']:,}</td><td>{values['tokens_per_word']:.3f}</td>"
        f"<td>{values['bytes_per_token']:.3f}</td></tr>"
        for name, values in test["tokenizers"].items()
    )
    handpicked_rows = ""
    for name, item in comparison["handpicked"].items():
        values = item["metrics"]["tokenizers"]
        handpicked_rows += (
            f"<tr><td>{name.replace('_', ' ')}</td><td>{item['metrics']['words']}</td>"
            f"<td>{values['ours']['tokens_per_word']:.2f}</td>"
            f"<td>{values['o200k_base']['tokens_per_word']:.2f}</td>"
            f"<td>{values['gpt2']['tokens_per_word']:.2f}</td></tr>"
        )
    failure_notes = {
        "roman_chat": "Training is overwhelmingly Perso-Arabic, so shortened Latin spellings and repeated punctuation fall back to small byte pieces.",
        "rare_diacritics": "The regex correctly keeps marks with the word, but rare fully-vowelled combinations did not occur often enough to earn long merges.",
        "technical_identifier": "Mixed identifiers are intentionally lossless, yet uncommon letter-digit-symbol sequences remain fragmented.",
    }
    failure_titles = {
        "roman_chat": "Case 1: Roman Urdu with shortened spellings",
        "rare_diacritics": "Case 2: Heavily vowelled Arabic-style text",
        "technical_identifier": "Case 3: Mixed technical identifier",
    }
    failure_blocks = ""
    for name, item in comparison["failure_cases"].items():
        values = item["metrics"]["tokenizers"]
        preview = " | ".join(display_token(piece) for piece in item["our_token_pieces"][:12])
        failure_blocks += (
            f"<h3>{failure_titles[name]}</h3>"
            f"<p class='sample rtl'>{html.escape(item['text'])}</p>"
            f"<p>Ours: {values['ours']['tokens_per_word']:.2f} tokens/word; "
            f"o200k_base: {values['o200k_base']['tokens_per_word']:.2f}; "
            f"GPT-2: {values['gpt2']['tokens_per_word']:.2f}. {failure_notes[name]}</p>"
            f"<p class='pieces'>Pieces: {preview}</p>"
        )

    ours = test["tokenizers"]["ours"]
    gpt2 = test["tokenizers"]["gpt2"]
    modern = test["tokenizers"]["o200k_base"]
    training_minutes = metadata["wall_clock_seconds"] / 60
    total_clean = corpus["total"]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 12mm 16mm; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; color:#202020; font-family:'Times New Roman','Noto Nastaliq Urdu',serif; font-size:8.65pt; line-height:1.23; }}
.page {{ height:273mm; page-break-after:always; overflow:hidden; position:relative; }}
.page:last-child {{ page-break-after:auto; }}
.running-head {{ display:flex; justify-content:space-between; border-bottom:.45px solid #a9a9a9; color:#666; font-size:7pt; padding-bottom:1mm; margin-bottom:5mm; }}
h1 {{ font-size:17pt; margin:0 0 2mm; color:#111; }}
h2 {{ font-size:11.5pt; margin:3mm 0 1mm; color:#111; }}
h3 {{ font-size:9pt; margin:2mm 0 .5mm; color:#111; }}
p {{ margin:.8mm 0 1.5mm; text-align:justify; }}
table {{ width:100%; border-collapse:collapse; margin:1mm 0 1.8mm; }}
th {{ background:#e8ebee; font-weight:bold; }}
th,td {{ border-top:.45px solid #aaa; border-bottom:.45px solid #d1d1d1; padding:.65mm 1mm; vertical-align:top; }}
thead tr:first-child th {{ border-top:1px solid #555; border-bottom:.7px solid #777; }}
tbody tr:last-child td {{ border-bottom:1px solid #555; }}
.compact td,.compact th {{ padding:.45mm .8mm; font-size:7.25pt; }}
.merge-grid {{ table-layout:fixed; }}
.merge-grid td {{ width:10%; font-size:6.25pt; padding:.55mm; border-left:.35px solid #ccc; }}
.muted {{ color:#666; font-size:6.25pt; }}
.rtl {{ direction:rtl; text-align:right; font-family:'Noto Nastaliq Urdu','Segoe UI',serif; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:4mm; align-items:start; }}
.plot {{ display:block; width:100%; max-height:61mm; object-fit:contain; }}
.caption {{ text-align:center; color:#666; font-size:7pt; font-style:italic; margin:.5mm 0 2mm; }}
.token {{ max-width:102mm; overflow-wrap:anywhere; }}
.sample {{ text-align:center; font-size:8.2pt; margin:.5mm 5mm; }}
.pieces {{ color:#666; font-size:6.8pt; margin-top:0; }}
.footer {{ position:absolute; bottom:0; width:100%; font-size:7.3pt; display:flex; justify-content:space-between; }}
code {{ font-family:Consolas,monospace; font-size:6.7pt; }}
ul {{ margin:1mm 0 1mm 4mm; padding-left:3mm; }} li {{ margin:.4mm 0; }}
.cover {{ text-align:center; padding-top:19mm; }}
.cover .university {{ color:#17365d; font-size:16pt; font-weight:bold; }}
.cover .school {{ font-size:11pt; margin-top:2mm; }}
.cover hr {{ border:0; border-top:1px solid #555; margin:10mm 8mm 13mm; }}
.cover .course {{ font-size:13pt; font-weight:bold; }}
.cover .term {{ font-size:10pt; font-style:italic; margin-top:1mm; }}
.cover .assignment {{ margin-top:18mm; color:#666; font-size:11pt; }}
.cover .title {{ font-size:23pt; font-weight:bold; line-height:1.2; margin:4mm auto 2mm; max-width:150mm; }}
.cover .subtitle {{ font-size:11pt; font-style:italic; }}
.members {{ width:80%; margin:16mm auto 9mm; text-align:left; font-size:9pt; }}
.members caption {{ font-weight:bold; font-size:10pt; margin-bottom:2mm; }}
.submitted {{ font-size:10pt; margin-top:7mm; }}
.links {{ margin-top:14mm; font-size:8.5pt; line-height:1.55; }}
.links a {{ color:#0563c1; text-decoration:underline; }}
</style></head><body>

<section class="page cover">
  <div class="university">National University of Sciences &amp; Technology</div>
  <div class="school">School of Electrical Engineering and Computer Science</div>
  <div class="school">Department of Computing</div>
  <hr>
  <div class="course">CS 417: Large Language Models (2+1)</div>
  <div class="term">Fall 2026</div>
  <div class="assignment">ASSIGNMENT 1</div>
  <div class="title">Building an Urdu Byte-Pair Encoding<br>Tokenizer from Scratch</div>
  <div class="subtitle">Assignment Report</div>
  <table class="members">
    <caption>Group Members</caption>
    <thead><tr><th>No.</th><th>Name</th><th>Registration No.</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>Ch Daniyal</td><td>453835</td></tr>
      <tr><td>2</td><td>Malik Muhammad Hassaan</td><td>460233</td></tr>
      <tr><td>3</td><td>Syed Danish Abbas</td><td>463657</td></tr>
    </tbody>
  </table>
  <div class="submitted"><b>Submitted to:</b>&nbsp;&nbsp; Dr. Naima Iltaf</div>
  <div class="links">
    <div><b>Code repository:</b>&nbsp; <a href="https://github.com/Ch-Daniyal-Akhtar/Urdu-Byte-Pair-Encoding-Tokenizer">https://github.com/Ch-Daniyal-Akhtar/Urdu-Byte-Pair-Encoding-Tokenizer</a></div>
    <div><b>Large data files:</b>&nbsp; <a href="https://drive.google.com/drive/folders/1Rdh_UYRDMJN5Psq1aVz167d3KMfgtiyc?usp=sharing">https://drive.google.com/drive/folders/1Rdh_UYRDMJN5Psq1aVz167d3KMfgtiyc</a></div>
  </div>
</section>

<section class="page">
  <div class="running-head"><span>CS 417 · Assignment 1</span><span>Urdu BPE Tokenizer from Scratch</span></div>
  <h1>Urdu BPE Tokenizer from Scratch</h1>
  <h2>1. Overview</h2>
  <p>This report describes a byte-level BPE tokenizer written without a tokenizer library. It was trained on a new Urdu corpus assembled for this submission. The corpus uses Urdu Wikipedia, Urdu Wikisource and Urdu mC4, so it does not reuse Daily Express, Daily Jang, BBC Urdu, Makhzan or CC-100 from the supplied solution. The final vocabulary contains exactly 50,000 token IDs.</p>
  <table class="compact"><thead><tr><th>Design choice</th><th>Implementation</th></tr></thead><tbody>
    <tr><td><b>Base vocabulary</b></td><td>IDs 0-255 are raw byte values. This makes every Unicode string lossless and removes the need for an unknown token.</td></tr>
    <tr><td><b>Special token</b></td><td><code>&lt;|endoftext|&gt;</code> has ID 256 and never enters pair counting or merging.</td></tr>
    <tr><td><b>Learned merges</b></td><td>49,743 ordered merges occupy IDs 257-49,999. A SHA-256 checksum protects their order in <code>tokenizer.json</code>.</td></tr>
    <tr><td><b>Tie-breaking</b></td><td>The pair with greatest frequency is selected; tied pairs use the lexicographically greater integer pair.</td></tr>
    <tr><td><b>Pre-tokenization</b></td><td>The Urdu-aware regex keeps combining marks and internal ZWNJ inside a word and attaches one leading space to the next unit.</td></tr>
    <tr><td><b>Training</b></td><td>The reference trainer recounts all pairs. The fast trainer uses weighted unique types, a pair-to-word index, local count updates and an encoding cache.</td></tr>
    <tr><td><b>Normalization</b></td><td>NFC normalization maps Arabic yeh, kaf and heh variants, uses ASCII digits, removes tatweel and controls, and preserves meaningful Urdu heh forms.</td></tr>
  </tbody></table>

  <h2>2. Corpus and Training Setup</h2>
  <p>The clean corpus contains <b>{total_clean['mebibytes']:.2f} MiB</b>, {total_clean['documents']:,} documents, {total_clean['words']:,} words and {total_clean['unique_words']:,} unique word forms. The two Wikimedia collectors produced 118.34 MiB of raw self-collected text with per-page provenance records.</p>
  <table class="compact"><thead><tr><th>Source</th><th>Genre</th><th>Raw MiB</th><th>Clean MiB</th><th>Documents</th><th>Licence or terms</th></tr></thead><tbody>{source_rows}</tbody></table>
  <p>Cleaning removes wiki and HTML markup, URLs, email addresses, phone-like strings and IP addresses. It applies the idempotent normalizer, filters low-Urdu lines, removes lines found in at least {cleaning['boilerplate_document_frequency_threshold']} documents, and removes exact normalized duplicates by SHA-256. A source-stratified shuffle with seed 4,172,026 produced a 98/1/1 document split. The three split hash sets are pairwise disjoint.</p>
  <table class="compact"><thead><tr><th>Item</th><th>Measured value</th></tr></thead><tbody>
    <tr><td>Training split</td><td>{corpus['train']['mebibytes']:.2f} MiB; {corpus['train']['documents']:,} documents; {corpus['train']['words']:,} words</td></tr>
    <tr><td>Validation and test</td><td>{corpus['val']['mebibytes']:.2f} MiB and {corpus['test']['mebibytes']:.2f} MiB</td></tr>
    <tr><td>Reference versus fast check</td><td>{metadata['reference_fast_verification']['merges_compared']} identical merges on {metadata['reference_fast_verification']['sample_bytes']/1024/1024:.2f} MiB; {metadata['reference_fast_verification']['reference_seconds']:.2f} s versus {metadata['reference_fast_verification']['fast_seconds']:.2f} s</td></tr>
    <tr><td>Full training</td><td>{training_minutes:.2f} minutes; {metadata['hardware']['logical_cpu_count']} logical CPUs; {metadata['hardware']['ram_gib']:.1f} GiB RAM; Python {metadata['hardware']['python']}</td></tr>
    <tr><td>Encoded output</td><td>{encoding['train']['tokens']:,} train IDs and {encoding['validation']['tokens']:,} validation IDs, stored as flat <code>uint16</code></td></tr>
  </tbody></table>
  <div class="footer"><span></span><span>1</span></div>
</section>

<section class="page">
  <div class="running-head"><span>CS 417 · Assignment 1</span><span>Urdu BPE Tokenizer from Scratch</span></div>
  <h2>3. Question 1: What Did the Merges Learn?</h2>
  <p>Urdu letters usually occupy two UTF-8 bytes. Early merges therefore rebuild letters before they learn complete words. In the tables below, <b>␠</b> denotes a leading space and byte escapes denote intermediate tokens that are not valid UTF-8 by themselves.</p>
  <h3>Merges 1-50</h3>
  <table class="merge-grid">{''.join(merge_cells)}</table>
  <div class="two">
    <div><h3>Token created at every 5,000th merge</h3>
      <table class="compact"><thead><tr><th>Merge</th><th>ID</th><th>Token</th><th>Bytes</th></tr></thead><tbody>{every_5k_rows}</tbody></table>
    </div>
    <div><h3>Early reconstruction</h3>
      <p><b>Letters rebuilt in the first 200 merges:</b> {letter_text}.</p>
      <p>The first attested space-prefixed whole Urdu word is <b class="rtl">{display_token(first_word['decoded'])}</b> at merge {first_word['merge_index']} (ID {first_word['token_id']}). Earlier merges mainly reconstruct individual letters and space-plus-lead-byte fragments.</p>
      <p>There are {merge_data['invalid_utf8_count']:,} learned tokens that are invalid UTF-8 alone. They are legal internal byte fragments and can be completed by later merges. Examples: {invalid_examples}.</p>
    </div>
  </div>
  <p>The progression is consistent with byte BPE. Frequent letter bytes are joined first, followed by common suffixes, function words and longer domain-specific strings. By merge 5,000, several regular Urdu words and names are already represented by single tokens.</p>
  <div class="footer"><span></span><span>2</span></div>
</section>

<section class="page">
  <div class="running-head"><span>CS 417 · Assignment 1</span><span>Urdu BPE Tokenizer from Scratch</span></div>
  <h2>3.1 Vocabulary Shape and Long Tokens</h2>
  <div class="two">
    <div><h3>Twenty longest learned tokens</h3><table class="compact"><thead><tr><th>#</th><th>ID</th><th>Token</th><th>B</th><th>Chars</th></tr></thead><tbody>{longest_rows}</tbody></table></div>
    <div><img class="plot" src="{(ROOT / 'data/merge_lengths.png').as_uri()}"><div class="caption">Figure 1. Learned token length across the 49,743 merge iterations.</div>
      <p>Later merges combine recurring morphemes, space-prefixed words, names, punctuation and web-specific strings. Very long tokens also reveal residual boilerplate or unusual web text. For this reason, the longest-token table is useful both as a tokenizer result and as a corpus-quality check.</p>
      <h3>Implementation checks</h3><ul><li>All 23 focused unit tests pass.</li><li>Forty randomized reference and fast trainer checks pass.</li><li>Empty strings, emoji, English, Urdu, newlines and the special marker round trip without loss.</li><li>Maximum encoded token ID is 49,999, which fits <code>uint16</code>.</li></ul>
    </div>
  </div>

  <h2>4. Question 2: Comparison with Existing Tokenizers</h2>
  <p>The held-out test split contains {test['words']:,} words and {mb(test['utf8_bytes'])} MiB after removing document markers. GPT-2 and <code>o200k_base</code> are used only for comparison, never for training or encoding our corpus.</p>
  <div class="two">
    <div><table class="compact"><thead><tr><th>Tokenizer</th><th>Total tokens</th><th>Tokens/word</th><th>Bytes/token</th></tr></thead><tbody>{comparison_rows}</tbody></table>
      <p>Our tokenizer uses {gpt2['tokens']/ours['tokens']:.2f} times fewer tokens than GPT-2 and {modern['tokens']/ours['tokens']:.2f} times fewer than <code>o200k_base</code> on this test set. The result reflects specialization to the training domains rather than a general ranking across languages.</p>
    </div>
    <div><img class="plot" src="{(ROOT / 'data/tokens_per_word.png').as_uri()}"><div class="caption">Figure 2. Tokens per word on the test split and seven required input types; lower is better.</div></div>
  </div>
  <div class="footer"><span></span><span>3</span></div>
</section>

<section class="page">
  <div class="running-head"><span>CS 417 · Assignment 1</span><span>Urdu BPE Tokenizer from Scratch</span></div>
  <h2>4.1 Results on the Seven Required Inputs</h2>
  <table class="compact"><thead><tr><th>Input type</th><th>Words</th><th>Ours</th><th>o200k_base</th><th>GPT-2</th></tr></thead><tbody>{handpicked_rows}</tbody></table>
  <p>News, poetry, dates and personal names benefit from the Urdu-focused vocabulary. Roman Urdu is less efficient because abbreviated Latin chat spellings are rare in the corpus. Fully vowelled Arabic-style text also needs more pieces because its combining-mark sequences are sparse.</p>

  <h2>5. Question 3: Inputs the Tokenizer Handles Badly</h2>
  {failure_blocks}

  <h2>6. Deliverables and Reproduction</h2>
  <p>The repository includes the normalizer, both BPE trainers, collection and cleaning scripts, deterministic splitter, streaming encoder, tests, <code>tokenizer.json</code>, data card, analysis JSON and this report. The large-file folder should contain <code>train.txt</code>, <code>val.txt</code>, <code>test.txt</code>, <code>train.bin</code> and <code>val.bin</code>.</p>
  <pre><code>python -m unittest discover -s tests -v
python train_tokenizer.py --input corpus/train.txt --vocab-size 50000
python encode_corpus.py --tokenizer tokenizer.json
python scripts/analyze_tokenizer.py
python scripts/generate_report_pdf.py</code></pre>
  <div class="footer"><span></span><span>4</span></div>
</section>
</body></html>"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORT_DIR / "report.html"
    pdf_path = REPORT_DIR / "report.pdf"
    html_path.write_text(build_html(), encoding="utf-8")
    profile = REPORT_DIR / ".browser-profile"
    profile.mkdir(exist_ok=True)
    command = [
        str(browser_path()),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--user-data-dir={profile.resolve()}",
        f"--print-to-pdf={pdf_path.resolve()}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=True, timeout=120)
    shutil.copy2(pdf_path, ROOT / "report.pdf")
    print(f"report={pdf_path} bytes={pdf_path.stat().st_size}")


if __name__ == "__main__":
    main()
