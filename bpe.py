"""A byte-level BPE tokenizer implemented from first principles.

No tokenizer package is used here. IDs 0..255 are raw bytes, ID 256 is the
literal ``<|endoftext|>`` marker, and learned merges start at ID 257.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import regex


END_OF_TEXT = "<|endoftext|>"
END_OF_TEXT_ID = 256
FIRST_MERGE_ID = 257

# A letter together with any combining marks remains one pre-token. Internal
# ZWNJ-separated components also remain together. An optional ASCII space is
# attached to the following word/number/punctuation unit, GPT-2 style.
URDU_PRETOKEN_PATTERN = (
    r" ?[\p{L}\p{M}]+(?:\u200c[\p{L}\p{M}]+)*"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{M}\p{N}]+"
    r"|\r\n|[\r\n]|[^\S\r\n]+"
)

Pair = tuple[int, int]
Merge = tuple[Pair, int]
ProgressCallback = Callable[[int, int, Pair, int], None]


def _replace_pair(sequence: Sequence[int], pair: Pair, replacement: int) -> tuple[int, ...]:
    """Replace non-overlapping occurrences from left to right."""

    output: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index] == pair[0] and sequence[index + 1] == pair[1]:
            output.append(replacement)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return tuple(output)


def _adjacent_counts(sequence: Sequence[int]) -> Counter[Pair]:
    return Counter(zip(sequence, sequence[1:]))


def iter_documents(path: str | Path, *, chunk_chars: int = 4 * 1024 * 1024) -> Iterator[str]:
    """Stream documents from an assignment-format corpus without splitting one."""

    carry = ""
    with Path(path).open("r", encoding="utf-8", errors="strict", newline="") as stream:
        while block := stream.read(chunk_chars):
            carry += block
            while END_OF_TEXT in carry:
                document, carry = carry.split(END_OF_TEXT, 1)
                yield document
        # A formatting newline after the final EOT marker is not a document.
        if carry.strip():
            yield carry


class BPETokenizer:
    """Urdu-aware byte BPE with deterministic reference and fast trainers."""

    def __init__(self, vocab_size: int = 50_000, pattern: str = URDU_PRETOKEN_PATTERN) -> None:
        if vocab_size < FIRST_MERGE_ID:
            raise ValueError("vocab_size must be at least 257")
        self.target_vocab_size = vocab_size
        self.pattern = pattern
        self._pretokens = regex.compile(pattern)
        self.merges: list[Merge] = []
        self.merge_ranks: dict[Pair, int] = {}
        self.token_bytes: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        self.word_cache: dict[str, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        return 257 + len(self.merges)

    def pre_tokenize(self, text: str) -> list[str]:
        pieces = self._pretokens.findall(text)
        if "".join(pieces) != text:
            raise RuntimeError("pre-tokenizer did not cover every input character")
        return pieces

    def _reset_merges(self) -> None:
        self.merges.clear()
        self.merge_ranks.clear()
        self.token_bytes = {index: bytes([index]) for index in range(256)}
        self.word_cache.clear()

    def _record_merge(self, pair: Pair, token_id: int) -> None:
        expected = FIRST_MERGE_ID + len(self.merges)
        if token_id != expected:
            raise ValueError(f"merge ID {token_id} is not sequential; expected {expected}")
        self.merges.append((pair, token_id))
        self.merge_ranks[pair] = token_id
        self.token_bytes[token_id] = self.token_bytes[pair[0]] + self.token_bytes[pair[1]]

    @staticmethod
    def _word_frequencies(documents: Iterable[str], pattern: regex.Pattern) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for document in documents:
            frequencies.update(pattern.findall(document))
        return frequencies

    def train_reference(
        self,
        text: str,
        *,
        vocab_size: int | None = None,
        min_frequency: int = 1,
    ) -> list[Merge]:
        """Slow reference algorithm: recount every occurrence after every merge."""

        target = vocab_size or self.target_vocab_size
        if target < FIRST_MERGE_ID:
            raise ValueError("target vocabulary must be at least 257")
        words = [tuple(piece.encode("utf-8")) for part in text.split(END_OF_TEXT) for piece in self.pre_tokenize(part)]
        self._reset_merges()

        for token_id in range(FIRST_MERGE_ID, target):
            pair_counts: Counter[Pair] = Counter()
            for word in words:
                pair_counts.update(zip(word, word[1:]))
            if not pair_counts:
                break
            pair, count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            if count < min_frequency:
                break
            words = [_replace_pair(word, pair, token_id) for word in words]
            self._record_merge(pair, token_id)
        return list(self.merges)

    def train_fast(
        self,
        word_frequencies: Mapping[str, int],
        *,
        vocab_size: int | None = None,
        min_frequency: int = 1,
        progress: ProgressCallback | None = None,
    ) -> list[Merge]:
        """Train with weighted word types and a pair-to-word inverted index."""

        target = vocab_size or self.target_vocab_size
        if target < FIRST_MERGE_ID:
            raise ValueError("target vocabulary must be at least 257")
        self._reset_merges()

        words = [tuple(word.encode("utf-8")) for word in word_frequencies]
        weights = [word_frequencies[word] for word in word_frequencies]
        pair_counts: dict[Pair, int] = defaultdict(int)
        pair_words: dict[Pair, set[int]] = defaultdict(set)

        for word_index, (sequence, weight) in enumerate(zip(words, weights)):
            local = _adjacent_counts(sequence)
            for pair, occurrences in local.items():
                pair_counts[pair] += occurrences * weight
                pair_words[pair].add(word_index)

        heap: list[tuple[int, int, int, Pair]] = [
            (-count, -pair[0], -pair[1], pair) for pair, count in pair_counts.items() if count > 0
        ]
        heapq.heapify(heap)
        total_merges = target - FIRST_MERGE_ID

        for step, token_id in enumerate(range(FIRST_MERGE_ID, target), start=1):
            selected: Pair | None = None
            selected_count = 0
            while heap:
                negative_count, _, _, candidate = heapq.heappop(heap)
                current = pair_counts.get(candidate, 0)
                if current > 0 and current == -negative_count:
                    selected = candidate
                    selected_count = current
                    break
            if selected is None or selected_count < min_frequency:
                break

            affected = tuple(pair_words.get(selected, ()))
            changed_pairs: set[Pair] = set()
            for word_index in affected:
                before = words[word_index]
                before_counts = _adjacent_counts(before)
                after = _replace_pair(before, selected, token_id)
                after_counts = _adjacent_counts(after)
                words[word_index] = after

                before_pairs = set(before_counts)
                after_pairs = set(after_counts)
                for pair in before_pairs - after_pairs:
                    pair_words[pair].discard(word_index)
                for pair in after_pairs:
                    pair_words[pair].add(word_index)

                weight = weights[word_index]
                for pair in before_pairs | after_pairs:
                    difference = after_counts.get(pair, 0) - before_counts.get(pair, 0)
                    if difference:
                        pair_counts[pair] += difference * weight
                        changed_pairs.add(pair)

            self._record_merge(selected, token_id)
            for pair in changed_pairs:
                count = pair_counts.get(pair, 0)
                if count > 0:
                    heapq.heappush(heap, (-count, -pair[0], -pair[1], pair))
            if progress is not None:
                progress(step, total_merges, selected, selected_count)

        return list(self.merges)

    def train_fast_text(self, text: str, **kwargs: object) -> list[Merge]:
        documents = text.split(END_OF_TEXT)
        frequencies = self._word_frequencies(documents, self._pretokens)
        return self.train_fast(frequencies, **kwargs)

    def train_fast_file(self, path: str | Path, **kwargs: object) -> list[Merge]:
        frequencies = self._word_frequencies(iter_documents(path), self._pretokens)
        return self.train_fast(frequencies, **kwargs)

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        cached = self.word_cache.get(piece)
        if cached is not None:
            return cached
        sequence: tuple[int, ...] = tuple(piece.encode("utf-8"))
        while len(sequence) > 1:
            ranked = ((self.merge_ranks[pair], pair) for pair in zip(sequence, sequence[1:]) if pair in self.merge_ranks)
            try:
                _, pair = min(ranked)
            except ValueError:
                break
            sequence = _replace_pair(sequence, pair, self.merge_ranks[pair])
        self.word_cache[piece] = sequence
        return sequence

    def _encode_plain(self, text: str) -> list[int]:
        encoded: list[int] = []
        for piece in self.pre_tokenize(text):
            encoded.extend(self._encode_piece(piece))
        return encoded

    def encode(self, text: str, *, allow_special: bool = True) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("encode() expects str input")
        if not text:
            return []
        if not allow_special or END_OF_TEXT not in text:
            return self._encode_plain(text)

        result: list[int] = []
        cursor = 0
        for match in regex.finditer(regex.escape(END_OF_TEXT), text):
            result.extend(self._encode_plain(text[cursor : match.start()]))
            result.append(END_OF_TEXT_ID)
            cursor = match.end()
        result.extend(self._encode_plain(text[cursor:]))
        return result

    def decode(self, token_ids: Iterable[int]) -> str:
        output: list[str] = []
        pending = bytearray()
        for token_id in token_ids:
            if token_id == END_OF_TEXT_ID:
                if pending:
                    output.append(pending.decode("utf-8", errors="strict"))
                    pending.clear()
                output.append(END_OF_TEXT)
            elif token_id in self.token_bytes:
                pending.extend(self.token_bytes[token_id])
            else:
                raise ValueError(f"unknown token ID: {token_id}")
        if pending:
            output.append(pending.decode("utf-8", errors="strict"))
        return "".join(output)

    def save(self, path: str | Path, *, metadata: Mapping[str, object] | None = None) -> None:
        payload: dict[str, object] = {
            "format": "urdu-byte-bpe-v1",
            "target_vocab_size": self.target_vocab_size,
            "actual_vocab_size": self.vocab_size,
            "special_token": {"text": END_OF_TEXT, "id": END_OF_TEXT_ID},
            "pretoken_pattern": self.pattern,
            "merges": [{"pair": list(pair), "id": token_id} for pair, token_id in self.merges],
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        canonical = json.dumps(payload["merges"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        payload["merges_sha256"] = hashlib.sha256(canonical).hexdigest()
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    @classmethod
    def from_file(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(
            vocab_size=int(payload["target_vocab_size"]),
            pattern=str(payload["pretoken_pattern"]),
        )
        for item in payload["merges"]:
            tokenizer._record_merge(tuple(item["pair"]), int(item["id"]))
        expected_hash = payload.get("merges_sha256")
        if expected_hash:
            canonical = json.dumps(payload["merges"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if hashlib.sha256(canonical).hexdigest() != expected_hash:
                raise ValueError("tokenizer merge checksum does not match")
        return tokenizer
