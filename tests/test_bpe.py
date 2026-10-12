import tempfile
import unittest
from collections import Counter
from pathlib import Path

from bpe import BPETokenizer, END_OF_TEXT, END_OF_TEXT_ID, FIRST_MERGE_ID, iter_documents


TRAINING_TEXT = (
    " اردو اُردُو بے\u200cچین کتاب کتاب\n"
    " اردو زبان خوبصورت ہے۔ English 2026 😀\n"
    f"{END_OF_TEXT}"
    " ایک دوسرا دستاویز کتاب اور زبان۔"
)


class BPETests(unittest.TestCase):
    def test_trailing_newline_after_final_eot_is_not_a_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            corpus.write_text(
                f"first{END_OF_TEXT}\nsecond{END_OF_TEXT}\n",
                encoding="utf-8",
            )
            documents = list(iter_documents(corpus))
            self.assertEqual(len(documents), 2)
            self.assertEqual([document.strip() for document in documents], ["first", "second"])

    def test_initial_vocabulary_layout(self) -> None:
        tokenizer = BPETokenizer()
        self.assertEqual(tokenizer.vocab_size, 257)
        self.assertEqual(tokenizer.token_bytes[0], b"\x00")
        self.assertEqual(tokenizer.token_bytes[255], b"\xff")
        self.assertNotIn(END_OF_TEXT_ID, tokenizer.token_bytes)

    def test_pretokenizer_keeps_diacritics_and_zwnj(self) -> None:
        tokenizer = BPETokenizer()
        text = " اُردُو بے\u200cچین 2026!\n"
        pieces = tokenizer.pre_tokenize(text)
        self.assertIn(" اُردُو", pieces)
        self.assertIn(" بے\u200cچین", pieces)
        self.assertEqual("".join(pieces), text)

    def test_lossless_before_and_after_training(self) -> None:
        tokenizer = BPETokenizer(vocab_size=300)
        inputs = ["", "کتاب", "English 😀\nاُردُو", END_OF_TEXT, f"x{END_OF_TEXT}y"]
        for value in inputs:
            self.assertEqual(tokenizer.decode(tokenizer.encode(value)), value)
        tokenizer.train_fast_text(TRAINING_TEXT, vocab_size=300)
        for value in inputs:
            self.assertEqual(tokenizer.decode(tokenizer.encode(value)), value)

    def test_special_token_never_joins_a_merge(self) -> None:
        tokenizer = BPETokenizer(vocab_size=280)
        tokenizer.train_fast_text(f"الف{END_OF_TEXT}الف", vocab_size=280)
        self.assertEqual(tokenizer.encode(END_OF_TEXT), [END_OF_TEXT_ID])
        self.assertTrue(all(END_OF_TEXT_ID not in pair for pair, _ in tokenizer.merges))

    def test_lexicographically_greater_pair_wins_tie(self) -> None:
        tokenizer = BPETokenizer(vocab_size=258)
        tokenizer.train_reference("abac", vocab_size=258)
        self.assertEqual(tokenizer.merges[0], ((ord("b"), ord("a")), FIRST_MERGE_ID))

    def test_fast_trainer_matches_reference(self) -> None:
        reference = BPETokenizer(vocab_size=330)
        fast = BPETokenizer(vocab_size=330)
        self.assertEqual(
            reference.train_reference(TRAINING_TEXT, vocab_size=330),
            fast.train_fast_text(TRAINING_TEXT, vocab_size=330),
        )

    def test_weighted_word_types_match_expanded_text(self) -> None:
        text = " کتاب کتاب زبان کتاب زبان"
        tokenizer_a = BPETokenizer(vocab_size=280)
        tokenizer_b = BPETokenizer(vocab_size=280)
        expanded = tokenizer_a.train_reference(text, vocab_size=280)
        pieces = tokenizer_b.pre_tokenize(text)
        weighted = tokenizer_b.train_fast(Counter(pieces), vocab_size=280)
        self.assertEqual(expanded, weighted)

    def test_save_load_and_checksum(self) -> None:
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train_fast_text(TRAINING_TEXT, vocab_size=300)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path, metadata={"purpose": "unit test"})
            restored = BPETokenizer.from_file(path)
            sample = "نئی کتاب 😀"
            self.assertEqual(restored.merges, tokenizer.merges)
            self.assertEqual(restored.decode(restored.encode(sample)), sample)


if __name__ == "__main__":
    unittest.main()
