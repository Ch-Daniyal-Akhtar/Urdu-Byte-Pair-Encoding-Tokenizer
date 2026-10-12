import unittest

from normalize import normalize


class NormalizeTests(unittest.TestCase):
    def test_required_urdu_letter_mappings(self) -> None:
        self.assertEqual(normalize("ي ى ك ه"), "ی ی ک ہ")

    def test_two_heh_letters_remain_distinct(self) -> None:
        self.assertEqual(normalize("ہ ھ"), "ہ ھ")

    def test_formatting_junk_is_removed(self) -> None:
        self.assertEqual(normalize("کـ\ufeffتاب\u200f"), "کتاب")

    def test_all_digits_use_ascii_convention(self) -> None:
        self.assertEqual(normalize("١٢٣ ۴۵۶ 789"), "123 456 789")

    def test_only_internal_zwnj_is_preserved(self) -> None:
        self.assertEqual(normalize("\u200cبے\u200c\u200cچین\u200c !"), "بے\u200cچین !")

    def test_stray_zwnj_next_to_marks_does_not_fail(self) -> None:
        self.assertEqual(normalize("\u200c\u0650\u200cلفظ"), "ِلفظ")

    def test_whitespace_is_canonical(self) -> None:
        self.assertEqual(normalize("  ایک\u00a0\tدو\r\n\r\n\r\n تین  "), "ایک دو\n\nتین")

    def test_nfc_combining_sequence(self) -> None:
        self.assertEqual(normalize("ا\u0653"), normalize(normalize("ا\u0653")))

    def test_empty_and_type_error(self) -> None:
        self.assertEqual(normalize(""), "")
        with self.assertRaises(TypeError):
            normalize(None)  # type: ignore[arg-type]

    def test_idempotent_mixed_sample(self) -> None:
        sample = "\ufeffتـحقیـق\u200f  يہ ۱۲٣\n\n\nبے\u200cچین ھ ہ"
        once = normalize(sample)
        self.assertEqual(normalize(once), once)


if __name__ == "__main__":
    unittest.main()
