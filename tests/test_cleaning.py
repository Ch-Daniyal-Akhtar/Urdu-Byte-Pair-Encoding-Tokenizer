import unittest

from scripts.clean_corpus import line_is_urdu, normalize, redact_personal_patterns, strip_wikitext


class CleaningTests(unittest.TestCase):
    def test_wiki_markup_is_removed_but_labels_remain(self) -> None:
        source = "== عنوان ==\nیہ [[پاکستان|ملک]] {{سانچہ|مواد}} خوبصورت ہے۔<ref>حوالہ</ref>"
        cleaned = strip_wikitext(source)
        self.assertIn("عنوان", cleaned)
        self.assertIn("یہ ملک", cleaned)
        self.assertNotIn("سانچہ", cleaned)
        self.assertNotIn("حوالہ", cleaned)

    def test_personal_patterns_are_redacted(self) -> None:
        text = "رابطہ person@example.com یا +92 300 1234567 اور 192.168.0.1"
        cleaned, count = redact_personal_patterns(text)
        self.assertEqual(count, 3)
        self.assertNotIn("example.com", cleaned)
        self.assertNotIn("1234567", cleaned)

    def test_urdu_density_keeps_code_mixing(self) -> None:
        self.assertTrue(line_is_urdu("یہ ہماری نئی AI research laboratory کی مکمل رپورٹ ہے۔"))
        self.assertFalse(line_is_urdu("Sign in and click the navigation menu to continue"))

    def test_cleaning_normalizer_is_idempotent(self) -> None:
        text = "يہ كـتاب ١٩٤٧ میں لکھی گئی۔"
        self.assertEqual(normalize(normalize(text)), normalize(text))


if __name__ == "__main__":
    unittest.main()
