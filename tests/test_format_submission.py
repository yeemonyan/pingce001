import unittest

from scripts.format_submission import convert_records, normalize_answer


class FormatSubmissionTest(unittest.TestCase):
    def test_normalize_answer_keeps_valid_unique_labels(self):
        self.assertEqual(normalize_answer(["A", "A", "c", "Z"], "D"), ["A", "C"])

    def test_normalize_answer_uses_fallback_for_empty(self):
        self.assertEqual(normalize_answer([], "D"), ["D"])

    def test_convert_records_default_shape(self):
        records = [{"id": "x", "answer": ["B"]}]
        self.assertEqual(convert_records(records, include_id=False, fallback="D"), [{"answer": ["B"]}])

    def test_convert_records_can_keep_id(self):
        records = [{"id": "x", "answer": ["B"]}]
        self.assertEqual(
            convert_records(records, include_id=True, fallback="D"),
            [{"answer": ["B"], "id": "x"}],
        )


if __name__ == "__main__":
    unittest.main()
