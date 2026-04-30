import unittest

from scripts.format_submission import convert_records, normalize_answer


class FormatSubmissionTest(unittest.TestCase):
    def test_normalize_answer_keeps_valid_unique_labels(self):
        self.assertEqual(normalize_answer(["A", "A", "c", "Z"], "D"), ["A", "C"])

    def test_normalize_answer_uses_fallback_for_empty(self):
        self.assertEqual(normalize_answer([], "D"), ["D"])

    def test_convert_records_default_shape(self):
        records = [{"id": "x", "answer": ["B"]}]
        self.assertEqual(
            convert_records(records, official_format="official_json", fallback="D"),
            [{"id": "x", "answers": ["B"]}],
        )

    def test_convert_records_accepts_answers_field(self):
        records = [{"id": "x", "answers": ["A", "C"]}]
        self.assertEqual(
            convert_records(records, official_format="official_json", fallback="D"),
            [{"id": "x", "answers": ["A", "C"]}],
        )

    def test_convert_records_can_write_jsonl_with_id_shape(self):
        records = [{"id": "x", "answer": ["B"]}]
        self.assertEqual(
            convert_records(records, official_format="jsonl_with_id", fallback="D"),
            [{"id": "x", "answer": ["B"]}],
        )

    def test_convert_records_can_write_system_json_shape(self):
        records = [{"id": "x", "answer": ["B"]}]
        self.assertEqual(
            convert_records(records, official_format="system_json", fallback="D"),
            [{"answer": ["B"]}],
        )


if __name__ == "__main__":
    unittest.main()
