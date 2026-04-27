import unittest

from scripts.parse_score_json import ScoreFormatError, normalize_record


class ParseScoreJsonTest(unittest.TestCase):
    def test_normalize_record_with_legacy_answer(self):
        record = {
            "text": "Given: A is left of B.",
            "question": "A is ___ B.",
            "options": {"A": "left of", "B": "right of"},
            "answer": ["A"],
        }

        result = normalize_record(record, 0)

        self.assertEqual(result["id"], 0)
        self.assertEqual(result["answer"], ["A"])
        self.assertTrue(result["has_answer"])
        self.assertIn("A. left of", result["prompt"])

    def test_normalize_record_with_official_answers(self):
        record = {
            "id": "SCoRE2026-train-1",
            "text": "Given: A is left of B.",
            "question": "Which options are correct?",
            "options": {"A": "left of", "B": "right of", "C": "overlap"},
            "answers": ["a", "C"],
        }

        result = normalize_record(record, 0)

        self.assertEqual(result["id"], "SCoRE2026-train-1")
        self.assertEqual(result["answer"], ["A", "C"])
        self.assertTrue(result["has_answer"])

    def test_answer_is_optional_for_test_data(self):
        record = {
            "id": "test-1",
            "text": "A person has one father.",
            "question": "The statement is ___.",
            "options": {"A": "true", "B": "false"},
        }

        result = normalize_record(record, 3)

        self.assertEqual(result["id"], "test-1")
        self.assertIsNone(result["answer"])
        self.assertFalse(result["has_answer"])

    def test_rejects_invalid_option_label(self):
        record = {
            "text": "x",
            "question": "y",
            "options": {"E": "bad", "A": "ok"},
        }

        with self.assertRaises(ScoreFormatError):
            normalize_record(record, 0)


if __name__ == "__main__":
    unittest.main()
