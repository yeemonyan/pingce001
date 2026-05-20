import unittest

from scripts.format_submission import format_submission


class FormatSubmissionTest(unittest.TestCase):
    def test_formats_answer_field_to_answers(self):
        records = [
            {"id": "SCoRE2026-test-1", "answer": ["B", "A"], "raw_output": "ignored"},
            {"id": "SCoRE2026-test-2", "answers": ["C"]},
        ]

        submission = format_submission(records)

        self.assertEqual(
            submission,
            [
                {"id": "SCoRE2026-test-1", "answers": ["B", "A"]},
                {"id": "SCoRE2026-test-2", "answers": ["C"]},
            ],
        )

    def test_rejects_empty_answers(self):
        with self.assertRaises(ValueError):
            format_submission([{"id": "SCoRE2026-test-1", "answer": []}])


if __name__ == "__main__":
    unittest.main()
