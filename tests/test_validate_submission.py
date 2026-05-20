import unittest

from scripts.validate_submission import validate_submission_records


class ValidateSubmissionTest(unittest.TestCase):
    def test_accepts_complete_official_like_submission(self):
        records = [
            {"id": "SCoRE2026-test-1", "answers": ["A"]},
            {"id": "SCoRE2026-test-2", "answers": ["B", "C"]},
        ]

        report = validate_submission_records(records, expected_count=2)

        self.assertTrue(report["ok"])
        self.assertEqual(report["errors"], [])

    def test_rejects_missing_and_duplicate_ids(self):
        records = [
            {"id": "SCoRE2026-test-1", "answers": ["A"]},
            {"id": "SCoRE2026-test-1", "answers": ["B"]},
        ]

        report = validate_submission_records(records, expected_count=2)

        self.assertFalse(report["ok"])
        self.assertTrue(any("duplicate id" in error for error in report["errors"]))
        self.assertTrue(any("missing ids" in error for error in report["errors"]))

    def test_rejects_empty_answers(self):
        records = [{"id": "SCoRE2026-test-1", "answers": []}]

        report = validate_submission_records(records, expected_count=1)

        self.assertFalse(report["ok"])
        self.assertTrue(any("empty or invalid answers" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
