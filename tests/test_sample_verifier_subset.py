import unittest

from scripts.sample_verifier_subset import sample_by_question


class SampleVerifierSubsetTest(unittest.TestCase):
    def test_sample_by_question_keeps_complete_questions(self):
        records = [
            {"question_id": "q1", "option_label": "A"},
            {"question_id": "q1", "option_label": "B"},
            {"question_id": "q2", "option_label": "A"},
            {"question_id": "q2", "option_label": "B"},
            {"question_id": "q3", "option_label": "A"},
            {"question_id": "q3", "option_label": "B"},
        ]

        sampled = sample_by_question(records, questions=2, seed=2026)
        question_ids = {item["question_id"] for item in sampled}

        self.assertEqual(len(question_ids), 2)
        for question_id in question_ids:
            labels = [item["option_label"] for item in sampled if item["question_id"] == question_id]
            self.assertEqual(labels, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
