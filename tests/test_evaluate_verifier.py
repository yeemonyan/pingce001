import unittest

from scripts.evaluate_verifier import evaluate_verifier


class EvaluateVerifierTest(unittest.TestCase):
    def test_evaluate_verifier_tracks_answer_kind_and_error_modes(self):
        gold = [
            {"id": "1", "domain": "temporal", "answer": ["A"]},
            {"id": "2", "domain": "spatial", "answer": ["A", "C"]},
            {"id": "3", "domain": "natural", "answer": ["B"]},
        ]
        pred = [
            {"id": "1", "answers": ["A"]},
            {"id": "2", "answers": ["A"]},
            {"id": "3", "answers": ["A", "B"]},
        ]
        report = evaluate_verifier(gold, pred)
        self.assertEqual(report["correct"], 1)
        self.assertEqual(report["per_answer_kind"]["single"]["total"], 2)
        self.assertEqual(report["per_answer_kind"]["multi"]["total"], 1)
        self.assertEqual(report["error_modes"]["under_predict"], 1)
        self.assertEqual(report["error_modes"]["over_predict"], 1)


if __name__ == "__main__":
    unittest.main()
