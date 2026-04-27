import unittest

from scripts.evaluate_score import align_predictions, evaluate


class EvaluateScoreTest(unittest.TestCase):
    def test_aligns_by_id_when_available(self):
        gold = [{"id": "a", "answer": ["A"]}, {"id": "b", "answer": ["B"]}]
        pred = [{"id": "b", "answer": ["B"]}, {"id": "a", "answer": ["A"]}]

        aligned = align_predictions(gold, pred)

        self.assertEqual(aligned[0][1]["id"], "a")
        self.assertEqual(aligned[1][1]["id"], "b")

    def test_evaluate_accuracy_and_domain(self):
        gold = [
            {"id": "a", "domain": "spatial", "answer": ["A"]},
            {"id": "b", "domain": "spatial", "answer": ["B", "C"]},
        ]
        pred = [
            {"id": "a", "answer": ["A"]},
            {"id": "b", "answer": ["C", "B"]},
        ]

        report = evaluate(gold, pred)

        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["per_domain"]["spatial"]["accuracy"], 1.0)

    def test_evaluate_missing_prediction(self):
        gold = [{"id": "a", "answer": ["A"]}]
        pred = []

        report = evaluate(gold, pred)

        self.assertEqual(report["accuracy"], 0.0)
        self.assertEqual(report["missing_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
