import unittest

from scripts.analyze_errors import analyze, classify_failure


class AnalyzeErrorsTest(unittest.TestCase):
    def test_classifies_single_to_multi(self):
        gold = {"domain": "temporal", "text": "", "question": "", "options": {}}
        self.assertEqual(classify_failure(gold, {"answer": ["A", "B"]}, ["A"], ["A", "B"]), "single_to_multi")

    def test_classifies_multi_missing(self):
        gold = {"domain": "natural", "text": "", "question": "", "options": {}}
        self.assertEqual(classify_failure(gold, {"answer": ["A"]}, ["A", "C"], ["A"]), "multi_missing")

    def test_analyze_counts_domain_and_failures(self):
        gold = [
            {"id": "1", "domain": "temporal", "answer": ["A"], "text": "A before B", "question": "q", "options": {"A": "x", "B": "y"}},
            {"id": "2", "domain": "natural", "answer": ["A", "B"], "text": "food item", "question": "q", "options": {"A": "x", "B": "y"}},
        ]
        pred = [
            {"id": "1", "answer": ["A", "B"]},
            {"id": "2", "answer": ["A"]},
        ]
        report, cases = analyze(gold, pred, max_cases_per_type=5)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["correct"], 0)
        self.assertEqual(report["failure_counts"]["single_to_multi"], 1)
        self.assertEqual(report["failure_counts"]["multi_missing"], 1)
        self.assertEqual(len(cases), 2)


if __name__ == "__main__":
    unittest.main()
