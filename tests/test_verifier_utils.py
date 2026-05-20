import json
import unittest

from scripts.infer_verifier import GoldMockVerifierBackend, run_verifier
from scripts.verifier_utils import (
    build_option_prompt,
    build_verifier_item,
    detailed_metrics,
    extract_verdict,
    merge_option_predictions,
)


class VerifierUtilsTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "id": "q1",
            "domain": "temporal",
            "language": "en",
            "text": "A happens before B.",
            "question": "Select the correct statement(s).",
            "options": {"A": "A before B", "B": "B before A", "C": "A after B", "D": "None"},
            "answer": ["A"],
        }

    def test_build_verifier_item_yes_and_no(self):
        yes_item = build_verifier_item(self.record, "A")
        no_item = build_verifier_item(self.record, "B")

        self.assertEqual(yes_item["label"], "yes")
        self.assertEqual(no_item["label"], "no")
        self.assertEqual(yes_item["target_label"], "yes")
        self.assertEqual(no_item["target_label"], "no")
        self.assertEqual(json.loads(yes_item["messages"][2]["content"]), {"target_label": "yes"})

    def test_build_option_prompt_variants(self):
        all_options = build_option_prompt(self.record, "A")
        candidate_only = build_option_prompt(self.record, "A", include_all_options=False)
        hinted = build_option_prompt(self.record, "A", instruction_variant="domain_hint")

        self.assertIn("Options:", all_options)
        self.assertNotIn("Options:", candidate_only)
        self.assertIn("timeline", hinted)

    def test_extract_verdict_from_json_and_text(self):
        self.assertEqual(extract_verdict('{"label":"yes"}'), "yes")
        self.assertEqual(extract_verdict('{"target_label":"no"}'), "no")
        self.assertEqual(extract_verdict("not entailed"), "no")
        self.assertEqual(extract_verdict("ĊĠyes"), "yes")
        self.assertIsNone(extract_verdict("maybe"))

    def test_merge_option_predictions(self):
        merged = merge_option_predictions(
            [
                {"question_id": "q1", "domain": "temporal", "option_label": "A", "target_label": "yes", "gold_answer": ["A"]},
                {"question_id": "q1", "domain": "temporal", "option_label": "B", "target_label": "no", "gold_answer": ["A"]},
            ]
        )

        self.assertEqual(merged[0]["answer"], ["A"])

    def test_merge_option_predictions_can_use_threshold_scores(self):
        merged = merge_option_predictions(
            [
                {"question_id": "q1", "domain": "temporal", "option_label": "A", "target_label": "no", "yes_score": 0.7},
                {"question_id": "q1", "domain": "temporal", "option_label": "B", "target_label": "yes", "yes_score": 0.2},
            ],
            yes_threshold=0.5,
        )

        self.assertEqual(merged[0]["answer"], ["A"])

    def test_detailed_metrics_counts_over_under(self):
        gold = [
            {**self.record, "id": "q1", "answer": ["A"]},
            {**self.record, "id": "q2", "answer": ["A", "B"]},
        ]
        pred = [
            {"id": "q1", "answer": ["A", "B"]},
            {"id": "q2", "answer": ["A"]},
        ]

        report = detailed_metrics(gold, pred)

        self.assertEqual(report["single_vs_multi"]["single"]["accuracy"], 0.0)
        self.assertEqual(report["single_vs_multi"]["multi"]["accuracy"], 0.0)
        self.assertEqual(report["prediction_error_types"]["over_predict"], 1)
        self.assertEqual(report["prediction_error_types"]["under_predict"], 1)

    def test_run_verifier_mock_merges_gold_answers(self):
        option_outputs, merged = run_verifier([self.record], GoldMockVerifierBackend())

        self.assertEqual(len(option_outputs), 4)
        self.assertEqual(merged[0]["answer"], ["A"])


if __name__ == "__main__":
    unittest.main()
