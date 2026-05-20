import unittest

from scripts.selective_vote_infer import (
    aggregate_votes,
    build_system_prompt_variants,
    is_multi_answer_like,
    none_of_above_label,
    run_selective_vote,
    should_vote,
)
from scripts.infer_score import MockBackend


class SelectiveVoteInferTest(unittest.TestCase):
    def test_should_vote_for_temporal_statement_question(self):
        record = {
            "domain": "temporal",
            "text": "A happens 3 days before B.",
            "question": "Select the correct statement(s): ____",
            "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
        }

        use_vote, reason = should_vote(record, ["A"])

        self.assertTrue(use_vote)
        self.assertEqual(reason, "temporal_uncertain")

    def test_multi_answer_like_detects_statement_cues(self):
        record = {
            "text": "",
            "question": "以下选项正确的是____",
            "options": {"A": "x", "B": "y"},
        }
        self.assertTrue(is_multi_answer_like(record))

    def test_aggregate_votes_for_multi_answer_like_question(self):
        record = {
            "text": "",
            "question": "Select the correct statement(s): ____",
            "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
        }

        final_prediction = aggregate_votes(
            record,
            [["A"], ["A", "C"], ["C"]],
            multi_min_votes=2,
            none_margin=2,
        )

        self.assertEqual(final_prediction, ["A", "C"])

    def test_aggregate_votes_for_single_answer_uses_majority(self):
        record = {
            "text": "The answer is unique.",
            "question": "Which option is correct?",
            "options": {"A": "x", "B": "y", "C": "z"},
        }

        final_prediction = aggregate_votes(
            record,
            [["A"], ["B"], ["B"]],
            multi_min_votes=2,
            none_margin=2,
        )

        self.assertEqual(final_prediction, ["B"])

    def test_aggregate_votes_tie_falls_back_to_first_prediction(self):
        record = {
            "text": "The answer is unique.",
            "question": "Which option is correct?",
            "options": {"A": "x", "B": "y", "C": "z"},
        }

        final_prediction = aggregate_votes(
            record,
            [["A"], ["B"]],
            multi_min_votes=2,
            none_margin=2,
        )

        self.assertEqual(final_prediction, ["A"])

    def test_none_of_above_margin_falls_back_to_first_prediction(self):
        record = {
            "text": "The answer is unique.",
            "question": "Which option is correct?",
            "options": {"A": "x", "B": "y", "C": "z", "D": "None of the above"},
        }

        final_prediction = aggregate_votes(
            record,
            [["B"], ["D"], ["D"]],
            multi_min_votes=2,
            none_margin=2,
        )

        self.assertEqual(final_prediction, ["B"])

    def test_should_not_vote_for_social_domain(self):
        record = {
            "domain": "social",
            "text": "Tom is Jerry's friend.",
            "question": "Who is Jerry's friend?",
            "options": {"A": "Tom", "B": "Amy"},
        }

        use_vote, reason = should_vote(record, ["A"])

        self.assertFalse(use_vote)
        self.assertEqual(reason, "domain_off")

    def test_build_system_prompt_variants_dedupes_general_fallback(self):
        variants = build_system_prompt_variants(
            {"general": "g1", "spatial": "s1"},
            [{"general": "g1", "spatial": "s2"}, {"general": "g3"}],
            "spatial",
        )

        self.assertEqual(variants, ["s1", "s2", "g3"])

    def test_run_selective_vote_tracks_vote_reasons(self):
        record = {
            "id": "demo-1",
            "domain": "hybrid",
            "text": "A is left of B and before C.",
            "question": "Select the correct statement(s): ____",
            "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
            "answer": ["A"],
        }
        backend = MockBackend()

        outputs, metrics = run_selective_vote(
            [record],
            {"general": "g1", "hybrid": "h1"},
            [{"general": "g2", "hybrid": "h2"}],
            backend,
            multi_min_votes=2,
            none_margin=2,
        )

        self.assertEqual(outputs[0]["answer"], ["A"])
        self.assertTrue(outputs[0]["vote_used"])
        self.assertEqual(metrics["vote_used"], 1)
        self.assertEqual(metrics["domain_vote_used"], {"hybrid": 1})
        self.assertEqual(metrics["vote_reason_counts"], {"hybrid_default": 1})
        self.assertEqual(metrics["domain_counts"], {"hybrid": 1})
        self.assertEqual(none_of_above_label(record), None)


if __name__ == "__main__":
    unittest.main()
