import json
import tempfile
import unittest
from pathlib import Path

from scripts.infer_verifier import (
    build_candidate_prompt,
    merge_labels,
    parse_verifier_label,
)


class InferVerifierTest(unittest.TestCase):
    def test_build_candidate_prompt_mentions_label(self):
        record = {
            "text": "A before B.",
            "question": "Which option is supported?",
            "options": {"A": "A before B", "B": "B before A"},
        }
        prompt = build_candidate_prompt(record, "A")
        self.assertIn("Candidate Option:", prompt)
        self.assertIn("A. A before B", prompt)

    def test_parse_verifier_label(self):
        self.assertEqual(parse_verifier_label('{"label":"yes"}'), "yes")
        self.assertEqual(parse_verifier_label('{"label":"no"}'), "no")
        self.assertEqual(parse_verifier_label("yes"), "yes")
        self.assertEqual(parse_verifier_label("unknown"), "no")

    def test_merge_labels_uses_fallback(self):
        self.assertEqual(merge_labels([("A", "no"), ("B", "yes"), ("C", "no")]), ["B"])
        self.assertEqual(merge_labels([("A", "no"), ("B", "no")]), ["D"])


if __name__ == "__main__":
    unittest.main()
