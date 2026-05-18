import json
import unittest

from scripts.build_verifier_data import (
    build_verifier_assistant,
    build_verifier_item,
    build_verifier_user_prompt,
    validate_items,
)


class BuildVerifierDataTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "id": "SCoRE2026-train-1",
            "domain": "temporal",
            "language": "en",
            "text": "A happened before B; C happened after B.",
            "question": "Which is correct?",
            "options": {"A": "A before B", "B": "B before A", "C": "C after B", "D": "A after C"},
            "answer": ["A", "C"],
        }

    def test_build_verifier_user_prompt_contains_candidate(self):
        prompt = build_verifier_user_prompt(self.record, "B")
        self.assertIn("Candidate Option:", prompt)
        self.assertIn("B. B before A", prompt)

    def test_build_verifier_assistant_yes_no(self):
        yes_payload = json.loads(build_verifier_assistant(self.record, "A"))
        no_payload = json.loads(build_verifier_assistant(self.record, "B"))
        self.assertEqual(yes_payload, {"label": "yes"})
        self.assertEqual(no_payload, {"label": "no"})

    def test_build_verifier_item_and_validate(self):
        item = build_verifier_item(self.record, "C")
        self.assertEqual(item["source_id"], "SCoRE2026-train-1")
        self.assertEqual(item["candidate_label"], "C")
        self.assertEqual(item["target_label"], "yes")
        self.assertEqual([message["role"] for message in item["messages"]], ["system", "user", "assistant"])
        validate_items([item])

    def test_validate_rejects_test_leak(self):
        item = build_verifier_item({**self.record, "id": "SCoRE2026-test-1"}, "A")
        with self.assertRaises(ValueError):
            validate_items([item])


if __name__ == "__main__":
    unittest.main()
