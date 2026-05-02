import json
import unittest

from scripts.build_sft_data import (
    build_answer_only_assistant,
    build_rationale_assistant,
    build_sft_item,
    compact_constraints,
    normalize_system_prompt_schema,
    validate_items,
)


class BuildSftDataTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "id": "SCoRE2026-train-1",
            "domain": "temporal",
            "language": "en",
            "text": "A happened before B; C happened after B.",
            "question": "Which is correct?",
            "options": {"A": "A before B", "B": "B before A", "C": "C after B"},
            "answers": ["C", "A"],
            "answer": ["C", "A"],
        }

    def test_answer_only_assistant_uses_answers_key(self):
        payload = json.loads(build_answer_only_assistant(self.record))
        self.assertEqual(payload, {"answers": ["A", "C"]})

    def test_rationale_assistant_is_structured_json(self):
        payload = json.loads(build_rationale_assistant(self.record))
        self.assertEqual(payload["analysis"]["domain"], "temporal")
        self.assertEqual(payload["answers"], ["A", "C"])
        self.assertTrue(payload["analysis"]["key_constraints"])

    def test_compact_constraints_limits_count(self):
        constraints = compact_constraints({"text": "1;2;3;4;5;6;7"}, limit=3)
        self.assertEqual(constraints, ["1", "2", "3"])

    def test_build_sft_item_and_validate(self):
        item = build_sft_item(self.record, "system prompt", "answer_only")
        self.assertEqual(item["answers"], ["A", "C"])
        self.assertEqual([message["role"] for message in item["messages"]], ["system", "user", "assistant"])
        validate_items([item])

    def test_normalize_system_prompt_schema(self):
        prompt = 'Return only {"answer":["A"]}.'
        self.assertEqual(normalize_system_prompt_schema(prompt), 'Return only {"answers":["A"]}.')

    def test_validate_rejects_test_leak(self):
        item = build_sft_item({**self.record, "id": "SCoRE2026-test-1"}, "system prompt", "answer_only")
        with self.assertRaises(ValueError):
            validate_items([item])


if __name__ == "__main__":
    unittest.main()
