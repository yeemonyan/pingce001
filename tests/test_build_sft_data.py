import json
import unittest

from scripts.build_sft_data import (
    answer_count_instruction,
    build_answer_only_assistant,
    build_rationale_assistant,
    build_reasoning_short_assistant,
    build_sft_item,
    compact_constraints,
    normalize_system_prompt_schema,
    reasoning_steps,
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

    def test_reasoning_short_assistant_contains_steps_and_answer_kind(self):
        payload = json.loads(build_reasoning_short_assistant(self.record))
        self.assertEqual(payload["analysis"]["answer_kind"], "multi")
        self.assertEqual(payload["answers"], ["A", "C"])
        self.assertGreaterEqual(len(payload["analysis"]["reasoning_steps"]), 3)

    def test_answer_count_instruction_changes_with_answer_kind(self):
        multi_instruction = answer_count_instruction(self.record)
        single_instruction = answer_count_instruction({**self.record, "answers": ["A"], "answer": ["A"]})
        self.assertIn("multi-answer", multi_instruction.lower())
        self.assertIn("single-answer", single_instruction.lower())

    def test_reasoning_steps_are_domain_aware(self):
        steps = reasoning_steps(self.record)
        self.assertIn("timeline", steps[0].lower())
        self.assertIn("multi-answer", steps[-1].lower())

    def test_compact_constraints_limits_count(self):
        constraints = compact_constraints({"text": "1;2;3;4;5;6;7"}, limit=3)
        self.assertEqual(constraints, ["1", "2", "3"])

    def test_build_sft_item_and_validate(self):
        item = build_sft_item(self.record, "system prompt", "answer_only")
        self.assertEqual(item["answers"], ["A", "C"])
        self.assertEqual([message["role"] for message in item["messages"]], ["system", "user", "assistant"])
        self.assertIn("multi-answer question", item["messages"][1]["content"])
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
