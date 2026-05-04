import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import scripts.infer_score as infer_score
from scripts.infer_score import (
    MockBackend,
    extract_answer,
    infer_domain,
    run_inference,
    vote_answers,
)


class InferScoreTest(unittest.TestCase):
    def test_extract_answer_from_json(self):
        output = '{"answer":["B","A"]}'
        self.assertEqual(extract_answer(output, {"A", "B", "C"}), ["B", "A"])

    def test_extract_answer_from_text(self):
        output = "推理完成，最终答案是 A 和 C。"
        self.assertEqual(extract_answer(output, {"A", "B", "C", "D"}), ["A", "C"])

    def test_infer_domain_from_explicit_field(self):
        record = {"domain": "时间", "text": "", "question": ""}
        self.assertEqual(infer_domain(record), "temporal")

    def test_infer_domain_from_official_space_nature_field(self):
        record = {"domain": "space+nature", "text": "", "question": ""}
        self.assertEqual(infer_domain(record), "hybrid")

    def test_infer_hybrid_from_mixed_clues(self):
        record = {
            "text": "The animal is on the left shelf and watches TV 3 days after Monday.",
            "question": "Which item is correct?",
        }
        self.assertEqual(infer_domain(record), "hybrid")

    def test_run_inference_mock_computes_accuracy(self):
        records = [
            {
                "id": 1,
                "domain": "social",
                "text": "Given: A is B's friend.",
                "question": "Which is correct?",
                "options": {"A": "A is B's friend", "B": "A is B's father"},
                "answer": ["A"],
            }
        ]
        prompts = {"social": "social prompt", "general": "general prompt"}

        outputs, metrics = run_inference(records, prompts, MockBackend())

        self.assertEqual(outputs[0]["answer"], ["A"])
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_run_inference_force_domain_uses_general_prompt(self):
        records = [
            {
                "id": 1,
                "domain": "social",
                "text": "Given: A is B's friend.",
                "question": "Which is correct?",
                "options": {"A": "A is B's friend", "B": "A is B's father"},
                "answer": ["A"],
            }
        ]
        prompts = {"social": "social prompt", "general": "general prompt"}

        outputs, metrics = run_inference(records, prompts, MockBackend(), force_domain="general")

        self.assertEqual(outputs[0]["domain"], "social")
        self.assertEqual(outputs[0]["prompt_domain"], "general")
        self.assertEqual(metrics["prompt_domain_counts"], {"general": 1})

    def test_run_inference_fallback_general_maps_hybrid_prompt(self):
        records = [
            {
                "id": 1,
                "text": "The animal is on the left shelf and watches TV 3 days after Monday.",
                "question": "Which item is correct?",
                "options": {"A": "A", "B": "B"},
                "answer": ["A"],
            }
        ]
        prompts = {"hybrid": "hybrid prompt", "general": "general prompt"}

        outputs, metrics = run_inference(records, prompts, MockBackend(), fallback_general=True)

        self.assertEqual(outputs[0]["domain"], "hybrid")
        self.assertEqual(outputs[0]["prompt_domain"], "general")
        self.assertEqual(metrics["domain_counts"], {"hybrid": 1})
        self.assertEqual(metrics["prompt_domain_counts"], {"general": 1})

    def test_vote_answers_prefers_majority_sequence(self):
        voted = vote_answers([["A"], ["A"], ["B"]], ["A", "B", "C", "D"])
        self.assertEqual(voted, ["A"])

    def test_vote_answers_breaks_tie_by_label_frequency(self):
        voted = vote_answers([["A", "C"], ["A"], ["B"]], ["A", "B", "C", "D"])
        self.assertEqual(voted, ["A"])

    def test_run_inference_records_sample_answers(self):
        records = [
            {
                "id": 1,
                "domain": "social",
                "text": "Given: A is B's friend.",
                "question": "Which is correct?",
                "options": {"A": "A is B's friend", "B": "A is B's father"},
                "answer": ["A"],
            }
        ]
        prompts = {"social": "social prompt", "general": "general prompt"}

        outputs, metrics = run_inference(records, prompts, MockBackend(), num_samples=3)

        self.assertEqual(outputs[0]["answer"], ["A"])
        self.assertEqual(len(outputs[0]["sample_answers"]), 3)
        self.assertEqual(metrics["num_samples"], 3)

    def test_transformers_backend_requires_cuda_for_non_cpu_device_map(self):
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.float16 = "float16"
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.float32 = "float32"
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = object()
        fake_transformers.AutoTokenizer = object()
        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
                infer_score.TransformersBackend(
                    model_path="dummy-model",
                    adapter_path=None,
                    max_new_tokens=8,
                    temperature=0.0,
                    top_p=1.0,
                    dtype="float16",
                    device_map="auto",
                )


if __name__ == "__main__":
    unittest.main()
