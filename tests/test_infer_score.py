import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.infer_score import (
    MockBackend,
    TransformersBackend,
    answer_count_hint,
    build_user_prompt_with_count_hint,
    extract_answer,
    infer_domain,
    run_inference,
)


class InferScoreTest(unittest.TestCase):
    def test_extract_answer_from_json(self):
        output = '{"answer":["B","A"]}'
        self.assertEqual(extract_answer(output, {"A", "B", "C"}), ["B", "A"])

    def test_extract_answer_from_text(self):
        output = "推理完成，最终答案是 A 和 C。"
        self.assertEqual(extract_answer(output, {"A", "B", "C", "D"}), ["A", "C"])

    def test_build_user_prompt_with_answer_count_hint(self):
        record = {
            "text": "A happened before B.",
            "question": "Which is correct?",
            "options": {"A": "A before B", "B": "B before A"},
            "answer": ["A"],
            "language": "en",
        }
        prompt = build_user_prompt_with_count_hint(record, include_answer_count_hint=True)
        self.assertIn("single-answer question", prompt)

    def test_answer_count_hint_for_multi_zh(self):
        record = {
            "text": "甲在乙左边。",
            "question": "以下选项正确的是____",
            "options": {"A": "甲在左", "B": "乙在右"},
            "answer": ["A", "B"],
            "language": "zh",
        }
        hint = answer_count_hint(record)
        self.assertIn("多选题", hint)

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

    def test_run_inference_can_include_answer_count_hint(self):
        records = [
            {
                "id": 1,
                "domain": "temporal",
                "text": "A happened before B.",
                "question": "Which is correct?",
                "options": {"A": "A before B", "B": "B before A"},
                "answer": ["A"],
                "language": "en",
            }
        ]
        prompts = {"temporal": "temporal prompt", "general": "general prompt"}

        outputs, metrics = run_inference(records, prompts, MockBackend(), include_answer_count_hint=True)

        self.assertEqual(outputs[0]["answer"], ["A"])
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_transformers_backend_loads_adapter_when_requested(self):
        fake_tokenizer = SimpleNamespace()
        fake_model = object()
        fake_wrapped_model = object()

        with patch.dict(
            "sys.modules",
            {
                "torch": SimpleNamespace(float16="fp16", bfloat16="bf16", float32="fp32"),
                "transformers": SimpleNamespace(
                    AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: fake_tokenizer),
                    AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: fake_model),
                ),
                "peft": SimpleNamespace(
                    PeftModel=SimpleNamespace(from_pretrained=lambda model, adapter_path: fake_wrapped_model)
                ),
            },
        ):
            backend = TransformersBackend(
                model_path="models/base",
                adapter_path="checkpoints/adapter",
                max_new_tokens=32,
                temperature=0.0,
                top_p=1.0,
                dtype="bfloat16",
                device_map="auto",
            )

        self.assertIs(backend.tokenizer, fake_tokenizer)
        self.assertIs(backend.model, fake_wrapped_model)


if __name__ == "__main__":
    unittest.main()
