import unittest

from scripts.infer_score import (
    MockBackend,
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


if __name__ == "__main__":
    unittest.main()
