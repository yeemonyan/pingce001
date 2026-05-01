import unittest

from scripts.make_dev_split import (
    answer_kind,
    build_report,
    detect_language,
    split_records,
)


def record(index, domain, language, answers):
    text = "中文材料" if language == "zh" else "English text"
    return {
        "id": f"r{index}",
        "domain": domain,
        "text": text,
        "question": "问题" if language == "zh" else "Question",
        "options": {"A": "one", "B": "two"},
        "answer": answers,
        "has_answer": True,
    }


class MakeDevSplitTest(unittest.TestCase):
    def test_detect_language(self):
        self.assertEqual(detect_language({"text": "中文", "question": ""}), "zh")
        self.assertEqual(detect_language({"text": "English", "question": ""}), "en")

    def test_answer_kind(self):
        self.assertEqual(answer_kind({"answer": ["A"]}), "single")
        self.assertEqual(answer_kind({"answer": ["A", "B"]}), "multi")

    def test_split_is_stable_and_stratified(self):
        records = []
        index = 0
        for domain in ("temporal", "natural"):
            for language in ("zh", "en"):
                for answers in (["A"], ["A", "B"]):
                    for _ in range(10):
                        records.append(record(index, domain, language, answers))
                        index += 1

        train_a, dev_a = split_records(records, dev_ratio=0.2, seed=2026)
        train_b, dev_b = split_records(records, dev_ratio=0.2, seed=2026)

        self.assertEqual([item["id"] for item in train_a], [item["id"] for item in train_b])
        self.assertEqual([item["id"] for item in dev_a], [item["id"] for item in dev_b])
        self.assertEqual(len(train_a), 64)
        self.assertEqual(len(dev_a), 16)

        report = build_report(records, train_a, dev_a, seed=2026, dev_ratio=0.2)
        self.assertEqual(report["train_count"], 64)
        self.assertEqual(report["dev_count"], 16)
        self.assertEqual(report["dev"]["language"], {"en": 8, "zh": 8})
        self.assertEqual(report["dev"]["answer_kind"], {"multi": 8, "single": 8})


if __name__ == "__main__":
    unittest.main()
