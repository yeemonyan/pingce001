import unittest

from scripts.audit_verifier_data import audit


class AuditVerifierDataTest(unittest.TestCase):
    def test_audit_passes_and_samples(self):
        items = []
        for qid, domain, answer_kind, gold in [
            ("q1", "temporal", "single", ["A"]),
            ("q2", "spatial", "multi", ["A", "C"]),
            ("q3", "hybrid", "single", ["D"]),
        ]:
            for label in ["A", "B", "C", "D"]:
                target = "yes" if label in gold else "no"
                items.append(
                    {
                        "id": f"{qid}::option::{label}",
                        "question_id": qid,
                        "domain": domain,
                        "answer_kind": answer_kind,
                        "gold_answer": gold,
                        "option_label": label,
                        "option_text": label,
                        "target_label": target,
                        "label": target,
                    }
                )

        report = audit(items, seed=1, sample_count=1, domain_count=1)

        self.assertTrue(report["validation"]["passed"])
        sampled = report["samples"]["single"][0]
        for label, target in sampled["targets"].items():
            expected = "yes" if label in sampled["gold_answer"] else "no"
            self.assertEqual(target, expected)
        self.assertEqual(len(report["samples"]["multi"]), 1)
        self.assertEqual(len(report["samples"]["hybrid"]), 1)

    def test_audit_catches_target_mismatch(self):
        items = [
            {
                "id": "q1::option::A",
                "question_id": "q1",
                "domain": "temporal",
                "answer_kind": "single",
                "gold_answer": ["A"],
                "option_label": "A",
                "target_label": "no",
                "label": "no",
            }
        ]

        report = audit(items, seed=1, sample_count=1, domain_count=1)

        self.assertFalse(report["validation"]["passed"])
        self.assertGreater(report["validation"]["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
