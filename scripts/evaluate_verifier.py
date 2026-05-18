"""Evaluate verifier-style question-level predictions for SCoRE2026."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_score import align_predictions, load_json_or_jsonl, normalize_answer


def answer_kind(answer: list[str]) -> str:
    return "multi" if len(answer) > 1 else "single"


def evaluate_verifier(gold_records: list[dict[str, Any]], pred_records: list[dict[str, Any]]) -> dict[str, Any]:
    aligned = align_predictions(gold_records, pred_records)
    total = 0
    correct = 0
    invalid_gold = 0
    missing_predictions = 0
    per_domain_total: Counter[str] = Counter()
    per_domain_correct: Counter[str] = Counter()
    per_kind_total: Counter[str] = Counter()
    per_kind_correct: Counter[str] = Counter()
    error_modes: Counter[str] = Counter()
    mistakes: list[dict[str, Any]] = []

    for index, (gold, pred) in enumerate(aligned):
        gold_answer = normalize_answer(gold.get("answer"))
        if not gold_answer:
            invalid_gold += 1
            continue
        pred_answer = normalize_answer((pred or {}).get("answers", (pred or {}).get("answer")))
        if pred is None:
            missing_predictions += 1

        domain = str(gold.get("domain") or "unknown")
        kind = answer_kind(gold_answer)
        matched = set(gold_answer) == set(pred_answer)

        total += 1
        correct += int(matched)
        per_domain_total[domain] += 1
        per_domain_correct[domain] += int(matched)
        per_kind_total[kind] += 1
        per_kind_correct[kind] += int(matched)

        if matched:
            continue

        gold_set = set(gold_answer)
        pred_set = set(pred_answer)
        if not pred_answer:
            mode = "empty_prediction"
        elif pred_set > gold_set:
            mode = "over_predict"
        elif pred_set < gold_set:
            mode = "under_predict"
        else:
            mode = "wrong_combination"
        error_modes[mode] += 1
        mistakes.append(
            {
                "index": index,
                "id": gold.get("id"),
                "domain": domain,
                "answer_kind": kind,
                "gold": gold_answer,
                "prediction": pred_answer,
                "error_mode": mode,
            }
        )

    per_domain = {
        domain: {
            "total": count,
            "correct": per_domain_correct[domain],
            "accuracy": per_domain_correct[domain] / count if count else None,
        }
        for domain, count in per_domain_total.items()
    }
    per_kind = {
        kind: {
            "total": count,
            "correct": per_kind_correct[kind],
            "accuracy": per_kind_correct[kind] / count if count else None,
        }
        for kind, count in per_kind_total.items()
    }
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "missing_predictions": missing_predictions,
        "invalid_gold_records": invalid_gold,
        "per_domain": per_domain,
        "per_answer_kind": per_kind,
        "error_modes": dict(sorted(error_modes.items())),
        "mistake_count": len(mistakes),
        "mistakes": mistakes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate verifier-style predictions.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--max-mistakes", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_verifier(
        load_json_or_jsonl(Path(args.gold)),
        load_json_or_jsonl(Path(args.pred)),
    )
    report["mistakes"] = report["mistakes"][: args.max_mistakes]
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
