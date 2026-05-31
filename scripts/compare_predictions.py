"""Compare two SCoRE2026 prediction files against the same gold split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_score import align_predictions, load_json_or_jsonl, normalize_answer


def answer_kind(record: dict[str, Any]) -> str:
    return "multi" if len(normalize_answer(record.get("answer"))) > 1 else "single"


def domain_of(record: dict[str, Any]) -> str:
    return str(record.get("domain") or record.get("category") or record.get("type") or "unknown")


def is_match(gold: dict[str, Any], pred: dict[str, Any] | None) -> bool:
    if pred is None:
        return False
    return set(normalize_answer(gold.get("answer"))) == set(normalize_answer(pred.get("answer")))


def safe_acc(correct: int, total: int) -> float | None:
    return correct / total if total else None


def compare(
    gold_records: list[dict[str, Any]],
    base_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    base_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    base_aligned = align_predictions(gold_records, base_records)
    candidate_aligned = align_predictions(gold_records, candidate_records)

    total = 0
    base_correct = 0
    candidate_correct = 0
    both_correct = 0
    both_wrong = 0
    candidate_gain = 0
    candidate_loss = 0

    buckets: dict[str, Counter[str]] = {}
    flips: list[dict[str, Any]] = []

    for index, ((gold, base), (_, candidate)) in enumerate(zip(base_aligned, candidate_aligned)):
        if not normalize_answer(gold.get("answer")):
            continue
        total += 1
        b_ok = is_match(gold, base)
        c_ok = is_match(gold, candidate)
        base_correct += int(b_ok)
        candidate_correct += int(c_ok)
        both_correct += int(b_ok and c_ok)
        both_wrong += int((not b_ok) and (not c_ok))
        candidate_gain += int((not b_ok) and c_ok)
        candidate_loss += int(b_ok and (not c_ok))

        keys = [
            f"domain:{domain_of(gold)}",
            f"kind:{answer_kind(gold)}",
            f"domain_kind:{domain_of(gold)}:{answer_kind(gold)}",
        ]
        for key in keys:
            counter = buckets.setdefault(key, Counter())
            counter["total"] += 1
            counter[f"{base_name}_correct"] += int(b_ok)
            counter[f"{candidate_name}_correct"] += int(c_ok)
            counter[f"{candidate_name}_gain"] += int((not b_ok) and c_ok)
            counter[f"{candidate_name}_loss"] += int(b_ok and (not c_ok))

        if b_ok != c_ok:
            flips.append(
                {
                    "index": index,
                    "id": gold.get("id"),
                    "domain": domain_of(gold),
                    "kind": answer_kind(gold),
                    "gold": normalize_answer(gold.get("answer")),
                    base_name: normalize_answer(base.get("answer")) if base else [],
                    candidate_name: normalize_answer(candidate.get("answer")) if candidate else [],
                    "winner": candidate_name if c_ok else base_name,
                }
            )

    bucket_report = {}
    for key, counts in sorted(buckets.items()):
        total_in_bucket = counts["total"]
        base_bucket_correct = counts[f"{base_name}_correct"]
        candidate_bucket_correct = counts[f"{candidate_name}_correct"]
        bucket_report[key] = {
            "total": total_in_bucket,
            f"{base_name}_accuracy": safe_acc(base_bucket_correct, total_in_bucket),
            f"{candidate_name}_accuracy": safe_acc(candidate_bucket_correct, total_in_bucket),
            "delta": safe_acc(candidate_bucket_correct, total_in_bucket)
            - safe_acc(base_bucket_correct, total_in_bucket),
            f"{candidate_name}_gain": counts[f"{candidate_name}_gain"],
            f"{candidate_name}_loss": counts[f"{candidate_name}_loss"],
        }

    return {
        "total": total,
        base_name: {"correct": base_correct, "accuracy": safe_acc(base_correct, total)},
        candidate_name: {"correct": candidate_correct, "accuracy": safe_acc(candidate_correct, total)},
        "delta": safe_acc(candidate_correct, total) - safe_acc(base_correct, total),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        f"{candidate_name}_gain": candidate_gain,
        f"{candidate_name}_loss": candidate_loss,
        "buckets": bucket_report,
        "flips": flips,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two SCoRE prediction files.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--report", default=None)
    parser.add_argument("--max-flips", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(
        load_json_or_jsonl(Path(args.gold)),
        load_json_or_jsonl(Path(args.base)),
        load_json_or_jsonl(Path(args.candidate)),
        base_name=args.base_name,
        candidate_name=args.candidate_name,
    )
    report["flips"] = report["flips"][: args.max_flips]
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
