"""Evaluate SCoRE2026 predictions against a labeled validation set."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_LABELS = ("A", "B", "C", "D")


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            for key in ("data", "records", "examples", "predictions"):
                if isinstance(data.get(key), list):
                    records = data[key]
                    break
            else:
                records = [data]
        else:
            raise ValueError(f"Unsupported JSON top-level type: {type(data).__name__}")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Every item must be an object: {path}")
    return records


def normalize_answer(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    labels = []
    for item in value:
        label = str(item).strip().upper()
        if label in VALID_LABELS and label not in labels:
            labels.append(label)
    return labels


def align_predictions(
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    gold_ids = [record.get("id") for record in gold_records]
    pred_by_id = {
        record.get("id"): record
        for record in pred_records
        if record.get("id") is not None
    }

    if gold_ids and all(item is not None for item in gold_ids) and pred_by_id:
        return [(gold, pred_by_id.get(gold.get("id"))) for gold in gold_records]

    return [
        (gold, pred_records[index] if index < len(pred_records) else None)
        for index, gold in enumerate(gold_records)
    ]


def evaluate(
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
) -> dict[str, Any]:
    aligned = align_predictions(gold_records, pred_records)
    total = 0
    correct = 0
    missing = 0
    invalid_gold = 0
    per_domain_total: Counter[str] = Counter()
    per_domain_correct: Counter[str] = Counter()
    mistakes: list[dict[str, Any]] = []

    for index, (gold, pred) in enumerate(aligned):
        gold_answer = normalize_answer(gold.get("answer"))
        if not gold_answer:
            invalid_gold += 1
            continue

        domain = str(gold.get("domain") or gold.get("category") or gold.get("type") or "unknown")
        pred_answer = normalize_answer(pred.get("answer")) if pred else []
        if pred is None:
            missing += 1

        matched = set(pred_answer) == set(gold_answer)
        total += 1
        correct += int(matched)
        per_domain_total[domain] += 1
        per_domain_correct[domain] += int(matched)

        if not matched:
            mistakes.append(
                {
                    "index": index,
                    "id": gold.get("id", pred.get("id") if pred else None),
                    "domain": domain,
                    "gold": gold_answer,
                    "prediction": pred_answer,
                }
            )

    per_domain = {}
    for domain, count in per_domain_total.items():
        domain_correct = per_domain_correct[domain]
        per_domain[domain] = {
            "total": count,
            "correct": domain_correct,
            "accuracy": domain_correct / count if count else None,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "missing_predictions": missing,
        "invalid_gold_records": invalid_gold,
        "per_domain": per_domain,
        "mistake_count": len(mistakes),
        "mistakes": mistakes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SCoRE2026 predictions.")
    parser.add_argument("--gold", required=True, help="Labeled validation JSON/JSONL.")
    parser.add_argument("--pred", required=True, help="Prediction/submission JSON/JSONL.")
    parser.add_argument("--report", default=None, help="Optional report JSON output path.")
    parser.add_argument(
        "--max-mistakes",
        type=int,
        default=20,
        help="Number of mistakes to keep in stdout/report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_records = load_json_or_jsonl(Path(args.gold))
    pred_records = load_json_or_jsonl(Path(args.pred))
    report = evaluate(gold_records, pred_records)
    report["mistakes"] = report["mistakes"][: args.max_mistakes]

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
