"""Analyze failed teacher distillation responses."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def normalize_pred_answers(payload: dict[str, Any]) -> list[str]:
    answers = payload.get("answers")
    if not isinstance(answers, list):
        return []
    return [str(item).strip().upper() for item in answers]


def classify_answer_error(gold: list[str], pred: list[str]) -> str:
    gold_set = set(gold)
    pred_set = set(pred)
    if not pred:
        return "empty_answer"
    if len(gold) == 1 and len(pred) > 1:
        return "single_to_multi"
    if len(gold) > 1 and len(pred) == 1:
        return "multi_to_single"
    if gold_set < pred_set:
        return "extra_options"
    if pred_set < gold_set:
        return "missing_options"
    return "wrong_combination"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze teacher distillation failures.")
    parser.add_argument("--requests", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sample-limit", type=int, default=10)
    args = parser.parse_args()

    request_map = {item["id"]: item for item in load_jsonl(Path(args.requests))}
    responses = load_jsonl(Path(args.responses))

    status_counter = Counter()
    domain_counter = Counter()
    kind_counter = Counter()
    parse_error_counter = Counter()
    answer_error_counter = Counter()

    invalid_samples: list[dict[str, Any]] = []
    incorrect_samples: list[dict[str, Any]] = []

    for item in responses:
        request = request_map.get(item["id"], {})
        payload = item.get("response_json")
        status: str
        if not isinstance(payload, dict):
            status = "invalid_json"
            parse_error = item.get("parse_error") or "json_not_found"
            parse_error_counter[str(parse_error)] += 1
            if len(invalid_samples) < args.sample_limit:
                invalid_samples.append(
                    {
                        "id": item["id"],
                        "domain": item.get("domain"),
                        "answer_kind": item.get("answer_kind"),
                        "raw_output_preview": str(item.get("raw_output", ""))[:1200],
                    }
                )
        else:
            gold = [str(label).strip().upper() for label in request.get("gold_answers", [])]
            pred = normalize_pred_answers(payload)
            if pred == gold:
                status = "accepted"
            else:
                status = "incorrect_answer"
                error_type = classify_answer_error(gold, pred)
                answer_error_counter[error_type] += 1
                if len(incorrect_samples) < args.sample_limit:
                    incorrect_samples.append(
                        {
                            "id": item["id"],
                            "domain": item.get("domain"),
                            "answer_kind": item.get("answer_kind"),
                            "gold_answers": gold,
                            "pred_answers": pred,
                            "answer_error_type": error_type,
                            "response_json": payload,
                            "raw_output_preview": str(item.get("raw_output", ""))[:1200],
                        }
                    )

        status_counter[status] += 1
        domain_counter[f"{status}|{item.get('domain')}"] += 1
        kind_counter[f"{status}|{item.get('answer_kind')}"] += 1

    report = {
        "status_counts": dict(sorted(status_counter.items())),
        "domain_counts": dict(sorted(domain_counter.items())),
        "answer_kind_counts": dict(sorted(kind_counter.items())),
        "parse_error_counts": dict(sorted(parse_error_counter.items())),
        "answer_error_counts": dict(sorted(answer_error_counter.items())),
        "invalid_samples": invalid_samples,
        "incorrect_samples": incorrect_samples,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
