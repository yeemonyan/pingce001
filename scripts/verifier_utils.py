"""Shared utilities for option-level SCoRE2026 verifier experiments."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.evaluate_score import align_predictions, normalize_answer
from scripts.infer_score import VALID_LABELS, infer_domain


VERIFIER_SYSTEM_PROMPT = (
    "You are a SCoRE2026 option verifier. Decide whether one option is entailed "
    "by the text and question. Return only JSON: {\"label\":\"yes\"} or {\"label\":\"no\"}."
)

YES_VALUES = {"yes", "y", "true", "1", "entailed", "correct"}
NO_VALUES = {"no", "n", "false", "0", "not_entailed", "incorrect"}


def option_labels(options: dict[str, Any]) -> list[str]:
    return [label for label in VALID_LABELS if label in options]


def ordered_answers(record: dict[str, Any]) -> list[str]:
    answers = set(normalize_answer(record.get("answer") or record.get("answers")))
    return [label for label in option_labels(record.get("options") or {}) if label in answers]


def answer_kind(answer: list[str]) -> str:
    return "multi" if len(answer) > 1 else "single"


def build_option_prompt(record: dict[str, Any], option_label: str) -> str:
    options = record["options"]
    return (
        f"Text:\n{record['text']}\n\n"
        f"Question:\n{record['question']}\n\n"
        f"Option {option_label}:\n{options[option_label]}\n\n"
        "Does this option correctly answer the question? Output only JSON."
    )


def verifier_item_id(record_id: Any, option_label: str) -> str:
    return f"{record_id}::option::{option_label}"


def build_verifier_item(record: dict[str, Any], option_label: str, system_prompt: str = VERIFIER_SYSTEM_PROMPT) -> dict[str, Any]:
    answers = ordered_answers(record)
    label = "yes" if option_label in answers else "no"
    return {
        "id": verifier_item_id(record["id"], option_label),
        "question_id": record["id"],
        "option_label": option_label,
        "option_text": record["options"][option_label],
        "domain": infer_domain(record),
        "language": record.get("language"),
        "answer_kind": answer_kind(answers),
        "gold_answer": answers,
        "label": label,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_option_prompt(record, option_label)},
            {"role": "assistant", "content": json.dumps({"label": label}, ensure_ascii=False)},
        ],
    }


def extract_verdict(output: str) -> str | None:
    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            value = parsed.get("label") or parsed.get("verdict") or parsed.get("answer")
            if isinstance(value, bool):
                return "yes" if value else "no"
            if isinstance(value, str):
                normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
                if normalized in YES_VALUES:
                    return "yes"
                if normalized in NO_VALUES:
                    return "no"
    except json.JSONDecodeError:
        pass

    lowered = output.strip().lower()
    match = re.search(r"\b(yes|no|true|false|entailed|not entailed|correct|incorrect)\b", lowered)
    if not match:
        return None
    value = match.group(1).replace(" ", "_")
    if value in YES_VALUES:
        return "yes"
    if value in NO_VALUES:
        return "no"
    return None


def merge_option_predictions(option_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    metadata: dict[Any, dict[str, Any]] = {}
    for item in option_records:
        question_id = item.get("question_id")
        grouped.setdefault(question_id, []).append(item)
        metadata.setdefault(
            question_id,
            {
                "id": question_id,
                "domain": item.get("domain"),
                "gold": item.get("gold_answer"),
            },
        )

    outputs = []
    for question_id, items in grouped.items():
        by_label = {str(item.get("option_label")): item for item in items}
        answer = [
            label
            for label in VALID_LABELS
            if by_label.get(label, {}).get("label") == "yes"
        ]
        if not answer:
            scored = [
                item for item in items
                if isinstance(item.get("yes_score"), (int, float))
            ]
            if scored:
                best = max(scored, key=lambda item: item["yes_score"])
                answer = [str(best["option_label"])]
        outputs.append(
            {
                **metadata[question_id],
                "answer": answer,
                "option_verdicts": {
                    label: {
                        "label": by_label[label].get("label"),
                        "raw_output": by_label[label].get("raw_output"),
                        "yes_score": by_label[label].get("yes_score"),
                    }
                    for label in VALID_LABELS
                    if label in by_label
                },
            }
        )
    return outputs


def detailed_metrics(gold_records: list[dict[str, Any]], pred_records: list[dict[str, Any]]) -> dict[str, Any]:
    aligned = align_predictions(gold_records, pred_records)
    total = correct = missing = invalid_gold = 0
    by_domain_total: Counter[str] = Counter()
    by_domain_correct: Counter[str] = Counter()
    by_kind_total: Counter[str] = Counter()
    by_kind_correct: Counter[str] = Counter()
    error_types: Counter[str] = Counter()

    for gold, pred in aligned:
        gold_answer = normalize_answer(gold.get("answer"))
        if not gold_answer:
            invalid_gold += 1
            continue
        pred_answer = normalize_answer(pred.get("answer")) if pred else []
        if pred is None:
            missing += 1

        matched = set(gold_answer) == set(pred_answer)
        domain = infer_domain(gold)
        kind = answer_kind(gold_answer)
        total += 1
        correct += int(matched)
        by_domain_total[domain] += 1
        by_domain_correct[domain] += int(matched)
        by_kind_total[kind] += 1
        by_kind_correct[kind] += int(matched)

        if matched:
            continue
        gold_set = set(gold_answer)
        pred_set = set(pred_answer)
        if not pred_set:
            error_types["empty_prediction"] += 1
        elif pred_set > gold_set:
            error_types["over_predict"] += 1
        elif pred_set < gold_set:
            error_types["under_predict"] += 1
        elif len(pred_set) > len(gold_set):
            error_types["over_predict"] += 1
        elif len(pred_set) < len(gold_set):
            error_types["under_predict"] += 1
        else:
            error_types["wrong_label_set"] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else None,
        "missing_predictions": missing,
        "invalid_gold_records": invalid_gold,
        "per_domain": {
            domain: {
                "total": count,
                "correct": by_domain_correct[domain],
                "accuracy": round(by_domain_correct[domain] / count, 6) if count else None,
            }
            for domain, count in sorted(by_domain_total.items())
        },
        "single_vs_multi": {
            kind: {
                "total": count,
                "correct": by_kind_correct[kind],
                "accuracy": round(by_kind_correct[kind] / count, 6) if count else None,
            }
            for kind, count in sorted(by_kind_total.items())
        },
        "prediction_error_types": dict(sorted(error_types.items())),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
