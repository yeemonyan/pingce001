"""V6 single-first post-processing for SCoRE2026 test predictions.

This script is intentionally conservative:
- keep explicit multi-answer questions untouched
- collapse over-selected predictions to a single label only when the
  question looks likely-single with high precision

The goal is to repair V5's remaining "single_to_multi" failure mode without
retraining or re-running the model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def dump_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_domain(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "time" in text or "temporal" in text:
        return "temporal"
    if "space" in text or "spatial" in text:
        return "spatial"
    if "social" in text:
        return "social"
    if "nature" in text or "natural" in text:
        return "natural"
    if "hybrid" in text or "fusion" in text or "mixed" in text:
        return "hybrid"
    return text or "general"


def strong_multi_signal(question: str) -> bool:
    question_lower = question.lower()
    patterns = (
        r"statement\(s\)",
        r"which of the following statements",
        r"select the correct statements?",
        r"select the incorrect statements?",
        r"all that apply",
        r"以下选项中",
        r"下列.*(正确|错误)",
        r"哪些",
        r"哪几项",
        r"多选",
    )
    return any(re.search(pattern, question_lower) for pattern in patterns)


def single_question_marker(question: str) -> bool:
    question_lower = question.lower()
    patterns = (
        r"which one",
        r"which option",
        r"which item",
        r"which person",
        r"who\b",
        r"where\b",
        r"when\b",
        r"what is\b",
        r"哪一个",
        r"哪个",
        r"哪位",
        r"谁",
        r"什么时候",
        r"何时",
        r"哪项",
        r"哪种",
        r"哪件",
        r"哪只",
    )
    return any(re.search(pattern, question_lower) for pattern in patterns)


def should_force_single(source_record: dict[str, Any]) -> tuple[bool, str]:
    question = str(source_record.get("question") or "")
    domain = normalize_domain(source_record.get("domain"))

    if strong_multi_signal(question):
        return False, "explicit_multi_cue"

    if single_question_marker(question):
        return True, "single_marker"

    if domain in {"natural", "social", "general"}:
        return True, f"{domain}_default_single"

    if domain == "spatial":
        return True, "spatial_default_single"

    return False, f"{domain}_uncertain"


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply V6 single-first post-processing.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL from infer_score.py")
    parser.add_argument("--source", required=True, help="Source prompt JSONL with question/domain fields")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    predictions = load_jsonl(Path(args.predictions))
    source_records = {record["id"]: record for record in load_jsonl(Path(args.source))}

    changed = 0
    kept_multi = 0
    missing_source = 0

    for record in predictions:
        answers = list(record.get("answer") or record.get("answers") or [])
        if len(answers) <= 1:
            continue

        source_record = source_records.get(record.get("id"))
        if source_record is None:
            missing_source += 1
            continue

        force_single, reason = should_force_single(source_record)
        record["v6_decision"] = reason
        if force_single:
            record["answer_before_v6"] = answers
            record["answer"] = answers[:1]
            changed += 1
        else:
            kept_multi += 1

    dump_jsonl(predictions, Path(args.output))
    print(
        json.dumps(
            {
                "total": len(predictions),
                "changed_to_single": changed,
                "kept_multi": kept_multi,
                "missing_source": missing_source,
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
