"""Build option-level verifier SFT datasets from SCoRE2026 train/dev splits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_sft_data import (
    DEFAULT_DEV_IDS,
    DEFAULT_INPUT,
    DEFAULT_TRAIN_IDS,
    DEFAULT_OUTPUT_DIR,
    load_json,
    normalize_records_by_id,
    select_records,
)
from scripts.infer_score import infer_domain
from scripts.make_dev_split import answer_kind, detect_language


DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "verifier_data_report.json"
TRAIN_OUTPUT = DEFAULT_OUTPUT_DIR / "sft_train_verifier.jsonl"
VALID_OUTPUT = DEFAULT_OUTPUT_DIR / "sft_valid_verifier.jsonl"


def ordered_labels(record: dict[str, Any]) -> list[str]:
    return [str(label) for label in record["options"].keys()]


def ordered_answers(record: dict[str, Any]) -> list[str]:
    answer_set = {str(answer).strip().upper() for answer in record["answer"]}
    return [label for label in ordered_labels(record) if label in answer_set]


def build_verifier_user_prompt(record: dict[str, Any], option_label: str) -> str:
    option_text = record["options"][option_label]
    option_lines = "\n".join(
        f"{label}. {value}"
        for label, value in record["options"].items()
    )
    return (
        f"Text:\n{record['text']}\n\n"
        f"Question:\n{record['question']}\n\n"
        f"All Options:\n{option_lines}\n\n"
        f"Candidate Option:\n{option_label}. {option_text}\n\n"
        "Decide whether the candidate option is supported by the text and question. "
        'Return only JSON: {"label":"yes"} or {"label":"no"}.'
    )


def build_verifier_assistant(record: dict[str, Any], option_label: str) -> str:
    label = "yes" if option_label in ordered_answers(record) else "no"
    return json.dumps({"label": label}, ensure_ascii=False, separators=(",", ":"))


def build_verifier_item(record: dict[str, Any], option_label: str) -> dict[str, Any]:
    system_prompt = (
        "You are solving one candidate option from a SCoRE2026 reasoning question. "
        "Judge only whether the candidate option is supported. "
        'Return only JSON: {"label":"yes"} or {"label":"no"}.'
    )
    return {
        "id": f"{record['id']}::{option_label}",
        "source_id": record["id"],
        "domain": record["domain"],
        "language": record["language"],
        "answer_kind": answer_kind(record),
        "candidate_label": option_label,
        "candidate_text": record["options"][option_label],
        "gold_answers": ordered_answers(record),
        "target_label": "yes" if option_label in ordered_answers(record) else "no",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_verifier_user_prompt(record, option_label)},
            {"role": "assistant", "content": build_verifier_assistant(record, option_label)},
        ],
    }


def build_verifier_dataset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        for option_label in ordered_labels(record):
            items.append(build_verifier_item(record, option_label))
    return items


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(items),
        "domain": dict(sorted(Counter(item["domain"] for item in items).items())),
        "language": dict(sorted(Counter(item["language"] for item in items).items())),
        "answer_kind": dict(sorted(Counter(item["answer_kind"] for item in items).items())),
        "target_label": dict(sorted(Counter(item["target_label"] for item in items).items())),
        "avg_user_length": round(
            sum(len(item["messages"][1]["content"]) for item in items) / len(items), 2
        ) if items else 0,
    }


def build_report(train_items: list[dict[str, Any]], valid_items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "files": {
            str(TRAIN_OUTPUT): {"split": "train", **summarize(train_items)},
            str(VALID_OUTPUT): {"split": "valid", **summarize(valid_items)},
        },
        "source": {
            "input": str(DEFAULT_INPUT),
            "train_ids": str(DEFAULT_TRAIN_IDS),
            "dev_ids": str(DEFAULT_DEV_IDS),
            "task": "option_level_verifier",
            "test_data_used": False,
        },
    }


def validate_items(items: list[dict[str, Any]]) -> None:
    for item in items:
        if str(item["source_id"]).startswith("SCoRE2026-test-"):
            raise ValueError(f"test sample leaked into verifier data: {item['source_id']}")
        if item["target_label"] not in {"yes", "no"}:
            raise ValueError(f"bad target label: {item['id']}")
        if [message["role"] for message in item["messages"]] != ["system", "user", "assistant"]:
            raise ValueError(f"bad message roles: {item['id']}")
        parsed = json.loads(item["messages"][2]["content"])
        if parsed.get("label") != item["target_label"]:
            raise ValueError(f"assistant/target mismatch: {item['id']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCoRE2026 option-level verifier SFT data.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--train-ids", default=str(DEFAULT_TRAIN_IDS))
    parser.add_argument("--dev-ids", default=str(DEFAULT_DEV_IDS))
    parser.add_argument("--train-output", default=str(TRAIN_OUTPUT))
    parser.add_argument("--valid-output", default=str(VALID_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"missing official training data: {input_path}. "
            "Place the official train.json at data/raw/train.json or pass --input explicitly."
        )
    by_id = normalize_records_by_id(input_path)
    train_records = select_records(by_id, load_json(Path(args.train_ids)))
    valid_records = select_records(by_id, load_json(Path(args.dev_ids)))

    for record in train_records + valid_records:
        record["domain"] = infer_domain(record)
        record["language"] = detect_language(record)

    train_items = build_verifier_dataset(train_records)
    valid_items = build_verifier_dataset(valid_records)
    validate_items(train_items)
    validate_items(valid_items)

    train_output = Path(args.train_output)
    valid_output = Path(args.valid_output)
    report_output = Path(args.report)

    write_jsonl(train_items, train_output)
    write_jsonl(valid_items, valid_output)
    report = build_report(train_items, valid_items)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "train_output": str(train_output),
                "valid_output": str(valid_output),
                "train_count": len(train_items),
                "valid_count": len(valid_items),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
