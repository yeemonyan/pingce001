"""Build SFT train/valid datasets from the official SCoRE2026 train split."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.infer_score import build_user_prompt, infer_domain, load_prompts
from scripts.make_dev_split import answer_kind, detect_language
from scripts.parse_score_json import load_records, normalize_record


DEFAULT_INPUT = Path("data/raw/train.json")
DEFAULT_TRAIN_IDS = Path("data/splits/train_ids.json")
DEFAULT_DEV_IDS = Path("data/splits/dev_ids.json")
DEFAULT_PROMPTS = Path("configs/system_prompts_v2.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs")

OUTPUTS = {
    ("train", "answer_only"): DEFAULT_OUTPUT_DIR / "sft_train_answer_only.jsonl",
    ("valid", "answer_only"): DEFAULT_OUTPUT_DIR / "sft_valid_answer_only.jsonl",
    ("train", "rationale_json"): DEFAULT_OUTPUT_DIR / "sft_train_rationale_json.jsonl",
    ("valid", "rationale_json"): DEFAULT_OUTPUT_DIR / "sft_valid_rationale_json.jsonl",
    ("train", "reasoning_short_json"): DEFAULT_OUTPUT_DIR / "sft_train_reasoning_short_json.jsonl",
    ("valid", "reasoning_short_json"): DEFAULT_OUTPUT_DIR / "sft_valid_reasoning_short_json.jsonl",
}
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "sft_data_report.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_records_by_id(input_path: Path) -> dict[str, dict[str, Any]]:
    raw_records = load_records(input_path)
    records = [normalize_record(record, index) for index, record in enumerate(raw_records)]
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not record.get("has_answer"):
            continue
        record_id = str(record["id"])
        if record_id in by_id:
            raise ValueError(f"duplicate record id: {record_id}")
        record["answers"] = record["answer"]
        record["domain"] = infer_domain(record)
        record["language"] = detect_language(record)
        by_id[record_id] = record
    return by_id


def select_records(by_id: dict[str, dict[str, Any]], ids: list[Any]) -> list[dict[str, Any]]:
    records = []
    missing = []
    for raw_id in ids:
        record_id = str(raw_id)
        record = by_id.get(record_id)
        if record is None:
            missing.append(record_id)
            continue
        records.append(record)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} split ids were not found in training data: {preview}")
    return records


def option_ordered_answers(record: dict[str, Any]) -> list[str]:
    answers = {str(answer).strip().upper() for answer in record["answers"]}
    return [label for label in record["options"] if label in answers]


def compact_constraints(record: dict[str, Any], limit: int = 6) -> list[str]:
    text = record["text"].replace("\r\n", "\n")
    candidates = re.split(r"(?:\n+|[;；。])", text)
    constraints = []
    for candidate in candidates:
        item = re.sub(r"\s+", " ", candidate).strip(" :：,，")
        if not item:
            continue
        if len(item) > 120:
            item = item[:117].rstrip() + "..."
        constraints.append(item)
        if len(constraints) >= limit:
            break
    return constraints


def metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "domain": record["domain"],
        "language": record["language"],
        "answer_kind": answer_kind(record),
        "answers": option_ordered_answers(record),
    }


def answer_count_instruction(record: dict[str, Any]) -> str:
    kind = answer_kind(record)
    language = record["language"]
    if kind == "single":
        if language == "zh":
            return "这是一道单选题。你只能输出一个最符合题意的选项。"
        return "This is a single-answer question. You must output exactly one best-supported option."
    if language == "zh":
        return "这是一道多选题。你必须保留所有被材料支持的选项，且不要多选。"
    return "This is a multi-answer question. Keep every supported option and do not add unsupported ones."


def build_sft_user_prompt(record: dict[str, Any]) -> str:
    return build_user_prompt(record) + "\n\n" + answer_count_instruction(record)


def build_answer_only_assistant(record: dict[str, Any]) -> str:
    return json.dumps({"answers": option_ordered_answers(record)}, ensure_ascii=False, separators=(",", ":"))


def build_rationale_assistant(record: dict[str, Any]) -> str:
    payload = {
        "analysis": {
            "domain": record["domain"],
            "language": record["language"],
            "key_constraints": compact_constraints(record),
        },
        "answers": option_ordered_answers(record),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def reasoning_steps(record: dict[str, Any]) -> list[str]:
    domain = record["domain"]
    language = record["language"]
    kind = answer_kind(record)

    if language == "zh":
        domain_steps = {
            "temporal": [
                "先按题干中的先后、间隔和星期信息整理时间线。",
                "逐个核对选项是否与时间线完全一致。",
            ],
            "spatial": [
                "先固定参考系，再整理左右、上下、同层和相邻关系。",
                "逐个核对选项是否与位置关系完全一致。",
            ],
            "social": [
                "先整理人物关系链，再统一正向和逆向称谓。",
                "逐个核对选项是否与关系链完全一致。",
            ],
            "natural": [
                "先整理类别、属性、用途或位置等事实。",
                "逐个核对选项是否与已知事实完全一致。",
            ],
            "hybrid": [
                "先按题型分别整理关键约束，再联立所有条件。",
                "逐个核对选项是否同时满足全部约束。",
            ],
            "general": [
                "先整理题干中的关键约束。",
                "逐个核对选项是否与约束完全一致。",
            ],
        }
        count_step = (
            "这是单选题，所以最终只能保留一个最符合条件的选项。"
            if kind == "single"
            else "这是多选题，所以最终要保留所有正确选项，不能漏选也不能多选。"
        )
    else:
        domain_steps = {
            "temporal": [
                "First build a timeline from before/after, gap, and weekday clues.",
                "Then check each option against the completed timeline.",
            ],
            "spatial": [
                "First fix the reference frame, then map left/right, above/below, same-tier, and adjacency clues.",
                "Then check each option against the completed layout.",
            ],
            "social": [
                "First build the relation chain, then normalize forward and inverse kinship or role terms.",
                "Then check each option against the completed relation chain.",
            ],
            "natural": [
                "First organize category, property, function, or location facts.",
                "Then check each option against the known facts.",
            ],
            "hybrid": [
                "First separate the constraints by sub-domain, then merge them into one joint solution.",
                "Then check each option against all merged constraints.",
            ],
            "general": [
                "First organize the key constraints from the passage.",
                "Then check each option against those constraints.",
            ],
        }
        count_step = (
            "This is a single-answer question, so keep exactly one supported option."
            if kind == "single"
            else "This is a multi-answer question, so keep all supported options and avoid both misses and extras."
        )

    steps = list(domain_steps.get(domain, domain_steps["general"]))
    steps.append(count_step)
    return steps


def build_reasoning_short_assistant(record: dict[str, Any]) -> str:
    payload = {
        "analysis": {
            "domain": record["domain"],
            "language": record["language"],
            "answer_kind": answer_kind(record),
            "key_constraints": compact_constraints(record, limit=4),
            "reasoning_steps": reasoning_steps(record),
        },
        "answers": option_ordered_answers(record),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_sft_item(record: dict[str, Any], system_prompt: str, variant: str) -> dict[str, Any]:
    if variant == "answer_only":
        assistant = build_answer_only_assistant(record)
    elif variant == "rationale_json":
        assistant = build_rationale_assistant(record)
    elif variant == "reasoning_short_json":
        assistant = build_reasoning_short_assistant(record)
    else:
        raise ValueError(f"unsupported SFT variant: {variant}")

    return {
        "id": record["id"],
        "domain": record["domain"],
        "language": record["language"],
        "text": record["text"],
        "question": record["question"],
        "options": record["options"],
        "answers": option_ordered_answers(record),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_sft_user_prompt(record)},
            {"role": "assistant", "content": assistant},
        ],
    }


def build_dataset(records: list[dict[str, Any]], prompts: dict[str, str], variant: str) -> list[dict[str, Any]]:
    items = []
    for record in records:
        system_prompt = normalize_system_prompt_schema(prompts.get(record["domain"]) or prompts["general"])
        items.append(build_sft_item(record, system_prompt, variant))
    return items


def normalize_system_prompt_schema(system_prompt: str) -> str:
    return system_prompt.replace('{"answer":["A"]}', '{"answers":["A"]}')


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_lengths = [
        len(item["messages"][0]["content"]) + len(item["messages"][1]["content"])
        for item in items
    ]
    assistant_lengths = [len(item["messages"][2]["content"]) for item in items]
    return {
        "count": len(items),
        "domain": dict(sorted(Counter(record["domain"] for record in records).items())),
        "language": dict(sorted(Counter(record["language"] for record in records).items())),
        "answer_kind": dict(sorted(Counter(answer_kind(record) for record in records).items())),
        "avg_prompt_length": round(sum(prompt_lengths) / len(prompt_lengths), 2) if prompt_lengths else 0,
        "avg_assistant_length": round(sum(assistant_lengths) / len(assistant_lengths), 2) if assistant_lengths else 0,
    }


def build_report(
    train_records: list[dict[str, Any]],
    valid_records: list[dict[str, Any]],
    outputs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    report = {"files": {}}
    for key, path in OUTPUTS.items():
        split, variant = key
        records = train_records if split == "train" else valid_records
        items = outputs[f"{split}_{variant}"]
        report["files"][str(path)] = {
            "split": split,
            "variant": variant,
            **summarize(records, items),
        }
    report["source"] = {
        "input": str(DEFAULT_INPUT),
        "train_ids": str(DEFAULT_TRAIN_IDS),
        "dev_ids": str(DEFAULT_DEV_IDS),
        "test_data_used": False,
    }
    return report


def validate_items(items: list[dict[str, Any]]) -> None:
    for item in items:
        if str(item["id"]).startswith("SCoRE2026-test-"):
            raise ValueError(f"test sample leaked into SFT data: {item['id']}")
        if not item.get("answers"):
            raise ValueError(f"missing answers: {item['id']}")
        if item["domain"] not in {"spatial", "temporal", "social", "natural", "hybrid", "general"}:
            raise ValueError(f"bad domain: {item['id']} {item['domain']}")
        if item["language"] not in {"zh", "en"}:
            raise ValueError(f"bad language: {item['id']} {item['language']}")
        if [message["role"] for message in item["messages"]] != ["system", "user", "assistant"]:
            raise ValueError(f"bad messages: {item['id']}")
        json.loads(item["messages"][2]["content"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCoRE2026 SFT train/valid JSONL files.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Official training JSON file.")
    parser.add_argument("--train-ids", default=str(DEFAULT_TRAIN_IDS), help="Train id split JSON.")
    parser.add_argument("--dev-ids", default=str(DEFAULT_DEV_IDS), help="Dev id split JSON.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="System prompts YAML.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="SFT data report JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_paths = {
        ("train", "answer_only"): output_dir / "sft_train_answer_only.jsonl",
        ("valid", "answer_only"): output_dir / "sft_valid_answer_only.jsonl",
        ("train", "rationale_json"): output_dir / "sft_train_rationale_json.jsonl",
        ("valid", "rationale_json"): output_dir / "sft_valid_rationale_json.jsonl",
        ("train", "reasoning_short_json"): output_dir / "sft_train_reasoning_short_json.jsonl",
        ("valid", "reasoning_short_json"): output_dir / "sft_valid_reasoning_short_json.jsonl",
    }

    by_id = normalize_records_by_id(Path(args.input))
    train_records = select_records(by_id, load_json(Path(args.train_ids)))
    valid_records = select_records(by_id, load_json(Path(args.dev_ids)))
    prompts = load_prompts(Path(args.prompts))

    outputs = {
        "train_answer_only": build_dataset(train_records, prompts, "answer_only"),
        "valid_answer_only": build_dataset(valid_records, prompts, "answer_only"),
        "train_rationale_json": build_dataset(train_records, prompts, "rationale_json"),
        "valid_rationale_json": build_dataset(valid_records, prompts, "rationale_json"),
        "train_reasoning_short_json": build_dataset(train_records, prompts, "reasoning_short_json"),
        "valid_reasoning_short_json": build_dataset(valid_records, prompts, "reasoning_short_json"),
    }
    for items in outputs.values():
        validate_items(items)

    for key, path in output_paths.items():
        split, variant = key
        write_jsonl(outputs[f"{split}_{variant}"], path)

    global OUTPUTS
    OUTPUTS = output_paths
    report = build_report(train_records, valid_records, outputs)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "train": len(train_records),
                "valid": len(valid_records),
                "files": {str(path): len(outputs[f"{split}_{variant}"]) for (split, variant), path in output_paths.items()},
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
