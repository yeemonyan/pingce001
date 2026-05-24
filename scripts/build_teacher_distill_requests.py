"""Build API teacher distillation requests from the official SCoRE2026 train split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.infer_score import build_user_prompt, infer_domain, parse_simple_block_yaml
from scripts.make_dev_split import answer_kind, detect_language
from scripts.parse_score_json import load_records, normalize_record


DEFAULT_INPUT = Path("data/raw/train.json")
DEFAULT_TRAIN_IDS = Path("data/splits/train_ids.json")
DEFAULT_DEV_IDS = Path("data/splits/dev_ids.json")
DEFAULT_PROMPTS = Path("configs/teacher_distill_prompts.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/distill")


def load_prompt_config(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        data = parse_simple_block_yaml(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt config must be a mapping: {path}")
    return {str(key): str(value).strip() for key, value in data.items()}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_records_by_id(input_path: Path) -> dict[str, dict[str, Any]]:
    records = [normalize_record(record, index) for index, record in enumerate(load_records(input_path))]
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not record.get("has_answer"):
            continue
        record_id = str(record["id"])
        record["answers"] = list(record["answer"] or [])
        record["domain"] = infer_domain(record)
        record["language"] = detect_language(record)
        record["answer_kind"] = answer_kind(record)
        by_id[record_id] = record
    return by_id


def select_records(by_id: dict[str, dict[str, Any]], ids: list[Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for raw_id in ids:
        record = by_id.get(str(raw_id))
        if record is not None:
            selected.append(record)
    return selected


def teacher_output_schema(record: dict[str, Any]) -> dict[str, Any]:
    options = list(record["options"].keys())
    return {
        "option_judgments": {
            label: {
                "label": "true_or_false",
                "reason": "very short reason",
            }
            for label in options
        },
        "final_reasoning": "one short concluding sentence",
        "answers": options[:1],
    }


def build_teacher_user_prompt(record: dict[str, Any]) -> str:
    language = record["language"]
    option_lines = "\n".join(f"{label}. {text}" for label, text in record["options"].items())
    kind = record["answer_kind"]
    if language == "zh":
        schema_hint = json.dumps(teacher_output_schema(record), ensure_ascii=False, indent=2)
        kind_hint = "单选题，只能输出一个答案。" if kind == "single" else "多选题，必须保留所有正确答案。"
        return (
            f"题型领域：{record['domain']}\n"
            f"答案类型：{kind}\n"
            f"{kind_hint}\n\n"
            f"材料：\n{record['text']}\n\n"
            f"问题：\n{record['question']}\n\n"
            f"选项：\n{option_lines}\n\n"
            "请对每个选项分别判断 true/false，并给出不超过 30 个汉字的短理由。"
            "最终再给出一句极短结论和标准答案数组。"
            "不要输出 markdown，不要输出额外解释，只输出 JSON。\n\n"
            f"输出 JSON 模板：\n{schema_hint}"
        )

    schema_hint = json.dumps(teacher_output_schema(record), ensure_ascii=False, indent=2)
    kind_hint = "This is a single-answer question. Keep exactly one answer." if kind == "single" else "This is a multi-answer question. Keep all correct answers."
    return (
        f"Domain: {record['domain']}\n"
        f"Answer type: {kind}\n"
        f"{kind_hint}\n\n"
        f"Passage:\n{record['text']}\n\n"
        f"Question:\n{record['question']}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Judge each option as true or false and keep each reason very short. "
        "Then give one short concluding sentence and the final answers array. "
        "Return JSON only, with no markdown and no extra prose.\n\n"
        f"JSON template:\n{schema_hint}"
    )


def build_request(record: dict[str, Any], prompts: dict[str, str], split: str) -> dict[str, Any]:
    system_key = "teacher_system_zh" if record["language"] == "zh" else "teacher_system_en"
    return {
        "id": record["id"],
        "split": split,
        "domain": record["domain"],
        "language": record["language"],
        "answer_kind": record["answer_kind"],
        "gold_answers": list(record["answers"]),
        "request_version": "teacher_distill_v1",
        "messages": [
            {"role": "system", "content": prompts[system_key]},
            {"role": "user", "content": build_teacher_user_prompt(record)},
        ],
        "record": {
            "text": record["text"],
            "question": record["question"],
            "options": record["options"],
        },
    }


def write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher distillation requests.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--train-ids", default=str(DEFAULT_TRAIN_IDS))
    parser.add_argument("--dev-ids", default=str(DEFAULT_DEV_IDS))
    parser.add_argument("--prompt-config", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--train-output", default=str(DEFAULT_OUTPUT_DIR / "teacher_requests_train.jsonl"))
    parser.add_argument("--valid-output", default=str(DEFAULT_OUTPUT_DIR / "teacher_requests_valid.jsonl"))
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_DIR / "teacher_requests_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompt_config(Path(args.prompt_config))
    by_id = normalize_records_by_id(Path(args.input))
    train_records = select_records(by_id, load_json(Path(args.train_ids)))
    valid_records = select_records(by_id, load_json(Path(args.dev_ids)))
    train_items = [build_request(record, prompts, "train") for record in train_records]
    valid_items = [build_request(record, prompts, "valid") for record in valid_records]
    write_jsonl(train_items, Path(args.train_output))
    write_jsonl(valid_items, Path(args.valid_output))
    report = {
        "input": str(args.input),
        "train_count": len(train_items),
        "valid_count": len(valid_items),
        "prompt_config": str(args.prompt_config),
        "request_version": "teacher_distill_v1",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
