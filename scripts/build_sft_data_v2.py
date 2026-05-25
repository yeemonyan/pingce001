"""Build SFT data v2: remove answer_count_instruction dependency.

Key change: Instead of baking "这是一道单选题/多选题" into every training
example, we use a neutral instruction that teaches the model to self-determine
cardinality from the question text.

This makes the model robust to inference without gold cardinality hints.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# ---- Paths (same as original) ----
DEFAULT_INPUT = Path("data/raw/train.json")
DEFAULT_TRAIN_IDS = Path("data/splits/train_ids.json")
DEFAULT_DEV_IDS = Path("data/splits/dev_ids.json")
DEFAULT_PROMPTS = Path("configs/system_prompts_v4_self_determine.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def detect_language(record: dict[str, Any]) -> str:
    text = f"{record.get('text', '')} {record.get('question', '')}"
    return "zh" if any("一" <= ch <= "鿿" for ch in text) else "en"


def answer_kind(record: dict[str, Any]) -> str:
    answers = record.get("answer") or record.get("answers") or []
    return "multi" if len(answers) > 1 else "single"


def option_ordered_answers(record: dict[str, Any]) -> list[str]:
    answers = {str(a).strip().upper() for a in (record.get("answer") or record.get("answers") or [])}
    return [label for label in record.get("options", {}) if label in answers]


def infer_domain(record: dict[str, Any]) -> str:
    domain = str(record.get("domain", "")).strip().lower()
    mapping = {
        "space": "spatial", "spatial": "spatial",
        "temporal": "temporal", "time": "temporal",
        "social": "social", "society": "social",
        "natural": "natural", "nature": "natural",
        "hybrid": "hybrid", "fusion": "hybrid", "mixed": "hybrid",
        "space+nature": "hybrid", "time+nature": "hybrid",
        "time+social": "hybrid", "space+social": "hybrid",
    }
    return mapping.get(domain, "general")


def load_prompts(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    import yaml
    return yaml.safe_load(text)


# ---- V2: Neutral user prompt (no cardinality hint baked in) ----
def build_sft_user_prompt_v2(record: dict[str, Any]) -> str:
    """Build user prompt that teaches self-determination.
    
    Instead of telling the model whether it's single or multi,
    we tell it to determine this from the question.
    """
    option_lines = "\n".join(
        f"{label}. {value}"
        for label, value in record["options"].items()
    )
    lang = record.get("language", detect_language(record))
    
    if lang == "zh":
        instruction = (
            "请阅读题目，判断是单选题还是多选题，然后选出所有正确选项。"
            "只输出 JSON，格式为 {\"answers\":[\"A\"]} 或 {\"answers\":[\"A\",\"B\"]}。"
        )
    else:
        instruction = (
            "Read the question, determine whether it is single-answer or multi-answer, "
            "then select every correct option. "
            "Output only JSON as {\"answers\":[\"A\"]} or {\"answers\":[\"A\",\"B\"]}."
        )
    
    return (
        f"Text:\n{record['text']}\n\n"
        f"Question:\n{record['question']}\n\n"
        f"Options:\n{option_lines}\n\n"
        f"{instruction}"
    )


def build_assistant_output(record: dict[str, Any]) -> str:
    """V2 assistant output: includes cardinality determination + answers."""
    lang = record.get("language", detect_language(record))
    kind = answer_kind(record)
    answers = option_ordered_answers(record)
    
    # Assistant output includes the cardinality determination
    # This teaches the model to output its reasoning about single/multi
    if lang == "zh":
        cardinality = "单选题" if kind == "single" else "多选题"
        reasoning = f"这是一道{cardinality}。"
    else:
        cardinality = "single-answer" if kind == "single" else "multi-answer"
        reasoning = f"This is a {cardinality} question."
    
    payload = {
        "reasoning": reasoning,
        "answers": answers,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_sft_item_v2(record: dict[str, Any], system_prompt: str, variant: str) -> dict[str, Any]:
    return {
        "id": record["id"],
        "domain": record.get("domain", infer_domain(record)),
        "language": record.get("language", detect_language(record)),
        "variant": variant,
        "text": record["text"],
        "question": record["question"],
        "options": record["options"],
        "answers": option_ordered_answers(record),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_sft_user_prompt_v2(record)},
            {"role": "assistant", "content": build_assistant_output(record)},
        ],
    }


def build_dataset_v2(
    records: list[dict[str, Any]],
    prompts: dict[str, str],
    variant: str,
) -> list[dict[str, Any]]:
    items = []
    for record in records:
        domain = infer_domain(record)
        system_prompt = prompts.get(domain) or prompts["general"]
        items.append(build_sft_item_v2(record, system_prompt, variant))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCoRE2026 SFT data v2 (no cardinality hint dependency).")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Official training JSON file.")
    parser.add_argument("--train-ids", default=str(DEFAULT_TRAIN_IDS))
    parser.add_argument("--dev-ids", default=str(DEFAULT_DEV_IDS))
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--variant", default="v2_neutral", help="Variant name for output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load data
    from scripts.parse_score_json import load_records, normalize_record
    
    raw_records = load_records(Path(args.input))
    by_id = {}
    for i, rec in enumerate(raw_records):
        normalized = normalize_record(rec, i)
        if not normalized.get("has_answer"):
            continue
        rid = str(normalized["id"])
        normalized["domain"] = infer_domain(normalized)
        normalized["language"] = detect_language(normalized)
        by_id[rid] = normalized
    
    train_ids = load_json(Path(args.train_ids))
    dev_ids = load_json(Path(args.dev_ids))
    
    train_records = [by_id[str(rid)] for rid in train_ids if str(rid) in by_id]
    dev_records = [by_id[str(rid)] for rid in dev_ids if str(rid) in by_id]
    
    prompts = load_prompts(Path(args.prompts))
    
    # Build datasets
    train_items = build_dataset_v2(train_records, prompts, args.variant)
    dev_items = build_dataset_v2(dev_records, prompts, args.variant)
    
    # Write outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / f"sft_train_{args.variant}.jsonl"
    dev_path = output_dir / f"sft_valid_{args.variant}.jsonl"
    
    with train_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with dev_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in dev_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Stats
    stats = {
        "train_count": len(train_items),
        "valid_count": len(dev_items),
        "train_path": str(train_path),
        "valid_path": str(dev_path),
        "variant": args.variant,
        "cardinality_distribution": {
            "train": dict(Counter(answer_kind(r) for r in train_records)),
            "valid": dict(Counter(answer_kind(r) for r in dev_records)),
        },
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
