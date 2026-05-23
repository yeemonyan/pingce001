"""Rebuild mixed-reasoning SFT files from existing answer-only SFT files.

This is a recovery utility for cases where the raw official training JSON is
temporarily unavailable but the normalized answer-only SFT files already exist.
It reproduces the mixed-reasoning variant used by train_qwen_mixed_reasoning.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def answer_kind(record: dict[str, Any]) -> str:
    return "single" if len(record["answers"]) == 1 else "multi"


def compact_constraints(text: str, limit: int = 4) -> list[str]:
    candidates = re.split(r"(?:\n+|[;；。])", text.replace("\r\n", "\n"))
    items: list[str] = []
    for candidate in candidates:
        item = re.sub(r"\s+", " ", candidate).strip(" :：,，")
        if not item:
            continue
        if len(item) > 120:
            item = item[:117].rstrip() + "..."
        items.append(item)
        if len(items) >= limit:
            break
    return items


def reasoning_steps(domain: str, language: str, kind: str) -> list[str]:
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    domain = record["domain"]
    kind = answer_kind(record)
    if domain in {"temporal", "spatial", "hybrid"}:
        assistant = {
            "analysis": {
                "domain": domain,
                "language": record["language"],
                "answer_kind": kind,
                "key_constraints": compact_constraints(record["text"], limit=4),
                "reasoning_steps": reasoning_steps(domain, record["language"], kind),
            },
            "answers": record["answers"],
        }
        variant = "reasoning_short_json"
    else:
        assistant = {"answers": record["answers"]}
        variant = "answer_only"

    updated = dict(record)
    updated["variant"] = variant
    updated["messages"] = list(record["messages"])
    updated["messages"][2] = {
        "role": "assistant",
        "content": json.dumps(assistant, ensure_ascii=False, separators=(",", ":")),
    }
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild mixed-reasoning SFT files from answer-only files.")
    parser.add_argument("--train-input", default="outputs/sft_train_answer_only.jsonl")
    parser.add_argument("--valid-input", default="outputs/sft_valid_answer_only.jsonl")
    parser.add_argument("--train-output", default="outputs/sft_train_mixed_reasoning_json.jsonl")
    parser.add_argument("--valid-output", default="outputs/sft_valid_mixed_reasoning_json.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_records = [convert_record(record) for record in load_jsonl(Path(args.train_input))]
    valid_records = [convert_record(record) for record in load_jsonl(Path(args.valid_input))]
    dump_jsonl(train_records, Path(args.train_output))
    dump_jsonl(valid_records, Path(args.valid_output))
    print(
        json.dumps(
            {
                "train_output": args.train_output,
                "train_count": len(train_records),
                "valid_output": args.valid_output,
                "valid_count": len(valid_records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
