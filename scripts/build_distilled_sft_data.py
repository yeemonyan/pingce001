"""Build distilled SFT data from API teacher responses."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_sft_data import answer_count_instruction
from scripts.build_teacher_distill_requests import build_request as build_teacher_request
from scripts.build_teacher_distill_requests import detect_question_polarity
from scripts.build_teacher_distill_requests import load_prompt_config as load_teacher_prompt_config
from scripts.infer_score import build_user_prompt, infer_domain, load_prompts
from scripts.make_dev_split import answer_kind, detect_language
from scripts.parse_score_json import load_records, normalize_record


DEFAULT_INPUT = Path("data/raw/train.json")
DEFAULT_TRAIN_IDS = Path("data/splits/train_ids.json")
DEFAULT_DEV_IDS = Path("data/splits/dev_ids.json")
DEFAULT_PROMPTS = Path("configs/system_prompts_v3_mixed_focus.yaml")
DEFAULT_TEACHER_PROMPTS = Path("configs/teacher_distill_prompts.yaml")
DEFAULT_TEACHER_TRAIN = Path("outputs/distill/teacher_responses_train.jsonl")
DEFAULT_TEACHER_VALID = Path("outputs/distill/teacher_responses_valid.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "distill_data_report.json"
RETRYABLE_STATUSES = {
    "invalid_json",
    "missing_option_judgments",
    "missing_option",
    "invalid_option_label",
    "inconsistent_selected_answers",
    "missing_teacher_output",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def normalize_records_by_id(input_path: Path) -> dict[str, dict[str, Any]]:
    raw_records = load_records(input_path)
    records = [normalize_record(record, index) for index, record in enumerate(raw_records)]
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


def build_sft_user_prompt(record: dict[str, Any]) -> str:
    return build_user_prompt(record) + "\n\n" + answer_count_instruction(record)


def load_split(by_id: dict[str, dict[str, Any]], ids_path: Path) -> list[dict[str, Any]]:
    ids = load_json(ids_path)
    return [by_id[str(item)] for item in ids if str(item) in by_id]


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "correct", "supported"}:
        return True
    if text in {"false", "no", "n", "0", "incorrect", "unsupported"}:
        return False
    return None


def normalize_reason(text: Any, limit: int = 80) -> str:
    cleaned = " ".join(str(text).replace("\r", " ").replace("\n", " ").split()).strip()
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 3].rstrip() + "..."
    return cleaned


def extract_payload(item: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(item.get("response_json"), dict):
        return item["response_json"]
    for key in ("response", "output", "content", "assistant", "raw_output"):
        value = item.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                continue
    return None


def normalize_teacher_result(record: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    payload = extract_payload(item)
    options = list(record["options"].keys())
    polarity = detect_question_polarity(record)
    if payload is None:
        return {"status": "invalid_json", "id": record["id"]}
    judgments = payload.get("option_judgments")
    if not isinstance(judgments, dict):
        return {"status": "missing_option_judgments", "id": record["id"], "payload": payload}

    normalized_judgments: dict[str, dict[str, Any]] = {}
    selected_answers: list[str] = []
    statement_truth_answers: list[str] = []
    for label in options:
        entry = judgments.get(label)
        if not isinstance(entry, dict):
            return {"status": "missing_option", "id": record["id"], "payload": payload, "option": label}
        selected_flag = normalize_bool(entry.get("selected"))
        truth_flag = normalize_bool(entry.get("label"))
        if selected_flag is None and truth_flag is None:
            return {"status": "invalid_option_label", "id": record["id"], "payload": payload, "option": label}
        if selected_flag is None and truth_flag is not None:
            selected_flag = (not truth_flag) if polarity == "select_incorrect" else truth_flag
        reason = normalize_reason(entry.get("reason", ""))
        if not reason:
            reason = "Insufficient explanation."
        normalized_judgments[label] = {"label": bool(selected_flag), "reason": reason}
        if selected_flag:
            selected_answers.append(label)
        if truth_flag:
            statement_truth_answers.append(label)

    payload_answers = []
    if isinstance(payload.get("answers"), list):
        payload_answers = [str(label).strip().upper() for label in payload["answers"] if str(label).strip().upper() in options]

    predicted_answers = payload_answers or selected_answers
    if predicted_answers and not payload_answers:
        payload_answers = predicted_answers
    if payload_answers and selected_answers and payload_answers != selected_answers:
        return {
            "status": "inconsistent_selected_answers",
            "id": record["id"],
            "payload": payload,
            "selected_answers": selected_answers,
            "payload_answers": payload_answers,
            "statement_truth_answers": statement_truth_answers,
        }
    if not predicted_answers:
        predicted_answers = selected_answers

    final_reasoning = normalize_reason(payload.get("final_reasoning", ""))
    gold_answers = list(record["answers"])
    if predicted_answers == gold_answers:
        return {
            "status": "accepted",
            "id": record["id"],
            "option_judgments": normalized_judgments,
            "final_reasoning": final_reasoning or "All selected options match the passage constraints.",
            "answers": gold_answers,
        }

    return {
        "status": "incorrect_answer",
        "id": record["id"],
        "option_judgments": normalized_judgments,
        "final_reasoning": final_reasoning,
        "predicted_answers": predicted_answers,
        "answers": gold_answers,
    }


def build_answer_only_assistant(record: dict[str, Any]) -> str:
    return json.dumps({"answers": list(record["answers"])}, ensure_ascii=False, separators=(",", ":"))


def build_short_reasoning_assistant(result: dict[str, Any], record: dict[str, Any]) -> str:
    reasoning = result.get("final_reasoning") or " ".join(
        f"{label}{'对' if entry['label'] else '错'}" if record["language"] == "zh" else f"{label}:{'T' if entry['label'] else 'F'}"
        for label, entry in result["option_judgments"].items()
    )
    return json.dumps({"reasoning": reasoning, "answers": list(record["answers"])}, ensure_ascii=False, separators=(",", ":"))


def build_option_level_assistant(result: dict[str, Any], record: dict[str, Any]) -> str:
    option_judgments = {
        label: {
            "label": bool(entry["label"]),
            "reason": entry["reason"],
        }
        for label, entry in result["option_judgments"].items()
    }
    payload = {
        "option_judgments": option_judgments,
        "reasoning": result.get("final_reasoning") or "",
        "answers": list(record["answers"]),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_item(record: dict[str, Any], system_prompt: str, variant: str, assistant_content: str) -> dict[str, Any]:
    return {
        "id": record["id"],
        "domain": record["domain"],
        "language": record["language"],
        "variant": variant,
        "answer_kind": record["answer_kind"],
        "answers": list(record["answers"]),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_sft_user_prompt(record)},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def matches_bucket(record: dict[str, Any], buckets: set[str]) -> bool:
    if not buckets:
        return False
    domain = record["domain"]
    kind = record["answer_kind"]
    return domain in buckets or f"{domain}:{kind}" in buckets


def build_mix(
    pools: dict[str, list[dict[str, Any]]],
    ratios: dict[str, float],
    target_count: int,
    boost_buckets: set[str],
    boost_factor: int,
) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    counts: dict[str, int] = {}
    assigned = 0
    items = sorted(ratios.items(), key=lambda item: item[0])
    for index, (variant, ratio) in enumerate(items):
        if index == len(items) - 1:
            count = target_count - assigned
        else:
            count = int(math.floor(target_count * ratio))
            assigned += count
        counts[variant] = count

    missing_budget = 0
    for variant, count in list(counts.items()):
        if count <= 0:
            continue
        if pools.get(variant):
            continue
        missing_budget += count
        counts[variant] = 0
    if missing_budget > 0:
        fallback_variant = "answer_only" if pools.get("answer_only") else next(
            (variant for variant, pool in pools.items() if pool),
            None,
        )
        if fallback_variant is not None:
            counts[fallback_variant] = counts.get(fallback_variant, 0) + missing_budget

    mixed: list[dict[str, Any]] = []
    for variant, count in counts.items():
        base_pool = list(pools.get(variant, []))
        if boost_buckets and boost_factor > 1:
            boosted = [item for item in base_pool if matches_bucket(item, boost_buckets)]
            base_pool.extend(boosted * (boost_factor - 1))
        if not base_pool:
            continue
        for index in range(count):
            source = dict(base_pool[index % len(base_pool)])
            source["mix_variant"] = variant
            source["mix_repeat_index"] = index
            mixed.append(source)
    mixed.sort(key=lambda item: (item["id"], item["variant"], item.get("mix_repeat_index", 0)))
    return mixed


def write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_teacher_file(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    items = load_jsonl(path)
    by_id = {}
    for item in items:
        record_id = str(item.get("id", "")).strip()
        if record_id:
            by_id[record_id] = item
    return by_id


def parse_ratio(text: str) -> dict[str, float]:
    pairs = [segment.strip() for segment in text.split(",") if segment.strip()]
    data: dict[str, float] = {}
    total = 0.0
    for pair in pairs:
        key, value = pair.split("=", 1)
        ratio = float(value)
        data[key.strip()] = ratio
        total += ratio
    if not data or total <= 0:
        raise ValueError("mix ratios must be non-empty")
    return {key: value / total for key, value in data.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build distilled SFT data from API teacher responses.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--train-ids", default=str(DEFAULT_TRAIN_IDS))
    parser.add_argument("--dev-ids", default=str(DEFAULT_DEV_IDS))
    parser.add_argument("--prompt-config", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--teacher-prompt-config", default=str(DEFAULT_TEACHER_PROMPTS))
    parser.add_argument("--teacher-train", default=str(DEFAULT_TEACHER_TRAIN))
    parser.add_argument("--teacher-valid", default=str(DEFAULT_TEACHER_VALID))
    parser.add_argument("--mix-ratios", default="answer_only=0.4,short_reasoning=0.3,option_level=0.3")
    parser.add_argument("--boost-buckets", default="spatial:single,temporal:single,spatial:multi,temporal:multi")
    parser.add_argument("--boost-factor", type=int, default=2)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--retry-output", default="outputs/distill/teacher_retry_requests.jsonl")
    parser.add_argument("--fallback-incorrect-to-answer-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(Path(args.prompt_config))
    teacher_prompts = load_teacher_prompt_config(Path(args.teacher_prompt_config))
    ratios = parse_ratio(args.mix_ratios)
    boost_buckets = {item.strip() for item in args.boost_buckets.split(",") if item.strip()}

    by_id = normalize_records_by_id(Path(args.input))
    split_records = {
        "train": load_split(by_id, Path(args.train_ids)),
        "valid": load_split(by_id, Path(args.dev_ids)),
    }
    teacher_maps = {
        "train": parse_teacher_file(Path(args.teacher_train)),
        "valid": parse_teacher_file(Path(args.teacher_valid)),
    }

    generated: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {"answer_only": [], "short_reasoning": [], "option_level": []}
        for split in ("train", "valid")
    }
    retry_requests: list[dict[str, Any]] = []
    status_counter: Counter[str] = Counter()

    for split, records in split_records.items():
        for record in records:
            system_prompt = prompts.get(record["domain"], prompts.get("general", ""))
            answer_item = build_item(record, system_prompt, "answer_only", build_answer_only_assistant(record))
            generated[split]["answer_only"].append(answer_item)

            teacher_item = teacher_maps[split].get(str(record["id"]))
            if teacher_item is None:
                status_counter["missing_teacher_output"] += 1
                retry_item = build_teacher_request(record, teacher_prompts, split)
                retry_item["retry_reason"] = "missing_teacher_output"
                retry_requests.append(retry_item)
                continue

            normalized = normalize_teacher_result(record, teacher_item)
            status = str(normalized["status"])
            status_counter[status] += 1
            if status == "accepted":
                generated[split]["short_reasoning"].append(
                    build_item(record, system_prompt, "short_reasoning", build_short_reasoning_assistant(normalized, record))
                )
                generated[split]["option_level"].append(
                    build_item(record, system_prompt, "option_level", build_option_level_assistant(normalized, record))
                )
                continue

            if status in RETRYABLE_STATUSES:
                retry_requests.append(
                    dict(
                        build_teacher_request(record, teacher_prompts, split),
                        retry_reason=status,
                        predicted_answers=normalized.get("predicted_answers", []),
                    )
                )
                continue

            if status == "incorrect_answer" and args.fallback_incorrect_to_answer_only:
                continue

            retry_requests.append(
                dict(
                    build_teacher_request(record, teacher_prompts, split),
                    retry_reason=status,
                    predicted_answers=normalized.get("predicted_answers", []),
                )
            )

    mix_outputs: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid"):
        target_count = len(generated[split]["answer_only"])
        mix_outputs[split] = build_mix(
            pools=generated[split],
            ratios=ratios,
            target_count=target_count,
            boost_buckets=boost_buckets,
            boost_factor=max(args.boost_factor, 1),
        )

    output_map = {
        ("train", "answer_only"): Path("outputs/sft_train_distill_answer_only.jsonl"),
        ("valid", "answer_only"): Path("outputs/sft_valid_distill_answer_only.jsonl"),
        ("train", "short_reasoning"): Path("outputs/sft_train_distill_short_reasoning.jsonl"),
        ("valid", "short_reasoning"): Path("outputs/sft_valid_distill_short_reasoning.jsonl"),
        ("train", "option_level"): Path("outputs/sft_train_distill_option_level.jsonl"),
        ("valid", "option_level"): Path("outputs/sft_valid_distill_option_level.jsonl"),
        ("train", "mix"): Path("outputs/sft_train_distill_mix.jsonl"),
        ("valid", "mix"): Path("outputs/sft_valid_distill_mix.jsonl"),
    }
    for split in ("train", "valid"):
        for variant in ("answer_only", "short_reasoning", "option_level"):
            write_jsonl(generated[split][variant], output_map[(split, variant)])
        write_jsonl(mix_outputs[split], output_map[(split, "mix")])

    retry_path = Path(args.retry_output)
    write_jsonl(retry_requests, retry_path)

    report = {
        "teacher_train": str(args.teacher_train),
        "teacher_valid": str(args.teacher_valid),
        "teacher_prompt_config": str(args.teacher_prompt_config),
        "prompt_config": str(args.prompt_config),
        "mix_ratios": ratios,
        "boost_buckets": sorted(boost_buckets),
        "boost_factor": max(args.boost_factor, 1),
        "status_counts": dict(sorted(status_counter.items())),
        "retryable_statuses": sorted(RETRYABLE_STATUSES),
        "retryable_count": sum(status_counter.get(status, 0) for status in RETRYABLE_STATUSES),
        "split_counts": {
            split: {
                variant: len(items)
                for variant, items in generated[split].items()
            }
            for split in ("train", "valid")
        },
        "mix_counts": {split: len(mix_outputs[split]) for split in ("train", "valid")},
        "retry_count": len(retry_requests),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
