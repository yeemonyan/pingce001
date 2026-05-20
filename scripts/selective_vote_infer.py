"""Run selective vote inference for SCoRE2026 on high-risk domains only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.infer_score import (
    GenerationBackend,
    MockBackend,
    TransformersBackend,
    VALID_LABELS,
    build_user_prompt,
    dump_jsonl,
    extract_answer,
    infer_domain,
    is_correct,
    load_jsonl,
    load_prompts,
)


STATEMENT_CUES = (
    "select the correct",
    "select the incorrect",
    "statement(s)",
    "以下选项",
    "正确的是",
    "不正确的是",
)

TEMPORAL_CUES = (
    "before",
    "after",
    "gap",
    "days",
    "day",
    "years",
    "year",
    "weekday",
    "之前",
    "之后",
    "相差",
)

SPATIAL_CUES = (
    "left",
    "right",
    "above",
    "below",
    "neighbor",
    "same layer",
    "adjacent",
    "clockwise",
    "左",
    "右",
    "上方",
    "下方",
    "同层",
    "相邻",
)

NONE_OF_ABOVE_CUES = (
    "none of the above",
    "以上都不",
    "以上都不是",
    "都不正确",
)


def build_system_prompt_variants(primary: dict[str, str], extras: list[dict[str, str]], domain: str) -> list[str]:
    variants = [primary.get(domain) or primary["general"]]
    for prompt_map in extras:
        variants.append(prompt_map.get(domain) or prompt_map["general"])
    deduped: list[str] = []
    for item in variants:
        if item not in deduped:
            deduped.append(item)
    return deduped


def text_blob(record: dict[str, Any]) -> str:
    return f"{record.get('text', '')} {record.get('question', '')}".lower()


def is_multi_answer_like(record: dict[str, Any]) -> bool:
    blob = text_blob(record)
    return any(cue in blob for cue in STATEMENT_CUES)


def none_of_above_label(record: dict[str, Any]) -> str | None:
    for label, text in record.get("options", {}).items():
        normalized = str(text).strip().lower()
        if any(cue in normalized for cue in NONE_OF_ABOVE_CUES):
            return str(label).strip().upper()
    return None


def should_vote(record: dict[str, Any], first_prediction: list[str]) -> tuple[bool, str]:
    domain = infer_domain(record)
    if domain not in {"temporal", "spatial", "hybrid"}:
        return False, "domain_off"

    blob = text_blob(record)
    multi_cue = any(cue in blob for cue in STATEMENT_CUES)
    temporal_cue = any(cue in blob for cue in TEMPORAL_CUES)
    spatial_cue = any(cue in blob for cue in SPATIAL_CUES)
    output_uncertain = len(first_prediction) != 1
    none_label = none_of_above_label(record)
    none_selected = bool(none_label and first_prediction == [none_label])

    if domain == "hybrid":
        return True, "hybrid_default"
    if domain == "temporal" and (multi_cue or temporal_cue or output_uncertain):
        return True, "temporal_uncertain"
    if domain == "spatial" and (multi_cue or spatial_cue or output_uncertain or none_selected):
        return True, "spatial_uncertain"
    return False, "stable_first_pass"


def aggregate_votes(
    record: dict[str, Any],
    predictions: list[list[str]],
    *,
    multi_min_votes: int,
    none_margin: int,
) -> list[str]:
    first_prediction = predictions[0] if predictions else []
    counts: Counter[str] = Counter()
    for prediction in predictions:
        for label in prediction:
            counts[label] += 1

    if is_multi_answer_like(record):
        chosen = [label for label in VALID_LABELS if label in record["options"] and counts[label] >= multi_min_votes]
        if chosen:
            return chosen
        return first_prediction or ([max(counts, key=counts.get)] if counts else [])

    if not counts:
        return first_prediction

    none_label = none_of_above_label(record)
    ordered_labels = [label for label in VALID_LABELS if label in record["options"]]
    best_label = max(ordered_labels, key=lambda label: (counts[label], label in first_prediction, -ordered_labels.index(label)))
    best_count = counts[best_label]
    tied = [label for label in ordered_labels if counts[label] == best_count]
    if len(tied) > 1 and first_prediction:
        return first_prediction[:1]

    if none_label and best_label == none_label:
        runner_up = max((counts[label] for label in ordered_labels if label != none_label), default=0)
        if best_count - runner_up < none_margin and first_prediction:
            return first_prediction[:1]
    return [best_label]


def run_selective_vote(
    records: list[dict[str, Any]],
    primary_prompts: dict[str, str],
    extra_prompt_maps: list[dict[str, str]],
    backend: GenerationBackend,
    *,
    multi_min_votes: int,
    none_margin: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    total = correct = scored = 0
    vote_used = 0
    domain_counts: Counter[str] = Counter()
    domain_vote_used: Counter[str] = Counter()
    vote_reason_counts: Counter[str] = Counter()

    for record in records:
        if isinstance(backend, MockBackend):
            backend.current_record = record

        domain = infer_domain(record)
        domain_counts[domain] += 1
        allowed_labels = set(record["options"].keys())
        prompt_variants = build_system_prompt_variants(primary_prompts, extra_prompt_maps, domain)
        user_prompt = build_user_prompt(record)

        raw_outputs: list[str] = []
        parsed_predictions: list[list[str]] = []
        for system_prompt in prompt_variants:
            raw_output = backend.generate(system_prompt, user_prompt)
            raw_outputs.append(raw_output)
            parsed_predictions.append(extract_answer(raw_output, allowed_labels))

        first_prediction = parsed_predictions[0] if parsed_predictions else []
        use_vote, vote_reason = should_vote(record, first_prediction)
        final_prediction = first_prediction
        if use_vote and len(parsed_predictions) > 1:
            final_prediction = aggregate_votes(
                record,
                parsed_predictions,
                multi_min_votes=multi_min_votes,
                none_margin=none_margin,
            )
            vote_used += 1
            domain_vote_used[domain] += 1
            vote_reason_counts[vote_reason] += 1

        matched = is_correct(final_prediction, record.get("answer"))
        total += 1
        if matched is not None:
            scored += 1
            correct += int(matched)

        outputs.append(
            {
                "id": record.get("id"),
                "domain": domain,
                "answer": final_prediction,
                "first_answer": first_prediction,
                "vote_used": use_vote and len(parsed_predictions) > 1,
                "vote_reason": vote_reason,
                "vote_candidates": parsed_predictions,
                "raw_output": raw_outputs[0] if raw_outputs else "",
                "raw_outputs": raw_outputs,
                "gold": record.get("answer"),
                "correct": matched,
            }
        )

    metrics = {
        "total": total,
        "scored": scored,
        "correct": correct,
        "accuracy": (correct / scored) if scored else None,
        "domain_counts": dict(domain_counts),
        "vote_used": vote_used,
        "domain_vote_used": dict(domain_vote_used),
        "vote_reason_counts": dict(vote_reason_counts),
    }
    return outputs, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selective vote inference for Qwen answer_only LoRA.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", default="configs/system_prompts_v2.yaml")
    parser.add_argument(
        "--vote-prompts",
        nargs="*",
        default=["configs/system_prompts_round3_short.yaml"],
        help="Additional prompt YAML files used only for vote variants.",
    )
    parser.add_argument("--backend", choices=("mock", "transformers"), default="transformers")
    parser.add_argument("--model-path", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--report", default=None)
    parser.add_argument("--multi-min-votes", type=int, default=2)
    parser.add_argument("--none-margin", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.input))
    primary_prompts = load_prompts(Path(args.prompts))
    extra_prompt_maps = [load_prompts(Path(path)) for path in args.vote_prompts]

    if args.backend == "mock":
        backend: GenerationBackend = MockBackend()
    else:
        backend = TransformersBackend(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            dtype=args.dtype,
            device_map=args.device_map,
        )

    outputs, metrics = run_selective_vote(
        records,
        primary_prompts,
        extra_prompt_maps,
        backend,
        multi_min_votes=args.multi_min_votes,
        none_margin=args.none_margin,
    )
    dump_jsonl(outputs, Path(args.output))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
