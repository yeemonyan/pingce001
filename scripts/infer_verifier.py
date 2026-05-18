"""Run option-level verifier inference and merge yes/no verdicts into answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.infer_score import MockBackend, TransformersBackend, infer_domain, load_jsonl
from scripts.verifier_utils import (
    VERIFIER_SYSTEM_PROMPT,
    build_option_prompt,
    detailed_metrics,
    extract_verdict,
    merge_option_predictions,
    option_labels,
    ordered_answers,
    write_json,
    write_jsonl,
)


class VerifierBackend(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a yes/no verifier response."""


class GoldMockVerifierBackend:
    """Mock verifier that emits gold yes/no labels for local flow tests."""

    def __init__(self) -> None:
        self.current_record: dict[str, Any] | None = None
        self.current_label: str | None = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        answers = set(ordered_answers(self.current_record or {}))
        label = "yes" if self.current_label in answers else "no"
        return json.dumps({"target_label": label}, ensure_ascii=False)


def run_verifier(
    records: list[dict[str, Any]],
    backend: VerifierBackend,
    yes_threshold: float | None = None,
    include_all_options: bool = True,
    instruction_variant: str = "base",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    option_outputs = []
    for record in records:
        domain = infer_domain(record)
        for option_label in option_labels(record["options"]):
            if isinstance(backend, GoldMockVerifierBackend):
                backend.current_record = record
                backend.current_label = option_label
            raw_output = backend.generate(
                VERIFIER_SYSTEM_PROMPT,
                build_option_prompt(
                    record,
                    option_label,
                    include_all_options=include_all_options,
                    instruction_variant=instruction_variant,
                ),
            )
            verdict = extract_verdict(raw_output)
            option_outputs.append(
                {
                    "id": f"{record.get('id')}::option::{option_label}",
                    "question_id": record.get("id"),
                    "domain": domain,
                    "option_label": option_label,
                    "target_label": verdict or "no",
                    "label": verdict or "no",
                    "raw_output": raw_output,
                    "gold_answer": record.get("answer"),
                    "input_variant": "all_options" if include_all_options else "candidate_only",
                    "instruction_variant": instruction_variant,
                }
            )
    return option_outputs, merge_option_predictions(option_outputs, yes_threshold=yes_threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCoRE2026 option-level verifier inference.")
    parser.add_argument("--input", required=True, help="Question-level JSONL.")
    parser.add_argument("--output", required=True, help="Merged question-level prediction JSONL.")
    parser.add_argument("--option-output", default=None, help="Optional option-level verdict JSONL.")
    parser.add_argument("--report", default=None, help="Optional detailed metric report JSON.")
    parser.add_argument("--backend", choices=("mock", "transformers"), default="transformers")
    parser.add_argument("--model-path", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--yes-threshold",
        type=float,
        default=None,
        help="Reserved for future probability/logit outputs. If yes_score exists, select options above this threshold.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--candidate-only", action="store_true", help="Only include current candidate option in user prompt.")
    parser.add_argument("--instruction-variant", choices=("base", "domain_hint"), default="base")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.input))
    if args.backend == "mock":
        backend: VerifierBackend = GoldMockVerifierBackend()
    else:
        backend = TransformersBackend(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            dtype=args.dtype,
            device_map=args.device_map,
        )

    option_outputs, merged_outputs = run_verifier(
        records,
        backend,
        yes_threshold=args.yes_threshold,
        include_all_options=not args.candidate_only,
        instruction_variant=args.instruction_variant,
    )
    write_jsonl(merged_outputs, Path(args.output))
    if args.option_output:
        write_jsonl(option_outputs, Path(args.option_output))

    metrics = detailed_metrics(records, merged_outputs)
    if args.report:
        write_json(metrics, Path(args.report))
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
