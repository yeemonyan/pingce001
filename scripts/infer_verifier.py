"""Run option-level verifier inference for SCoRE2026."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.infer_score import (
    DEFAULT_PROMPTS_PATH,
    TransformersBackend,
    load_jsonl,
    load_prompts,
)


VALID_LABELS = ("A", "B", "C", "D")


def build_candidate_prompt(record: dict[str, Any], candidate_label: str) -> str:
    option_lines = "\n".join(
        f"{label}. {value}"
        for label, value in record["options"].items()
    )
    return (
        f"Text:\n{record['text']}\n\n"
        f"Question:\n{record['question']}\n\n"
        f"All Options:\n{option_lines}\n\n"
        f"Candidate Option:\n{candidate_label}. {record['options'][candidate_label]}\n\n"
        "Decide whether the candidate option is supported.\n"
        'Return only JSON: {"label":"yes"} or {"label":"no"}.'
    )


def parse_verifier_label(text: str) -> str:
    payload = text.strip().lower()
    if '"label"' in payload:
        if '"yes"' in payload:
            return "yes"
        if '"no"' in payload:
            return "no"
    if "yes" in payload and "no" not in payload:
        return "yes"
    if "no" in payload and "yes" not in payload:
        return "no"
    return "no"


def merge_labels(candidates: list[tuple[str, str]]) -> list[str]:
    selected = [label for label, verdict in candidates if verdict == "yes"]
    return selected or ["D"]


class VerifierMockBackend:
    """Deterministic backend for local smoke tests.

    It returns yes when the candidate label belongs to the gold answer list,
    otherwise no. This is only for pipeline verification.
    """

    def __init__(self) -> None:
        self.current_record: dict[str, Any] | None = None
        self.current_label: str | None = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        gold = set((self.current_record or {}).get("answer") or [])
        verdict = "yes" if self.current_label in gold else "no"
        return json.dumps({"label": verdict}, ensure_ascii=False)


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run option-level verifier inference.")
    parser.add_argument("--input", required=True, help="Normalized dev/test JSONL.")
    parser.add_argument("--output", required=True, help="Merged question-level output JSONL.")
    parser.add_argument("--raw-output", default=None, help="Optional option-level raw JSONL.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH))
    parser.add_argument("--backend", choices=("mock", "transformers"), default="transformers")
    parser.add_argument("--model-path", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(Path(args.prompts))
    records = load_jsonl(Path(args.input))
    if args.backend == "mock":
        backend: Any = VerifierMockBackend()
    else:
        backend = TransformersBackend(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            dtype=args.dtype,
            device_map=args.device_map,
        )

    merged_outputs: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []

    for record in records:
        prompt_domain = str(record.get("domain") or "general")
        system_prompt = prompts.get(prompt_domain) or prompts["general"]
        option_results: list[tuple[str, str]] = []
        option_raw: dict[str, Any] = {}
        for label in VALID_LABELS:
            if label not in record["options"]:
                continue
            if isinstance(backend, VerifierMockBackend):
                backend.current_record = record
                backend.current_label = label
            user_prompt = build_candidate_prompt(record, label)
            raw_output = backend.generate(system_prompt, user_prompt)
            verdict = parse_verifier_label(raw_output)
            option_results.append((label, verdict))
            option_raw[label] = {"raw_output": raw_output, "verdict": verdict}

        merged = merge_labels(option_results)
        merged_outputs.append(
            {
                "id": record.get("id"),
                "answers": merged,
                "domain": record.get("domain"),
                "option_results": option_raw,
            }
        )
        raw_outputs.append(
            {
                "id": record.get("id"),
                "domain": record.get("domain"),
                "option_results": option_raw,
            }
        )

    write_jsonl(merged_outputs, Path(args.output))
    if args.raw_output:
        write_jsonl(raw_outputs, Path(args.raw_output))
    print(json.dumps({"count": len(merged_outputs), "output": args.output, "raw_output": args.raw_output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
