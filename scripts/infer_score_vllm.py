"""Run SCoRE2026 inference with vLLM.

This keeps the same input/output contract as infer_score.py, but batches
prompts through vLLM so MoE models can be tested quickly on multi-GPU servers.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
import types
from pathlib import Path
from typing import Any

def ensure_sklearn_runtime() -> None:
    """Provide a tiny sklearn.metrics fallback for broken inference envs."""

    try:
        from sklearn.metrics import roc_curve  # noqa: F401
        return
    except Exception:
        metrics_module = types.ModuleType("sklearn.metrics")
        metrics_module.__spec__ = importlib.machinery.ModuleSpec(
            "sklearn.metrics",
            loader=None,
        )

        def roc_curve(*args: object, **kwargs: object) -> None:
            raise RuntimeError("sklearn.metrics.roc_curve is unavailable in this runtime.")

        metrics_module.roc_curve = roc_curve

        sklearn_module = sys.modules.get("sklearn") or types.ModuleType("sklearn")
        sklearn_module.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
        sklearn_module.metrics = metrics_module
        sys.modules["sklearn"] = sklearn_module
        sys.modules["sklearn.metrics"] = metrics_module


ensure_sklearn_runtime()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.infer_score import (  # noqa: E402
    build_user_prompt_with_count_hint,
    dump_jsonl,
    extract_answer,
    infer_domain,
    is_correct,
    load_jsonl,
    load_prompts,
)


def build_chat_prompt(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    enable_thinking: bool,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def run_inference(
    *,
    model_path: str,
    records: list[dict[str, Any]],
    prompts: dict[str, str],
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    include_answer_count_hint: bool,
    enable_thinking: bool,
    enforce_eager: bool,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if limit is not None:
        records = records[:limit]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    request_meta: list[dict[str, Any]] = []
    request_texts: list[str] = []

    for record in records:
        domain = infer_domain(record)
        system_prompt = prompts.get(domain) or prompts["general"]
        user_prompt = build_user_prompt_with_count_hint(
            record,
            include_answer_count_hint=include_answer_count_hint,
        )
        request_meta.append({"record": record, "domain": domain})
        request_texts.append(
            build_chat_prompt(
                tokenizer,
                system_prompt,
                user_prompt,
                enable_thinking=enable_thinking,
            )
        )

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    generations = llm.generate(request_texts, sampling_params)

    outputs = []
    correct = 0
    scored = 0
    for meta, generation in zip(request_meta, generations):
        record = meta["record"]
        domain = meta["domain"]
        raw_output = generation.outputs[0].text.strip() if generation.outputs else ""
        prediction = extract_answer(raw_output, set(record["options"].keys()))
        matched = is_correct(prediction, record.get("answer"))
        if matched is not None:
            scored += 1
            correct += int(matched)
        outputs.append(
            {
                "id": record.get("id"),
                "domain": domain,
                "answer": prediction,
                "raw_output": raw_output,
                "gold": record.get("answer"),
                "correct": matched,
            }
        )

    metrics = {
        "total": len(outputs),
        "scored": scored,
        "correct": correct,
        "accuracy": correct / scored if scored else None,
        "model_path": model_path,
    }
    return outputs, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCoRE2026 inference with vLLM.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", default="configs/system_prompts_v2.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--answer-count-hint", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.input))
    prompts = load_prompts(Path(args.prompts))
    outputs, metrics = run_inference(
        model_path=args.model_path,
        records=records,
        prompts=prompts,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        include_answer_count_hint=bool(args.answer_count_hint),
        enable_thinking=bool(args.enable_thinking),
        enforce_eager=bool(args.enforce_eager),
        limit=args.limit,
    )
    dump_jsonl(outputs, Path(args.output))
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
