"""Run a one-sample inference smoke test against a local causal LM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-sample local inference smoke test.")
    parser.add_argument("--model-dir", required=True, help="Local model directory.")
    parser.add_argument("--input", required=True, help="Input JSONL file produced by parse_score_json.py.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation length cap.")
    return parser.parse_args()


def load_first_record(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file:
        first_line = next((line for line in file if line.strip()), "")
    if not first_line:
        raise ValueError(f"No non-empty records found in {path}")
    return json.loads(first_line)


def main() -> None:
    args = parse_args()
    record = load_first_record(Path(args.input))
    prompt = (
        f"{record['prompt']}\n"
        '请只输出一个 JSON 对象，例如 {"answer":["A"]} 或 {"answer":["A","C"]}。'
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model_inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    prompt_len = model_inputs["input_ids"].shape[1]
    completion = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)

    print(f"id={record.get('id')}")
    print("completion:")
    print(completion)


if __name__ == "__main__":
    main()
