"""Merge a PEFT LoRA adapter into the base causal LM and save the merged model."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base-model", required=True, help="Path to the base Hugging Face model.")
    parser.add_argument("--adapter", required=True, help="Path to the PEFT adapter directory.")
    parser.add_argument("--output", required=True, help="Directory to save the merged model.")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="Torch dtype used while loading the base model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map passed to from_pretrained.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    merged_model = peft_model.merge_and_unload()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"merged_model_saved={output_dir}")


if __name__ == "__main__":
    main()
