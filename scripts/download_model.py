"""Download the Day 1 baseline model from Hugging Face Hub.

The default model is Qwen2.5-7B-Instruct, a dense 7B-class model that fits
the SCoRE2026 Dense <= 8B rule. Model weights are saved outside Git tracking.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LOCAL_DIR = Path("models") / "Qwen2.5-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Model repository id.")
    parser.add_argument(
        "--local-dir",
        default=str(DEFAULT_LOCAL_DIR),
        help="Local directory for model files.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision, tag, or commit SHA for reproducibility.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional file pattern to include. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved download plan without downloading files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)

    if args.dry_run:
        print(f"repo_id={args.repo_id}")
        print(f"revision={args.revision or 'default'}")
        print(f"local_dir={local_dir}")
        print(f"allow_patterns={args.allow_pattern or 'all'}")
        return

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=args.allow_pattern,
    )
    print(f"Downloaded {args.repo_id} to {path}")


if __name__ == "__main__":
    main()
