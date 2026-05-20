#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PREDICTION_INPUT="${PREDICTION_INPUT:-outputs/dev_deepseek_verifier_probe_predictions.jsonl}"
SUBMISSION_OUTPUT="${SUBMISSION_OUTPUT:-outputs/submissions/deepseek_verifier_submission.json}"
REPORT_OUTPUT="${REPORT_OUTPUT:-outputs/submissions/deepseek_verifier_submission_report.json}"

python3 scripts/format_submission.py --input "$PREDICTION_INPUT" --output "$SUBMISSION_OUTPUT"
python3 scripts/validate_submission.py --input "$SUBMISSION_OUTPUT" --report "$REPORT_OUTPUT"

echo "submission_ready=$SUBMISSION_OUTPUT"
