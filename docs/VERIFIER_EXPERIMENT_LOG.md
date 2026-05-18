# Verifier Experiment Log

## Fixed Metrics

Every verifier experiment must record:

- overall ACC
- per-domain ACC
- single vs multi ACC
- `over_predict`
- `under_predict`
- `empty_prediction`
- temporal ACC
- multi-answer ACC

## Experiment Template

```markdown
## YYYY-MM-DD experiment-name

- Branch:
- Model:
- Data version:
- Config:
- Input variant:
- Instruction variant:
- Oversampling:
- Train command:
- Inference command:
- Dev result:
  - overall:
  - temporal:
  - spatial:
  - hybrid:
  - single:
  - multi:
  - over_predict:
  - under_predict:
  - empty_prediction:
- Online result:
- Notes:
```

## 2026-05-18 Local Verifier Data QA

- Branch: `codex/round4-verifier-probe`
- Model: none, data/pipeline QA only
- Data version:
  - `outputs/sft_train_verifier.jsonl`
  - `outputs/sft_valid_verifier.jsonl`
- Config: `configs/train_lora_verifier.yaml`
- Input variant: all options
- Instruction variant: base
- Oversampling: `1`
- Commands:
  - `python scripts/build_verifier_data.py --compat-output-names`
  - `python scripts/audit_verifier_data.py --input outputs/sft_valid_verifier.jsonl --output outputs/verifier_data_audit.json`
  - `python scripts/infer_verifier.py --input outputs/dev_prompts.jsonl --backend mock --output outputs/dev_verifier_mock_predictions.jsonl --option-output outputs/dev_verifier_mock_option_predictions.jsonl --report outputs/dev_verifier_mock_eval.json`
- Dev result: mock flow only, expected `1.0`
- Data audit:
  - single samples: 10
  - multi samples: 10
  - temporal samples: 5
  - spatial samples: 5
  - hybrid samples: 5
  - Chinese samples: 10
  - long-constraint samples: 10
  - target alignment errors: 0
- Notes:
  - Server inspection was attempted but SSH host/password interaction blocked non-interactive access in this session.
  - No training was launched.
