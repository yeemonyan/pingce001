# Dense Bucket Route Notes

Status: historical dev-only exploration. This note is retained to document an
attempted routing direction, but its generated outputs are not tracked in the
public branch and it is not the final paper-facing method. Use `README.md` and
`docs/PAPER_MAINLINE.md` for the public mainline.

## Goal

Push the dense Qwen2.5-7B line beyond the plain mixed_reasoning adapter by
routing each example to the better adapter family:

- `answer_only` stays stronger on some conservative buckets.
- `mixed_reasoning` stays stronger on the hard reasoning buckets.

This is a low-cost dense boost because it reuses existing checkpoints instead
of starting another full LoRA training run.

## Dev Routing Rule Explored Here

Use `mixed_reasoning` for:

- `temporal:*`
- `social:*`
- `hybrid:*`
- `spatial:multi`
- `natural:single`

Keep `answer_only` as the fallback for the other buckets.

On the fixed dev split, this routing reaches:

- `283 / 720`
- `0.3930555556`

Compared with prior dense checkpoints:

- `answer_only`: `261 / 720 = 0.3625`
- `mixed_reasoning`: `277 / 720 = 0.3847222222`

## Why This Works

- `mixed_reasoning` improves hard domains, especially temporal and hybrid.
- `answer_only` is still safer on conservative spatial single-answer cases.
- Routing by bucket is a dense analogue of selective expert use: keep the
  reasoning-heavy adapter only where it already proved stronger.

## Reproduction

Run:

```bash
bash scripts/run_qwen_dense_bucket_route_dev.sh
```

Key outputs:

- `outputs/dev_lora_dense_bucket_route_predictions.jsonl`
- `outputs/dev_lora_dense_bucket_route_report.json`
- `outputs/dev_lora_dense_bucket_route_eval.json`
- `outputs/dev_lora_dense_bucket_route_vs_mixed.json`
