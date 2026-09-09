import json, sys, time, os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.infer_score import (
    TransformersBackend, load_jsonl, load_prompts,
    run_inference, dump_jsonl
)

MODEL = os.environ.get('MODEL_PATH', 'models/Qwen2.5-7B-Instruct')
ADAPTER = os.environ.get('ADAPTER_PATH', 'checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026')
PROMPTS = 'configs/system_prompts_v4_self_determine.yaml'
INPUT = 'outputs/test_prompts_with_cardinality.jsonl'
OUTPUT = 'outputs/test_lora_v4_predictions.jsonl'

print('Loading backend...', flush=True)
backend = TransformersBackend(
    model_path=MODEL, adapter_path=ADAPTER,
    max_new_tokens=256, temperature=0.0, top_p=1.0,
    dtype='bfloat16', device_map='cuda:0'
)

print('Loading data...', flush=True)
records = load_jsonl(Path(INPUT))
prompts = load_prompts(Path(PROMPTS))
print(f'Loaded {len(records)} records', flush=True)

print('Running inference...', flush=True)
t0 = time.time()
outputs, metrics = run_inference(records, prompts, backend, include_auto_count_hint=True)
t1 = time.time()

print(f'Done in {t1-t0:.1f}s ({len(records)/(t1-t0):.1f} samples/s)', flush=True)
print(json.dumps(metrics, ensure_ascii=False), flush=True)

dump_jsonl(outputs, Path(OUTPUT))
print(f'Written to {OUTPUT}', flush=True)
