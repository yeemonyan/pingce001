"""Build SFT data with temporal multi oversampling (3x), keeping original format."""
import json, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_sft_data import (
    normalize_records_by_id, select_records, load_json,
    build_sft_item, build_dataset, write_jsonl, validate_items
)
from scripts.infer_score import load_prompts

INPUT = Path('data/raw/train.json')
TRAIN_IDS = Path('data/splits/train_ids.json')
DEV_IDS = Path('data/splits/dev_ids.json')
PROMPTS = Path('configs/system_prompts_v3_mixed_focus.yaml')

# Load data
by_id = normalize_records_by_id(INPUT)
train_records = select_records(by_id, load_json(TRAIN_IDS))
valid_records = select_records(by_id, load_json(DEV_IDS))

# Identify temporal + multi questions
temporal_multi = []
other = []
for r in train_records:
    dom = r.get('domain', '')
    is_temporal = dom in ('temporal',) or 'time' in str(dom).lower()
    is_multi = len(r.get('answer', r.get('answers', []))) > 1
    if is_temporal and is_multi:
        temporal_multi.append(r)
    else:
        other.append(r)

print(f'Temporal multi: {len(temporal_multi)}')
print(f'Other: {len(other)}')
print(f'After 3x oversample: {len(temporal_multi)*3 + len(other)} total')

# Build oversampled training set
oversampled = other + temporal_multi * 3
prompts = load_prompts(PROMPTS)
items = build_dataset(oversampled, prompts, 'answer_only')
validate_items(items)

# Build validation set (no oversampling)
valid_items = build_dataset(valid_records, prompts, 'answer_only')

# Write
write_jsonl(items, Path('outputs/sft_train_ao_temporal3x.jsonl'))
write_jsonl(valid_items, Path('outputs/sft_valid_ao_temporal3x.jsonl'))

# Stats
from scripts.build_sft_data import summarize
stats_train = summarize(oversampled, items)
stats_valid = summarize(valid_records, valid_items)
print(json.dumps({'train': stats_train, 'valid': stats_valid}, ensure_ascii=False, indent=2))
print('DONE')
