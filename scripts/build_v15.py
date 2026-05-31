import json, os
from collections import Counter

ao = [json.loads(l) for l in open('outputs/test_lora_answer_only_predictions.jsonl')]
v5 = [json.loads(l) for l in open('outputs/test_lora_v5_predictions.jsonl')]
cards = [json.loads(l) for l in open('outputs/test_prompts_with_cardinality.jsonl')]
card_by_id = {c['id']: c for c in cards}

def get_ans(p):
    return p.get('answer', p.get('answers', []))

def has_multi_cue(q):
    ql = q.lower()
    return any(x in ql for x in ['statement(s)', 'select the incorrect', 'select the correct']) or '以下选项中' in q

# V15: Smart AO post-processing
r15 = []
stats = Counter()
for pa, pv in zip(ao, v5):
    pid = pa['id']
    ao_ans = get_ans(pa)
    mix_ans = get_ans(pv)
    q = card_by_id.get(pid, {}).get('question', '')
    dom = pa.get('domain', '')
    d = (dom or '').lower()
    is_temporal = 'time' in d or 'temporal' in d
    
    ao_n = len(ao_ans)
    final = ao_ans
    
    # Rule 1: AO 4 answers → too many, use mixed if mixed has 1-2
    if ao_n == 4 and len(mix_ans) <= 2:
        final = mix_ans
        stats['ao4_fix'] += 1
    # Rule 2: AO 3 answers on non-temporal → likely wrong, trim to 1
    elif ao_n == 3 and not is_temporal:
        final = ao_ans[:1]
        stats['ao3_trim'] += 1
    # Rule 3: AO 1 answer but explicit multi cue → use mixed
    elif ao_n == 1 and has_multi_cue(q) and len(mix_ans) >= 2:
        final = mix_ans
        stats['ao1_multi_fix'] += 1
    # Rule 4: AO 2+ answers on temporal, mixed has more → trust mixed
    elif is_temporal and ao_n >= 2 and len(mix_ans) > ao_n:
        final = mix_ans
        stats['temporal_mix'] += 1
    else:
        stats['ao_keep'] += 1
    
    r15.append({'id': pid, 'answers': final})

c = Counter(len(x['answers']) for x in r15)
print(f'V15: {dict(stats)}')
print(f'Dist: {dict(c)}')

os.makedirs('outputs/submissions', exist_ok=True)
with open('outputs/submissions/v15_submission.json', 'w') as f:
    json.dump(r15, f, ensure_ascii=False, indent=2)
print('DONE')
