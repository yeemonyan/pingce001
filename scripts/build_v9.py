import json, os
from collections import Counter

v5 = [json.loads(l) for l in open('outputs/test_lora_v5_predictions.jsonl')]
ao = [json.loads(l) for l in open('outputs/test_lora_answer_only_predictions.jsonl')]
cards = [json.loads(l) for l in open('outputs/test_prompts_with_cardinality.jsonl')]
card_by_id = {c['id']: c for c in cards}

def get_ans(p):
    return p.get('answer', p.get('answers', []))

def has_explicit_multi_cue(q):
    ql = q.lower()
    return any(x in ql for x in ['statement(s)', 'select the incorrect', 'select the correct']) or '以下选项中' in q

r9 = []
stats = Counter()

for p_v5, p_ao in zip(v5, ao):
    pid = p_v5['id']
    ao_ans = get_ans(p_ao)
    mix_ans = get_ans(p_v5)
    domain = p_v5.get('domain', '')
    q = card_by_id.get(pid, {}).get('question', '')
    
    d = (domain or '').lower()
    is_temporal = 'time' in d or 'temporal' in d
    ao_n = len(ao_ans)
    mix_n = len(mix_ans)
    multi_cue = has_explicit_multi_cue(q)
    
    # Default: AO
    use_mixed = False
    reason = 'ao_default'
    
    # Only use mixed when BOTH conditions met:
    # 1. temporal domain
    # 2. explicit multi cue (statement(s), Select..., 以下选项中)
    if is_temporal and multi_cue:
        use_mixed = True
        reason = 'temporal+cue'
    
    # Backstop: AO clearly broken (3-4 answers on temporal question)
    if is_temporal and ao_n >= 3:
        use_mixed = True
        reason = 'ao_broken'
    
    if use_mixed:
        r9.append({'id': pid, 'answers': mix_ans})
        stats[reason] += 1
    else:
        r9.append({'id': pid, 'answers': ao_ans})
        stats[reason] += 1

c = Counter(len(x['answers']) for x in r9)
print(f'V9: {dict(stats)}')
print(f'Answer dist: {dict(c)}')

os.makedirs('outputs/submissions', exist_ok=True)
with open('outputs/submissions/v9_submission.json', 'w') as f:
    json.dump(r9, f, ensure_ascii=False, indent=2)
print('Wrote v9_submission.json')
