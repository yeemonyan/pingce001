"""V17: V7d routing + V15 post-processing on AO predictions."""
import json, os
from collections import Counter

v5 = [json.loads(l) for l in open('outputs/test_lora_v5_predictions.jsonl')]
ao = [json.loads(l) for l in open('outputs/test_lora_answer_only_predictions.jsonl')]
cards = [json.loads(l) for l in open('outputs/test_prompts_with_cardinality.jsonl')]
card_by_id = {c['id']: c for c in cards}

def get_ans(p):
    return p.get('answer', p.get('answers', []))

# V17: V7d routing, then trim AO over-selection on non-temporal
r17 = []
s = Counter()
for pv, pa in zip(v5, ao):
    pid = pv['id']
    dom = pv.get('domain', '')
    card = card_by_id.get(pid, {}).get('predicted_cardinality', 'single')
    d = (dom or '').lower()
    
    if ('time' in d or 'temporal' in d or 'hybrid' in d) and card == 'multi':
        ans = get_ans(pv)  # mixed
        s['v5'] += 1
    else:
        ans = get_ans(pa)  # AO
        # V15 fix: AO=3 on non-temporal -> trim to 1
        if len(ans) == 3 and not ('time' in d or 'temporal' in d):
            ans = ans[:1]
            s['ao_trim'] += 1
        else:
            s['ao_keep'] += 1
    
    r17.append({'id': pid, 'answers': ans})

c = Counter(len(x['answers']) for x in r17)
print(f'V17: {dict(s)}')
print(f'Dist: {dict(c)}')

# V17b: V7d + also fix AO=4
r17b = []
s2 = Counter()
for pv, pa in zip(v5, ao):
    pid = pv['id']
    dom = pv.get('domain', '')
    card = card_by_id.get(pid, {}).get('predicted_cardinality', 'single')
    d = (dom or '').lower()
    
    if ('time' in d or 'temporal' in d or 'hybrid' in d) and card == 'multi':
        ans = get_ans(pv)
        s2['v5'] += 1
    else:
        ans = get_ans(pa)
        ao_n = len(ans)
        # All over-selection fixes on AO
        if ao_n == 4:
            ans = ans[:2]  # reduce to 2
            s2['ao4_trim'] += 1
        elif ao_n == 3 and not ('time' in d or 'temporal' in d):
            ans = ans[:1]
            s2['ao3_trim'] += 1
        else:
            s2['ao_keep'] += 1
    
    r17b.append({'id': pid, 'answers': ans})

c2 = Counter(len(x['answers']) for x in r17b)
print(f'V17b: {dict(s2)}')
print(f'Dist: {dict(c2)}')

os.makedirs('outputs/submissions', exist_ok=True)
with open('outputs/submissions/v17_submission.json', 'w') as f:
    json.dump(r17, f, ensure_ascii=False, indent=2)
with open('outputs/submissions/v17b_submission.json', 'w') as f:
    json.dump(r17b, f, ensure_ascii=False, indent=2)
print('DONE')
