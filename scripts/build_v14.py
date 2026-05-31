import json, os
from collections import Counter

new = [json.loads(l) for l in open('outputs/test_lora_ao_temporal3x_predictions.jsonl')]
ao  = [json.loads(l) for l in open('outputs/test_lora_answer_only_predictions.jsonl')]
cards = [json.loads(l) for l in open('outputs/test_prompts_with_cardinality.jsonl')]
card_by_id = {c['id']: c for c in cards}

def get_ans(p):
    return p.get('answer', p.get('answers', []))

# V14: V7d routing, temporal3x adapter in mixed slots
r14 = []
s = Counter()
for pn, pa in zip(new, ao):
    pid = pn['id']
    dom = pn.get('domain', '')
    card = card_by_id.get(pid, {}).get('predicted_cardinality', 'single')
    d = (dom or '').lower()
    if ('time' in d or 'temporal' in d or 'hybrid' in d) and card == 'multi':
        r14.append({'id': pid, 'answers': get_ans(pn)})
        s['new'] += 1
    else:
        r14.append({'id': pid, 'answers': get_ans(pa)})
        s['ao'] += 1

c14 = Counter(len(x['answers']) for x in r14)
print(f'V14: new={s["new"]}, ao={s["ao"]}, dist={dict(c14)}')

# V14b: new adapter standalone
r14b = [{'id': p['id'], 'answers': get_ans(p)} for p in new]
c14b = Counter(len(x['answers']) for x in r14b)
print(f'V14b (standalone): dist={dict(c14b)}')

os.makedirs('outputs/submissions', exist_ok=True)
with open('outputs/submissions/v14_submission.json', 'w') as f:
    json.dump(r14, f, ensure_ascii=False, indent=2)
with open('outputs/submissions/v14b_submission.json', 'w') as f:
    json.dump(r14b, f, ensure_ascii=False, indent=2)
print('DONE')
