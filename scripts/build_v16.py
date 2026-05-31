import json, os
from collections import Counter

rs = [json.loads(l) for l in open('outputs/test_lora_reasoning_short_predictions.jsonl')]
ao = [json.loads(l) for l in open('outputs/test_lora_answer_only_predictions.jsonl')]
cards = [json.loads(l) for l in open('outputs/test_prompts_with_cardinality.jsonl')]
card_by_id = {c['id']: c for c in cards}

def get_ans(p):
    return p.get('answer', p.get('answers', []))

# V16: V7d routing, reasoning_short in mixed slots
r16 = []
s = Counter()
for pr, pa in zip(rs, ao):
    pid = pr['id']
    dom = pr.get('domain', '')
    card = card_by_id.get(pid, {}).get('predicted_cardinality', 'single')
    d = (dom or '').lower()
    if ('time' in d or 'temporal' in d or 'hybrid' in d) and card == 'multi':
        r16.append({'id': pid, 'answers': get_ans(pr)})
        s['rs'] += 1
    else:
        r16.append({'id': pid, 'answers': get_ans(pa)})
        s['ao'] += 1

c16 = Counter(len(x['answers']) for x in r16)
print(f'V16: rs={s["rs"]}, ao={s["ao"]}, dist={dict(c16)}')

# V16b: standalone
r16b = [{'id': p['id'], 'answers': get_ans(p)} for p in rs]
c16b = Counter(len(x['answers']) for x in r16b)
print(f'V16b (standalone): dist={dict(c16b)}')

os.makedirs('outputs/submissions', exist_ok=True)
with open('outputs/submissions/v16_submission.json', 'w') as f:
    json.dump(r16, f, ensure_ascii=False, indent=2)
with open('outputs/submissions/v16b_submission.json', 'w') as f:
    json.dump(r16b, f, ensure_ascii=False, indent=2)
print('DONE')
