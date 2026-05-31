import json, os
from collections import Counter

v2 = [json.loads(l) for l in open('outputs/test_lora_v2_neutral_predictions.jsonl')]
ao = [json.loads(l) for l in open('outputs/test_lora_answer_only_predictions.jsonl')]
cards = [json.loads(l) for l in open('outputs/test_prompts_with_cardinality.jsonl')]
card_by_id = {c['id']: c for c in cards}

def get_ans(p):
    return p.get('answer', p.get('answers', []))

# V11: V7d routing, V2 neutral in mixed slots
r11 = []
s = Counter()
for pv, pa in zip(v2, ao):
    pid = pv['id']
    dom = pv.get('domain', '')
    card = card_by_id.get(pid, {}).get('predicted_cardinality', 'single')
    d = (dom or '').lower()
    if ('time' in d or 'temporal' in d or 'hybrid' in d) and card == 'multi':
        r11.append({'id': pid, 'answers': get_ans(pv)})
        s['v2'] += 1
    else:
        r11.append({'id': pid, 'answers': get_ans(pa)})
        s['ao'] += 1

c = Counter(len(x['answers']) for x in r11)
print(f'V11: v2={s["v2"]}, ao={s["ao"]}, dist={dict(c)}')

# Also build V11b: V2 for ALL questions (standalone)
r11b = [{'id': p['id'], 'answers': get_ans(p)} for p in v2]
c2 = Counter(len(x['answers']) for x in r11b)
print(f'V11b (V2 standalone): dist={dict(c2)}')

os.makedirs('outputs/submissions', exist_ok=True)
with open('outputs/submissions/v11_submission.json', 'w') as f:
    json.dump(r11, f, ensure_ascii=False, indent=2)
with open('outputs/submissions/v11b_submission.json', 'w') as f:
    json.dump(r11b, f, ensure_ascii=False, indent=2)
print('DONE')
