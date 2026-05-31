import json, os
from collections import Counter

v2 = [json.loads(l) for l in open('outputs/test_lora_v2_neutral_predictions.jsonl')]
ao = [json.loads(l) for l in open('outputs/test_lora_answer_only_predictions.jsonl')]
cards = [json.loads(l) for l in open('outputs/test_prompts_with_cardinality.jsonl')]
card_by_id = {c['id']: c for c in cards}

def get_ans(p):
    return p.get('answer', p.get('answers', []))

# V12: REVERSE V7d - temporal+multi -> AO (experienced with multi), rest -> V2 (conservative for single)
r12 = []
s = Counter()
for pv, pa in zip(v2, ao):
    pid = pv['id']
    dom = pv.get('domain', '')
    card = card_by_id.get(pid, {}).get('predicted_cardinality', 'single')
    d = (dom or '').lower()
    if ('time' in d or 'temporal' in d or 'hybrid' in d) and card == 'multi':
        r12.append({'id': pid, 'answers': get_ans(pa)})  # AO for multi
        s['ao_multi'] += 1
    else:
        r12.append({'id': pid, 'answers': get_ans(pv)})  # V2 for single
        s['v2_single'] += 1

c = Counter(len(x['answers']) for x in r12)
print(f'V12: ao_multi={s["ao_multi"]}, v2_single={s["v2_single"]}, dist={dict(c)}')

# V12b: pure cardinality routing: multi->AO, single->V2
r12b = []
for pv, pa in zip(v2, ao):
    pid = pv['id']
    card = card_by_id.get(pid, {}).get('predicted_cardinality', 'single')
    if card == 'multi':
        r12b.append({'id': pid, 'answers': get_ans(pa)})
    else:
        r12b.append({'id': pid, 'answers': get_ans(pv)})

c2 = Counter(len(x['answers']) for x in r12b)
print(f'V12b (cardinality): dist={dict(c2)}')

os.makedirs('outputs/submissions', exist_ok=True)
with open('outputs/submissions/v12_submission.json', 'w') as f:
    json.dump(r12, f, ensure_ascii=False, indent=2)
with open('outputs/submissions/v12b_submission.json', 'w') as f:
    json.dump(r12b, f, ensure_ascii=False, indent=2)
print('DONE')
