import json, os
from collections import Counter

v5 = [json.loads(l) for l in open('outputs/test_lora_mixed_original_prompt.jsonl')]
ao = [json.loads(l) for l in open('outputs/test_lora_answer_only_predictions.jsonl')]
cards = [json.loads(l) for l in open('outputs/test_prompts_with_cardinality.jsonl')]
card_by_id = {c['id']: c for c in cards}

def get_ans(p):
    return p.get('answer', p.get('answers', []))

# V7g: improved domain routing
# social always mixed (dev: 59% vs 55%), space+social try mixed
def route_g(dom):
    d = (dom or '').lower()
    if 'time' in d or 'temporal' in d: return 'mixed'
    if d == 'social' or d == 'hybrid': return 'mixed'
    if 'space' in d and 'social' in d: return 'mixed'
    return 'ao'

# V7h: only fix social
def route_h(dom):
    d = (dom or '').lower()
    if 'time' in d or 'temporal' in d: return 'mixed'
    if d == 'social' or d == 'hybrid': return 'mixed'
    return 'ao'

for name, route_fn in [('v10', route_g), ('v7h', route_h)]:
    result = []
    for p in v5:
        pid, dom = p['id'], p.get('domain','?')
        if route_fn(dom) == 'mixed':
            result.append({'id': pid, 'answers': get_ans(p)})
        else:
            ao_p = next((a for a in ao if a['id'] == pid), p)
            result.append({'id': pid, 'answers': get_ans(ao_p)})
    
    c = Counter(len(x['answers']) for x in result)
    mixed_count = sum(1 for p in v5 if route_fn(p.get('domain','?')) == 'mixed')
    print(f'{name}: mixed={mixed_count}, ao={1000-mixed_count}, dist={dict(c)}')
    
    os.makedirs('outputs/submissions', exist_ok=True)
    with open(f'outputs/submissions/{name}_submission.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'Wrote {name}_submission.json')

print('DONE')
