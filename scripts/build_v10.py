import json, os
from collections import Counter

mix = [json.loads(l) for l in open('outputs/test_lora_mixed_original_prompt.jsonl')]
ao = [json.loads(l) for l in open('outputs/test_lora_answer_only_predictions.jsonl')]
cards = [json.loads(l) for l in open('outputs/test_prompts_with_cardinality.jsonl')]
card_by_id = {c['id']: c for c in cards}

def get_ans(p):
    return p.get('answer', p.get('answers', []))

r10 = []
stats = Counter()
for pm, pa in zip(mix, ao):
    pid = pm['id']
    domain = pm.get('domain', '')
    card = card_by_id.get(pid, {}).get('predicted_cardinality', 'single')
    d = (domain or '').lower()
    if ('time' in d or 'temporal' in d or 'hybrid' in d) and card == 'multi':
        r10.append({'id': pid, 'answers': get_ans(pm)})
        stats['mixed'] += 1
    else:
        r10.append({'id': pid, 'answers': get_ans(pa)})
        stats['ao'] += 1

c = Counter(len(x['answers']) for x in r10)
print(f'V10: mixed={stats["mixed"]}, ao={stats["ao"]}')
print(f'Answer dist: {dict(c)}')

os.makedirs('outputs/submissions', exist_ok=True)
with open('outputs/submissions/v10_submission.json', 'w') as f:
    json.dump(r10, f, ensure_ascii=False, indent=2)
print('DONE')
