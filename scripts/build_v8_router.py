import json, os
from collections import Counter

v5 = [json.loads(l) for l in open('outputs/test_lora_v5_predictions.jsonl')]
ao = [json.loads(l) for l in open('outputs/test_lora_answer_only_predictions.jsonl')]
cards = [json.loads(l) for l in open('outputs/test_prompts_with_cardinality.jsonl')]
card_by_id = {c['id']: c for c in cards}

def get_ans(p):
    return p.get('answer', p.get('answers', []))

def has_multi_cue(q):
    ql = q.lower()
    return any(x in ql for x in ['statement(s)', 'select the incorrect', 'select the correct']) or '以下选项中' in q

def router(ao_ans, mix_ans, domain, question):
    ao_n = len(ao_ans)
    mix_n = len(mix_ans)
    if ao_n >= 3 and mix_n <= 2:
        return 'mixed'
    if mix_n >= 3 and ao_n <= 2:
        return 'ao'
    if has_multi_cue(question) and ao_n == 1 and mix_n >= 2:
        return 'mixed'
    d = (domain or '').lower()
    if ('time' in d or 'temporal' in d) and mix_n >= 2:
        return 'mixed'
    if ao_n == 1 and mix_n == 1:
        return 'ao'
    return 'ao'

r8 = []
agree = router_mixed = router_ao = 0
for p_v5, p_ao in zip(v5, ao):
    pid = p_v5['id']
    ao_ans = get_ans(p_ao)
    mix_ans = get_ans(p_v5)
    domain = p_v5.get('domain', '')
    q = card_by_id.get(pid, {}).get('question', '')
    
    if set(ao_ans) == set(mix_ans):
        r8.append({'id': pid, 'answers': ao_ans})
        agree += 1
    else:
        choice = router(ao_ans, mix_ans, domain, q)
        if choice == 'mixed':
            r8.append({'id': pid, 'answers': mix_ans})
            router_mixed += 1
        else:
            r8.append({'id': pid, 'answers': ao_ans})
            router_ao += 1

c = Counter(len(x['answers']) for x in r8)
print(f'V8: agree={agree}, router_mixed={router_mixed}, router_ao={router_ao}')
print(f'Answer dist: {dict(c)}')

os.makedirs('outputs/submissions', exist_ok=True)
with open('outputs/submissions/v8_submission.json', 'w') as f:
    json.dump(r8, f, ensure_ascii=False, indent=2)
print('Wrote v8_submission.json')
