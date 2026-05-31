import json, os
from collections import Counter

# Three adapters
ao = [json.loads(l) for l in open('outputs/test_lora_answer_only_predictions.jsonl')]
v5 = [json.loads(l) for l in open('outputs/test_lora_v5_predictions.jsonl')]
orig = [json.loads(l) for l in open('outputs/test_lora_mixed_original_prompt.jsonl')]

def get_ans(p):
    return p.get('answer', p.get('answers', []))

# V13: majority vote among 3 adapters
r13 = []
for a, v, o in zip(ao, v5, orig):
    pid = a['id']
    ans_a = tuple(sorted(get_ans(a)))
    ans_v = tuple(sorted(get_ans(v)))
    ans_o = tuple(sorted(get_ans(o)))
    
    votes = {}
    for ans in [ans_a, ans_v, ans_o]:
        votes[ans] = votes.get(ans, 0) + 1
    
    # Find majority (2+ votes)
    winner = max(votes, key=votes.get)
    if votes[winner] >= 2:
        r13.append({'id': pid, 'answers': list(winner)})
    else:
        # 3-way tie → AO (safest)
        r13.append({'id': pid, 'answers': list(ans_a)})

c = Counter(len(x['answers']) for x in r13)
print(f'V13 (majority vote): dist={dict(c)}')

# Count agreement patterns
all3 = sum(1 for a,v,o in zip(ao,v5,orig) if tuple(sorted(get_ans(a)))==tuple(sorted(get_ans(v)))==tuple(sorted(get_ans(o))))
two = sum(1 for a,v,o in zip(ao,v5,orig) if len({tuple(sorted(get_ans(a))),tuple(sorted(get_ans(v))),tuple(sorted(get_ans(o)))})==2)
all_diff = sum(1 for a,v,o in zip(ao,v5,orig) if len({tuple(sorted(get_ans(a))),tuple(sorted(get_ans(v))),tuple(sorted(get_ans(o)))})==3)
print(f'All3 agree: {all3}, 2of3 agree: {two}, all differ: {all_diff}')

os.makedirs('outputs/submissions', exist_ok=True)
with open('outputs/submissions/v13_submission.json', 'w') as f:
    json.dump(r13, f, ensure_ascii=False, indent=2)
print('DONE')
