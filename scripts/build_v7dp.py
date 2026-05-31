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

# === V7d baseline (for comparison) ===
def route_v7d(domain, card):
    d = (domain or '').lower()
    is_temporal = 'time' in d or 'temporal' in d
    is_hybrid = 'hybrid' in d
    return (is_temporal or is_hybrid) and card == 'multi'

# === V7d+: temporal + explicit_multi_cue only ===
def route_v7dp(domain, q):
    d = (domain or '').lower()
    is_temporal = 'time' in d or 'temporal' in d
    return is_temporal and has_explicit_multi_cue(q)

# === V7d++: temporal + (explicit_cue OR classifier=multi) ===
def route_v7dpp(domain, q, card):
    d = (domain or '').lower()
    is_temporal = 'time' in d or 'temporal' in d
    return is_temporal and (has_explicit_multi_cue(q) or card == 'multi')

# Build all three and compare
for name, route_fn in [
    ('v7d_plus', lambda d,q,c: route_v7dp(d,q)),
    ('v7d_plusplus', lambda d,q,c: route_v7dpp(d,q,c)),
]:
    result = []
    stats = Counter()
    for p_v5, p_ao in zip(v5, ao):
        pid = p_v5['id']
        domain = p_v5.get('domain', '')
        card = card_by_id.get(pid, {}).get('predicted_cardinality', 'single')
        q = card_by_id.get(pid, {}).get('question', '')
        
        if route_fn(domain, q, card):
            result.append({'id': pid, 'answers': get_ans(p_v5)})
            stats['mixed'] += 1
        else:
            result.append({'id': pid, 'answers': get_ans(p_ao)})
            stats['ao'] += 1
    
    c = Counter(len(x['answers']) for x in result)
    print(f'{name}: mixed={stats["mixed"]}, ao={stats["ao"]}')
    print(f'  Answer dist: {dict(c)}')
    
    os.makedirs('outputs/submissions', exist_ok=True)
    with open(f'outputs/submissions/{name}_submission.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'  Wrote {name}_submission.json')

# === Error analysis on dev ===
print('\n=== Dev analysis: where mixed helps vs hurts ===')
dev_preds_mixed = [json.loads(l) for l in open('outputs/dev_lora_mixed_reasoning_predictions.jsonl')]
dev_preds_ao = [json.loads(l) for l in open('outputs/dev_lora_answer_only_predictions.jsonl')]
dev_gold = [json.loads(l) for l in open('outputs/dev_prompts.jsonl') if json.loads(l).get('answer')]
gold_by_id = {g['id']: g for g in dev_gold}

# Compare mixed vs AO on dev: where is each better?
mixed_better = 0
ao_better = 0
both_right = 0
both_wrong = 0

for pm, pa in zip(dev_preds_mixed, dev_preds_ao):
    pid = pm['id']
    gold = gold_by_id.get(pid, {})
    gold_ans = set(gold.get('answer', []))
    if not gold_ans or pm.get('correct') is None:
        continue
    
    m_correct = set(pm.get('answer', [])) == gold_ans
    a_correct = set(pa.get('answer', [])) == gold_ans
    
    if m_correct and a_correct:
        both_right += 1
    elif m_correct:
        mixed_better += 1
    elif a_correct:
        ao_better += 1
    else:
        both_wrong += 1

total = mixed_better + ao_better + both_right + both_wrong
print(f'Dev total compared: {total}')
print(f'  Both right: {both_right} ({100*both_right/total:.1f}%)')
print(f'  Mixed better: {mixed_better} ({100*mixed_better/total:.1f}%)')
print(f'  AO better: {ao_better} ({100*ao_better/total:.1f}%)')
print(f'  Both wrong: {both_wrong} ({100*both_wrong/total:.1f}%)')

# Per domain
print('\n  Per domain (Mixed better / AO better):')
from collections import defaultdict
per_dom = defaultdict(lambda: {'mixed':0, 'ao':0, 'both':0, 'neither':0})
for pm, pa in zip(dev_preds_mixed, dev_preds_ao):
    pid = pm['id']
    gold = gold_by_id.get(pid, {})
    gold_ans = set(gold.get('answer', []))
    if not gold_ans or pm.get('correct') is None:
        continue
    domain = pm.get('domain', '?')
    m_correct = set(pm.get('answer', [])) == gold_ans
    a_correct = set(pa.get('answer', [])) == gold_ans
    if m_correct and a_correct: per_dom[domain]['both'] += 1
    elif m_correct: per_dom[domain]['mixed'] += 1
    elif a_correct: per_dom[domain]['ao'] += 1
    else: per_dom[domain]['neither'] += 1

for d in sorted(per_dom):
    s = per_dom[d]
    t = sum(s.values())
    print(f'  {d:10s}: mixed+{s["mixed"]:3d}  ao+{s["ao"]:3d}  both={s["both"]:3d}  neither={s["neither"]:3d}  total={t}')

print('\nDONE')
