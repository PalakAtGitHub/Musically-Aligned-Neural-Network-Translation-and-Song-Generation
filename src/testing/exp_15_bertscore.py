"""
Compute BERTScore for exp_15 results.
Run this AFTER exp_15_eval_only.py (which saves hypotheses/references).
Does NOT load IndicTrans2 — avoids the tokenizer conflict.
"""

import json
from pathlib import Path
from bert_score import score as bert_score_fn

bs_path = Path("logs/exp_15_bertscore_inputs.json")
with open(bs_path) as f:
    data = json.load(f)

for mode in ['beam', 'sample']:
    hyps = data[f'{mode}_hypotheses']
    refs = data[f'{mode}_references']
    # Replace empty strings — bert_score crashes on empty inputs
    hyps = [h if h.strip() else "।" for h in hyps]
    refs = [r if r.strip() else "।" for r in refs]
    n_empty = sum(1 for h in data[f'{mode}_hypotheses'] if not h.strip())
    print(f"Computing BERTScore ({mode}, {len(hyps)} examples, {n_empty} empty replaced)...")
    P, R, F1 = bert_score_fn(hyps, refs, lang='hi', verbose=True)
    mean_f1 = F1.mean().item()
    print(f"  BERTScore F1 mean: {mean_f1:.4f}")

    # Update the results file
    tag = 'a' if mode == 'beam' else 'b'
    results_path = Path(f"logs/exp_15{tag}_rl_1epoch_{mode}_results.json")
    with open(results_path) as f:
        results = json.load(f)
    results['aggregate_metrics']['mean_bertscore_f1'] = round(mean_f1, 4)
    results['per_example_bertscore_f1'] = [round(f.item(), 4) for f in F1]
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Updated {results_path}")

print("\nBERTScore computation complete.")
