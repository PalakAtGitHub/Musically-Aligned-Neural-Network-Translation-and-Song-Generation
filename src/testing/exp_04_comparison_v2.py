"""
Experiment 04 v2: Four-way comparison report (includes no-fusion ablation).

Loads results from Experiments 1, 2, 3, 5 and produces:
  - Comparison table
  - Statistical tests (McNemar, paired t-test)
  - Agreement/disagreement analysis
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

print("=" * 70)
print("EXPERIMENT 04 v2: Four-Way Comparison Report")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Load all four result files
# ------------------------------------------------------------------
print("\n[1/4] Loading result files...")

with open(LOG_DIR / "exp_01_indictrans2_baseline_results.json") as f:
    exp1 = json.load(f)
with open(LOG_DIR / "exp_02_mcnst_constrained_results.json") as f:
    exp2 = json.load(f)
with open(LOG_DIR / "exp_03_mcnst_default_results.json") as f:
    exp3 = json.load(f)
with open(LOG_DIR / "exp_05_no_fusion_results.json") as f:
    exp5 = json.load(f)

systems = {
    'IndicTrans2 (baseline)': exp1,
    'MCNST default': exp3,
    'MCNST constrained': exp2,
    'MCNST no-fusion': exp5,
}

for name, exp in systems.items():
    print(f"  {name:<25} {exp['num_examples']} examples")

# ------------------------------------------------------------------
# 2. Verify same 198 examples in same order
# ------------------------------------------------------------------
print("\n[2/4] Verifying example ordering...")

ref_texts = [r['english_text'] for r in exp1['per_example_results']]
for name, exp in systems.items():
    texts = [r['english_text'] for r in exp['per_example_results']]
    assert len(texts) == 198, f"{name} has {len(texts)} examples, expected 198"
    assert texts == ref_texts, f"{name} ordering mismatch!"
print("  All four use identical 198 examples in identical order.")

# ------------------------------------------------------------------
# 3. Extract per-example arrays
# ------------------------------------------------------------------
def extract_arrays(exp):
    r = exp['per_example_results']
    return {
        'syl_err': np.array([x['syl_error'] for x in r]),
        'syl_ok': np.array([x['syl_match_within_2'] for x in r], dtype=int),
        'bleu': np.array([x['sentence_bleu'] for x in r]),
        'bs': np.array(exp['per_example_bertscore_f1']),
        'char': np.array([x['char_overlap_ratio'] for x in r]),
    }

arrays = {name: extract_arrays(exp) for name, exp in systems.items()}

# ------------------------------------------------------------------
# 4. Comparison table
# ------------------------------------------------------------------
print("\n[3/4] Comparison table:")
print()

header = f"{'System':<25} {'BLEU':>7} {'BERTSc F1':>10} {'Syl Acc%':>9} {'Syl Err':>8} {'Char Ovlp':>10}"
sep = "-" * len(header)
print(sep)
print(header)
print(sep)

for name, exp in systems.items():
    a = exp['aggregate_metrics']
    print(f"{name:<25} {a['corpus_bleu']:>7.2f} {a['mean_bertscore_f1']:>10.4f} "
          f"{a['syllable_accuracy_within_2']*100:>8.1f}% {a['mean_syllable_error']:>8.2f} "
          f"{a['mean_char_overlap']:>10.4f}")

print(sep)

# Constraint tracking for exp2
ct = exp2['constraint_tracking_summary']
print(f"\nConstrained beam search (Exp 2):")
print(f"  Constraint satisfied: {ct['examples_with_any_candidate_within_2']}/198 ({ct['constraint_success_rate']*100:.1f}%)")
print(f"  Constraint fallback:  {ct['examples_with_no_candidate_within_2']}/198 ({ct['constraint_fallback_rate']*100:.1f}%)")

# No-fusion training info
print(f"\nNo-fusion ablation (Exp 5):")
print(f"  Best val_loss: {exp5.get('checkpoint_val_loss', 'N/A')}")
print(f"  Best epoch:    {exp5.get('checkpoint_epoch', 'N/A')}")
print(f"  Training time: {exp5.get('training_time_seconds', 'N/A')}s")

# ------------------------------------------------------------------
# 5. Statistical tests
# ------------------------------------------------------------------
print(f"\n[4/4] Statistical tests...")
print()

def mcnemar_test(a, b, name):
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    if n01 + n10 == 0:
        chi2, p = 0.0, 1.0
    else:
        chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        p = stats.chi2.sf(chi2, df=1)
    print(f"  {name}")
    print(f"    both_ok={n11}, A_only={n10}, B_only={n01}, neither={n00}")
    print(f"    chi2={chi2:.4f}, p={p:.6f}")

def paired_ttest(a, b, name):
    t_stat, p_val = stats.ttest_rel(a, b)
    print(f"  {name}")
    print(f"    mean_A={a.mean():.4f}, mean_B={b.mean():.4f}, diff={a.mean()-b.mean():.4f}")
    print(f"    t={t_stat:.4f}, p={p_val:.6f}")

# All pairwise comparisons involving no-fusion
pairs = [
    ('MCNST default', 'MCNST no-fusion'),
    ('MCNST constrained', 'MCNST no-fusion'),
    ('IndicTrans2 (baseline)', 'MCNST no-fusion'),
    ('MCNST constrained', 'MCNST default'),
    ('MCNST constrained', 'IndicTrans2 (baseline)'),
    ('MCNST default', 'IndicTrans2 (baseline)'),
]

print("McNemar's test for syllable accuracy (within +-2):")
print("-" * 60)
for a_name, b_name in pairs:
    mcnemar_test(arrays[a_name]['syl_ok'], arrays[b_name]['syl_ok'],
                 f"{a_name} vs {b_name}")

print()
print("Paired t-test for syllable error:")
print("-" * 60)
for a_name, b_name in pairs:
    paired_ttest(arrays[a_name]['syl_err'], arrays[b_name]['syl_err'],
                 f"{a_name} vs {b_name}")

print()
print("Paired t-test for sentence BLEU:")
print("-" * 60)
for a_name, b_name in pairs:
    paired_ttest(arrays[a_name]['bleu'], arrays[b_name]['bleu'],
                 f"{a_name} vs {b_name}")

print()
print("Paired t-test for BERTScore F1:")
print("-" * 60)
for a_name, b_name in pairs:
    paired_ttest(arrays[a_name]['bs'], arrays[b_name]['bs'],
                 f"{a_name} vs {b_name}")

# ------------------------------------------------------------------
# 6. Side-by-side examples (all 4 systems)
# ------------------------------------------------------------------
print()
print("SIDE-BY-SIDE EXAMPLES (5 examples, all 4 systems)")
print("=" * 70)

r1 = exp1['per_example_results']
r2 = exp2['per_example_results']
r3 = exp3['per_example_results']
r5 = exp5['per_example_results']

for idx in [0, 2, 49, 99, 149]:
    print(f"\n--- Example {idx} ---")
    print(f"english_text: {r1[idx]['english_text']}")
    print(f"ground_truth_hi: {r1[idx]['ground_truth_hi']}")
    print(f"target_notes: {r1[idx]['target_notes']}")
    print()
    for label, r in [('IndicTrans2', r1), ('MCNST default', r3),
                     ('MCNST constrained', r2), ('MCNST no-fusion', r5)]:
        print(f"{label:<20} {r[idx]['generated_hi']}")
        print(f"  syl={r[idx]['gen_syl_count']} err={r[idx]['syl_error']} "
              f"bleu={r[idx]['sentence_bleu']:.1f}")

# ------------------------------------------------------------------
# 7. Save report
# ------------------------------------------------------------------
report_path = LOG_DIR / "exp_04_comparison_report_v2.txt"

# Rebuild as text
lines = []
lines.append("EXPERIMENT 04 v2: Four-Way Comparison Report")
lines.append(f"Generated: {datetime.now().isoformat()}")
lines.append(f"Examples: 198 strict held-out")
lines.append("")
lines.append(sep)
lines.append(header)
lines.append(sep)
for name, exp in systems.items():
    a = exp['aggregate_metrics']
    lines.append(f"{name:<25} {a['corpus_bleu']:>7.2f} {a['mean_bertscore_f1']:>10.4f} "
                 f"{a['syllable_accuracy_within_2']*100:>8.1f}% {a['mean_syllable_error']:>8.2f} "
                 f"{a['mean_char_overlap']:>10.4f}")
lines.append(sep)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"\nReport saved to {report_path}")

print()
print("EXPERIMENT 04 v2 COMPLETE")
