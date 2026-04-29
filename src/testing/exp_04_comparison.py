"""
Experiment 04: Three-way comparison report.

Loads results from Experiments 1, 2, 3 and produces:
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
print("EXPERIMENT 04: Three-Way Comparison Report")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Load all three result files
# ------------------------------------------------------------------
print("\n[1/4] Loading result files...")

with open(LOG_DIR / "exp_01_indictrans2_baseline_results.json") as f:
    exp1 = json.load(f)
with open(LOG_DIR / "exp_02_mcnst_constrained_results.json") as f:
    exp2 = json.load(f)
with open(LOG_DIR / "exp_03_mcnst_default_results.json") as f:
    exp3 = json.load(f)

print(f"  Exp1 (IndicTrans2):       {exp1['num_examples']} examples")
print(f"  Exp2 (MCNST constrained): {exp2['num_examples']} examples")
print(f"  Exp3 (MCNST default):     {exp3['num_examples']} examples")

# ------------------------------------------------------------------
# 2. Verify same 198 examples in same order
# ------------------------------------------------------------------
print("\n[2/4] Verifying example ordering...")

e1_texts = [r['english_text'] for r in exp1['per_example_results']]
e2_texts = [r['english_text'] for r in exp2['per_example_results']]
e3_texts = [r['english_text'] for r in exp3['per_example_results']]

assert len(e1_texts) == 198, f"Exp1 has {len(e1_texts)} examples, expected 198"
assert len(e2_texts) == 198, f"Exp2 has {len(e2_texts)} examples, expected 198"
assert len(e3_texts) == 198, f"Exp3 has {len(e3_texts)} examples, expected 198"
assert e1_texts == e2_texts, "Exp1 vs Exp2 ordering mismatch!"
assert e1_texts == e3_texts, "Exp1 vs Exp3 ordering mismatch!"
print("  All three use identical 198 examples in identical order.")

# ------------------------------------------------------------------
# 3. Extract per-example arrays
# ------------------------------------------------------------------
r1 = exp1['per_example_results']
r2 = exp2['per_example_results']
r3 = exp3['per_example_results']

syl_err_1 = np.array([r['syl_error'] for r in r1])
syl_err_2 = np.array([r['syl_error'] for r in r2])
syl_err_3 = np.array([r['syl_error'] for r in r3])

syl_ok_1 = np.array([r['syl_match_within_2'] for r in r1], dtype=int)
syl_ok_2 = np.array([r['syl_match_within_2'] for r in r2], dtype=int)
syl_ok_3 = np.array([r['syl_match_within_2'] for r in r3], dtype=int)

bleu_1 = np.array([r['sentence_bleu'] for r in r1])
bleu_2 = np.array([r['sentence_bleu'] for r in r2])
bleu_3 = np.array([r['sentence_bleu'] for r in r3])

bs_1 = np.array(exp1['per_example_bertscore_f1'])
bs_2 = np.array(exp2['per_example_bertscore_f1'])
bs_3 = np.array(exp3['per_example_bertscore_f1'])

char_1 = np.array([r['char_overlap_ratio'] for r in r1])
char_2 = np.array([r['char_overlap_ratio'] for r in r2])
char_3 = np.array([r['char_overlap_ratio'] for r in r3])

# ------------------------------------------------------------------
# 4. Comparison table
# ------------------------------------------------------------------
a1 = exp1['aggregate_metrics']
a2 = exp2['aggregate_metrics']
a3 = exp3['aggregate_metrics']

header = f"{'System':<25} {'BLEU':>7} {'BERTSc F1':>10} {'Syl Acc%':>9} {'Syl Err':>8} {'Char Ovlp':>10}"
sep = "-" * len(header)

table_lines = []
table_lines.append(sep)
table_lines.append(header)
table_lines.append(sep)
table_lines.append(
    f"{'IndicTrans2 (baseline)':<25} {a1['corpus_bleu']:>7.2f} {a1['mean_bertscore_f1']:>10.4f} "
    f"{a1['syllable_accuracy_within_2']*100:>8.1f}% {a1['mean_syllable_error']:>8.2f} {a1['mean_char_overlap']:>10.4f}"
)
table_lines.append(
    f"{'MCNST default':<25} {a3['corpus_bleu']:>7.2f} {a3['mean_bertscore_f1']:>10.4f} "
    f"{a3['syllable_accuracy_within_2']*100:>8.1f}% {a3['mean_syllable_error']:>8.2f} {a3['mean_char_overlap']:>10.4f}"
)
table_lines.append(
    f"{'MCNST constrained':<25} {a2['corpus_bleu']:>7.2f} {a2['mean_bertscore_f1']:>10.4f} "
    f"{a2['syllable_accuracy_within_2']*100:>8.1f}% {a2['mean_syllable_error']:>8.2f} {a2['mean_char_overlap']:>10.4f}"
)
table_lines.append(sep)

# Add constraint tracking
ct = exp2['constraint_tracking_summary']
table_lines.append(f"")
table_lines.append(f"Constrained beam search tracking (Exp 2 only):")
table_lines.append(f"  Constraint satisfied (>=1 candidate within +-2): {ct['examples_with_any_candidate_within_2']}/198 ({ct['constraint_success_rate']*100:.1f}%)")
table_lines.append(f"  Constraint fallback (no candidate within +-2):   {ct['examples_with_no_candidate_within_2']}/198 ({ct['constraint_fallback_rate']*100:.1f}%)")

print("\n[3/4] Comparison table:")
print()
for line in table_lines:
    print(line)

# ------------------------------------------------------------------
# 5. Statistical tests
# ------------------------------------------------------------------
print(f"\n[4/4] Statistical tests...")
print()

stat_lines = []
stat_lines.append("STATISTICAL TESTS")
stat_lines.append("=" * 70)

# --- McNemar's test for syllable accuracy ---
stat_lines.append("")
stat_lines.append("McNemar's test for syllable accuracy (within +-2):")
stat_lines.append("-" * 50)

def mcnemar_test(a, b, name):
    # a[i]=1 means system A got example i correct, b[i]=1 means system B got it correct
    # Contingency: n01 = A wrong, B right; n10 = A right, B wrong
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    # McNemar's chi-squared (with continuity correction)
    if n01 + n10 == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        p = stats.chi2.sf(chi2, df=1)
    lines = []
    lines.append(f"  {name}")
    lines.append(f"    Contingency: both_ok={n11}, A_only={n10}, B_only={n01}, neither={n00}")
    lines.append(f"    chi2={chi2:.4f}, p={p:.6f}")
    return lines

stat_lines.extend(mcnemar_test(syl_ok_2, syl_ok_3, "MCNST constrained vs MCNST default"))
stat_lines.extend(mcnemar_test(syl_ok_2, syl_ok_1, "MCNST constrained vs IndicTrans2"))
stat_lines.extend(mcnemar_test(syl_ok_3, syl_ok_1, "MCNST default vs IndicTrans2"))

# --- Paired t-test for syllable error ---
stat_lines.append("")
stat_lines.append("Paired t-test for mean syllable error:")
stat_lines.append("-" * 50)

def paired_ttest(a, b, name):
    t_stat, p_val = stats.ttest_rel(a, b)
    lines = []
    lines.append(f"  {name}")
    lines.append(f"    mean_A={a.mean():.4f}, mean_B={b.mean():.4f}, diff={a.mean()-b.mean():.4f}")
    lines.append(f"    t={t_stat:.4f}, p={p_val:.6f}")
    return lines

stat_lines.extend(paired_ttest(syl_err_2, syl_err_3, "MCNST constrained vs MCNST default"))
stat_lines.extend(paired_ttest(syl_err_2, syl_err_1, "MCNST constrained vs IndicTrans2"))
stat_lines.extend(paired_ttest(syl_err_3, syl_err_1, "MCNST default vs IndicTrans2"))

# --- Paired t-test for sentence BLEU ---
stat_lines.append("")
stat_lines.append("Paired t-test for sentence BLEU:")
stat_lines.append("-" * 50)

stat_lines.extend(paired_ttest(bleu_2, bleu_3, "MCNST constrained vs MCNST default"))
stat_lines.extend(paired_ttest(bleu_2, bleu_1, "MCNST constrained vs IndicTrans2"))
stat_lines.extend(paired_ttest(bleu_3, bleu_1, "MCNST default vs IndicTrans2"))

# --- Paired t-test for BERTScore F1 ---
stat_lines.append("")
stat_lines.append("Paired t-test for BERTScore F1:")
stat_lines.append("-" * 50)

stat_lines.extend(paired_ttest(bs_2, bs_3, "MCNST constrained vs MCNST default"))
stat_lines.extend(paired_ttest(bs_2, bs_1, "MCNST constrained vs IndicTrans2"))
stat_lines.extend(paired_ttest(bs_3, bs_1, "MCNST default vs IndicTrans2"))

for line in stat_lines:
    print(line)

# ------------------------------------------------------------------
# 6. Agreement / disagreement analysis
# ------------------------------------------------------------------
print()
print("AGREEMENT / DISAGREEMENT ANALYSIS")
print("=" * 70)

# All three agree on syllable count (within +-2 of each other)
agree_lines = []
disagree_constrained_better = []
constrained_worse_than_baseline = []

for i in range(198):
    s1 = r1[i]['gen_syl_count']
    s2 = r2[i]['gen_syl_count']
    s3 = r3[i]['gen_syl_count']
    target = r1[i]['target_notes']

    # All three within +-2 of each other on syl_count
    if abs(s1 - s2) <= 2 and abs(s1 - s3) <= 2 and abs(s2 - s3) <= 2:
        agree_lines.append(i)

    # Constrained dramatically better than default (>10 syl difference)
    err_constrained = abs(s2 - target)
    err_default = abs(s3 - target)
    err_baseline = abs(s1 - target)
    if err_default - err_constrained > 10:
        disagree_constrained_better.append(i)

    # Constrained worse than baseline IndicTrans2
    if err_constrained > err_baseline:
        constrained_worse_than_baseline.append(i)

print(f"All three systems agree (syl counts within +-2 of each other): {len(agree_lines)}/198")
print(f"MCNST constrained dramatically better than default (>10 syl diff): {len(disagree_constrained_better)}/198")
print(f"MCNST constrained worse syl error than IndicTrans2 baseline: {len(constrained_worse_than_baseline)}/198")

# Show up to 3 examples where constrained is dramatically better
if disagree_constrained_better:
    print(f"\nExamples where constrained is >10 syl closer than default:")
    for idx in disagree_constrained_better[:3]:
        print(f"  idx={idx}: target={r1[idx]['target_notes']}, "
              f"constrained_syl={r2[idx]['gen_syl_count']} (err={abs(r2[idx]['gen_syl_count']-r1[idx]['target_notes'])}), "
              f"default_syl={r3[idx]['gen_syl_count']} (err={abs(r3[idx]['gen_syl_count']-r1[idx]['target_notes'])})")

# Show up to 3 examples where constrained is worse than baseline
if constrained_worse_than_baseline:
    print(f"\nExamples where constrained is worse than IndicTrans2 (first 3):")
    for idx in constrained_worse_than_baseline[:3]:
        print(f"  idx={idx}: target={r1[idx]['target_notes']}, "
              f"baseline_syl={r1[idx]['gen_syl_count']} (err={syl_err_1[idx]}), "
              f"constrained_syl={r2[idx]['gen_syl_count']} (err={syl_err_2[idx]})")

# ------------------------------------------------------------------
# 7. Five-way example comparison
# ------------------------------------------------------------------
print()
print("SIDE-BY-SIDE EXAMPLES (5 examples, all 3 systems)")
print("=" * 70)

# Pick 5 diverse examples: idx 0, 2, 49, 99, 149
for idx in [0, 2, 49, 99, 149]:
    print(f"\n--- Example {idx} ---")
    print(f"english_text: {r1[idx]['english_text']}")
    print(f"ground_truth_hi: {r1[idx]['ground_truth_hi']}")
    print(f"target_notes: {r1[idx]['target_notes']}")
    print(f"")
    print(f"IndicTrans2:       {r1[idx]['generated_hi']}")
    print(f"  syl={r1[idx]['gen_syl_count']} err={syl_err_1[idx]} bleu={bleu_1[idx]:.1f} bert={bs_1[idx]:.4f}")
    print(f"MCNST default:     {r3[idx]['generated_hi']}")
    print(f"  syl={r3[idx]['gen_syl_count']} err={syl_err_3[idx]} bleu={bleu_3[idx]:.1f} bert={bs_3[idx]:.4f}")
    print(f"MCNST constrained: {r2[idx]['generated_hi']}")
    print(f"  syl={r2[idx]['gen_syl_count']} err={syl_err_2[idx]} bleu={bleu_2[idx]:.1f} bert={bs_2[idx]:.4f}")

# ------------------------------------------------------------------
# 8. Save report
# ------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = LOG_DIR / f"exp_04_comparison_report.txt"

all_lines = []
all_lines.append("EXPERIMENT 04: Three-Way Comparison Report")
all_lines.append(f"Generated: {datetime.now().isoformat()}")
all_lines.append(f"Examples: 198 strict held-out (english_text not in training)")
all_lines.append("")
all_lines.extend(table_lines)
all_lines.append("")
all_lines.extend(stat_lines)
all_lines.append("")
all_lines.append("AGREEMENT / DISAGREEMENT ANALYSIS")
all_lines.append(f"All three agree (syl within +-2 of each other): {len(agree_lines)}/198")
all_lines.append(f"Constrained dramatically better than default (>10 syl diff): {len(disagree_constrained_better)}/198")
all_lines.append(f"Constrained worse syl error than baseline: {len(constrained_worse_than_baseline)}/198")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_lines))
print(f"\nReport saved to {report_path}")

print()
print("EXPERIMENT 04 COMPLETE")
