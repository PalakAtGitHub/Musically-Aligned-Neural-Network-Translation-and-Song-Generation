"""
Experiment 08: Six-way full ablation comparison report.

Systems:
1. IndicTrans2 baseline (exp_01)
2. Decoder-fine-tuning only (exp_07)
3. MCNST no-fusion (exp_05)
4. MCNST no-8a (exp_06)
5. MCNST full / default (exp_03)
6. MCNST constrained (exp_02)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

print("=" * 70)
print("EXPERIMENT 08: Six-Way Full Ablation Comparison")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Load all result files
# ------------------------------------------------------------------
print("\n[1/3] Loading result files...")

files = {
    'IndicTrans2 baseline':   'exp_01_indictrans2_baseline_results.json',
    'Decoder-tuning only':    'exp_07_decoder_only_results.json',
    'MCNST no-fusion':        'exp_05_no_fusion_results.json',
    'MCNST no-8a':            'exp_06_no_8a_results.json',
    'MCNST full (default)':   'exp_03_mcnst_default_results.json',
    'MCNST constrained':      'exp_02_mcnst_constrained_results.json',
}

data = {}
for name, fname in files.items():
    with open(LOG_DIR / fname) as f:
        data[name] = json.load(f)
    print(f"  {name:<25} {data[name]['num_examples']} examples")

# ------------------------------------------------------------------
# 2. Verify ordering
# ------------------------------------------------------------------
print("\n[2/3] Verifying example ordering...")
ref = [r['english_text'] for r in data['IndicTrans2 baseline']['per_example_results']]
for name, d in data.items():
    texts = [r['english_text'] for r in d['per_example_results']]
    assert len(texts) == 198, f"{name}: {len(texts)} != 198"
    assert texts == ref, f"{name}: ordering mismatch!"
print("  All six use identical 198 examples in identical order.")

# ------------------------------------------------------------------
# 3. Extract arrays
# ------------------------------------------------------------------
def get_arrays(d):
    r = d['per_example_results']
    return {
        'syl_err': np.array([x['syl_error'] for x in r]),
        'syl_ok':  np.array([x['syl_match_within_2'] for x in r], dtype=int),
        'bleu':    np.array([x['sentence_bleu'] for x in r]),
        'bs':      np.array(d['per_example_bertscore_f1']),
        'char':    np.array([x['char_overlap_ratio'] for x in r]),
    }

arrays = {name: get_arrays(d) for name, d in data.items()}

# ------------------------------------------------------------------
# 4. Comparison table
# ------------------------------------------------------------------
names_ordered = [
    'IndicTrans2 baseline',
    'Decoder-tuning only',
    'MCNST no-fusion',
    'MCNST no-8a',
    'MCNST full (default)',
    'MCNST constrained',
]

header = f"{'System':<25} {'BLEU':>7} {'BERTSc F1':>10} {'Syl Acc%':>9} {'Syl Err':>8} {'Char Ovlp':>10}"
sep = "-" * len(header)

print(f"\n[3/3] Results:\n")
print(sep)
print(header)
print(sep)
for name in names_ordered:
    a = data[name]['aggregate_metrics']
    print(f"{name:<25} {a['corpus_bleu']:>7.2f} {a['mean_bertscore_f1']:>10.4f} "
          f"{a['syllable_accuracy_within_2']*100:>8.1f}% {a['mean_syllable_error']:>8.2f} "
          f"{a['mean_char_overlap']:>10.4f}")
print(sep)

# ------------------------------------------------------------------
# 5. Statistical tests — adjacent pairs
# ------------------------------------------------------------------
print()
print("STATISTICAL TESTS (adjacent pairs in table order)")
print("=" * 70)

adjacent_pairs = [
    ('IndicTrans2 baseline', 'Decoder-tuning only'),
    ('Decoder-tuning only', 'MCNST no-fusion'),
    ('MCNST no-fusion', 'MCNST no-8a'),
    ('MCNST no-8a', 'MCNST full (default)'),
    ('MCNST full (default)', 'MCNST constrained'),
]

def mcnemar(a_ok, b_ok, a_name, b_name):
    n01 = int(np.sum((a_ok == 0) & (b_ok == 1)))
    n10 = int(np.sum((a_ok == 1) & (b_ok == 0)))
    n11 = int(np.sum((a_ok == 1) & (b_ok == 1)))
    n00 = int(np.sum((a_ok == 0) & (b_ok == 0)))
    if n01 + n10 == 0:
        chi2, p = 0.0, 1.0
    else:
        chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        p = stats.chi2.sf(chi2, df=1)
    print(f"  {a_name} vs {b_name}")
    print(f"    both_ok={n11} A_only={n10} B_only={n01} neither={n00}")
    print(f"    chi2={chi2:.4f} p={p:.6f}")

def paired_t(a_vals, b_vals, a_name, b_name, metric):
    t_stat, p_val = stats.ttest_rel(a_vals, b_vals)
    print(f"  {a_name} vs {b_name}")
    print(f"    mean_A={a_vals.mean():.4f} mean_B={b_vals.mean():.4f} diff={a_vals.mean()-b_vals.mean():.4f}")
    print(f"    t={t_stat:.4f} p={p_val:.6f}")

print()
print("McNemar's test — syllable accuracy (within +-2):")
print("-" * 60)
for a_name, b_name in adjacent_pairs:
    mcnemar(arrays[a_name]['syl_ok'], arrays[b_name]['syl_ok'], a_name, b_name)

print()
print("Paired t-test — syllable error:")
print("-" * 60)
for a_name, b_name in adjacent_pairs:
    paired_t(arrays[a_name]['syl_err'], arrays[b_name]['syl_err'], a_name, b_name, 'syl_err')

print()
print("Paired t-test — sentence BLEU:")
print("-" * 60)
for a_name, b_name in adjacent_pairs:
    paired_t(arrays[a_name]['bleu'], arrays[b_name]['bleu'], a_name, b_name, 'bleu')

print()
print("Paired t-test — BERTScore F1:")
print("-" * 60)
for a_name, b_name in adjacent_pairs:
    paired_t(arrays[a_name]['bs'], arrays[b_name]['bs'], a_name, b_name, 'bs')

# ------------------------------------------------------------------
# 6. Save report
# ------------------------------------------------------------------
report_lines = []
report_lines.append("EXPERIMENT 08: Six-Way Full Ablation Comparison")
report_lines.append(f"Generated: {datetime.now().isoformat()}")
report_lines.append(f"Examples: 198 strict held-out")
report_lines.append("")
report_lines.append(sep)
report_lines.append(header)
report_lines.append(sep)
for name in names_ordered:
    a = data[name]['aggregate_metrics']
    report_lines.append(
        f"{name:<25} {a['corpus_bleu']:>7.2f} {a['mean_bertscore_f1']:>10.4f} "
        f"{a['syllable_accuracy_within_2']*100:>8.1f}% {a['mean_syllable_error']:>8.2f} "
        f"{a['mean_char_overlap']:>10.4f}"
    )
report_lines.append(sep)

report_path = LOG_DIR / "exp_08_full_ablation_comparison.txt"
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"\nReport saved to {report_path}")
print()
print("EXPERIMENT 08 COMPLETE")
