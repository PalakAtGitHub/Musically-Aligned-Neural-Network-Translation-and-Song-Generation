"""
Experiment 01: Plain IndicTrans2 baseline (no melody conditioning).

Translates strict held-out test examples using vanilla IndicTrans2,
then computes the same metrics as MCNST experiments for direct comparison.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score_fn
from src.utils.syllable_utils import count_hindi_syllables

print("=" * 70)
print("EXPERIMENT 01: Plain IndicTrans2 Baseline")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Load vanilla IndicTrans2 (NOT MCNST)
# ------------------------------------------------------------------
print("\n[1/5] Loading vanilla IndicTrans2...")
MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

device = torch.device("cpu")
model.to(device)
model.eval()
print(f"  Model: {MODEL_NAME}")
print(f"  Device: {device}")

# ------------------------------------------------------------------
# 2. Load data and build strict held-out set
# ------------------------------------------------------------------
print("\n[2/5] Loading datasets and filtering strict held-out...")
test_data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)
train_data = torch.load("src/data/processed/fma_train_data.pt", weights_only=False)

train_english = {ex['english_text'] for ex in train_data}
strict_heldout = [ex for ex in test_data if ex['english_text'] not in train_english]

print(f"  Test total:           {len(test_data)}")
print(f"  Train unique english: {len(train_english)}")
print(f"  Strict held-out:      {len(strict_heldout)}")
assert len(strict_heldout) == 198, f"Expected 198 strict held-out, got {len(strict_heldout)}"

# ------------------------------------------------------------------
# 3. Generate translations
# ------------------------------------------------------------------
print(f"\n[3/5] Generating on all {len(strict_heldout)} strict held-out examples...")
print("=" * 70)

SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"
bleu_scorer = BLEU(effective_order=True)

results = []
start_time = time.time()

for i, ex in enumerate(strict_heldout):
    english_text = ex['english_text']
    hindi_ref = ex['hindi_text']
    num_notes = ex['num_notes']

    # IndicTrans2 expects: "src_lang tgt_lang text"
    input_text = f"{SRC_LANG} {TGT_LANG} {english_text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            use_cache=False,
        )

    # Decode with target-mode tokenizer
    tokenizer._switch_to_target_mode()
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    tokenizer._switch_to_input_mode()

    # Metrics
    gen_syl = count_hindi_syllables(gen_text)
    syl_error = abs(gen_syl - num_notes)
    syl_match = syl_error <= 2

    # Character overlap
    max_len = max(len(gen_text), len(hindi_ref), 1)
    matching_chars = sum(1 for a, b in zip(gen_text, hindi_ref) if a == b)
    char_overlap = matching_chars / max_len

    # Sentence BLEU
    sent_bleu = bleu_scorer.sentence_score(gen_text, [hindi_ref]).score

    result = {
        'index': i,
        'song_name': ex['song_name'],
        'english_text': english_text,
        'ground_truth_hi': hindi_ref,
        'generated_hi': gen_text,
        'gen_syl_count': gen_syl,
        'target_notes': int(num_notes),
        'syl_error': syl_error,
        'syl_match_within_2': syl_match,
        'char_overlap_ratio': round(char_overlap, 4),
        'sentence_bleu': round(sent_bleu, 2),
    }
    results.append(result)

    if i < 3 or (i + 1) % 50 == 0 or i == len(strict_heldout) - 1:
        print(f"\n--- Example {i} (song: {ex['song_name']}) ---")
        print(f"  EN:  {english_text}")
        print(f"  REF: {hindi_ref}")
        print(f"  GEN: {gen_text}")
        print(f"  syl: {gen_syl} (target: {num_notes}, err: {syl_error}) {'within2' if syl_match else 'MISS'}")
        print(f"  char_overlap: {char_overlap:.3f}  sent_BLEU: {sent_bleu:.1f}")

elapsed = time.time() - start_time
print(f"\n  Generation complete in {elapsed:.1f}s ({elapsed/len(strict_heldout):.2f}s/example)")

# ------------------------------------------------------------------
# 4. Corpus-level metrics
# ------------------------------------------------------------------
print(f"\n[4/5] Computing corpus-level metrics...")

hypotheses = [r['generated_hi'] for r in results]
references = [r['ground_truth_hi'] for r in results]

# Corpus BLEU
corpus_bleu = BLEU().corpus_score(hypotheses, [references]).score

# BERTScore
print("  Computing BERTScore (this may take a moment)...")
P, R, F1 = bert_score_fn(hypotheses, references, lang='hi', verbose=False)
mean_bertscore_f1 = F1.mean().item()

# Aggregate syllable metrics
syl_acc = sum(1 for r in results if r['syl_match_within_2']) / len(results)
mean_syl_err = sum(r['syl_error'] for r in results) / len(results)
mean_char_overlap = sum(r['char_overlap_ratio'] for r in results) / len(results)

# ------------------------------------------------------------------
# 5. Save and report
# ------------------------------------------------------------------
print(f"\n[5/5] Saving results...")

output = {
    'experiment': 'exp_01_indictrans2_baseline',
    'model': MODEL_NAME,
    'description': 'Plain IndicTrans2 (no melody conditioning, no MCNST)',
    'timestamp': datetime.now().isoformat(),
    'num_examples': len(results),
    'generation_time_seconds': round(elapsed, 1),
    'aggregate_metrics': {
        'corpus_bleu': round(corpus_bleu, 2),
        'mean_bertscore_f1': round(mean_bertscore_f1, 4),
        'syllable_accuracy_within_2': round(syl_acc, 4),
        'mean_syllable_error': round(mean_syl_err, 2),
        'mean_char_overlap': round(mean_char_overlap, 4),
    },
    'per_example_results': results,
    'per_example_bertscore_f1': [round(f.item(), 4) for f in F1],
}

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
out_path = log_dir / "exp_01_indictrans2_baseline_results.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  Saved to {out_path}")

# Summary table
print(f"\n{'=' * 70}")
print(f"EXPERIMENT 01 SUMMARY: Plain IndicTrans2 Baseline")
print(f"{'=' * 70}")
print(f"  Examples:              {len(results)}")
print(f"  Corpus BLEU:           {corpus_bleu:.2f}")
print(f"  BERTScore F1 (mean):   {mean_bertscore_f1:.4f}")
print(f"  Syllable Acc (within 2): {syl_acc*100:.1f}%")
print(f"  Mean Syllable Error:   {mean_syl_err:.2f}")
print(f"  Mean Char Overlap:     {mean_char_overlap:.4f}")
print(f"{'=' * 70}")
print("EXPERIMENT 01 COMPLETE")
print(f"{'=' * 70}")
