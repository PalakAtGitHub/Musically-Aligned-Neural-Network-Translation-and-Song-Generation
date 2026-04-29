"""
Experiment 03: MCNST with default generation (no constrained reranking).

Same checkpoint, same strict held-out set as Experiments 1 and 2.
Uses model.generate() — the standard melody-fused beam search path.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score_fn
from src.models.mcnst_model import MCNST
from src.utils.syllable_utils import count_hindi_syllables

print("=" * 70)
print("EXPERIMENT 03: MCNST Default Generation")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Load trained MCNST
# ------------------------------------------------------------------
print("\n[1/5] Loading trained MCNST...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=10)
device = torch.device("cpu")
ckpt = torch.load("checkpoints/best_model.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()
tokenizer = model.tokenizer
print(f"  Checkpoint epoch={ckpt['epoch']}, val_loss={ckpt.get('val_loss', 'N/A')}")

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

# Verify ordering matches Experiment 1
with open("logs/exp_01_indictrans2_baseline_results.json") as f:
    exp1 = json.load(f)
exp1_texts = [r['english_text'] for r in exp1['per_example_results']]
exp3_texts = [ex['english_text'] for ex in strict_heldout]
assert exp1_texts == exp3_texts, "Ordering mismatch with Experiment 1!"
print("  Ordering verified: matches Experiment 1")

# ------------------------------------------------------------------
# 3. Generate with default model.generate()
# ------------------------------------------------------------------
print(f"\n[3/5] Generating on all {len(strict_heldout)} strict held-out examples...")
print("=" * 70)

bleu_scorer = BLEU(effective_order=True)
results = []
start_time = time.time()

for i, ex in enumerate(strict_heldout):
    english_text = ex['english_text']
    hindi_ref = ex['hindi_text']
    num_notes = ex['num_notes']

    src_ids = ex['src_ids'].unsqueeze(0).to(device)
    melody = ex['melody_features'].unsqueeze(0).to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=src_ids,
            melody_features=melody,
            max_length=50,
            num_beams=5,
        )

    tokenizer._switch_to_target_mode()
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    tokenizer._switch_to_input_mode()
    gen_text = model.postprocess_tgt(gen_text)[0]

    # Metrics
    gen_syl = count_hindi_syllables(gen_text)
    syl_error = abs(gen_syl - num_notes)
    syl_match = syl_error <= 2

    max_clen = max(len(gen_text), len(hindi_ref), 1)
    matching_chars = sum(1 for a, b in zip(gen_text, hindi_ref) if a == b)
    char_overlap = matching_chars / max_clen

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
        print(f"  syl: {gen_syl} (target: {num_notes}, err: {syl_error}) "
              f"{'within2' if syl_match else 'MISS'}")
        print(f"  char_overlap: {char_overlap:.3f}  sent_BLEU: {sent_bleu:.1f}")

elapsed = time.time() - start_time
print(f"\n  Generation complete in {elapsed:.1f}s ({elapsed/len(strict_heldout):.2f}s/example)")

# ------------------------------------------------------------------
# 4. Corpus-level metrics
# ------------------------------------------------------------------
print(f"\n[4/5] Computing corpus-level metrics...")

hypotheses = [r['generated_hi'] for r in results]
references = [r['ground_truth_hi'] for r in results]

corpus_bleu = BLEU().corpus_score(hypotheses, [references]).score

print("  Computing BERTScore...")
P, R, F1 = bert_score_fn(hypotheses, references, lang='hi', verbose=False)
mean_bertscore_f1 = F1.mean().item()

syl_acc = sum(1 for r in results if r['syl_match_within_2']) / len(results)
mean_syl_err = sum(r['syl_error'] for r in results) / len(results)
mean_char_overlap = sum(r['char_overlap_ratio'] for r in results) / len(results)

# ------------------------------------------------------------------
# 5. Save and report
# ------------------------------------------------------------------
print(f"\n[5/5] Saving results...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = {
    'experiment': 'exp_03_mcnst_default',
    'model': 'MCNST (checkpoints/best_model.pt)',
    'description': 'MCNST with default melody-fused generation (no constrained reranking)',
    'timestamp': datetime.now().isoformat(),
    'checkpoint_epoch': ckpt['epoch'],
    'num_examples': len(results),
    'generation_time_seconds': round(elapsed, 1),
    'generation_params': {
        'max_length': 50,
        'num_beams': 5,
        'repetition_penalty': 2.5,
        'no_repeat_ngram_size': 2,
        'use_cache': False,
    },
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
out_path = log_dir / f"exp_03_mcnst_default_results.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  Saved to {out_path}")

# Print raw aggregate
print()
print("AGGREGATE METRICS")
print(f"corpus_bleu: {round(corpus_bleu, 2)}")
print(f"bertscore_f1_mean: {round(mean_bertscore_f1, 4)}")
print(f"syllable_accuracy_pm2: {round(syl_acc, 4)}")
print(f"mean_syllable_error: {round(mean_syl_err, 2)}")
print(f"mean_char_overlap: {round(mean_char_overlap, 4)}")
print(f"total_examples: {len(results)}")

# Print 3 examples
print()
print("EXAMPLE OUTPUTS (first 3)")
for ex in results[:3]:
    print()
    print(f"english_text: {ex['english_text']}")
    print(f"ground_truth_hi: {ex['ground_truth_hi']}")
    print(f"generated_hi: {ex['generated_hi']}")
    print(f"gen_syl_count: {ex['gen_syl_count']}")
    print(f"target_notes: {ex['target_notes']}")
    print(f"syl_error: {ex['syl_error']}")
    print(f"syl_match_within_2: {ex['syl_match_within_2']}")
    print(f"char_overlap_ratio: {ex['char_overlap_ratio']}")
    print(f"sentence_bleu: {ex['sentence_bleu']}")

print()
print("EXPERIMENT 03 COMPLETE")
