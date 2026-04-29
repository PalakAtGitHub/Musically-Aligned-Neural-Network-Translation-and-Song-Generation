"""
Experiment 02: MCNST with constrained beam search (syllable reranking).

Uses the same algorithm as SyllableConstrainedBeamSearch in mcnst_model.py
but reimplemented here so we can inspect ALL beam candidates per example
and track how often at least one candidate falls within +-2 syllables
of the melody note target.
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
print("EXPERIMENT 02: MCNST + Constrained Beam Search")
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

# ------------------------------------------------------------------
# 3. Constrained beam search parameters (match SyllableConstrainedBeamSearch)
# ------------------------------------------------------------------
NUM_BEAMS = 8
SYLLABLE_PENALTY = 3.0
REPETITION_PENALTY = 2.5
NO_REPEAT_NGRAM = 2

# ------------------------------------------------------------------
# 4. Generate with per-candidate tracking
# ------------------------------------------------------------------
print(f"\n[3/5] Generating on all {len(strict_heldout)} strict held-out examples...")
print(f"  num_beams={NUM_BEAMS}, syl_penalty={SYLLABLE_PENALTY}")
print("=" * 70)

bleu_scorer = BLEU(effective_order=True)
results = []
start_time = time.time()

# Constraint tracking
n_any_within_2 = 0   # examples where >= 1 candidate within +-2
n_none_within_2 = 0  # examples where NO candidate within +-2 (fallback)

for i, ex in enumerate(strict_heldout):
    english_text = ex['english_text']
    hindi_ref = ex['hindi_text']
    num_notes = ex['num_notes']
    target_syl = int(num_notes)

    src_ids = ex['src_ids'].unsqueeze(0).to(device)
    melody = ex['melody_features'].unsqueeze(0).to(device)

    # Encode + fuse melody
    with torch.no_grad():
        encoder_outputs, attention_mask, _, _, _ = model._encode_and_fuse(
            src_ids, melody
        )

    # Generate N candidates
    max_len = max(50, target_syl * 3)
    with torch.no_grad():
        gen_output = model.seq2seq.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_BEAMS,
            max_length=max_len,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=False,
        )

    sequences = gen_output.sequences
    seq_scores = gen_output.sequences_scores

    # Score and inspect all candidates
    candidates = []
    best_score = float('-inf')
    best_idx = 0
    any_within_2 = False

    for j in range(sequences.size(0)):
        tokenizer._switch_to_target_mode()
        text = tokenizer.decode(sequences[j], skip_special_tokens=True)
        tokenizer._switch_to_input_mode()
        processed = model.postprocess_tgt(text)
        text = processed[0] if processed else text

        syl_count = count_hindi_syllables(text)
        syl_dev = abs(syl_count - target_syl) / max(target_syl, 1)
        model_score = seq_scores[j].item()
        joint_score = model_score - SYLLABLE_PENALTY * syl_dev

        within_2 = abs(syl_count - target_syl) <= 2
        if within_2:
            any_within_2 = True

        candidates.append({
            'text': text,
            'syl_count': syl_count,
            'model_score': round(model_score, 4),
            'joint_score': round(joint_score, 4),
            'within_2': within_2,
        })

        if joint_score > best_score:
            best_score = joint_score
            best_idx = j

    if any_within_2:
        n_any_within_2 += 1
    else:
        n_none_within_2 += 1

    # Best candidate
    best = candidates[best_idx]
    gen_text = best['text']
    gen_syl = best['syl_count']
    syl_error = abs(gen_syl - target_syl)
    syl_match = syl_error <= 2

    # Character overlap
    max_clen = max(len(gen_text), len(hindi_ref), 1)
    matching_chars = sum(1 for a, b in zip(gen_text, hindi_ref) if a == b)
    char_overlap = matching_chars / max_clen

    # Sentence BLEU
    sent_bleu = bleu_scorer.sentence_score(gen_text, [hindi_ref]).score

    # Count how many candidates were within +-2
    n_within_2_this = sum(1 for c in candidates if c['within_2'])

    result = {
        'index': i,
        'song_name': ex['song_name'],
        'english_text': english_text,
        'ground_truth_hi': hindi_ref,
        'generated_hi': gen_text,
        'gen_syl_count': gen_syl,
        'target_notes': target_syl,
        'syl_error': syl_error,
        'syl_match_within_2': syl_match,
        'char_overlap_ratio': round(char_overlap, 4),
        'sentence_bleu': round(sent_bleu, 2),
        'constraint_tracking': {
            'any_candidate_within_2': any_within_2,
            'n_candidates_within_2': n_within_2_this,
            'n_candidates_total': sequences.size(0),
            'selected_candidate_idx': best_idx,
            'selected_joint_score': round(best_score, 4),
        },
        'all_candidates': candidates,
    }
    results.append(result)

    if i < 3 or (i + 1) % 50 == 0 or i == len(strict_heldout) - 1:
        print(f"\n--- Example {i} (song: {ex['song_name']}) ---")
        print(f"  EN:  {english_text}")
        print(f"  REF: {hindi_ref}")
        print(f"  GEN: {gen_text}")
        print(f"  syl: {gen_syl} (target: {target_syl}, err: {syl_error}) "
              f"{'within2' if syl_match else 'MISS'}")
        print(f"  char_overlap: {char_overlap:.3f}  sent_BLEU: {sent_bleu:.1f}")
        print(f"  Beam candidates within +-2: {n_within_2_this}/{sequences.size(0)}"
              f"  {'CONSTRAINED' if any_within_2 else 'FALLBACK'}")
        # Show candidate syllable distribution
        syl_dist = [c['syl_count'] for c in candidates]
        print(f"  Candidate syl counts: {syl_dist}")

elapsed = time.time() - start_time
print(f"\n  Generation complete in {elapsed:.1f}s ({elapsed/len(strict_heldout):.2f}s/example)")

# ------------------------------------------------------------------
# 5. Corpus-level metrics
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
# 6. Save and report
# ------------------------------------------------------------------
print(f"\n[5/5] Saving results...")

output = {
    'experiment': 'exp_02_mcnst_constrained',
    'model': 'MCNST (checkpoints/best_model.pt)',
    'description': 'MCNST with syllable-constrained beam search (reranking)',
    'timestamp': datetime.now().isoformat(),
    'checkpoint_epoch': ckpt['epoch'],
    'num_examples': len(results),
    'generation_time_seconds': round(elapsed, 1),
    'constrained_search_params': {
        'num_beams': NUM_BEAMS,
        'syllable_penalty': SYLLABLE_PENALTY,
        'repetition_penalty': REPETITION_PENALTY,
        'no_repeat_ngram_size': NO_REPEAT_NGRAM,
    },
    'aggregate_metrics': {
        'corpus_bleu': round(corpus_bleu, 2),
        'mean_bertscore_f1': round(mean_bertscore_f1, 4),
        'syllable_accuracy_within_2': round(syl_acc, 4),
        'mean_syllable_error': round(mean_syl_err, 2),
        'mean_char_overlap': round(mean_char_overlap, 4),
    },
    'constraint_tracking_summary': {
        'examples_with_any_candidate_within_2': n_any_within_2,
        'examples_with_no_candidate_within_2': n_none_within_2,
        'constraint_success_rate': round(n_any_within_2 / len(results), 4),
        'constraint_fallback_rate': round(n_none_within_2 / len(results), 4),
    },
    'per_example_results': results,
    'per_example_bertscore_f1': [round(f.item(), 4) for f in F1],
}

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
out_path = log_dir / "exp_02_mcnst_constrained_results.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  Saved to {out_path}")

# Summary table
print(f"\n{'=' * 70}")
print(f"EXPERIMENT 02 SUMMARY: MCNST + Constrained Beam Search")
print(f"{'=' * 70}")
print(f"  Examples:                {len(results)}")
print(f"  Corpus BLEU:             {corpus_bleu:.2f}")
print(f"  BERTScore F1 (mean):     {mean_bertscore_f1:.4f}")
print(f"  Syllable Acc (within 2): {syl_acc*100:.1f}%")
print(f"  Mean Syllable Error:     {mean_syl_err:.2f}")
print(f"  Mean Char Overlap:       {mean_char_overlap:.4f}")
print(f"  ---")
print(f"  Constraint success:      {n_any_within_2}/{len(results)} "
      f"({n_any_within_2/len(results)*100:.1f}%) had >=1 candidate within +-2")
print(f"  Constraint fallback:     {n_none_within_2}/{len(results)} "
      f"({n_none_within_2/len(results)*100:.1f}%) had NO candidate within +-2")
print(f"{'=' * 70}")
print("EXPERIMENT 02 COMPLETE")
print(f"{'=' * 70}")
