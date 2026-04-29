"""
Experiment 14: MelodyAwareDecoderLayer — constrained reranking evaluation
on 198 strict held-out examples.

Loads the checkpoint from exp_13 (checkpoints/melody_aware_model.pt) and
evaluates with syllable-constrained beam search reranking.

Saves: logs/exp_14_melody_aware_constrained_results.json
"""

import sys
import os
import json
import time
import types
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score_fn

from src.models.mcnst_model import MCNST
from src.models.melody_aware_decoder import MelodyAwareDecoderLayer
from src.utils.syllable_utils import count_hindi_syllables


def patch_encode_and_fuse(model):
    _original = model._encode_and_fuse

    def _patched(self, input_ids, melody_features):
        out = _original(input_ids, melody_features)
        encoder_outputs, attention_mask, attn_weights, melody_encoded, melody_mask = out
        self._current_melody_features = melody_encoded
        self._current_melody_mask = melody_mask
        return out

    model._encode_and_fuse = types.MethodType(_patched, model)


print("=" * 70)
print("EXPERIMENT 14: MelodyAwareDecoderLayer — Constrained Reranking Eval")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Build model, wrap layers, load checkpoint
# ------------------------------------------------------------------
print("\n[1/5] Building model and loading melody-aware checkpoint...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=0)

# Freeze ALL original decoder layers
for layer in model.seq2seq.model.decoder.layers:
    for p in layer.parameters():
        p.requires_grad = False

model.seq2seq.model.decoder.layer_norm.weight.requires_grad = False
model.seq2seq.model.decoder.layer_norm.bias.requires_grad = False
model.seq2seq.lm_head.weight.requires_grad = True
model.seq2seq.model.decoder.embed_tokens.weight.requires_grad = True

num_layers = len(model.seq2seq.model.decoder.layers)
for i in range(num_layers):
    orig = model.seq2seq.model.decoder.layers[i]
    wrapped = MelodyAwareDecoderLayer(
        original_layer=orig, melody_dim=256, hidden_dim=1024,
        num_heads=16, parent_model=model,
    )
    for p in wrapped.original_layer.parameters():
        p.requires_grad = False
    model.seq2seq.model.decoder.layers[i] = wrapped

patch_encode_and_fuse(model)

# Load checkpoint
CHECKPOINT_PATH = Path("checkpoints/melody_aware_model.pt")
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
else:
    device = torch.device('cpu')

ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()
tokenizer = model.tokenizer
print(f"  Loaded epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
print(f"  Device: {device}")

# ------------------------------------------------------------------
# 2. Load strict held-out
# ------------------------------------------------------------------
print("\n[2/5] Loading strict held-out set...")
test_data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)
train_data_raw = torch.load("src/data/processed/fma_train_data.pt", weights_only=False)
train_english = {ex['english_text'] for ex in train_data_raw}
strict_heldout = [ex for ex in test_data if ex['english_text'] not in train_english]
assert len(strict_heldout) == 198

with open("logs/exp_01_indictrans2_baseline_results.json") as f:
    exp1 = json.load(f)
assert [r['english_text'] for r in exp1['per_example_results']] == \
       [ex['english_text'] for ex in strict_heldout]
print(f"  Strict held-out: {len(strict_heldout)} (ordering verified)")

# ------------------------------------------------------------------
# 3. Constrained beam search reranking
# ------------------------------------------------------------------
NUM_BEAMS = 8
SYLLABLE_PENALTY = 3.0
REPETITION_PENALTY = 2.5
NO_REPEAT_NGRAM = 2

print(f"\n[3/5] Constrained reranking on {len(strict_heldout)} examples...")
print(f"  num_beams={NUM_BEAMS}, syl_penalty={SYLLABLE_PENALTY}")
print("=" * 70)

bleu_scorer = BLEU(effective_order=True)
results = []
constraint_tracking = []
gen_start = time.time()

for i, ex in enumerate(strict_heldout):
    english_text = ex['english_text']
    hindi_ref = ex['hindi_text']
    num_notes = ex['num_notes']
    src_ids = ex['src_ids'].unsqueeze(0).to(device)
    melody = ex['melody_features'].unsqueeze(0).to(device)

    # Generate NUM_BEAMS candidates
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=src_ids, melody_features=melody,
            max_length=50, num_beams=NUM_BEAMS,
            num_return_sequences=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
        )

    # Decode all candidates
    candidates = []
    for seq in gen_ids:
        tokenizer._switch_to_target_mode()
        text = tokenizer.decode(seq, skip_special_tokens=True)
        tokenizer._switch_to_input_mode()
        text = model.postprocess_tgt(text)[0]
        syl = count_hindi_syllables(text)
        syl_err = abs(syl - num_notes)
        candidates.append({
            'text': text, 'syl': syl, 'syl_err': syl_err,
            'within_2': syl_err <= 2,
        })

    # Rerank: penalise syllable mismatch
    best_candidate = None
    best_score = float('-inf')
    for ci, c in enumerate(candidates):
        # Model score approximated by rank (beam 0 = best model score)
        model_score = -ci  # rank-based
        syl_penalty = SYLLABLE_PENALTY * c['syl_err'] / max(num_notes, 1)
        joint_score = model_score - syl_penalty
        if joint_score > best_score:
            best_score = joint_score
            best_candidate = c

    gen_text = best_candidate['text']
    gen_syl = best_candidate['syl']
    syl_error = best_candidate['syl_err']
    syl_match = syl_error <= 2

    any_within_2 = any(c['within_2'] for c in candidates)
    n_within_2 = sum(1 for c in candidates if c['within_2'])

    max_clen = max(len(gen_text), len(hindi_ref), 1)
    matching_chars = sum(1 for a, b in zip(gen_text, hindi_ref) if a == b)
    char_overlap = matching_chars / max_clen
    sent_bleu = bleu_scorer.sentence_score(gen_text, [hindi_ref]).score

    results.append({
        'index': i, 'song_name': ex['song_name'],
        'english_text': english_text, 'ground_truth_hi': hindi_ref,
        'generated_hi': gen_text, 'gen_syl_count': gen_syl,
        'target_notes': int(num_notes), 'syl_error': syl_error,
        'syl_match_within_2': syl_match,
        'char_overlap_ratio': round(char_overlap, 4),
        'sentence_bleu': round(sent_bleu, 2),
    })

    constraint_tracking.append({
        'index': i,
        'any_candidate_within_2': any_within_2,
        'n_candidates_within_2': n_within_2,
        'all_candidate_syls': [c['syl'] for c in candidates],
        'selected_syl': gen_syl,
        'target': int(num_notes),
    })

    if i < 3 or (i + 1) % 50 == 0 or i == len(strict_heldout) - 1:
        print(f"\n--- Example {i} (song: {ex['song_name']}) ---")
        print(f"  EN:  {english_text}")
        print(f"  REF: {hindi_ref}")
        print(f"  GEN: {gen_text}")
        print(f"  syl: {gen_syl} (target: {num_notes}, err: {syl_error}) "
              f"{'within2' if syl_match else 'MISS'}")
        print(f"  candidates_within_2: {n_within_2}/{NUM_BEAMS}")

gen_elapsed = time.time() - gen_start

# ------------------------------------------------------------------
# 4. Corpus metrics
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

# Constraint tracking summary
n_with_any = sum(1 for c in constraint_tracking if c['any_candidate_within_2'])
constraint_summary = {
    'examples_with_any_candidate_within_2': n_with_any,
    'constraint_success_rate': round(n_with_any / len(constraint_tracking), 4),
    'examples_with_no_candidate_within_2': len(constraint_tracking) - n_with_any,
    'constraint_fallback_rate': round(1 - n_with_any / len(constraint_tracking), 4),
}

# ------------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------------
print(f"\n[5/5] Saving results...")

output = {
    'experiment': 'exp_14_melody_aware_constrained',
    'model': 'MCNST + MelodyAwareDecoderLayer (checkpoints/melody_aware_model.pt)',
    'description': 'MelodyAwareDecoderLayer with constrained beam search reranking',
    'timestamp': datetime.now().isoformat(),
    'checkpoint_epoch': ckpt['epoch'],
    'checkpoint_val_loss': ckpt['val_loss'],
    'num_examples': len(results),
    'generation_time_seconds': round(gen_elapsed, 1),
    'constrained_search_params': {
        'num_beams': NUM_BEAMS,
        'syllable_penalty': SYLLABLE_PENALTY,
        'repetition_penalty': REPETITION_PENALTY,
        'no_repeat_ngram_size': NO_REPEAT_NGRAM,
    },
    'constraint_tracking_summary': constraint_summary,
    'aggregate_metrics': {
        'corpus_bleu': round(corpus_bleu, 2),
        'mean_bertscore_f1': round(mean_bertscore_f1, 4),
        'syllable_accuracy_within_2': round(syl_acc, 4),
        'mean_syllable_error': round(mean_syl_err, 2),
        'mean_char_overlap': round(mean_char_overlap, 4),
    },
    'per_example_results': results,
    'per_example_bertscore_f1': [round(f.item(), 4) for f in F1],
    'constraint_tracking': constraint_tracking,
}

out_path = Path("logs") / "exp_14_melody_aware_constrained_results.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  Saved to {out_path}")

print()
print("AGGREGATE METRICS")
print(f"corpus_bleu:            {round(corpus_bleu, 2)}")
print(f"bertscore_f1_mean:      {round(mean_bertscore_f1, 4)}")
print(f"syllable_accuracy_pm2:  {round(syl_acc, 4)}")
print(f"mean_syllable_error:    {round(mean_syl_err, 2)}")
print(f"mean_char_overlap:      {round(mean_char_overlap, 4)}")
print(f"total_examples:         {len(results)}")

print()
print("CONSTRAINT TRACKING")
print(f"  examples_with_any_candidate_within_2: {n_with_any}/198 "
      f"({constraint_summary['constraint_success_rate']*100:.1f}%)")
print(f"  constraint_fallback: {constraint_summary['examples_with_no_candidate_within_2']}/198 "
      f"({constraint_summary['constraint_fallback_rate']*100:.1f}%)")

print()
print("EXPERIMENT 14 COMPLETE")
