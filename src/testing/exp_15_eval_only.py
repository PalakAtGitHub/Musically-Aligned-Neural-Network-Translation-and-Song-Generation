"""
Experiment 15 — evaluation only (checkpoint already trained).
Loads checkpoints/rl_1epoch_model.pt and evaluates on 198 strict held-out
with both beam search and sampling.

Saves:
  logs/exp_15a_rl_1epoch_beam_results.json
  logs/exp_15b_rl_1epoch_sample_results.json
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
from sacrebleu.metrics import BLEU

from src.models.mcnst_model import MCNST
from src.utils.syllable_utils import count_hindi_syllables

print("=" * 70)
print("EXPERIMENT 15 (eval-only): RL 1-epoch — Beam + Sampling Eval")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Load model
# ------------------------------------------------------------------
print("\n[1/4] Loading model from checkpoints/rl_1epoch_model.pt...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=0)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
else:
    device = torch.device('cpu')

ckpt = torch.load("checkpoints/rl_1epoch_model.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()
tokenizer = model.tokenizer
print(f"  Loaded epoch={ckpt.get('epoch', '?')}, device={device}")

# Load training reward trajectory
batch_metrics = ckpt.get('batch_metrics', [])
training_time = ckpt.get('training_time_seconds', 0)

# ------------------------------------------------------------------
# 2. Load strict held-out
# ------------------------------------------------------------------
print("\n[2/4] Loading strict held-out set...")
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
# Shared evaluation function
# ------------------------------------------------------------------
def evaluate_heldout(mode='beam', num_beams=5, temperature=0.9, top_p=0.95):
    bleu_scorer = BLEU(effective_order=True)
    results = []
    gen_start = time.time()

    for i, ex in enumerate(strict_heldout):
        english_text = ex['english_text']
        hindi_ref = ex['hindi_text']
        num_notes = ex['num_notes']
        src_ids = ex['src_ids'].unsqueeze(0).to(device)
        mel = ex['melody_features'].unsqueeze(0).to(device)

        with torch.no_grad():
            if mode == 'beam':
                gen_ids = model.generate(
                    input_ids=src_ids, melody_features=mel,
                    max_length=50, num_beams=num_beams,
                )
            else:
                gen_ids = model.generate(
                    input_ids=src_ids, melody_features=mel,
                    max_length=50, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    num_return_sequences=1,
                )

        tokenizer._switch_to_target_mode()
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        tokenizer._switch_to_input_mode()
        gen_text = model.postprocess_tgt(gen_text)[0]

        gen_syl = count_hindi_syllables(gen_text)
        syl_error = abs(gen_syl - num_notes)
        syl_match = syl_error <= 2
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

        if i < 3 or (i + 1) % 50 == 0 or i == len(strict_heldout) - 1:
            print(f"    [{mode}] Ex {i}: \"{gen_text[:50]}...\" "
                  f"syl={gen_syl} (target={num_notes}, err={syl_error}) "
                  f"{'OK' if syl_match else 'MISS'}")

    gen_elapsed = time.time() - gen_start

    # Corpus metrics
    hypotheses = [r['generated_hi'] for r in results]
    references = [r['ground_truth_hi'] for r in results]
    corpus_bleu = BLEU().corpus_score(hypotheses, [references]).score
    syl_acc = sum(1 for r in results if r['syl_match_within_2']) / len(results)
    mean_syl_err = sum(r['syl_error'] for r in results) / len(results)
    mean_char_overlap = sum(r['char_overlap_ratio'] for r in results) / len(results)

    return {
        'results': results,
        'hypotheses': hypotheses,
        'references': references,
        'generation_time': round(gen_elapsed, 1),
        'aggregate': {
            'corpus_bleu': round(corpus_bleu, 2),
            'mean_bertscore_f1': None,  # computed in separate step
            'syllable_accuracy_within_2': round(syl_acc, 4),
            'mean_syllable_error': round(mean_syl_err, 2),
            'mean_char_overlap': round(mean_char_overlap, 4),
        },
    }


# ------------------------------------------------------------------
# 3. Evaluate
# ------------------------------------------------------------------
print("\n[3/4] Evaluating with beam search (num_beams=5)...")
beam_eval = evaluate_heldout(mode='beam', num_beams=5)

print("\n  Evaluating with sampling (temp=0.9, top_p=0.95)...")
sample_eval = evaluate_heldout(mode='sample', temperature=0.9, top_p=0.95)

# ------------------------------------------------------------------
# 4. Save results
# ------------------------------------------------------------------
print(f"\n[4/4] Saving results...")

rl_hyperparams = {
    'batch_size': 2, 'K': 5, 'lr': 1e-5,
    'temperature': 0.9, 'top_p': 0.95,
}

output_a = {
    'experiment': 'exp_15a_rl_1epoch_beam',
    'model': 'MCNST + 1-epoch MRT (checkpoints/rl_1epoch_model.pt)',
    'description': 'RL fine-tuned MCNST, beam search evaluation',
    'timestamp': datetime.now().isoformat(),
    'source_checkpoint': 'checkpoints/best_model.pt',
    'rl_epochs': 1,
    'rl_hyperparams': rl_hyperparams,
    'training_time_seconds': training_time,
    'num_examples': len(strict_heldout),
    'generation_time_seconds': beam_eval['generation_time'],
    'aggregate_metrics': beam_eval['aggregate'],
    'per_example_results': beam_eval['results'],
    'batch_reward_trajectory': batch_metrics,
}
out_a = Path("logs/exp_15a_rl_1epoch_beam_results.json")
out_a.parent.mkdir(exist_ok=True)
with open(out_a, 'w', encoding='utf-8') as f:
    json.dump(output_a, f, ensure_ascii=False, indent=2)
print(f"  Saved {out_a}")

output_b = {
    'experiment': 'exp_15b_rl_1epoch_sample',
    'model': 'MCNST + 1-epoch MRT (checkpoints/rl_1epoch_model.pt)',
    'description': 'RL fine-tuned MCNST, sampling evaluation',
    'timestamp': datetime.now().isoformat(),
    'source_checkpoint': 'checkpoints/best_model.pt',
    'rl_epochs': 1,
    'rl_hyperparams': rl_hyperparams,
    'training_time_seconds': training_time,
    'num_examples': len(strict_heldout),
    'generation_time_seconds': sample_eval['generation_time'],
    'aggregate_metrics': sample_eval['aggregate'],
    'per_example_results': sample_eval['results'],
}
out_b = Path("logs/exp_15b_rl_1epoch_sample_results.json")
with open(out_b, 'w', encoding='utf-8') as f:
    json.dump(output_b, f, ensure_ascii=False, indent=2)
print(f"  Saved {out_b}")

# Save hypotheses/references for BERTScore computation in separate script
bertscore_data = {
    'beam_hypotheses': beam_eval['hypotheses'],
    'beam_references': beam_eval['references'],
    'sample_hypotheses': sample_eval['hypotheses'],
    'sample_references': sample_eval['references'],
}
bs_path = Path("logs/exp_15_bertscore_inputs.json")
with open(bs_path, 'w', encoding='utf-8') as f:
    json.dump(bertscore_data, f, ensure_ascii=False, indent=2)
print(f"  Saved BERTScore inputs to {bs_path}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
print("=" * 70)
print("EXPERIMENT 15 RESULTS")
print("=" * 70)

if batch_metrics:
    all_rewards = [m['mean_reward'] for m in batch_metrics]
    print(f"\n  TRAINING (1 epoch MRT):")
    print(f"    Time: {training_time}s")
    print(f"    Reward: start={all_rewards[0]:.4f} → end={all_rewards[-1]:.4f}")
    print(f"    Mean reward: {sum(all_rewards)/len(all_rewards):.4f}")

print(f"\n  EVALUATION (198 strict held-out):")
print(f"  {'Metric':<28} {'Beam':>8} {'Sample':>8} {'Prior MCNST':>12}")
print(f"  {'-'*60}")
ba = beam_eval['aggregate']
sa = sample_eval['aggregate']
print(f"  {'corpus_bleu':<28} {ba['corpus_bleu']:>8.2f} "
      f"{sa['corpus_bleu']:>8.2f} {'36.14':>12}")
print(f"  {'bertscore_f1':<28} {ba['mean_bertscore_f1']:>8.4f} "
      f"{sa['mean_bertscore_f1']:>8.4f} {'0.8700':>12}")
print(f"  {'syllable_accuracy_pm2':<28} {ba['syllable_accuracy_within_2']:>8.4f} "
      f"{sa['syllable_accuracy_within_2']:>8.4f} {'0.2273':>12}")
print(f"  {'mean_syllable_error':<28} {ba['mean_syllable_error']:>8.2f} "
      f"{sa['mean_syllable_error']:>8.2f} {'7.01':>12}")
print(f"  {'mean_char_overlap':<28} {ba['mean_char_overlap']:>8.4f} "
      f"{sa['mean_char_overlap']:>8.4f} {'':>12}")

beam_syl_acc = ba['syllable_accuracy_within_2']
prior_syl_acc = 0.2273
delta_syl = beam_syl_acc - prior_syl_acc
beam_bleu = ba['corpus_bleu']
prior_bleu = 36.14
delta_bleu = beam_bleu - prior_bleu

print(f"\n  DECISION CRITERIA (beam search vs prior MCNST default):")
print(f"    Syl accuracy delta: {delta_syl*100:+.1f}pp "
      f"({prior_syl_acc*100:.1f}% → {beam_syl_acc*100:.1f}%)")
print(f"    BLEU delta: {delta_bleu:+.2f} "
      f"({prior_bleu:.2f} → {beam_bleu:.2f})")

if delta_syl >= 0.05 and delta_bleu >= -5.0:
    print(f"    → SUCCESS: syl acc improved by {delta_syl*100:.1f}pp, "
          f"BLEU within tolerance")
elif 0.02 <= delta_syl < 0.05:
    print(f"    → MARGINAL: syl acc improved by {delta_syl*100:.1f}pp")
else:
    print(f"    → INSUFFICIENT: syl acc delta = {delta_syl*100:.1f}pp")

print()
print("EXPERIMENT 15 COMPLETE")
