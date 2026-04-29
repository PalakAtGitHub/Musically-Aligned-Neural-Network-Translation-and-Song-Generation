"""
Diagnostic 13: RL reward distribution on strict held-out examples.

Picks 4 random held-out examples (seed=42), generates K=3 nucleus samples
each, computes rewards. Checks whether the model can produce high-reward
samples on novel inputs it hasn't memorized.
"""

import sys
import os
import math
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
from sacrebleu.metrics import BLEU

from src.models.mcnst_model import MCNST
from src.utils.syllable_utils import count_hindi_syllables

bleu_scorer = BLEU(effective_order=True)


def compute_reward(gen_text, ref_text, num_notes):
    syl_count = count_hindi_syllables(gen_text)
    syl_diff = abs(syl_count - num_notes)
    if syl_diff <= 2:
        syl_score = 1.0
    else:
        syl_score = math.exp(-(syl_diff - 2) / 5.0)
    bleu = bleu_scorer.sentence_score(gen_text, [ref_text]).score / 100.0
    total = min(1.0, 0.5 * syl_score + 0.5 * bleu)
    return total, syl_score, bleu, syl_count


print("=" * 70)
print("DIAGNOSTIC 13: RL Reward Distribution on Strict Held-Out")
print("=" * 70)

# Load model
print("\n[1/3] Loading model from checkpoints/best_model.pt...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=0)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
else:
    device = torch.device('cpu')

ckpt = torch.load("checkpoints/best_model.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()
tokenizer = model.tokenizer
print(f"  Loaded epoch={ckpt.get('epoch', '?')}, device={device}")

# Build strict held-out
print("\n[2/3] Building strict held-out set...")
test_data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)
train_data = torch.load("src/data/processed/fma_train_data.pt", weights_only=False)
train_english = {ex['english_text'] for ex in train_data}
strict_heldout = [ex for ex in test_data if ex['english_text'] not in train_english]
assert len(strict_heldout) == 198
print(f"  Strict held-out: {len(strict_heldout)}")

# Pick 4 random examples
random.seed(42)
indices = random.sample(range(len(strict_heldout)), 4)
examples = [strict_heldout[i] for i in indices]
print(f"  Selected indices: {indices}")
for i, ex in enumerate(examples):
    print(f"  Ex {i}: song={ex['song_name']}, notes={ex['num_notes']}, "
          f"en=\"{ex['english_text'][:60]}...\"")

# Generate K=3 samples per example
K = 3
print(f"\n[3/3] Generating K={K} nucleus samples per example...")
print("  (temperature=0.9, top_p=0.95, do_sample=True)")
print("=" * 70)

all_rewards = []
all_syl_scores = []
n_syl_perfect = 0  # syl_score == 1.0 (within ±2)
total_samples = 0

for i, ex in enumerate(examples):
    src_ids = ex['src_ids'].unsqueeze(0).to(device)
    melody = ex['melody_features'].unsqueeze(0).to(device)
    ref_text = ex['hindi_text']
    num_notes = int(ex['num_notes'])

    print(f"\n--- Example {i} (idx={indices[i]}, song={ex['song_name']}, "
          f"target={num_notes} notes) ---")
    print(f"  EN:  {ex['english_text']}")
    print(f"  REF: {ref_text} (syl={count_hindi_syllables(ref_text)})")

    for k in range(K):
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=src_ids, melody_features=melody,
                max_length=50, do_sample=True,
                temperature=0.9, top_p=0.95,
                num_return_sequences=1,
            )
        tokenizer._switch_to_target_mode()
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        tokenizer._switch_to_input_mode()
        gen_text = model.postprocess_tgt(gen_text)[0]

        reward, syl_score, bleu_score, syl_count = compute_reward(
            gen_text, ref_text, num_notes
        )
        all_rewards.append(reward)
        all_syl_scores.append(syl_score)
        if syl_score == 1.0:
            n_syl_perfect += 1
        total_samples += 1

        print(f"  Sample {k}: \"{gen_text}\"")
        print(f"    syl={syl_count} reward={reward:.4f} "
              f"(syl_score={syl_score:.4f}, bleu={bleu_score:.4f})"
              f"{' *** WITHIN ±2' if syl_score == 1.0 else ''}")

# Summary
print()
print("=" * 70)
print("DIAGNOSTIC 13 SUMMARY")
print("=" * 70)
print(f"  Total samples: {total_samples}")
print(f"  Reward range: [{min(all_rewards):.4f}, {max(all_rewards):.4f}]")
print(f"  Reward mean:  {sum(all_rewards)/len(all_rewards):.4f}")
print(f"  Syl_score=1.0 (within ±2): {n_syl_perfect}/{total_samples} "
      f"({n_syl_perfect/total_samples*100:.0f}%)")
print(f"  Mean syl_score: {sum(all_syl_scores)/len(all_syl_scores):.4f}")

any_above_03 = any(r > 0.3 for r in all_rewards)
any_above_05 = any(r > 0.5 for r in all_rewards)
print(f"  Any reward > 0.3: {any_above_03}")
print(f"  Any reward > 0.5: {any_above_05}")

if max(all_rewards) < 0.3:
    print("\n  VERDICT: All rewards below 0.3 — RL may struggle.")
    print("  STOP and report before proceeding to Stage 1.")
else:
    print(f"\n  VERDICT: Reward range [{min(all_rewards):.2f}, {max(all_rewards):.2f}] "
          f"provides signal for RL.")

print()
print("DIAGNOSTIC 13 COMPLETE")
