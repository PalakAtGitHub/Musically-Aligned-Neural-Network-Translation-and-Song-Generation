"""
Diagnostic test 04: cluster_loss and openness_reward.

What this script does:
  1. Construct a fake batch with crafted logits where we put nearly-all
     probability mass on a SPECIFIC TOKEN (one with known features from
     the precomputed table).
  2. Pair with a controlled alignment matrix and melody features.
  3. Compute expected_leading_consonants and expected_openness by hand.
  4. Compute them via cluster_loss and openness_reward.
  5. Verify the two agree and print every intermediate.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.training.loss import PentathlonLoss, _load_phoneme_features

print("=" * 70)
print("DIAGNOSTIC 04: cluster_loss and openness_reward")
print("=" * 70)


# ----------------------------------------------------------------------
# 1. Load phoneme features, pick two tokens with known properties
# ----------------------------------------------------------------------
print("\n[1/5] Load phoneme table, pick known tokens for test...")
device = torch.device('cpu')
feats = _load_phoneme_features(device, expected_vocab_size=122672)

# Earlier diagnostics showed:
#   token  8 = 'है'   leading=1 open=1 vowel=1
#   token 15 = 'छे'   leading=2 open=1 vowel=1
#   token 5136 = 'च्छ'  leading=4 open=0 vowel=1  (heavy cluster)
CLEAN_TOKEN = 8      # 'है'  : leading=1, open=1
OPEN_TOKEN  = 8      # same (ends in open vowel, light cluster)
HEAVY_TOKEN = 5136   # 'च्छ' : leading=4, open=0

print(f"  token {CLEAN_TOKEN}:  leading={feats['n_leading_consonants'][CLEAN_TOKEN].item()}  "
      f"open={feats['ends_in_open_vowel'][CLEAN_TOKEN].item()}")
print(f"  token {HEAVY_TOKEN}:  leading={feats['n_leading_consonants'][HEAVY_TOKEN].item()}  "
      f"open={feats['ends_in_open_vowel'][HEAVY_TOKEN].item()}")


# ----------------------------------------------------------------------
# 2. Build crafted inputs
# ----------------------------------------------------------------------
print("\n[2/5] Build crafted inputs...")
B, T_out, V, N = 2, 3, 122672, 4

# Logits: start with uniform-ish, then spike a chosen token
# (we use large positive logit at one token so softmax -> ~1 at that token)
logits = torch.zeros(B, T_out, V)

# Example 0: every output position is predicted to be the CLEAN_TOKEN
#   expected_leading ≈ 1,  expected_open ≈ 1
logits[0, :, CLEAN_TOKEN] = 20.0

# Example 1: every output position is predicted to be the HEAVY_TOKEN
#   expected_leading ≈ 4,  expected_open ≈ 0
logits[1, :, HEAVY_TOKEN] = 20.0

# Labels: all non-pad (we want the full T_out to count)
labels = torch.tensor([
    [CLEAN_TOKEN, CLEAN_TOKEN, CLEAN_TOKEN],
    [HEAVY_TOKEN, HEAVY_TOKEN, HEAVY_TOKEN],
])

# Alignment: uniform over 4 notes per output position (each output pos
# aligns equally with all 4 notes).
alignment = torch.full((B, T_out, N), 1.0 / N)

# Melody features [B, N, 5] where col 2 = duration, col 4 = beat_strength.
# Make example 0 have notes with duration=1.0 and beat_strength=1.0.
# Make example 1 have short notes (duration=0.25) and strong beats.
melody_features = torch.zeros(B, N, 5)
melody_features[0, :, 2] = 1.0    # all 1.0 beat long
melody_features[0, :, 4] = 1.0    # all strong beats
melody_features[0, :, 0] = 60.0   # just to make note_mask non-zero
melody_features[1, :, 2] = 0.25   # all very short
melody_features[1, :, 4] = 1.0    # all strong beats
melody_features[1, :, 0] = 60.0

print(f"  logits.shape:          {tuple(logits.shape)}")
print(f"  labels.shape:          {tuple(labels.shape)}")
print(f"  alignment.shape:       {tuple(alignment.shape)}")
print(f"  melody_features.shape: {tuple(melody_features.shape)}")
print(f"  example 0: predicts token {CLEAN_TOKEN} always, notes are long+strong")
print(f"  example 1: predicts token {HEAVY_TOKEN} always, notes are short+strong")


# ----------------------------------------------------------------------
# 3. Compute expected answers by hand
# ----------------------------------------------------------------------
print("\n[3/5] Hand-compute expected answers...")

# Example 0:
#   every output position has ~all prob on CLEAN_TOKEN which has leading=1
#   -> expected_cons_per_output = 1.0 per position
#   alignment uniform 1/N: expected_cons_per_note = 1/N * 3 per note
#     wait: alignment[b,t,n] = 1/N, so exp_cons_per_note[b,n] = sum_t (1/N)*1 = T_out/N = 3/4
#   duration 1.0 everywhere -> per_note_penalty = (3/4) / 1 = 0.75
#   averaged over notes -> 0.75
hand_cluster_ex0 = (T_out / N) / 1.0
hand_open_ex0 = (T_out / N) * 1.0     # open=1, beat=1

# Example 1:
#   every output position has leading=4 on HEAVY_TOKEN
#   exp_cons_per_note = (T_out / N) * 4 = 3/4 * 4 = 3.0
#   duration 0.25 -> per_note_penalty = 3.0 / 0.25 = 12.0
hand_cluster_ex1 = (T_out / N * 4.0) / 0.25
hand_open_ex1 = (T_out / N) * 0.0    # HEAVY_TOKEN not open

hand_cluster_mean = (hand_cluster_ex0 + hand_cluster_ex1) / 2.0
hand_open_mean    = (hand_open_ex0 + hand_open_ex1) / 2.0
hand_open_loss    = -hand_open_mean  # openness_reward returns negative

print(f"  example 0 cluster_penalty: {hand_cluster_ex0:.4f}")
print(f"  example 1 cluster_penalty: {hand_cluster_ex1:.4f}")
print(f"  mean cluster_loss (hand):  {hand_cluster_mean:.4f}")
print(f"  example 0 openness:         {hand_open_ex0:.4f}")
print(f"  example 1 openness:         {hand_open_ex1:.4f}")
print(f"  openness_reward (hand):     {hand_open_loss:.4f}  (negative mean)")


# ----------------------------------------------------------------------
# 4. Run the actual loss methods
# ----------------------------------------------------------------------
print("\n[4/5] Run loss methods...")
loss_fn = PentathlonLoss(pad_token_id=1)

actual_cluster = loss_fn.cluster_loss(logits, labels, alignment, melody_features)
actual_open    = loss_fn.openness_reward(logits, labels, alignment, melody_features)

print(f"  actual cluster_loss:     {actual_cluster.item():.4f}")
print(f"  actual openness_reward:  {actual_open.item():.4f}")


# ----------------------------------------------------------------------
# 5. Compare
# ----------------------------------------------------------------------
print("\n[5/5] Compare hand vs actual (tolerance 0.05)...")
cluster_diff = abs(actual_cluster.item() - hand_cluster_mean)
open_diff    = abs(actual_open.item() - hand_open_loss)

if cluster_diff < 0.05:
    print(f"  ✓ cluster_loss matches (diff: {cluster_diff:.4f})")
else:
    print(f"  ✗ cluster_loss MISMATCH (diff: {cluster_diff:.4f})")

if open_diff < 0.05:
    print(f"  ✓ openness_reward matches (diff: {open_diff:.4f})")
else:
    print(f"  ✗ openness_reward MISMATCH (diff: {open_diff:.4f})")


# Verify gradient flow
print("\n  Gradient check:")
total = actual_cluster + actual_open
# logits.requires_grad is False since they're leaves we built by hand with zeros
logits.requires_grad_(True)
actual_cluster2 = loss_fn.cluster_loss(logits, labels, alignment, melody_features)
actual_cluster2.backward()
print(f"  logits.grad not None:  {logits.grad is not None}")
print(f"  logits.grad.nonzero:   {(logits.grad.abs() > 1e-12).sum().item()} / {logits.numel()}")
print(f"  logits.grad.norm:      {logits.grad.norm().item():.6f}")

print("\n" + "=" * 70)
print("DIAGNOSTIC 04 COMPLETE")
print("=" * 70)
