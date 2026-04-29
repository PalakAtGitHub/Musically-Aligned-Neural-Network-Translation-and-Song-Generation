"""
Diagnostic test 01: check vocab size mismatch between tokenizer and logits.

Question we need to answer:
  - Does the current PentathlonLoss crash when the syllable_loss path runs,
    because the vowel_mask (size 32322) is multiplied by logits (size 122672)?
  - Or is something masking the problem?

What this script does:
  1. Load MCNST.
  2. Build a tiny fake batch: 2 examples, 6 source tokens each, 4 target tokens,
     and 8 notes of melody features.
  3. Run a forward pass requesting the loss.
  4. Print every intermediate shape so we can see what actually happens.

If the loss runs without crashing, we know the pipeline is silently falling
through to a path that doesn't hit the shape mismatch, and we need to fix
that too. If it crashes, we know where and why.
"""

import sys
from pathlib import Path

# Make sure project root is on the import path so `src.` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

print("=" * 70)
print("DIAGNOSTIC 01: Vocab size consistency across tokenizer and logits")
print("=" * 70)


# ----------------------------------------------------------------------
# 1. Load the model
# ----------------------------------------------------------------------
print("\n[1/4] Loading MCNST...")
from src.models.mcnst_model import MCNST

model = MCNST(freeze_encoder=True, freeze_decoder_layers=10)
model.eval()

# Use CPU for the diagnostic to keep things simple
device = torch.device('cpu')
model.to(device)

print(f"\n  Tokenizer type:   {type(model.tokenizer).__name__}")
print(f"  Tokenizer length: {len(model.tokenizer)}")
print(f"  Pad token id:     {model.pad_token_id}")
print(f"  Config vocab:     {model.seq2seq.config.vocab_size}")

lm_head = model.seq2seq.get_output_embeddings()
if lm_head is not None:
    print(f"  LM head out_features: {lm_head.out_features}")


# ----------------------------------------------------------------------
# 2. Build a tiny fake batch
# ----------------------------------------------------------------------
print("\n[2/4] Building tiny fake batch...")

# Source: 2 examples, 6 tokens each. Use small integers in the "safe" range.
# Token ids for IndicTrans2 start with special tokens, so 5..20 are safe normal tokens.
src_ids = torch.tensor([
    [2, 5, 6, 7, 8, 1],      # 1 = pad_token_id; this example has 5 real tokens
    [2, 9, 10, 11, 12, 13],  # this example has 6 real tokens
], dtype=torch.long)

# Target: 2 examples, 4 tokens each
tgt_ids = torch.tensor([
    [2, 14, 15, 1],
    [2, 16, 17, 18],
], dtype=torch.long)

# Melody: 2 examples, 8 notes, 5 features per note [pitch, pitch_class, duration_beats, duration_bin, beat_strength]
melody = torch.tensor([
    [[60.0, 0, 0.5, 2, 0.8], [62.0, 2, 0.5, 2, 0.4],
     [64.0, 4, 1.0, 3, 0.9], [65.0, 5, 0.5, 2, 0.3],
     [67.0, 7, 1.0, 3, 0.7], [65.0, 5, 0.5, 2, 0.4],
     [64.0, 4, 0.5, 2, 0.3], [62.0, 2, 1.5, 4, 0.6]],
    [[55.0, 7, 0.5, 2, 0.5], [57.0, 9, 0.5, 2, 0.3],
     [59.0, 11, 1.0, 3, 0.8], [60.0, 0, 0.5, 2, 0.2],
     [62.0, 2, 1.0, 3, 0.7], [64.0, 4, 0.5, 2, 0.4],
     [62.0, 2, 0.5, 2, 0.3], [60.0, 0, 2.0, 5, 0.9]],
])

num_notes = torch.tensor([8, 8], dtype=torch.long)

print(f"  src_ids.shape:    {tuple(src_ids.shape)}")
print(f"  tgt_ids.shape:    {tuple(tgt_ids.shape)}")
print(f"  melody.shape:     {tuple(melody.shape)}")
print(f"  num_notes:        {num_notes.tolist()}")


# ----------------------------------------------------------------------
# 3. Check the vowel mask that will be built
# ----------------------------------------------------------------------
print("\n[3/4] Checking vowel mask construction...")

# The loss lazy-builds the mask on first forward. Trigger that manually first
# so we can inspect it without running the whole loss.
try:
    mask = model.loss_fn._get_vowel_mask(device)
    print(f"  vowel_mask.shape: {tuple(mask.shape)}")
    print(f"  vowel_mask.sum:   {int(mask.sum().item())}  (count of vowel-bearing tokens)")
except Exception as e:
    print(f"  FAILED to build vowel_mask: {type(e).__name__}: {e}")


# ----------------------------------------------------------------------
# 4. Run forward pass and see what happens
# ----------------------------------------------------------------------
print("\n[4/4] Running forward pass with the fake batch...")

model.train()  # so dropout etc. is active but we're not computing gradients we care about

try:
    total_loss, loss_dict = model(
        input_ids=src_ids,
        melody_features=melody,
        labels=tgt_ids,
        num_notes=num_notes,
    )
    print(f"\n  ✓ Forward pass succeeded!")
    print(f"\n  Total loss:    {total_loss.item():.4f}")
    print(f"  Loss dict:")
    for k, v in loss_dict.items():
        print(f"    {k:30s} = {v:.4f}")
    print(f"\n  total_loss.requires_grad: {total_loss.requires_grad}")

    # Try backward to confirm gradient flow
    total_loss.backward()
    print(f"  ✓ Backward pass succeeded")

    # Check a trainable parameter got a gradient
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            nonzero = (p.grad.abs() > 1e-8).sum().item()
            total = p.grad.numel()
            print(f"  First trainable param with gradient: {name}")
            print(f"    gradient nonzero entries: {nonzero}/{total}")
            print(f"    gradient norm: {p.grad.norm().item():.6f}")
            break

except Exception as e:
    import traceback
    print(f"\n  ✗ Forward pass FAILED: {type(e).__name__}")
    print(f"    {e}")
    print("\n  Traceback:")
    traceback.print_exc()


print("\n" + "=" * 70)
print("DIAGNOSTIC 01 COMPLETE")
print("=" * 70)
