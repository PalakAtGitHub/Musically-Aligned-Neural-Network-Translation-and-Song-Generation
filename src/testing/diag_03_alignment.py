"""
Diagnostic test 03: LearnedAlignment module standalone.

What this script does:
  1. Instantiate LearnedAlignment with default dims.
  2. Feed a dummy batch of decoder hidden states and melody features.
  3. Print input and output shapes.
  4. Verify output sums to 1 along the note dimension (rows of the
     alignment matrix should be probability distributions).
  5. Verify mask handling (padded notes get zero attention).
  6. Run backward and confirm gradients flow into the alignment module's
     parameters.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.alignment import LearnedAlignment

print("=" * 70)
print("DIAGNOSTIC 03: LearnedAlignment module standalone test")
print("=" * 70)

torch.manual_seed(0)

# ----------------------------------------------------------------------
# 1. Instantiate
# ----------------------------------------------------------------------
print("\n[1/6] Instantiate LearnedAlignment...")
aligner = LearnedAlignment(text_dim=1024, melody_dim=256, attn_dim=256, num_heads=4)
total_params = sum(p.numel() for p in aligner.parameters())
trainable = sum(p.numel() for p in aligner.parameters() if p.requires_grad)
print(f"  Total params:     {total_params:,}")
print(f"  Trainable params: {trainable:,}")


# ----------------------------------------------------------------------
# 2. Build a dummy batch
# ----------------------------------------------------------------------
print("\n[2/6] Build dummy inputs...")
B, T_out, N = 2, 5, 8
decoder_hidden = torch.randn(B, T_out, 1024)
melody_encoded = torch.randn(B, N, 256)

# Simulate padding: example 0 has all 8 notes, example 1 has only 6 notes
melody_mask = torch.tensor([
    [False]*8,
    [False]*6 + [True]*2,
])

print(f"  decoder_hidden.shape: {tuple(decoder_hidden.shape)}")
print(f"  melody_encoded.shape: {tuple(melody_encoded.shape)}")
print(f"  melody_mask.shape:    {tuple(melody_mask.shape)}")
print(f"  melody_mask[0]: {melody_mask[0].tolist()}")
print(f"  melody_mask[1]: {melody_mask[1].tolist()}")


# ----------------------------------------------------------------------
# 3. Forward
# ----------------------------------------------------------------------
print("\n[3/6] Forward pass...")
alignment = aligner(decoder_hidden, melody_encoded, melody_mask)
print(f"  alignment.shape: {tuple(alignment.shape)}")
print(f"  (expected: [{B}, {T_out}, {N}])")


# ----------------------------------------------------------------------
# 4. Row-stochastic sanity check
# ----------------------------------------------------------------------
print("\n[4/6] Check rows sum to 1 along the note dim...")
row_sums = alignment.sum(dim=-1)  # [B, T_out]
print(f"  row_sums[0]: {row_sums[0].tolist()}")
print(f"  row_sums[1]: {row_sums[1].tolist()}")
max_dev = (row_sums - 1.0).abs().max().item()
if max_dev < 1e-5:
    print(f"  ✓ All rows sum to 1 (max dev: {max_dev:.2e})")
else:
    print(f"  ✗ Rows do NOT sum to 1 (max dev: {max_dev:.2e})")


# ----------------------------------------------------------------------
# 5. Mask handling: padded notes should get ~0 attention
# ----------------------------------------------------------------------
print("\n[5/6] Check mask: example 1 has notes 6,7 padded...")
padded_attn_example1 = alignment[1, :, 6:8]  # should be ~0
nonpadded_attn_example1 = alignment[1, :, :6].sum(dim=-1)  # should be ~1
padded_attn_example0 = alignment[0, :, :]   # no padding — sum should be 1

print(f"  example 1 attention on padded positions (should be ~0):")
print(f"    {padded_attn_example1.detach().tolist()}")
print(f"  example 1 attention on non-padded positions (sum, should be ~1):")
print(f"    {nonpadded_attn_example1.detach().tolist()}")
print(f"  example 0 attention total (sum over all 8 notes):")
print(f"    {alignment[0].sum(dim=-1).tolist()}")

if padded_attn_example1.abs().max().item() < 1e-5:
    print("  ✓ Padded positions correctly masked (attention ≈ 0)")
else:
    print(f"  ✗ Mask leak: padded positions got non-zero attention "
          f"(max: {padded_attn_example1.abs().max().item():.2e})")


# ----------------------------------------------------------------------
# 6. Gradient flow
# ----------------------------------------------------------------------
print("\n[6/6] Backward pass — verify gradients reach alignment params...")
# Use a simple scalar objective: -log(alignment[0, 0, 3])
# (encourage output position 0 to align with note 3)
loss = -torch.log(alignment[0, 0, 3] + 1e-8)
print(f"  loss: {loss.item():.4f}")
loss.backward()

for name, p in aligner.named_parameters():
    if p.grad is not None:
        norm = p.grad.norm().item()
        print(f"  {name:20s} grad_norm: {norm:.6f}")
    else:
        print(f"  {name:20s} NO GRADIENT")

print("\n" + "=" * 70)
print("DIAGNOSTIC 03 COMPLETE")
print("=" * 70)
