"""
Diagnostic test 05: smoke training — 3 optimiser steps on 4 real examples.

Checks:
  - Forward + backward + optimizer.step work end-to-end on real data
  - All 7 loss terms stay finite (no NaN / Inf)
  - total_loss trends downward (or at least stays in range)
  - Aligner gradients are non-zero (gradient flows through the 8a path)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.optim as optim

print("=" * 70)
print("DIAGNOSTIC 05: Smoke training (3 steps, 4 examples)")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Load 4 real training examples
# ------------------------------------------------------------------
print("\n[1/4] Loading data...")
from src.data.song_dataset import SongTranslationDataset

dataset = SongTranslationDataset("src/data/processed/fma_train_data.pt")
subset = torch.utils.data.Subset(dataset, range(min(4, len(dataset))))
loader = torch.utils.data.DataLoader(
    subset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn
)
batch = next(iter(loader))
print(f"  Batch keys: {list(batch.keys())}")
print(f"  src_ids:         {tuple(batch['src_ids'].shape)}")
print(f"  tgt_ids:         {tuple(batch['tgt_ids'].shape)}")
print(f"  melody_features: {tuple(batch['melody_features'].shape)}")
print(f"  num_notes:       {batch['num_notes'].tolist()}")

# ------------------------------------------------------------------
# 2. Instantiate model fresh
# ------------------------------------------------------------------
print("\n[2/4] Loading MCNST...")
from src.models.mcnst_model import MCNST

model = MCNST(freeze_encoder=True, freeze_decoder_layers=10)
device = torch.device("cpu")
model.to(device)
model.train()

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)

print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")

# ------------------------------------------------------------------
# 3. Run 3 training steps
# ------------------------------------------------------------------
print("\n[3/4] Running 3 training steps...\n")

losses = []
failed = False

for step in range(1, 4):
    optimizer.zero_grad()

    forward_kwargs = dict(
        input_ids=batch["src_ids"].to(device),
        melody_features=batch["melody_features"].to(device),
        labels=batch["tgt_ids"].to(device),
        num_notes=batch["num_notes"].to(device),
    )
    if "stress_pattern" in batch:
        forward_kwargs["stress_pattern"] = batch["stress_pattern"].to(device)

    total_loss, loss_dict = model(**forward_kwargs)

    # Check for NaN / Inf
    has_nan = False
    for k, v in loss_dict.items():
        if not torch.isfinite(torch.tensor(v)):
            print(f"  *** Step {step}: {k} = {v} (NaN/Inf!) ***")
            has_nan = True

    if has_nan:
        print("  STOPPING: NaN/Inf detected.")
        failed = True
        break

    total_loss.backward()

    # Gradient check on aligner
    aligner_grad_norm = 0.0
    if hasattr(model, "aligner") and hasattr(model.aligner, "q_proj"):
        g = model.aligner.q_proj.weight.grad
        if g is not None:
            aligner_grad_norm = g.norm().item()

    optimizer.step()
    losses.append(total_loss.item())

    print(f"  Step {step}:")
    print(f"    total_loss = {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"    {k:30s} = {v:.4f}")
    print(f"    aligner.q_proj.weight grad norm = {aligner_grad_norm:.6f}")
    print()

# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
print("[4/4] Summary")
print("=" * 70)

if failed:
    print("FAIL: NaN/Inf detected during training.")
    sys.exit(1)

print(f"  Step losses: {[f'{l:.4f}' for l in losses]}")
if losses[-1] < losses[0]:
    print(f"  Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}  GOOD")
elif losses[-1] < losses[0] * 1.5:
    print(f"  Loss did not strictly decrease but stayed in range (within 50%): OK")
    print(f"    {losses[0]:.4f} -> {losses[-1]:.4f}")
else:
    print(f"  WARNING: Loss increased significantly: {losses[0]:.4f} -> {losses[-1]:.4f}")

# Final pass/fail
all_finite = all(torch.isfinite(torch.tensor(l)) for l in losses)
gradients_flow = aligner_grad_norm > 0
in_range = losses[-1] < losses[0] * 2.0  # generous bound

if all_finite and gradients_flow and in_range:
    print("\n  PASS: All 3 steps completed, no NaN/Inf, gradients flowing.")
else:
    reasons = []
    if not all_finite:
        reasons.append("non-finite loss")
    if not gradients_flow:
        reasons.append("zero aligner gradient")
    if not in_range:
        reasons.append("loss diverged")
    print(f"\n  FAIL: {', '.join(reasons)}")
    sys.exit(1)

print("\n" + "=" * 70)
print("DIAGNOSTIC 05 COMPLETE")
print("=" * 70)
