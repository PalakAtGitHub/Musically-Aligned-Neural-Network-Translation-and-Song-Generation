"""
Diagnostic test 06: load best checkpoint, run one forward pass, print
the COMPLETE loss_dict including cluster_loss and openness_loss.

No backward pass, no training — just a snapshot of where every loss
term is right now.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

print("=" * 70)
print("DIAGNOSTIC 06: Full loss_dict from current best checkpoint")
print("=" * 70)

# 1. Load model from checkpoint
print("\n[1/3] Loading MCNST + best checkpoint...")
from src.models.mcnst_model import MCNST

model = MCNST(freeze_encoder=True, freeze_decoder_layers=16)
device = torch.device("cpu")

ckpt = torch.load("checkpoints/best_model.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"  Loaded checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
model.to(device)
model.eval()

# 2. Load 4 real examples
print("\n[2/3] Loading 4 training examples...")
from src.data.song_dataset import SongTranslationDataset

dataset = SongTranslationDataset("src/data/processed/fma_train_data.pt")
subset = torch.utils.data.Subset(dataset, range(4))
loader = torch.utils.data.DataLoader(
    subset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn
)
batch = next(iter(loader))

# 3. Forward pass (no grad)
print("\n[3/3] Running forward pass...")
with torch.no_grad():
    forward_kwargs = dict(
        input_ids=batch["src_ids"].to(device),
        melody_features=batch["melody_features"].to(device),
        labels=batch["tgt_ids"].to(device),
        num_notes=batch["num_notes"].to(device),
    )
    if "stress_pattern" in batch:
        forward_kwargs["stress_pattern"] = batch["stress_pattern"].to(device)

    total_loss, loss_dict = model(**forward_kwargs)

print(f"\n{'=' * 70}")
print("COMPLETE loss_dict:")
print(f"{'=' * 70}")
for k, v in loss_dict.items():
    print(f"  {k:35s} = {v:.6f}")
print(f"{'=' * 70}")
print("DIAGNOSTIC 06 COMPLETE")
print(f"{'=' * 70}")
