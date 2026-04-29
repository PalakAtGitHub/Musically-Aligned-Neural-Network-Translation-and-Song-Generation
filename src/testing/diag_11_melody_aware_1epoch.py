"""
Diagnostic 11: MelodyAwareDecoderLayer 1-epoch sanity training run.

Reports:
  - Train loss every 50 batches
  - Average train loss for epoch 1
  - Validation loss at end of epoch 1
  - melody_gate values for all 18 layers after training
"""

import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import types

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.models.mcnst_model import MCNST
from src.models.melody_aware_decoder import MelodyAwareDecoderLayer
from src.data.song_dataset import SongTranslationDataset


def patch_encode_and_fuse(model):
    """Patch _encode_and_fuse to stash melody_encoded on self so
    MelodyAwareDecoderLayer wrappers can access it during decoding."""
    _original = model._encode_and_fuse

    def _patched(self, input_ids, melody_features):
        out = _original(input_ids, melody_features)
        encoder_outputs, attention_mask, attn_weights, melody_encoded, melody_mask = out
        self._current_melody_features = melody_encoded
        self._current_melody_mask = melody_mask
        return out

    model._encode_and_fuse = types.MethodType(_patched, model)


print("=" * 70)
print("DIAGNOSTIC 11: MelodyAwareDecoderLayer 1-Epoch Sanity Run")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Build model and wrap decoder layers
# ------------------------------------------------------------------
print("\n[1/5] Building model and wrapping decoder layers...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=0)

# Freeze ALL original decoder layers
for layer in model.seq2seq.model.decoder.layers:
    for p in layer.parameters():
        p.requires_grad = False

# Freeze decoder layer_norm (but NOT lm_head or embed_tokens — those stay trainable)
model.seq2seq.model.decoder.layer_norm.weight.requires_grad = False
model.seq2seq.model.decoder.layer_norm.bias.requires_grad = False
# Explicitly ensure lm_head and embed_tokens are trainable
model.seq2seq.lm_head.weight.requires_grad = True
model.seq2seq.model.decoder.embed_tokens.weight.requires_grad = True

# Wrap all 18 layers
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

# Patch _encode_and_fuse to pass melody features to wrapped layers
patch_encode_and_fuse(model)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {trainable:,}")
print(f"  Wrapped {num_layers} decoder layers")
print(f"  Patched _encode_and_fuse to stash melody features")

# Print initial gate values
print("\n  Initial melody_gate values:")
for i in range(num_layers):
    layer = model.seq2seq.model.decoder.layers[i]
    print(f"    layer {i:2d}: {layer.melody_gate.item():.6f}")

# ------------------------------------------------------------------
# 2. Device
# ------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
else:
    device = torch.device('cpu')
model.to(device)
print(f"\n[2/5] Device: {device}")

# ------------------------------------------------------------------
# 3. Data
# ------------------------------------------------------------------
print("\n[3/5] Loading datasets...")
full_dataset = SongTranslationDataset("src/data/processed/fma_train_data.pt")
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=generator
)
train_loader = DataLoader(
    train_dataset, batch_size=4, shuffle=True,
    collate_fn=full_dataset.collate_fn,
)
val_loader = DataLoader(
    val_dataset, batch_size=4, shuffle=False,
    collate_fn=full_dataset.collate_fn,
)
print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ------------------------------------------------------------------
# 4. Optimizer
# ------------------------------------------------------------------
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)

# ------------------------------------------------------------------
# 5. Train 1 epoch
# ------------------------------------------------------------------
print("\n[4/5] Training for 1 epoch...")
print("=" * 70)

def unpack_batch(batch):
    src_ids = batch['src_ids'].to(device)
    tgt_ids = batch['tgt_ids'].to(device)
    melody = batch['melody_features'].to(device)
    num_notes = batch['num_notes'].to(device)
    tgt_syllables = batch['tgt_syllables'].to(device)
    stress_pattern = batch.get('stress_pattern')
    if stress_pattern is not None:
        stress_pattern = stress_pattern.to(device)
    beat_pattern = melody[:, :, 4]
    return dict(
        input_ids=src_ids, melody_features=melody, labels=tgt_ids,
        num_notes=num_notes, tgt_syllables=tgt_syllables,
        stress_pattern=stress_pattern, beat_pattern=beat_pattern,
    )

model.train()
total_train_loss = 0
batch_losses = []
train_start = time.time()

for step, batch in enumerate(train_loader):
    loss, loss_dict = model(**unpack_batch(batch))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()

    batch_loss = loss.item()
    total_train_loss += batch_loss
    batch_losses.append(batch_loss)

    if device.type == 'mps' and step % 50 == 0:
        torch.mps.empty_cache()

    if (step + 1) % 50 == 0 or step == 0 or step == len(train_loader) - 1:
        avg_so_far = total_train_loss / (step + 1)
        recent_avg = sum(batch_losses[-50:]) / len(batch_losses[-50:])
        elapsed = time.time() - train_start
        print(f"  batch {step+1:4d}/{len(train_loader)}  "
              f"loss={batch_loss:.4f}  recent_avg={recent_avg:.4f}  "
              f"running_avg={avg_so_far:.4f}  "
              f"trans={loss_dict['translation_loss']:.3f}  "
              f"syl={loss_dict['syllable_loss']:.3f}  "
              f"elapsed={elapsed:.0f}s")

avg_train_loss = total_train_loss / len(train_loader)
train_elapsed = time.time() - train_start
print(f"\n  Epoch 1 avg train loss: {avg_train_loss:.4f}")
print(f"  Training time: {train_elapsed:.0f}s")

# ------------------------------------------------------------------
# 6. Validation
# ------------------------------------------------------------------
print("\n[5/5] Validation...")
model.eval()
total_val_loss = 0
with torch.no_grad():
    for batch in val_loader:
        loss, _ = model(**unpack_batch(batch))
        total_val_loss += loss.item()

avg_val_loss = total_val_loss / max(len(val_loader), 1)
print(f"  Epoch 1 val loss: {avg_val_loss:.4f}")

# ------------------------------------------------------------------
# 7. Gate values after training
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("melody_gate values after 1 epoch:")
print("=" * 70)

gates_moved = 0
for i in range(num_layers):
    layer = model.seq2seq.model.decoder.layers[i]
    gate_val = layer.melody_gate.item()
    tanh_val = torch.tanh(layer.melody_gate).item()
    moved = abs(gate_val) > 1e-6
    if moved:
        gates_moved += 1
    print(f"  layer {i:2d}: gate={gate_val:+.6f}  tanh(gate)={tanh_val:+.6f}  "
          f"{'moved' if moved else 'static'}")

print(f"\n  Gates moved: {gates_moved}/{num_layers}")

# ------------------------------------------------------------------
# 7b. Gradient check on melody_attn projections (sample 3 layers)
# ------------------------------------------------------------------
# Run one more forward+backward to get current gradients
print("\n" + "=" * 70)
print("Gradient check on melody_attn projections (last batch):")
print("=" * 70)
model.train()
model.zero_grad()
# Use the last batch from training
last_batch_data = unpack_batch(batch)
check_loss, _ = model(**last_batch_data)
check_loss.backward()

sample_layers = [0, 9, 17]
proj_grads_ok = True
for i in sample_layers:
    layer = model.seq2seq.model.decoder.layers[i]
    g = layer.melody_attn.q_proj_weight.grad
    if g is None:
        print(f"  layer {i:2d} q_proj_weight.grad: *** NONE ***")
        proj_grads_ok = False
    else:
        norm = g.norm().item()
        ok = norm > 1e-12
        if not ok:
            proj_grads_ok = False
        print(f"  layer {i:2d} q_proj_weight.grad norm: {norm:.6e}  {'OK' if ok else '*** ZERO ***'}")

# ------------------------------------------------------------------
# 7c. Report trainable param count
# ------------------------------------------------------------------
print(f"\n  Total trainable params: {trainable:,}")

# ------------------------------------------------------------------
# 8. Summary / go/no-go
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("GO/NO-GO CHECKS")
print("=" * 70)

# Check 1: loss decreasing (compare first 50 batches avg to last 50)
first_50_avg = sum(batch_losses[:50]) / min(len(batch_losses), 50)
last_50_avg = sum(batch_losses[-50:]) / min(len(batch_losses), 50)
loss_decreasing = last_50_avg < first_50_avg

# Check 2: gates moved
gates_moved_check = gates_moved > 0

# Check 3: val_loss finite and not catastrophic (< 10.0, prior was ~6.50)
val_finite = torch.isfinite(torch.tensor(avg_val_loss)).item()
val_reasonable = avg_val_loss < 10.0

checks = {
    f'Loss decreasing (first50={first_50_avg:.4f} → last50={last_50_avg:.4f})': loss_decreasing,
    f'Gates moved ({gates_moved}/{num_layers} layers)': gates_moved_check,
    f'Val loss finite ({avg_val_loss:.4f})': val_finite,
    f'Val loss reasonable (<10.0, got {avg_val_loss:.4f})': val_reasonable,
    f'Projection weights have non-zero gradients': proj_grads_ok,
}

all_pass = True
for check, passed in checks.items():
    status = 'PASS' if passed else 'FAIL'
    if not passed:
        all_pass = False
    print(f"  [{status}] {check}")

print()
if all_pass:
    print("ALL CHECKS PASSED — safe to proceed to 10-epoch run.")
else:
    print("*** SOME CHECKS FAILED — do NOT proceed to 10-epoch run. ***")

print("\nDIAGNOSTIC 11 COMPLETE")
