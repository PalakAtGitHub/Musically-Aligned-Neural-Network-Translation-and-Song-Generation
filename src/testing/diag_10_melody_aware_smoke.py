"""
Diagnostic 10: MelodyAwareDecoderLayer forward+backward smoke test.

Confirms on a synthetic 2-example batch:
  - Loss is finite
  - All 7 loss terms are non-zero
  - Gradient flows to all 18 melody_gate scalars
  - Gradient flows to melody_attn projections in all 18 layers
  - Gradient does NOT flow to original IndicTrans2 decoder layer params,
    lm_head, or embed_tokens
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.chdir(PROJECT_ROOT)

import types

import torch
from src.models.mcnst_model import MCNST
from src.models.melody_aware_decoder import MelodyAwareDecoderLayer


def patch_encode_and_fuse(model):
    """Patch _encode_and_fuse to stash melody_encoded on self so
    MelodyAwareDecoderLayer wrappers can access it during decoding."""
    _original = model._encode_and_fuse

    def _patched(self, input_ids, melody_features):
        out = _original(input_ids, melody_features)
        encoder_outputs, attention_mask, attn_weights, melody_encoded, melody_mask = out
        # Stash for MelodyAwareDecoderLayer
        self._current_melody_features = melody_encoded
        self._current_melody_mask = melody_mask
        return out

    model._encode_and_fuse = types.MethodType(_patched, model)


print("=" * 70)
print("DIAGNOSTIC 10: MelodyAwareDecoderLayer Smoke Test")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Build model and wrap decoder layers
# ------------------------------------------------------------------
print("\n[1/4] Building model and wrapping decoder layers...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=0)

# Freeze ALL original decoder layers
for layer in model.seq2seq.model.decoder.layers:
    for p in layer.parameters():
        p.requires_grad = False

# Freeze decoder embeddings, layer_norm, LM head
model.seq2seq.model.decoder.embed_tokens.weight.requires_grad = False
model.seq2seq.model.decoder.layer_norm.weight.requires_grad = False
model.seq2seq.model.decoder.layer_norm.bias.requires_grad = False
model.seq2seq.lm_head.weight.requires_grad = False

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

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params:     {total:,}")
print(f"  Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
print(f"  Wrapped {num_layers} decoder layers with MelodyAwareDecoderLayer")
print(f"  Patched _encode_and_fuse to stash melody features")

# ------------------------------------------------------------------
# 2. Synthetic batch forward
# ------------------------------------------------------------------
print("\n[2/4] Forward pass on synthetic 2-example batch...")

B = 2
src_ids = torch.randint(0, 1000, (B, 15))
melody = torch.randn(B, 20, 5)
tgt_ids = torch.randint(0, 1000, (B, 10))
num_notes = torch.tensor([20, 18], dtype=torch.float)
tgt_syl = torch.tensor([8, 7], dtype=torch.float)
stress = torch.randn(B, 20)
beat = melody[:, :, 4]

loss, loss_dict = model(
    input_ids=src_ids, melody_features=melody, labels=tgt_ids,
    num_notes=num_notes, tgt_syllables=tgt_syl,
    stress_pattern=stress, beat_pattern=beat,
)

print(f"  Total loss: {loss.item():.4f} (finite: {torch.isfinite(loss).item()})")
print()
print("  Loss components:")
all_nonzero = True
for key in ['translation_loss', 'syllable_loss', 'naturalness_loss',
            'rhythm_loss', 'rhyme_loss', 'cluster_loss', 'openness_loss']:
    val = loss_dict.get(key, 0)
    if isinstance(val, torch.Tensor):
        val = val.item()
    nonzero = abs(val) > 1e-12
    if not nonzero:
        all_nonzero = False
    print(f"    {key:<25} {val:>10.6f}  {'OK' if nonzero else '*** ZERO ***'}")

# ------------------------------------------------------------------
# 3. Backward pass
# ------------------------------------------------------------------
print(f"\n[3/4] Backward pass...")
model.zero_grad()
loss.backward()

# Check melody_gate gradients
print("\n  melody_gate gradients (all 18 layers):")
gate_grads_ok = True
for i in range(num_layers):
    layer = model.seq2seq.model.decoder.layers[i]
    g = layer.melody_gate.grad
    if g is None:
        print(f"    layer {i:2d}: *** NO GRADIENT ***")
        gate_grads_ok = False
    else:
        val = g.item()
        ok = abs(val) > 1e-12
        if not ok:
            gate_grads_ok = False
        print(f"    layer {i:2d}: grad={val:+.6e}  {'OK' if ok else '*** ZERO ***'}")

# Check melody_attn projection gradients
print("\n  melody_attn projection gradients (spot-check q_proj_weight per layer):")
attn_grads_ok = True
for i in range(num_layers):
    layer = model.seq2seq.model.decoder.layers[i]
    g = layer.melody_attn.q_proj_weight.grad
    if g is None:
        print(f"    layer {i:2d}: *** NO GRADIENT ***")
        attn_grads_ok = False
    else:
        norm = g.norm().item()
        ok = norm > 1e-12
        if not ok:
            attn_grads_ok = False
        print(f"    layer {i:2d}: grad_norm={norm:.6e}  {'OK' if ok else '*** ZERO ***'}")

# Check NO gradient flows to original layer params
print("\n  Checking original IndicTrans2 decoder params have NO gradient...")
original_grad_leak = []
for n, p in model.named_parameters():
    if 'original_layer' in n and p.grad is not None and p.grad.abs().sum().item() > 0:
        original_grad_leak.append(n)

if original_grad_leak:
    print(f"  *** GRADIENT LEAK in {len(original_grad_leak)} original_layer params! ***")
    for name in original_grad_leak[:5]:
        print(f"    {name}")
else:
    print("  No gradient leak to original_layer params.")

# Check embed_tokens and lm_head
embed_grad = model.seq2seq.model.decoder.embed_tokens.weight.grad
lm_grad = model.seq2seq.lm_head.weight.grad
embed_ok = embed_grad is None or embed_grad.abs().sum().item() == 0
lm_ok = lm_grad is None or lm_grad.abs().sum().item() == 0
print(f"  embed_tokens gradient: {'None/zero (OK)' if embed_ok else '*** HAS GRADIENT ***'}")
print(f"  lm_head gradient:     {'None/zero (OK)' if lm_ok else '*** HAS GRADIENT ***'}")

# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
checks = {
    'Loss is finite': torch.isfinite(loss).item(),
    'All 7 loss terms non-zero': all_nonzero,
    'All 18 melody_gate grads non-zero': gate_grads_ok,
    'All 18 melody_attn q_proj grads non-zero': attn_grads_ok,
    'No gradient leak to original_layer': len(original_grad_leak) == 0,
    'No gradient to embed_tokens': embed_ok,
    'No gradient to lm_head': lm_ok,
}

all_pass = True
for check, passed in checks.items():
    status = 'PASS' if passed else 'FAIL'
    if not passed:
        all_pass = False
    print(f"  [{status}] {check}")

print()
if all_pass:
    print("ALL CHECKS PASSED.")
else:
    print("*** SOME CHECKS FAILED ***")

print("\nDIAGNOSTIC 10 COMPLETE")
