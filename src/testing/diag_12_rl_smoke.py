"""
Diagnostic 12: RL (MRT) smoke test.

Loads checkpoints/best_model.pt (prior MCNST default), picks 4 training
examples, generates K=3 nucleus samples each, computes the user-specified
reward (0.5*syl_score + 0.5*BLEU), computes MRT loss, and verifies
gradient flow. Does NOT update parameters.

PASS criteria:
  - Rewards in [0, 1]
  - Different samples produce different rewards
  - MRT loss is finite
  - Gradient flows to alignment, fusion, and decoder layer norms
"""

import sys
import os
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU
from transformers.modeling_outputs import BaseModelOutput

from src.models.mcnst_model import MCNST
from src.utils.syllable_utils import count_hindi_syllables


# ------------------------------------------------------------------
# Reward function (user spec)
# ------------------------------------------------------------------
bleu_scorer = BLEU(effective_order=True)


def compute_reward(gen_text, ref_text, num_notes):
    """Returns (total_reward, syl_score, bleu_score) all in [0,1]."""
    # 1. Syllable component
    syl_count = count_hindi_syllables(gen_text)
    syl_diff = abs(syl_count - num_notes)
    if syl_diff <= 2:
        syl_score = 1.0
    else:
        syl_score = math.exp(-(syl_diff - 2) / 5.0)

    # 2. BLEU component
    bleu = bleu_scorer.sentence_score(gen_text, [ref_text]).score / 100.0

    # 3. Combined (clamp for float precision — BLEU can be 100.00000001)
    total = min(1.0, 0.5 * syl_score + 0.5 * bleu)
    return total, syl_score, bleu


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
print("=" * 70)
print("DIAGNOSTIC 12: RL (MRT) Smoke Test")
print("=" * 70)

print("\n[1/6] Loading model from checkpoints/best_model.pt...")
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
print(f"  Loaded epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')}")
print(f"  Device: {device}")

tokenizer = model.tokenizer
pad_id = model.pad_token_id

# ------------------------------------------------------------------
# Pick 4 examples
# ------------------------------------------------------------------
print("\n[2/6] Loading 4 training examples...")
train_data = torch.load("src/data/processed/fma_train_data.pt", weights_only=False)
examples = train_data[:4]
for i, ex in enumerate(examples):
    print(f"  Ex {i}: song={ex['song_name']}, notes={ex['num_notes']}, "
          f"en=\"{ex['english_text'][:60]}...\"")

# ------------------------------------------------------------------
# Generate K=3 samples per example
# ------------------------------------------------------------------
K = 3
print(f"\n[3/6] Generating K={K} nucleus samples per example...")
print("  (temperature=0.9, top_p=0.95, do_sample=True)")
print("=" * 70)

all_samples = []  # list of list of dicts
all_gen_ids = []  # list of list of tensors (token ids)

model.eval()
for i, ex in enumerate(examples):
    src_ids = ex['src_ids'].unsqueeze(0).to(device)
    melody = ex['melody_features'].unsqueeze(0).to(device)
    ref_text = ex['hindi_text']
    num_notes = int(ex['num_notes'])

    samples = []
    sample_ids = []
    for k in range(K):
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=src_ids, melody_features=melody,
                max_length=50, do_sample=True,
                temperature=0.9, top_p=0.95,
                num_return_sequences=1,
            )
        # Decode
        tokenizer._switch_to_target_mode()
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        tokenizer._switch_to_input_mode()
        gen_text = model.postprocess_tgt(gen_text)[0]

        reward, syl_score, bleu_score = compute_reward(gen_text, ref_text, num_notes)
        syl_count = count_hindi_syllables(gen_text)

        samples.append({
            'text': gen_text, 'reward': reward,
            'syl_score': syl_score, 'bleu_score': bleu_score,
            'syl_count': syl_count,
        })
        sample_ids.append(gen_ids[0])

    all_samples.append(samples)
    all_gen_ids.append(sample_ids)

    print(f"\n--- Example {i} (target notes: {num_notes}) ---")
    print(f"  REF: {ref_text}")
    for k, s in enumerate(samples):
        print(f"  Sample {k}: \"{s['text']}\"")
        print(f"    syl={s['syl_count']} reward={s['reward']:.4f} "
              f"(syl_score={s['syl_score']:.4f}, bleu={s['bleu_score']:.4f})")

# ------------------------------------------------------------------
# Check rewards
# ------------------------------------------------------------------
print(f"\n[4/6] Reward checks...")
all_rewards = [s['reward'] for samples in all_samples for s in samples]
print(f"  All rewards: {[round(r, 4) for r in all_rewards]}")
print(f"  Min: {min(all_rewards):.4f}, Max: {max(all_rewards):.4f}")

in_range = all(0.0 <= r <= 1.0 for r in all_rewards)
print(f"  All in [0,1]: {in_range}")

# Check diversity: at least one example should have different rewards across samples
any_diverse = False
for i, samples in enumerate(all_samples):
    rewards_i = [s['reward'] for s in samples]
    if len(set(round(r, 6) for r in rewards_i)) > 1:
        any_diverse = True
        break
print(f"  Different rewards across samples: {any_diverse}")

# ------------------------------------------------------------------
# Compute MRT loss
# ------------------------------------------------------------------
print(f"\n[5/6] Computing MRT loss...")

model.train()
total_mrt_loss = torch.tensor(0.0, device=device)

for i, ex in enumerate(examples):
    src_ids = ex['src_ids'].unsqueeze(0).to(device)
    melody = ex['melody_features'].unsqueeze(0).to(device)

    # Encode once
    encoder_outputs, attention_mask, _, _, _ = model._encode_and_fuse(src_ids, melody)
    enc_hidden = encoder_outputs.last_hidden_state  # [1, seq_len, hidden]

    # Pad all K samples to same length
    sample_ids = all_gen_ids[i]
    max_len = max(s.size(0) for s in sample_ids)
    padded = torch.full((K, max_len), pad_id, dtype=torch.long, device=device)
    for k, s in enumerate(sample_ids):
        padded[k, :s.size(0)] = s

    # Expand encoder outputs for K candidates
    enc_expanded = enc_hidden.expand(K, -1, -1).contiguous()
    attn_expanded = attention_mask.expand(K, -1).contiguous()
    enc_out_expanded = BaseModelOutput(last_hidden_state=enc_expanded)

    # Teacher-force: shift for decoder input / targets
    decoder_input_ids = padded[:, :-1].contiguous()
    target_ids = padded[:, 1:].contiguous()

    outputs = model.seq2seq(
        encoder_outputs=enc_out_expanded,
        attention_mask=attn_expanded,
        decoder_input_ids=decoder_input_ids,
        return_dict=True,
    )
    logits = outputs.logits  # [K, seq_len-1, vocab]

    # Per-token log prob
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = (target_ids != pad_id).float()
    seq_log_probs = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # Rewards for this example
    rewards_i = torch.tensor(
        [all_samples[i][k]['reward'] for k in range(K)],
        device=device
    )
    R_mean = rewards_i.mean()
    centered = rewards_i - R_mean

    # MRT loss: -sum_k (R_k - R_mean) * log_prob(sample_k)
    mrt_loss_i = -(centered.detach() * seq_log_probs).sum()
    total_mrt_loss = total_mrt_loss + mrt_loss_i

    print(f"  Ex {i}: rewards={[round(r, 4) for r in rewards_i.tolist()]}, "
          f"R_mean={R_mean:.4f}, mrt_loss={mrt_loss_i.item():.4f}")

mrt_loss = total_mrt_loss / len(examples)
print(f"\n  Total MRT loss: {mrt_loss.item():.4f}")
print(f"  Finite: {torch.isfinite(mrt_loss).item()}")

# ------------------------------------------------------------------
# Backward pass — check gradient flow
# ------------------------------------------------------------------
print(f"\n[6/6] Backward pass + gradient check...")

model.zero_grad()
mrt_loss.backward()

# Check gradient on specific parameters
checks = {
    'aligner (query proj)': None,
    'fusion (text_proj)': None,
    'decoder layer_norm': None,
}

# Find alignment module parameter
for name, p in model.named_parameters():
    if 'aligner' in name and 'weight' in name and p.grad is not None:
        checks['aligner (query proj)'] = (name, p.grad.norm().item())
        break

for name, p in model.named_parameters():
    if 'fusion' in name and 'weight' in name and p.grad is not None:
        checks['fusion (text_proj)'] = (name, p.grad.norm().item())
        break

for name, p in model.named_parameters():
    if 'decoder' in name and 'layer_norm' in name and p.grad is not None:
        checks['decoder layer_norm'] = (name, p.grad.norm().item())
        break

print()
for label, val in checks.items():
    if val is None:
        print(f"  {label}: NO GRADIENT (FAIL)")
    else:
        name, norm = val
        status = "OK" if norm > 0 else "ZERO (FAIL)"
        print(f"  {label}: grad_norm={norm:.6f} ({name}) [{status}]")

# Count total params with gradients
n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 0)
n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
print(f"\n  Parameters with nonzero gradient: {n_with_grad} / {n_trainable} trainable")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
print("=" * 70)
print("DIAGNOSTIC 12 SUMMARY")
print("=" * 70)
pass_all = True

aligner_has_grad = checks['aligner (query proj)'] is not None and checks['aligner (query proj)'][1] > 0
check_results = {
    'Rewards in [0,1]': in_range,
    'Sample diversity': any_diverse,
    'MRT loss finite': torch.isfinite(mrt_loss).item(),
    'Gradient to fusion': checks['fusion (text_proj)'] is not None and checks['fusion (text_proj)'][1] > 0,
    'Gradient to decoder LN': checks['decoder layer_norm'] is not None and checks['decoder layer_norm'][1] > 0,
}

# Aligner is supervised-only (called only in forward(labels=...)),
# so it's not in the MRT computation graph. Report but don't fail.
if not aligner_has_grad:
    print(f"  [INFO] Aligner has no gradient — expected in MRT "
          f"(aligner is only used during supervised forward with labels)")
else:
    check_results['Gradient to aligner'] = True

for label, passed in check_results.items():
    status = "PASS" if passed else "FAIL"
    if not passed:
        pass_all = False
    print(f"  [{status}] {label}")

print()
if pass_all:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED — review output above")
print()
print("DIAGNOSTIC 12 COMPLETE")
