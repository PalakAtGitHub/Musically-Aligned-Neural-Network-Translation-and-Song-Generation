"""
Experiment 19: RL v2 — KL-regularized MRT with reweighted reward.

Changes from exp_15:
  1. Reward reweighted to 0.3*syl + 0.7*BLEU (was 0.5/0.5)
  2. Hard floor: zero reward if syl_count < max(3, num_notes//2)
  3. KL penalty (beta=0.1) against frozen base model
  4. Lower lr=5e-6

Warm-starts from checkpoints/best_model.pt (fresh, NOT rl_1epoch_model.pt).
"""

import sys
import os
import math
import json
import time
import random
import copy
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torch.optim as optim
from sacrebleu.metrics import BLEU
from transformers.modeling_outputs import BaseModelOutput

from src.models.mcnst_model import MCNST
from src.utils.syllable_utils import count_hindi_syllables

# ==================================================================
# Reward function v2
# ==================================================================
_bleu_scorer = BLEU(effective_order=True)


def compute_reward(gen_text, ref_text, num_notes):
    """Returns (total, syl_score, bleu_score, syl_count)."""
    syl_count = count_hindi_syllables(gen_text)

    # Hard floor: refuse trivial-length outputs
    min_syl = max(3, num_notes // 2)
    if syl_count < min_syl:
        return 0.0, 0.0, 0.0, syl_count

    syl_diff = abs(syl_count - num_notes)
    if syl_diff <= 2:
        syl_score = 1.0
    else:
        syl_score = math.exp(-(syl_diff - 2) / 5.0)

    bleu = _bleu_scorer.sentence_score(gen_text, [ref_text]).score / 100.0
    total = min(1.0, 0.3 * syl_score + 0.7 * bleu)
    return total, syl_score, bleu, syl_count


# ==================================================================
# Log-prob computation helper
# ==================================================================
def compute_seq_log_probs(seq2seq_model, enc_hidden, attn_mask, candidate_ids, pad_id, K):
    """Compute per-candidate mean log-prob via teacher-forcing."""
    B_total = candidate_ids.size(0)
    B = B_total // K

    enc_expanded = enc_hidden.repeat_interleave(K, dim=0)[:B_total]
    attn_expanded = attn_mask.repeat_interleave(K, dim=0)[:B_total]
    enc_out = BaseModelOutput(last_hidden_state=enc_expanded)

    decoder_input_ids = candidate_ids[:, :-1].contiguous()
    target_ids = candidate_ids[:, 1:].contiguous()

    outputs = seq2seq_model(
        encoder_outputs=enc_out,
        attention_mask=attn_expanded,
        decoder_input_ids=decoder_input_ids,
        return_dict=True,
    )

    log_probs = F.log_softmax(outputs.logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = (target_ids != pad_id).float()
    seq_lp = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return seq_lp.view(B, K)


# ==================================================================
# Setup
# ==================================================================
print("=" * 70)
print("EXPERIMENT 19: RL v2 — KL-Regularized MRT")
print("=" * 70)

BATCH_SIZE = 2
K = 5
LR = 1e-6
BETA = 0.5
TEMPERATURE = 0.9
TOP_P = 0.95

print(f"\n  batch_size={BATCH_SIZE}, K={K}, lr={LR}, beta={BETA}")
print(f"  temperature={TEMPERATURE}, top_p={TOP_P}")
print(f"  reward: 0.3*syl + 0.7*BLEU, hard floor on trivial outputs")

# ------------------------------------------------------------------
# 1. Load policy model
# ------------------------------------------------------------------
print("\n[1/7] Loading policy model from checkpoints/best_model.pt...")
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
tokenizer = model.tokenizer
pad_id = model.pad_token_id
print(f"  Loaded epoch={ckpt.get('epoch', '?')}, device={device}")

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")

# ------------------------------------------------------------------
# 2. Load frozen reference model (for KL penalty)
# ------------------------------------------------------------------
print("\n[2/7] Loading frozen reference model...")
ref_model = MCNST(freeze_encoder=True, freeze_decoder_layers=0)
ref_model.load_state_dict(ckpt['model_state_dict'])
ref_model.to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False
print("  Reference model loaded and frozen")

# ------------------------------------------------------------------
# 3. Load training data
# ------------------------------------------------------------------
print("\n[3/7] Loading training data...")
train_data_raw = torch.load("src/data/processed/fma_train_data.pt", weights_only=False)
print(f"  Training examples: {len(train_data_raw)}")

random.seed(42)
train_indices = list(range(len(train_data_raw)))
random.shuffle(train_indices)

batches = []
for start in range(0, len(train_indices), BATCH_SIZE):
    batch_idx = train_indices[start:start + BATCH_SIZE]
    batches.append([train_data_raw[i] for i in batch_idx])
print(f"  Batches: {len(batches)}")

# ------------------------------------------------------------------
# 4. RL Training — 1 epoch
# ------------------------------------------------------------------
print(f"\n[4/7] RL Training (KL-regularized MRT) — 1 epoch")
print("=" * 70)

optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
batch_metrics = []
n_skipped = 0
epoch_start = time.time()

for batch_idx, batch_examples in enumerate(batches):
    B = len(batch_examples)

    # --- Prepare padded inputs ---
    max_src = max(ex['src_ids'].size(0) for ex in batch_examples)
    max_mel = max(ex['melody_features'].size(0) for ex in batch_examples)

    src_ids = torch.zeros(B, max_src, dtype=torch.long, device=device)
    melody = torch.zeros(B, max_mel, 5, dtype=torch.float32, device=device)
    num_notes_list = []
    hindi_refs = []

    for i, ex in enumerate(batch_examples):
        src_ids[i, :ex['src_ids'].size(0)] = ex['src_ids']
        melody[i, :ex['melody_features'].size(0)] = ex['melody_features']
        num_notes_list.append(int(ex['num_notes']))
        hindi_refs.append(ex['hindi_text'])

    # --- 1. Encode with policy model ---
    model.eval()
    with torch.no_grad():
        enc_out, attn_mask, _, _, _ = model._encode_and_fuse(src_ids, melody)
        enc_hidden = enc_out.last_hidden_state.clone()
        attn_mask_orig = attn_mask.clone()

    # --- 2. Encode with reference model (for KL) ---
    with torch.no_grad():
        ref_enc_out, ref_attn_mask, _, _, _ = ref_model._encode_and_fuse(src_ids, melody)
        ref_enc_hidden = ref_enc_out.last_hidden_state.clone()
        ref_attn_mask_orig = ref_attn_mask.clone()

    # --- 3. Generate K samples per example ---
    all_candidates = []
    all_rewards = []
    all_syl_scores = []
    all_bleu_scores = []
    all_syl_counts = []
    n_empty = 0
    n_floored = 0
    sample_texts = []

    for i in range(B):
        enc_i = BaseModelOutput(last_hidden_state=enc_hidden[i:i+1])
        attn_i = attn_mask_orig[i:i+1]

        with torch.no_grad():
            gen_ids = model.seq2seq.generate(
                encoder_outputs=enc_i,
                attention_mask=attn_i,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_return_sequences=K,
                max_length=50,
                repetition_penalty=2.5,
                no_repeat_ngram_size=2,
                use_cache=False,
            )

        rewards_i = []
        syl_scores_i = []
        bleu_scores_i = []
        syl_counts_i = []
        texts_i = []

        for k_idx in range(gen_ids.size(0)):
            tokenizer._switch_to_target_mode()
            text = tokenizer.decode(gen_ids[k_idx], skip_special_tokens=True)
            tokenizer._switch_to_input_mode()
            text = model.postprocess_tgt(text)[0]

            r, ss, bs, sc = compute_reward(text, hindi_refs[i], num_notes_list[i])
            rewards_i.append(r)
            syl_scores_i.append(ss)
            bleu_scores_i.append(bs)
            syl_counts_i.append(sc)
            texts_i.append(text)
            if not text.strip():
                n_empty += 1
            if r == 0.0 and sc < max(3, num_notes_list[i] // 2):
                n_floored += 1

        all_candidates.append(gen_ids)
        all_rewards.append(rewards_i)
        all_syl_scores.extend(syl_scores_i)
        all_bleu_scores.extend(bleu_scores_i)
        all_syl_counts.extend(syl_counts_i)
        if i == 0:
            sample_texts = texts_i

    # --- 4. Compute policy log_probs (with gradients) ---
    model.train()
    max_cand_len = max(c.size(1) for c in all_candidates)
    padded_cands = torch.full((B * K, max_cand_len), pad_id, dtype=torch.long, device=device)
    for i, cands in enumerate(all_candidates):
        actual_k = cands.size(0)
        padded_cands[i * K:i * K + actual_k, :cands.size(1)] = cands

    policy_log_probs = compute_seq_log_probs(
        model.seq2seq, enc_hidden, attn_mask_orig, padded_cands, pad_id, K
    )  # [B, K]

    # --- 5. Compute reference log_probs (no gradients) ---
    with torch.no_grad():
        ref_log_probs = compute_seq_log_probs(
            ref_model.seq2seq, ref_enc_hidden, ref_attn_mask_orig,
            padded_cands, pad_id, K
        )  # [B, K]

    # --- 6. KL penalty per sample (clamped to prevent spikes) ---
    kl_per_sample_raw = policy_log_probs - ref_log_probs.detach()  # [B, K]
    kl_per_sample = kl_per_sample_raw.clamp(-3.0, 3.0)  # prevent outlier spikes
    mean_kl = kl_per_sample_raw.mean().item()  # report unclamped for monitoring

    # --- 7. MRT + KL loss ---
    rewards_flat = []
    for rr in all_rewards:
        rewards_flat.extend(rr)
    rewards_t = torch.tensor(rewards_flat, device=device).view(B, K)
    R_mean = rewards_t.mean(dim=1, keepdim=True)
    centered = rewards_t - R_mean

    mrt_loss = -(centered.detach() * policy_log_probs).mean()
    kl_loss = BETA * kl_per_sample.mean()
    total_loss = mrt_loss + kl_loss

    if not torch.isfinite(total_loss):
        print(f"\n  FATAL: loss is NaN/Inf at batch {batch_idx}. STOPPING.")
        sys.exit(1)

    # Skip divergent batches instead of crashing — occasional KL spikes
    # are outlier batches, not systematic drift
    if abs(mean_kl) > 3.0:
        n_skipped += 1
        if batch_idx % 50 == 0 or n_skipped % 10 == 0:
            print(f"    [SKIP] batch {batch_idx}: |mean_kl|={abs(mean_kl):.2f} > 3.0 "
                  f"(total skipped: {n_skipped})")
        continue  # don't update params, move to next batch

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
    optimizer.step()

    # --- Tracking ---
    mean_reward = sum(rewards_flat) / len(rewards_flat)
    mean_syl_score = sum(all_syl_scores) / max(len(all_syl_scores), 1)
    mean_bleu_score = sum(all_bleu_scores) / max(len(all_bleu_scores), 1)
    mean_syl_count = sum(all_syl_counts) / max(len(all_syl_counts), 1)
    mean_target = sum(num_notes_list) / len(num_notes_list)

    batch_metrics.append({
        'batch': batch_idx,
        'total_loss': round(total_loss.item(), 4),
        'mrt_loss': round(mrt_loss.item(), 4),
        'kl_loss': round(kl_loss.item(), 4),
        'mean_kl': round(mean_kl, 4),
        'mean_reward': round(mean_reward, 4),
        'mean_syl_score': round(mean_syl_score, 4),
        'mean_bleu_score': round(mean_bleu_score, 4),
        'mean_syl_count': round(mean_syl_count, 2),
        'mean_target_notes': round(mean_target, 2),
        'n_empty': n_empty,
        'n_floored': n_floored,
    })

    # --- Log every 50 batches ---
    if batch_idx % 50 == 0 or batch_idx == len(batches) - 1:
        elapsed = time.time() - epoch_start
        print(f"\n  Batch {batch_idx}/{len(batches)-1} ({elapsed:.0f}s)")
        print(f"    total_loss={total_loss.item():.4f} "
              f"(mrt={mrt_loss.item():.4f}, kl={kl_loss.item():.4f})")
        print(f"    mean_reward={mean_reward:.4f}  mean_kl={mean_kl:.4f}")
        print(f"    mean_syl_score={mean_syl_score:.4f}  "
              f"mean_bleu={mean_bleu_score:.4f}")
        print(f"    mean_syl={mean_syl_count:.1f} vs target={mean_target:.1f}  "
              f"empty={n_empty}  floored={n_floored}")
        best_k = max(range(len(all_rewards[0])),
                     key=lambda k: all_rewards[0][k])
        print(f"    sample: \"{sample_texts[best_k][:80]}\"")
        print(f"      reward={all_rewards[0][best_k]:.4f}")

epoch_elapsed = time.time() - epoch_start
all_batch_rewards = [m['mean_reward'] for m in batch_metrics]
all_batch_kl = [m['mean_kl'] for m in batch_metrics]
print(f"\n  Epoch complete in {epoch_elapsed:.0f}s")
print(f"  Reward: start={all_batch_rewards[0]:.4f} → "
      f"end={all_batch_rewards[-1]:.4f}")
print(f"  KL: start={all_batch_kl[0]:.4f} → end={all_batch_kl[-1]:.4f}")
print(f"  Skipped batches (|kl|>3): {n_skipped}/{len(batches)}")

# ------------------------------------------------------------------
# 5. Save checkpoint
# ------------------------------------------------------------------
print(f"\n[5/7] Saving checkpoint...")
ckpt_path = Path("checkpoints/rl_v2_1epoch_model.pt")
torch.save({
    'epoch': 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'batch_metrics': batch_metrics,
    'training_time_seconds': round(epoch_elapsed, 1),
    'source_checkpoint': 'checkpoints/best_model.pt',
    'hyperparams': {
        'batch_size': BATCH_SIZE, 'K': K, 'lr': LR, 'beta': BETA,
        'temperature': TEMPERATURE, 'top_p': TOP_P,
    },
}, ckpt_path)
print(f"  Saved to {ckpt_path}")

# Free reference model to save memory for evaluation
del ref_model
if device.type == 'mps':
    torch.mps.empty_cache()
elif device.type == 'cuda':
    torch.cuda.empty_cache()

# ------------------------------------------------------------------
# 6. Evaluate on 198 strict held-out
# ------------------------------------------------------------------
print(f"\n[6/7] Loading strict held-out set...")
test_data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)
train_english = {ex['english_text'] for ex in train_data_raw}
strict_heldout = [ex for ex in test_data if ex['english_text'] not in train_english]
assert len(strict_heldout) == 198

with open("logs/exp_01_indictrans2_baseline_results.json") as f:
    exp1 = json.load(f)
assert [r['english_text'] for r in exp1['per_example_results']] == \
       [ex['english_text'] for ex in strict_heldout]
print(f"  Strict held-out: {len(strict_heldout)} (ordering verified)")

model.eval()


def evaluate_heldout(mode='beam', num_beams=5, temperature=0.9, top_p=0.95):
    bleu_scorer = BLEU(effective_order=True)
    results = []
    gen_start = time.time()
    n_empty = 0

    for i, ex in enumerate(strict_heldout):
        english_text = ex['english_text']
        hindi_ref = ex['hindi_text']
        num_notes = ex['num_notes']
        src_ids_i = ex['src_ids'].unsqueeze(0).to(device)
        mel_i = ex['melody_features'].unsqueeze(0).to(device)

        with torch.no_grad():
            if mode == 'beam':
                gen_ids = model.generate(
                    input_ids=src_ids_i, melody_features=mel_i,
                    max_length=50, num_beams=num_beams,
                )
            else:
                gen_ids = model.generate(
                    input_ids=src_ids_i, melody_features=mel_i,
                    max_length=50, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    num_return_sequences=1,
                )

        tokenizer._switch_to_target_mode()
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        tokenizer._switch_to_input_mode()
        gen_text = model.postprocess_tgt(gen_text)[0]

        if not gen_text.strip():
            n_empty += 1

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
            print(f"    [{mode}] Ex {i}: syl={gen_syl} (target={num_notes}, "
                  f"err={syl_error}) {'OK' if syl_match else 'MISS'}")

    gen_elapsed = time.time() - gen_start

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
        'n_empty': n_empty,
        'aggregate': {
            'corpus_bleu': round(corpus_bleu, 2),
            'mean_bertscore_f1': None,
            'syllable_accuracy_within_2': round(syl_acc, 4),
            'mean_syllable_error': round(mean_syl_err, 2),
            'mean_char_overlap': round(mean_char_overlap, 4),
            'n_empty_outputs': n_empty,
        },
    }


print(f"\n  Evaluating with beam search (num_beams=5)...")
beam_eval = evaluate_heldout(mode='beam', num_beams=5)

print(f"\n  Evaluating with sampling (temp={TEMPERATURE}, top_p={TOP_P})...")
sample_eval = evaluate_heldout(mode='sample', temperature=TEMPERATURE, top_p=TOP_P)

# ------------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------------
print(f"\n[7/7] Saving results...")

output = {
    'experiment': 'exp_19_rl_v2',
    'model': 'MCNST + 1-epoch KL-regularized MRT (checkpoints/rl_v2_1epoch_model.pt)',
    'description': 'RL v2: 0.3*syl+0.7*BLEU reward, hard floor, KL penalty (beta=0.5, clamped, lr=1e-6)',
    'timestamp': datetime.now().isoformat(),
    'source_checkpoint': 'checkpoints/best_model.pt',
    'rl_epochs': 1,
    'hyperparams': {
        'batch_size': BATCH_SIZE, 'K': K, 'lr': LR, 'beta': BETA,
        'temperature': TEMPERATURE, 'top_p': TOP_P,
        'reward_weights': '0.3*syl + 0.7*BLEU',
    },
    'training_time_seconds': round(epoch_elapsed, 1),
    'num_examples': len(strict_heldout),
    'beam_eval': {
        'generation_time_seconds': beam_eval['generation_time'],
        'aggregate_metrics': beam_eval['aggregate'],
        'per_example_results': beam_eval['results'],
    },
    'sample_eval': {
        'generation_time_seconds': sample_eval['generation_time'],
        'aggregate_metrics': sample_eval['aggregate'],
        'per_example_results': sample_eval['results'],
    },
    'batch_reward_trajectory': batch_metrics,
}

out_path = Path("logs/exp_19_rl_v2_results.json")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  Saved to {out_path}")

# Save BERTScore inputs for separate computation
bs_data = {
    'beam_hypotheses': beam_eval['hypotheses'],
    'beam_references': beam_eval['references'],
    'sample_hypotheses': sample_eval['hypotheses'],
    'sample_references': sample_eval['references'],
}
bs_path = Path("logs/exp_19_bertscore_inputs.json")
with open(bs_path, 'w', encoding='utf-8') as f:
    json.dump(bs_data, f, ensure_ascii=False, indent=2)
print(f"  Saved BERTScore inputs to {bs_path}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
print("=" * 70)
print("EXPERIMENT 19 RESULTS")
print("=" * 70)

print(f"\n  TRAINING (1 epoch KL-regularized MRT):")
print(f"    Time: {epoch_elapsed:.0f}s")
print(f"    Reward: {all_batch_rewards[0]:.4f} → {all_batch_rewards[-1]:.4f}")
print(f"    KL: {all_batch_kl[0]:.4f} → {all_batch_kl[-1]:.4f}")

print(f"\n  EVALUATION (198 strict held-out):")
ba = beam_eval['aggregate']
sa = sample_eval['aggregate']
print(f"  {'Metric':<28} {'Beam':>8} {'Sample':>8} {'Prior MCNST':>12}")
print(f"  {'-'*60}")
print(f"  {'corpus_bleu':<28} {ba['corpus_bleu']:>8.2f} "
      f"{sa['corpus_bleu']:>8.2f} {'36.14':>12}")
print(f"  {'syllable_accuracy_pm2':<28} {ba['syllable_accuracy_within_2']:>8.4f} "
      f"{sa['syllable_accuracy_within_2']:>8.4f} {'0.2273':>12}")
print(f"  {'mean_syllable_error':<28} {ba['mean_syllable_error']:>8.2f} "
      f"{sa['mean_syllable_error']:>8.2f} {'7.01':>12}")
print(f"  {'empty_outputs':<28} {ba['n_empty_outputs']:>8d} "
      f"{sa['n_empty_outputs']:>8d} {'0':>12}")

beam_syl_acc = ba['syllable_accuracy_within_2']
prior_syl_acc = 0.2273
delta_syl = beam_syl_acc - prior_syl_acc
beam_bleu = ba['corpus_bleu']
prior_bleu = 36.14
delta_bleu = beam_bleu - prior_bleu
beam_empty = ba['n_empty_outputs']

print(f"\n  DECISION CRITERIA (beam search vs prior MCNST default):")
print(f"    Syl accuracy delta: {delta_syl*100:+.1f}pp "
      f"({prior_syl_acc*100:.1f}% → {beam_syl_acc*100:.1f}%)")
print(f"    BLEU delta: {delta_bleu:+.2f} ({prior_bleu:.2f} → {beam_bleu:.2f})")
print(f"    Empty outputs: {beam_empty}")

if beam_empty > 5 or delta_bleu < -10:
    print(f"    → STOP: BLEU drop={delta_bleu:.1f} or empty={beam_empty}")
elif delta_bleu >= -5 and delta_syl >= 0.10:
    print(f"    → SUCCESS: syl acc +{delta_syl*100:.1f}pp, BLEU drop {delta_bleu:.1f}")
elif delta_bleu >= -10 and delta_syl >= 0.15:
    print(f"    → MARGINAL: syl acc +{delta_syl*100:.1f}pp, BLEU drop {delta_bleu:.1f}")
else:
    print(f"    → INSUFFICIENT improvement")

print()
print("EXPERIMENT 19 COMPLETE")
