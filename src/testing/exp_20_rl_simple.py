"""
Experiment 20: Simple RL — single-model MRT, no KL penalty.

Final RL attempt. Single model (no reference copy), dual-floor reward
(0.3*syl + 0.7*BLEU), AdamW lr=5e-6, 1 epoch.

Warm-starts from checkpoints/best_model.pt.
"""

import sys
import os
import math
import json
import time
import random
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
# Reward function — dual floor (short + long), 0.3*syl + 0.7*BLEU
# ==================================================================
_bleu_scorer = BLEU(effective_order=True)


def compute_reward(gen_text, ref_text, num_notes):
    """Returns (total_reward, syl_score, bleu_score, syl_count)."""
    syl_count = count_hindi_syllables(gen_text)

    # Lower floor: prevent empty/short outputs
    min_syl = max(3, num_notes // 2)
    # Upper floor: prevent runaway long outputs
    max_syl = max(num_notes * 3, 30)

    if syl_count < min_syl or syl_count > max_syl:
        return 0.0, 0.0, 0.0, syl_count

    syl_diff = abs(syl_count - num_notes)
    if syl_diff <= 2:
        syl_score = 1.0
    else:
        syl_score = math.exp(-(syl_diff - 2) / 5.0)

    bleu = _bleu_scorer.sentence_score(gen_text, [ref_text]).score / 100.0
    total = 0.3 * syl_score + 0.7 * bleu
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
print("EXPERIMENT 20: Simple RL — Single-Model MRT (Final Attempt)")
print("=" * 70)

BATCH_SIZE = 2
K = 5
LR = 5e-6
TEMPERATURE = 0.9
TOP_P = 0.95
GRAD_CLIP = 1.0

print(f"\n  batch_size={BATCH_SIZE}, K={K}, lr={LR}")
print(f"  temperature={TEMPERATURE}, top_p={TOP_P}, grad_clip={GRAD_CLIP}")
print(f"  reward: 0.3*syl + 0.7*BLEU, dual floor (short+long)")
print(f"  NO KL penalty, single model only")

# ------------------------------------------------------------------
# 1. Load model
# ------------------------------------------------------------------
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
tokenizer = model.tokenizer
pad_id = model.pad_token_id
print(f"  Loaded epoch={ckpt.get('epoch', '?')}, device={device}")

trainable_params = [p for p in model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable_params)
print(f"  Trainable params: {n_trainable:,}")

# ------------------------------------------------------------------
# 2. Load training data
# ------------------------------------------------------------------
print("\n[2/6] Loading training data...")
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
# 3. RL Training — 1 epoch
# ------------------------------------------------------------------
print(f"\n[3/6] RL Training (simple MRT) — 1 epoch")
print("=" * 70)

optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
batch_metrics = []
epoch_start = time.time()

# Early stopping: track consecutive checkpoints with >50% zero-reward
consecutive_degenerate = 0

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

    # --- 1. Encode ---
    model.eval()
    with torch.no_grad():
        enc_out, attn_mask, _, _, _ = model._encode_and_fuse(src_ids, melody)
        enc_hidden = enc_out.last_hidden_state.clone()
        attn_mask_orig = attn_mask.clone()

    # --- 2. Generate K samples per example ---
    all_candidates = []
    all_rewards = []
    all_syl_scores = []
    all_bleu_scores = []
    all_syl_counts = []
    n_zero_reward = 0
    n_empty = 0
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
        if device.type == 'mps':
            torch.mps.empty_cache()

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
            if r == 0.0:
                n_zero_reward += 1
            if not text.strip():
                n_empty += 1

        all_candidates.append(gen_ids)
        all_rewards.append(rewards_i)
        all_syl_scores.extend(syl_scores_i)
        all_bleu_scores.extend(bleu_scores_i)
        all_syl_counts.extend(syl_counts_i)
        if i == 0:
            sample_texts = texts_i

    # --- 3. Compute policy log_probs (with gradients) ---
    model.train()
    max_cand_len = max(c.size(1) for c in all_candidates)
    padded_cands = torch.full((B * K, max_cand_len), pad_id, dtype=torch.long, device=device)
    for i, cands in enumerate(all_candidates):
        actual_k = cands.size(0)
        padded_cands[i * K:i * K + actual_k, :cands.size(1)] = cands

    policy_log_probs = compute_seq_log_probs(
        model.seq2seq, enc_hidden, attn_mask_orig, padded_cands, pad_id, K
    )  # [B, K]

    # --- 4. MRT loss ---
    rewards_flat = []
    for rr in all_rewards:
        rewards_flat.extend(rr)
    rewards_t = torch.tensor(rewards_flat, device=device).view(B, K)
    R_mean = rewards_t.mean(dim=1, keepdim=True)
    centered = rewards_t - R_mean

    mrt_loss = -(centered.detach() * policy_log_probs).mean()

    if not torch.isfinite(mrt_loss):
        print(f"\n  FATAL: loss is NaN/Inf at batch {batch_idx}. STOPPING.")
        break

    optimizer.zero_grad()
    mrt_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
    optimizer.step()
    if device.type == 'mps':
        torch.mps.empty_cache()

    # --- Tracking ---
    total_samples = B * K
    frac_zero = n_zero_reward / total_samples
    mean_reward = sum(rewards_flat) / len(rewards_flat)
    mean_syl_score = sum(all_syl_scores) / max(len(all_syl_scores), 1)
    mean_bleu_score = sum(all_bleu_scores) / max(len(all_bleu_scores), 1)
    mean_syl_count = sum(all_syl_counts) / max(len(all_syl_counts), 1)
    min_syl_count = min(all_syl_counts) if all_syl_counts else 0
    max_syl_count = max(all_syl_counts) if all_syl_counts else 0
    mean_target = sum(num_notes_list) / len(num_notes_list)

    batch_metrics.append({
        'batch': batch_idx,
        'mrt_loss': round(mrt_loss.item(), 4),
        'mean_reward': round(mean_reward, 4),
        'mean_syl_score': round(mean_syl_score, 4),
        'mean_bleu_score': round(mean_bleu_score, 4),
        'mean_syl_count': round(mean_syl_count, 2),
        'min_syl_count': min_syl_count,
        'max_syl_count': max_syl_count,
        'mean_target_notes': round(mean_target, 2),
        'frac_zero_reward': round(frac_zero, 4),
        'n_empty': n_empty,
    })

    # --- Log every 50 batches ---
    if batch_idx % 50 == 0 or batch_idx == len(batches) - 1:
        elapsed = time.time() - epoch_start
        print(f"\n  Batch {batch_idx}/{len(batches)-1} ({elapsed:.0f}s)")
        print(f"    mrt_loss={mrt_loss.item():.4f}  mean_reward={mean_reward:.4f}")
        print(f"    mean_syl_score={mean_syl_score:.4f}  mean_bleu={mean_bleu_score:.4f}")
        print(f"    frac_zero_reward={frac_zero:.4f}  n_empty={n_empty}")
        print(f"    syl: mean={mean_syl_count:.1f} min={min_syl_count} max={max_syl_count} "
              f"(target={mean_target:.1f})")
        best_k = max(range(len(all_rewards[0])),
                     key=lambda k: all_rewards[0][k])
        print(f"    sample: \"{sample_texts[best_k][:80]}\"")
        print(f"      reward={all_rewards[0][best_k]:.4f}")

        # Early stopping check at each checkpoint
        if frac_zero > 0.5:
            consecutive_degenerate += 1
            print(f"    WARNING: frac_zero_reward={frac_zero:.2f} > 0.5 "
                  f"({consecutive_degenerate}/3 consecutive)")
            if consecutive_degenerate >= 3:
                print(f"\n  EARLY STOP: 3 consecutive checkpoints with >50% zero-reward.")
                print(f"  Policy is producing degenerate outputs. STOPPING training.")
                break
        else:
            consecutive_degenerate = 0

epoch_elapsed = time.time() - epoch_start
all_batch_rewards = [m['mean_reward'] for m in batch_metrics]
print(f"\n  Epoch complete in {epoch_elapsed:.0f}s")
print(f"  Reward: start={all_batch_rewards[0]:.4f} → "
      f"end={all_batch_rewards[-1]:.4f}")

early_stopped = consecutive_degenerate >= 3

# ------------------------------------------------------------------
# 4. Save checkpoint
# ------------------------------------------------------------------
print(f"\n[4/6] Saving checkpoint...")
ckpt_path = Path("checkpoints/rl_simple_1epoch_model.pt")
torch.save({
    'epoch': 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'batch_metrics': batch_metrics,
    'training_time_seconds': round(epoch_elapsed, 1),
    'source_checkpoint': 'checkpoints/best_model.pt',
    'early_stopped': early_stopped,
    'hyperparams': {
        'batch_size': BATCH_SIZE, 'K': K, 'lr': LR,
        'temperature': TEMPERATURE, 'top_p': TOP_P,
        'grad_clip': GRAD_CLIP,
        'reward': '0.3*syl + 0.7*BLEU, dual floor',
    },
}, ckpt_path)
print(f"  Saved to {ckpt_path}")

if early_stopped:
    print("\n  Training was early-stopped due to degenerate outputs.")
    print("  Evaluating the partially-trained model anyway for comparison.")

# ------------------------------------------------------------------
# 5. Evaluate on 198 strict held-out
# ------------------------------------------------------------------
print(f"\n[5/6] Loading strict held-out set...")
test_data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)
train_english = {ex['english_text'] for ex in train_data_raw}
strict_heldout = [ex for ex in test_data if ex['english_text'] not in train_english]
assert len(strict_heldout) == 198, f"Expected 198, got {len(strict_heldout)}"
print(f"  Strict held-out: {len(strict_heldout)}")

# Verify ordering matches exp_01
with open("logs/exp_01_indictrans2_baseline_results.json") as f:
    exp1 = json.load(f)
assert [r['english_text'] for r in exp1['per_example_results']] == \
       [ex['english_text'] for ex in strict_heldout], "Ordering mismatch"
print(f"  Ordering verified against exp_01")

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
# 6. Save results
# ------------------------------------------------------------------
print(f"\n[6/6] Saving results...")

output = {
    'experiment': 'exp_20_rl_simple',
    'model': 'MCNST + 1-epoch simple MRT (checkpoints/rl_simple_1epoch_model.pt)',
    'description': 'Simple RL: 0.3*syl+0.7*BLEU, dual floor (short+long), no KL, lr=5e-6',
    'timestamp': datetime.now().isoformat(),
    'source_checkpoint': 'checkpoints/best_model.pt',
    'rl_epochs': 1,
    'early_stopped': early_stopped,
    'hyperparams': {
        'batch_size': BATCH_SIZE, 'K': K, 'lr': LR,
        'temperature': TEMPERATURE, 'top_p': TOP_P,
        'grad_clip': GRAD_CLIP,
        'reward_weights': '0.3*syl + 0.7*BLEU',
        'reward_floors': 'min_syl=max(3,notes//2), max_syl=max(notes*3,30)',
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

out_path = Path("logs/exp_20_rl_simple_results.json")
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
bs_path = Path("logs/exp_20_bertscore_inputs.json")
with open(bs_path, 'w', encoding='utf-8') as f:
    json.dump(bs_data, f, ensure_ascii=False, indent=2)
print(f"  Saved BERTScore inputs to {bs_path}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
print("=" * 70)
print("EXPERIMENT 20 RESULTS")
print("=" * 70)

print(f"\n  TRAINING (1 epoch simple MRT):")
print(f"    Time: {epoch_elapsed:.0f}s")
print(f"    Reward: {all_batch_rewards[0]:.4f} → {all_batch_rewards[-1]:.4f}")
if early_stopped:
    print(f"    *** EARLY STOPPED at batch {batch_metrics[-1]['batch']} ***")

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
    print(f"    → STOP: reward hacking (BLEU drop={delta_bleu:.1f}, empty={beam_empty})")
elif delta_bleu >= -5 and delta_syl >= 0.05:
    print(f"    → SUCCESS: syl acc +{delta_syl*100:.1f}pp, BLEU drop {delta_bleu:.1f}")
elif delta_bleu >= -10 and delta_syl >= 0.05:
    print(f"    → MARGINAL: syl acc +{delta_syl*100:.1f}pp, BLEU drop {delta_bleu:.1f}")
else:
    print(f"    → INSUFFICIENT improvement")

print()
print("EXPERIMENT 20 COMPLETE")
