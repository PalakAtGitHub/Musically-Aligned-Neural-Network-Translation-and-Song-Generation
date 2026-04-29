"""
Experiment 07: Ablation B — Decoder fine-tuning only baseline.

No fusion, no alignment, no melody encoder usage, no syllable/rhythm/rhyme/
cluster/openness losses. Only translation cross-entropy + naturalness.
Top 2 decoder layers fine-tuned (same freeze_decoder_layers=16 as full MCNST).

This isolates the contribution of just fine-tuning the decoder on song data.
"""

import sys
import os
import json
import time
import types
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score_fn

from src.data.song_dataset import SongTranslationDataset
from src.models.mcnst_model import MCNST
from src.utils.syllable_utils import count_hindi_syllables

print("=" * 70)
print("EXPERIMENT 07: Decoder-Only Baseline")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Build model and apply patches
# ------------------------------------------------------------------
print("\n[1/7] Initializing MCNST and patching to decoder-only mode...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=16)

# --- Patch _encode_and_fuse to skip fusion (like exp_05) ---
def _encode_and_fuse_no_fusion(self, input_ids, melody_features):
    attention_mask = (input_ids != self.pad_token_id).long()
    encoder_outputs = self._encoder(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    )
    min_notes = 3
    if melody_features.size(1) < min_notes:
        pad = min_notes - melody_features.size(1)
        melody_features = F.pad(melody_features, (0, 0, 0, pad))
    melody_encoded = self.melody_encoder(melody_features)
    melody_mask = (melody_features.sum(dim=-1) == 0)
    # SKIP FUSION — pass encoder output unchanged
    batch_size = input_ids.size(0)
    seq_len = encoder_outputs.last_hidden_state.size(1)
    num_notes = melody_encoded.size(1)
    attn_weights = torch.zeros(batch_size, seq_len, num_notes, device=input_ids.device)
    return encoder_outputs, attention_mask, attn_weights, melody_encoded, melody_mask

model._encode_and_fuse = types.MethodType(_encode_and_fuse_no_fusion, model)

# --- Patch forward to skip alignment AND use minimal loss ---
def _forward_decoder_only(self, input_ids, melody_features, labels=None,
                          tgt_syllables=None, num_notes=None,
                          stress_pattern=None, beat_pattern=None, **kwargs):
    encoder_outputs, attention_mask, _, _, _ = self._encode_and_fuse(input_ids, melody_features)
    if labels is not None:
        outputs = self.seq2seq(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True
        )
        translation_loss = outputs.loss
        logits = outputs.logits

        # Minimal loss: translation + naturalness only
        loss_dict = {}

        # Translation (weighted by log_var)
        precision_t = torch.exp(-self.loss_fn.log_var_translation)
        total = precision_t * translation_loss + self.loss_fn.log_var_translation
        loss_dict['translation_loss'] = float(translation_loss.detach())

        # Naturalness
        nat = self.loss_fn.naturalness_loss(logits, labels)
        precision_n = torch.exp(-self.loss_fn.log_var_naturalness)
        total = total + precision_n * nat + self.loss_fn.log_var_naturalness
        loss_dict['naturalness_loss'] = float(nat.detach())

        # Zero out unused loss fields for consistency
        loss_dict['syllable_loss'] = 0.0
        loss_dict['rhythm_loss'] = 0.0
        loss_dict['rhyme_loss'] = 0.0
        loss_dict['cluster_loss'] = 0.0
        loss_dict['openness_loss'] = 0.0
        loss_dict['total_loss'] = float(total.detach())

        return total, loss_dict
    else:
        return self.seq2seq(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            return_dict=True
        )

model.forward = types.MethodType(_forward_decoder_only, model)

# Freeze everything except: top 2 decoder layers, loss log_var params,
# decoder embeddings, LM head (same as full MCNST with freeze_decoder_layers=16)
# The MCNST constructor already froze encoder + bottom 16 decoder layers.
# Also freeze: melody_encoder, fusion, aligner (none of these are used)
for p in model.melody_encoder.parameters():
    p.requires_grad = False
for p in model.fusion.parameters():
    p.requires_grad = False
for p in model.aligner.parameters():
    p.requires_grad = False

# Remove unused loss log_var params
for name in ['log_var_syllable', 'log_var_rhythm', 'log_var_rhyme',
             'log_var_cluster', 'log_var_openness']:
    if hasattr(model.loss_fn, name):
        delattr(model.loss_fn, name)
        setattr(model.loss_fn, name, torch.tensor(0.0))

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} / {total_params:,}")
print("  Losses: translation + naturalness only")
print("  Fusion: OFF, Alignment: OFF, Melody encoder: frozen")

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
print(f"  Device: {device}")

# ------------------------------------------------------------------
# 3. Data
# ------------------------------------------------------------------
print("\n[2/7] Loading datasets...")
full_dataset = SongTranslationDataset("src/data/processed/fma_train_data.pt")
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                          collate_fn=full_dataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                        collate_fn=full_dataset.collate_fn)
print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ------------------------------------------------------------------
# 4. Optimizer
# ------------------------------------------------------------------
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)

# ------------------------------------------------------------------
# 5. Train
# ------------------------------------------------------------------
NUM_EPOCHS = 10
CHECKPOINT_PATH = Path("checkpoints/decoder_only_model.pt")

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

print(f"\n[3/7] Training for {NUM_EPOCHS} epochs (decoder-only)...")
print("=" * 70)

best_val_loss = float('inf')
history = []
train_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        loss, loss_dict = model(**unpack_batch(batch))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        total_train_loss += loss.item()
        if device.type == 'mps' and step % 50 == 0:
            torch.mps.empty_cache()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'trans': f"{loss_dict['translation_loss']:.3f}",
            'nat': f"{loss_dict['naturalness_loss']:.3f}",
        })

    avg_train = total_train_loss / len(train_loader)
    model.eval()
    total_val = 0
    with torch.no_grad():
        for batch in val_loader:
            loss, _ = model(**unpack_batch(batch))
            total_val += loss.item()
    avg_val = total_val / max(len(val_loader), 1)

    print(f"  Epoch {epoch}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")
    history.append({'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val})

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val,
            'ablation': 'decoder_only',
        }, CHECKPOINT_PATH)
        print(f"  Saved best (val_loss={avg_val:.4f})")

train_elapsed = time.time() - train_start
print(f"\nTraining complete in {train_elapsed:.0f}s. Best val_loss={best_val_loss:.4f}")

with open("checkpoints/decoder_only_training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

# ------------------------------------------------------------------
# 6. Evaluate on strict held-out
# ------------------------------------------------------------------
print(f"\n[4/7] Loading best decoder-only checkpoint...")
ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()
tokenizer = model.tokenizer
print(f"  Loaded epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

print(f"\n[5/7] Loading strict held-out set...")
test_data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)
train_data_raw = torch.load("src/data/processed/fma_train_data.pt", weights_only=False)
train_english = {ex['english_text'] for ex in train_data_raw}
strict_heldout = [ex for ex in test_data if ex['english_text'] not in train_english]
assert len(strict_heldout) == 198

with open("logs/exp_01_indictrans2_baseline_results.json") as f:
    exp1 = json.load(f)
assert [r['english_text'] for r in exp1['per_example_results']] == [ex['english_text'] for ex in strict_heldout]
print(f"  Strict held-out: {len(strict_heldout)} (ordering verified)")

print(f"\n[6/7] Generating on {len(strict_heldout)} examples...")
print("=" * 70)

bleu_scorer = BLEU(effective_order=True)
results = []
gen_start = time.time()

for i, ex in enumerate(strict_heldout):
    english_text = ex['english_text']
    hindi_ref = ex['hindi_text']
    num_notes = ex['num_notes']
    src_ids = ex['src_ids'].unsqueeze(0).to(device)
    melody = ex['melody_features'].unsqueeze(0).to(device)

    with torch.no_grad():
        gen_ids = model.generate(input_ids=src_ids, melody_features=melody,
                                 max_length=50, num_beams=5)
    tokenizer._switch_to_target_mode()
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    tokenizer._switch_to_input_mode()
    gen_text = model.postprocess_tgt(gen_text)[0]

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
        print(f"\n--- Example {i} (song: {ex['song_name']}) ---")
        print(f"  EN:  {english_text}")
        print(f"  REF: {hindi_ref}")
        print(f"  GEN: {gen_text}")
        print(f"  syl: {gen_syl} (target: {num_notes}, err: {syl_error}) "
              f"{'within2' if syl_match else 'MISS'}")

gen_elapsed = time.time() - gen_start

# ------------------------------------------------------------------
# 7. Corpus metrics and save
# ------------------------------------------------------------------
print(f"\n[7/7] Computing corpus-level metrics...")
hypotheses = [r['generated_hi'] for r in results]
references = [r['ground_truth_hi'] for r in results]
corpus_bleu = BLEU().corpus_score(hypotheses, [references]).score
print("  Computing BERTScore...")
P, R, F1 = bert_score_fn(hypotheses, references, lang='hi', verbose=False)
mean_bertscore_f1 = F1.mean().item()
syl_acc = sum(1 for r in results if r['syl_match_within_2']) / len(results)
mean_syl_err = sum(r['syl_error'] for r in results) / len(results)
mean_char_overlap = sum(r['char_overlap_ratio'] for r in results) / len(results)

output = {
    'experiment': 'exp_07_decoder_only',
    'model': 'MCNST decoder-only (checkpoints/decoder_only_model.pt)',
    'description': 'Decoder fine-tuning only — no fusion, no alignment, no melody losses',
    'timestamp': datetime.now().isoformat(),
    'checkpoint_epoch': ckpt['epoch'],
    'checkpoint_val_loss': ckpt['val_loss'],
    'training_time_seconds': round(train_elapsed, 1),
    'num_examples': len(results),
    'generation_time_seconds': round(gen_elapsed, 1),
    'training_history': history,
    'aggregate_metrics': {
        'corpus_bleu': round(corpus_bleu, 2),
        'mean_bertscore_f1': round(mean_bertscore_f1, 4),
        'syllable_accuracy_within_2': round(syl_acc, 4),
        'mean_syllable_error': round(mean_syl_err, 2),
        'mean_char_overlap': round(mean_char_overlap, 4),
    },
    'per_example_results': results,
    'per_example_bertscore_f1': [round(f.item(), 4) for f in F1],
}

out_path = Path("logs") / "exp_07_decoder_only_results.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  Saved to {out_path}")

print()
print("AGGREGATE METRICS")
print(f"corpus_bleu: {round(corpus_bleu, 2)}")
print(f"bertscore_f1_mean: {round(mean_bertscore_f1, 4)}")
print(f"syllable_accuracy_pm2: {round(syl_acc, 4)}")
print(f"mean_syllable_error: {round(mean_syl_err, 2)}")
print(f"mean_char_overlap: {round(mean_char_overlap, 4)}")
print(f"total_examples: {len(results)}")
print(f"best_val_loss: {best_val_loss:.4f}")
print(f"training_time_seconds: {train_elapsed:.0f}")

print()
print("EXAMPLE OUTPUTS (first 3)")
for ex in results[:3]:
    print()
    print(f"english_text: {ex['english_text']}")
    print(f"ground_truth_hi: {ex['ground_truth_hi']}")
    print(f"generated_hi: {ex['generated_hi']}")
    print(f"gen_syl_count: {ex['gen_syl_count']}")
    print(f"target_notes: {ex['target_notes']}")
    print(f"syl_error: {ex['syl_error']}")
    print(f"sentence_bleu: {ex['sentence_bleu']}")

print()
print("TRAINING HISTORY")
for h in history:
    print(f"  epoch {h['epoch']}: train={h['train_loss']:.4f} val={h['val_loss']:.4f}")

print()
print("EXPERIMENT 07 COMPLETE")
