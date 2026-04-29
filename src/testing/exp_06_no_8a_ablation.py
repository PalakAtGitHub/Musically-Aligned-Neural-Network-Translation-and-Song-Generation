"""
Experiment 06: Ablation A — MCNST without alignment + phonetic losses (no 8a).

Fusion is KEPT. LearnedAlignment is bypassed. cluster_loss and openness_reward
are disabled (alignment=None passed to loss, so those branches are skipped).
log_var_cluster and log_var_openness are deleted from the loss module so they
don't contribute parameters or regularisation.

All other components identical to the full MCNST training.
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
print("EXPERIMENT 06: No-8a Ablation (fusion ON, alignment/cluster/openness OFF)")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Build model and apply patches
# ------------------------------------------------------------------
print("\n[1/7] Initializing MCNST and disabling 8a components...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=16)

# --- Patch forward() to skip alignment computation ---
# Store original forward
_original_forward = model.forward

def _forward_no_alignment(self, input_ids, melody_features, labels=None,
                          tgt_syllables=None, num_notes=None,
                          stress_pattern=None, beat_pattern=None, **kwargs):
    """Forward pass that skips alignment computation and passes alignment=None."""
    encoder_outputs, attention_mask, attn_weights, melody_encoded, melody_mask = \
        self._encode_and_fuse(input_ids, melody_features)

    if labels is not None:
        outputs = self.seq2seq(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,  # no need — we're skipping alignment
            return_dict=True
        )

        translation_loss = outputs.loss
        logits = outputs.logits

        # NO alignment computation — pass alignment=None to loss
        loss_kwargs = dict(
            translation_loss=translation_loss,
            logits=logits,
            labels=labels,
            melody_features=melody_features,
            alignment=None,  # <-- THIS IS THE KEY CHANGE
        )
        if num_notes is not None:
            loss_kwargs['num_notes'] = num_notes.float()
        if stress_pattern is not None:
            loss_kwargs['stress_pattern'] = stress_pattern
            loss_kwargs['beat_pattern'] = (
                beat_pattern if beat_pattern is not None
                else melody_features[:, :, 4]
            )

        total_loss, loss_dict = self.loss_fn(**loss_kwargs)
        return total_loss, loss_dict
    else:
        return self.seq2seq(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            return_dict=True
        )

model.forward = types.MethodType(_forward_no_alignment, model)

# --- Remove log_var_cluster and log_var_openness from loss ---
# Delete them so they don't exist as parameters (no gradient, no regularisation)
if hasattr(model.loss_fn, 'log_var_cluster'):
    del model.loss_fn.log_var_cluster
if hasattr(model.loss_fn, 'log_var_openness'):
    del model.loss_fn.log_var_openness
# Replace with non-parameter attributes so the code doesn't crash if accessed
model.loss_fn.log_var_cluster = torch.tensor(0.0)
model.loss_fn.log_var_openness = torch.tensor(0.0)

print("  Patched: forward() skips alignment, loss has no cluster/openness params")

# Verify: count trainable params (should be fewer — no aligner grads needed,
# but aligner weights still exist as frozen since we didn't remove the module)
# Actually the aligner IS trainable, it just won't get gradient because we
# don't call it. Let's freeze it explicitly.
for p in model.aligner.parameters():
    p.requires_grad = False
print("  Froze aligner parameters (not used in this ablation)")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} / {total_params:,}")

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
CHECKPOINT_PATH = Path("checkpoints/no_8a_model.pt")

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

print(f"\n[3/7] Training for {NUM_EPOCHS} epochs (no 8a)...")
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
        postfix = {
            'loss': f"{loss.item():.4f}",
            'trans': f"{loss_dict['translation_loss']:.3f}",
            'syl': f"{loss_dict['syllable_loss']:.3f}",
        }
        if loss_dict.get('naturalness_loss', 0) > 0:
            postfix['nat'] = f"{loss_dict['naturalness_loss']:.3f}"
        pbar.set_postfix(postfix)

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
            'ablation': 'no_8a',
        }, CHECKPOINT_PATH)
        print(f"  Saved best (val_loss={avg_val:.4f})")

train_elapsed = time.time() - train_start
print(f"\nTraining complete in {train_elapsed:.0f}s. Best val_loss={best_val_loss:.4f}")

with open("checkpoints/no_8a_training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

# ------------------------------------------------------------------
# 6. Evaluate on strict held-out
# ------------------------------------------------------------------
print(f"\n[4/7] Loading best no-8a checkpoint...")
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
    'experiment': 'exp_06_no_8a_ablation',
    'model': 'MCNST no-8a (checkpoints/no_8a_model.pt)',
    'description': 'MCNST with fusion ON but alignment/cluster/openness OFF',
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

out_path = Path("logs") / "exp_06_no_8a_results.json"
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
print("EXPERIMENT 06 COMPLETE")
