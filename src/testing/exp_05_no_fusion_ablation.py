"""
Experiment 05: Cross-modal fusion ablation.

Trains MCNST with fusion disabled (encoder outputs pass through unchanged
to decoder — melody encoder still runs so alignment module can function,
but CrossModalFusion is bypassed).

Then evaluates on the same 198 strict held-out examples.
"""

import sys
import os
import json
import time
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


# ====================================================================
# PHASE 1: TRAIN WITH FUSION DISABLED
# ====================================================================

print("=" * 70)
print("EXPERIMENT 05: Cross-Modal Fusion Ablation")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Initialize model and monkey-patch _encode_and_fuse
# ------------------------------------------------------------------
print("\n[1/7] Initializing MCNST (fusion will be disabled)...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=16)

# Save reference to the original method
_original_encode_and_fuse = model._encode_and_fuse

def _encode_and_fuse_no_fusion(self_or_input_ids, melody_features_or_none=None):
    """
    Replacement _encode_and_fuse that skips CrossModalFusion.

    Encoder outputs pass through unchanged. Melody encoder still runs
    so alignment module can attend to melody features for cluster_loss
    and openness_reward.
    """
    # Handle both bound method and unbound call patterns
    if melody_features_or_none is None:
        # Called as model._encode_and_fuse(input_ids, melody_features)
        # but self_or_input_ids is actually input_ids because we replace
        # the bound method
        raise RuntimeError("Should not happen with bound method replacement")

    input_ids = self_or_input_ids
    melody_features = melody_features_or_none

    attention_mask = (input_ids != model.pad_token_id).long()

    # 1. Encode text with IndicTrans2 encoder (frozen)
    encoder_outputs = model._encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )

    # 2. Ensure melody is long enough for CNN kernel (min 3 notes)
    min_notes = 3
    if melody_features.size(1) < min_notes:
        pad = min_notes - melody_features.size(1)
        melody_features = F.pad(melody_features, (0, 0, 0, pad))

    # 3. Encode melody hierarchically (still needed for alignment module)
    melody_encoded = model.melody_encoder(melody_features)

    # 4. Create melody padding mask
    melody_mask = (melody_features.sum(dim=-1) == 0)

    # 5. SKIP FUSION — encoder output passes through unchanged
    # Create dummy attn_weights (zeros) matching expected shape
    batch_size = input_ids.size(0)
    seq_len = encoder_outputs.last_hidden_state.size(1)
    num_notes = melody_encoded.size(1)
    attn_weights = torch.zeros(batch_size, seq_len, num_notes,
                               device=input_ids.device)

    # DO NOT replace encoder_outputs.last_hidden_state
    # Decoder sees raw IndicTrans2 encoder representations

    return encoder_outputs, attention_mask, attn_weights, melody_encoded, melody_mask


# Replace the bound method
import types
model._encode_and_fuse = types.MethodType(
    lambda self, input_ids, melody_features: _encode_and_fuse_no_fusion(input_ids, melody_features),
    model
)

# Verify the patch works
print("  Verifying fusion bypass...")
test_ids = torch.tensor([[2, 100, 200, 3]])
test_melody = torch.randn(1, 5, 5)
with torch.no_grad():
    enc_out, attn_mask, attn_w, mel_enc, mel_mask = model._encode_and_fuse(test_ids, test_melody)
assert attn_w.sum().item() == 0.0, "Attention weights should be zero (fusion bypassed)"
print("  Fusion bypass verified (attn_weights are zero).")

# ------------------------------------------------------------------
# 2. Setup device
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
# 3. Load data (same as Trainer: 90/10 split of train file)
# ------------------------------------------------------------------
print("\n[2/7] Loading datasets...")
full_dataset = SongTranslationDataset("src/data/processed/fma_train_data.pt")
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Use a fixed seed so the split is reproducible
# (The original Trainer does NOT set a seed, so splits differ across runs.
#  We accept this — the comparison is still valid because both models see
#  the same amount of data from the same distribution.)
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                          collate_fn=full_dataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                        collate_fn=full_dataset.collate_fn)

print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ------------------------------------------------------------------
# 4. Optimizer (same as Trainer: AdamW, lr=5e-5, wd=0.01)
# ------------------------------------------------------------------
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")

# ------------------------------------------------------------------
# 5. Train 10 epochs
# ------------------------------------------------------------------
NUM_EPOCHS = 10
SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = SAVE_DIR / "no_fusion_model.pt"

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

print(f"\n[3/7] Training for {NUM_EPOCHS} epochs (fusion DISABLED)...")
print("=" * 70)

best_val_loss = float('inf')
history = []
train_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    # Train
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
        if 'cluster_loss' in loss_dict:
            postfix['clst'] = f"{loss_dict['cluster_loss']:.3f}"
        if 'openness_loss' in loss_dict:
            postfix['open'] = f"{loss_dict['openness_loss']:.4f}"
        pbar.set_postfix(postfix)

    avg_train_loss = total_train_loss / len(train_loader)

    # Validate
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            loss, _ = model(**unpack_batch(batch))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / max(len(val_loader), 1)

    print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

    history.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'ablation': 'no_fusion',
        }, CHECKPOINT_PATH)
        print(f"  Saved best no-fusion checkpoint (val_loss={avg_val_loss:.4f})")

train_elapsed = time.time() - train_start
print(f"\nTraining complete in {train_elapsed:.0f}s. Best val_loss={best_val_loss:.4f}")

# Save training history
with open(SAVE_DIR / "no_fusion_training_history.json", 'w') as f:
    json.dump(history, f, indent=2)


# ====================================================================
# PHASE 2: EVALUATE ON STRICT HELD-OUT
# ====================================================================

print(f"\n[4/7] Loading best no-fusion checkpoint for evaluation...")
ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()
tokenizer = model.tokenizer
print(f"  Loaded epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

# ------------------------------------------------------------------
# Load strict held-out
# ------------------------------------------------------------------
print(f"\n[5/7] Loading strict held-out set...")
test_data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)
train_data = torch.load("src/data/processed/fma_train_data.pt", weights_only=False)
train_english = {ex['english_text'] for ex in train_data}
strict_heldout = [ex for ex in test_data if ex['english_text'] not in train_english]
assert len(strict_heldout) == 198, f"Expected 198, got {len(strict_heldout)}"

# Verify ordering matches Experiment 1
with open("logs/exp_01_indictrans2_baseline_results.json") as f:
    exp1 = json.load(f)
exp1_texts = [r['english_text'] for r in exp1['per_example_results']]
exp5_texts = [ex['english_text'] for ex in strict_heldout]
assert exp1_texts == exp5_texts, "Ordering mismatch with Experiment 1!"
print(f"  Strict held-out: {len(strict_heldout)} (ordering verified)")

# ------------------------------------------------------------------
# Generate
# ------------------------------------------------------------------
print(f"\n[6/7] Generating on {len(strict_heldout)} strict held-out examples...")
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
        gen_ids = model.generate(
            input_ids=src_ids,
            melody_features=melody,
            max_length=50,
            num_beams=5,
        )

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

    result = {
        'index': i,
        'song_name': ex['song_name'],
        'english_text': english_text,
        'ground_truth_hi': hindi_ref,
        'generated_hi': gen_text,
        'gen_syl_count': gen_syl,
        'target_notes': int(num_notes),
        'syl_error': syl_error,
        'syl_match_within_2': syl_match,
        'char_overlap_ratio': round(char_overlap, 4),
        'sentence_bleu': round(sent_bleu, 2),
    }
    results.append(result)

    if i < 3 or (i + 1) % 50 == 0 or i == len(strict_heldout) - 1:
        print(f"\n--- Example {i} (song: {ex['song_name']}) ---")
        print(f"  EN:  {english_text}")
        print(f"  REF: {hindi_ref}")
        print(f"  GEN: {gen_text}")
        print(f"  syl: {gen_syl} (target: {num_notes}, err: {syl_error}) "
              f"{'within2' if syl_match else 'MISS'}")
        print(f"  char_overlap: {char_overlap:.3f}  sent_BLEU: {sent_bleu:.1f}")

gen_elapsed = time.time() - gen_start
print(f"\n  Generation complete in {gen_elapsed:.1f}s")

# ------------------------------------------------------------------
# Corpus-level metrics
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

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = {
    'experiment': 'exp_05_no_fusion_ablation',
    'model': 'MCNST no-fusion (checkpoints/no_fusion_model.pt)',
    'description': 'MCNST with CrossModalFusion bypassed — encoder outputs pass through unchanged',
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

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
out_path = log_dir / f"exp_05_no_fusion_results.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  Saved to {out_path}")

# Print raw aggregate
print()
print("AGGREGATE METRICS")
print(f"corpus_bleu: {round(corpus_bleu, 2)}")
print(f"bertscore_f1_mean: {round(mean_bertscore_f1, 4)}")
print(f"syllable_accuracy_pm2: {round(syl_acc, 4)}")
print(f"mean_syllable_error: {round(mean_syl_err, 2)}")
print(f"mean_char_overlap: {round(mean_char_overlap, 4)}")
print(f"total_examples: {len(results)}")
print(f"training_epochs: {NUM_EPOCHS}")
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
    print(f"syl_match_within_2: {ex['syl_match_within_2']}")
    print(f"char_overlap_ratio: {ex['char_overlap_ratio']}")
    print(f"sentence_bleu: {ex['sentence_bleu']}")

print()
print("TRAINING HISTORY")
for h in history:
    print(f"  epoch {h['epoch']}: train={h['train_loss']:.4f} val={h['val_loss']:.4f}")

print()
print("EXPERIMENT 05 COMPLETE")
