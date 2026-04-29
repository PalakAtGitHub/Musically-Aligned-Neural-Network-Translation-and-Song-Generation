"""
Diagnostic 09: strict held-out evaluation.

Filters test set to examples whose english_text never appears in training,
then generates and evaluates on that subset.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.mcnst_model import MCNST
from src.utils.syllable_utils import count_hindi_syllables

print("=" * 70)
print("DIAGNOSTIC 09: Strict held-out evaluation")
print("=" * 70)

# 1. Load model
print("\n[1/4] Loading model...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=16)
device = torch.device("cpu")
ckpt = torch.load("checkpoints/best_model.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()
tokenizer = model.tokenizer
print(f"  Checkpoint epoch={ckpt['epoch']}")

# 2. Load datasets
print("\n[2/4] Loading datasets...")
test_data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)
train_data = torch.load("src/data/processed/fma_train_data.pt", weights_only=False)
print(f"  Test examples:  {len(test_data)}")
print(f"  Train examples: {len(train_data)}")

# 3. Build strict held-out subset
print("\n[3/4] Building strict held-out subset...")
train_english = {ex['english_text'] for ex in train_data}
strict_heldout = [ex for ex in test_data if ex['english_text'] not in train_english]
print(f"  Train unique english texts: {len(train_english)}")
print(f"  Test examples total:        {len(test_data)}")
print(f"  Overlapping with train:     {len(test_data) - len(strict_heldout)}")
print(f"  Strict held-out remaining:  {len(strict_heldout)}")

# 4. Generate and evaluate
n = min(10, len(strict_heldout))
print(f"\n[4/4] Generating on {n} strict held-out examples...")
print("=" * 70)

identical_count = 0
char_overlaps = []
syl_errors = []

for i in range(n):
    ex = strict_heldout[i]
    english_text = ex['english_text']
    hindi_ref_text = ex['hindi_text']
    num_notes = ex['num_notes']
    tgt_ids = ex['tgt_ids']

    # Decode ground truth from tgt_ids
    tokenizer._switch_to_target_mode()
    gt_decoded = tokenizer.decode(tgt_ids, skip_special_tokens=True)
    tokenizer._switch_to_input_mode()

    # Generate
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=ex['src_ids'].unsqueeze(0),
            melody_features=ex['melody_features'].unsqueeze(0),
            max_length=50,
            num_beams=5,
        )
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    gen_text = model.postprocess_tgt(gen_text)[0]

    gen_syl = count_hindi_syllables(gen_text)
    syl_match = abs(gen_syl - num_notes) <= 2
    is_identical = (gen_text == hindi_ref_text)

    # Character-level overlap
    max_len = max(len(gen_text), len(gt_decoded), 1)
    matching_chars = sum(1 for a, b in zip(gen_text, gt_decoded) if a == b)
    char_overlap_pct = matching_chars / max_len * 100

    if is_identical:
        identical_count += 1
    char_overlaps.append(char_overlap_pct)
    syl_errors.append(abs(gen_syl - num_notes))

    print(f"\n--- Strict Held-out Example {i} (song: {ex['song_name']}) ---")
    print(f"  english_text:    {english_text}")
    print(f"  ground_truth_hi: {hindi_ref_text}")
    print(f"  gt_decoded:      {gt_decoded}")
    print(f"  generated_hi:    {gen_text}")
    print(f"  gen_syl_count:   {gen_syl}")
    print(f"  target_notes:    {num_notes}")
    print(f"  syl_match (±2):  {'✓' if syl_match else '✗'}")
    print(f"  char_overlap:    {char_overlap_pct:.1f}%  ({matching_chars}/{max_len})")
    print(f"  gen == gt:       {is_identical}")

# Summary
print(f"\n\n{'=' * 70}")
print(f"SUMMARY ({n} strict held-out examples)")
print(f"{'=' * 70}")
print(f"  Identical to ground truth: {identical_count}/{n}")
print(f"  Avg char overlap:          {sum(char_overlaps)/len(char_overlaps):.1f}%")
print(f"  Avg syllable error:        {sum(syl_errors)/len(syl_errors):.2f}")
syl_acc = sum(1 for e in syl_errors if e <= 2) / n
print(f"  Syllable accuracy (±2):    {syl_acc*100:.1f}%")
print(f"{'=' * 70}")
print("DIAGNOSTIC 09 COMPLETE")
print(f"{'=' * 70}")
