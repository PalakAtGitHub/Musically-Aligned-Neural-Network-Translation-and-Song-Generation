"""
Diagnostic 08: verify generation on held-out test data is NOT leaking labels.

Checks:
  - Uses fma_test_data.pt exclusively
  - Decodes src_ids, tgt_ids, and generated ids separately
  - Prints raw token IDs for comparison
  - Checks string equality between generated and ground truth
  - Cross-checks train vs test split
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.mcnst_model import MCNST

print("=" * 70)
print("DIAGNOSTIC 08: True held-out generation check")
print("=" * 70)

# 1. Load model
print("\n[1/5] Loading model...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=16)
device = torch.device("cpu")
ckpt = torch.load("checkpoints/best_model.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()
tokenizer = model.tokenizer
print(f"  Checkpoint epoch={ckpt['epoch']}")

# 2. Load TEST data
print("\n[2/5] Loading datasets...")
test_data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)
train_data = torch.load("src/data/processed/fma_train_data.pt", weights_only=False)
print(f"  fma_test_data.pt:  {len(test_data)} examples")
print(f"  fma_train_data.pt: {len(train_data)} examples")

# 3. Generate on 5 test examples
print("\n[3/5] Generating on test examples [0..4]...")
print("=" * 70)

for i in range(5):
    ex = test_data[i]
    english_text = ex['english_text']
    hindi_ref_text = ex['hindi_text']
    num_notes = ex['num_notes']
    src_ids = ex['src_ids']
    tgt_ids = ex['tgt_ids']

    # Decode ground truth tgt_ids using target-mode tokenizer
    tokenizer._switch_to_target_mode()
    gt_decoded = tokenizer.decode(tgt_ids, skip_special_tokens=True)
    tokenizer._switch_to_input_mode()

    # Generate
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=src_ids.unsqueeze(0),
            melody_features=ex['melody_features'].unsqueeze(0),
            max_length=50,
            num_beams=5,
        )
    gen_decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    gen_decoded = model.postprocess_tgt(gen_decoded)[0]

    identical = (gen_decoded == hindi_ref_text)

    print(f"\n--- Test Example {i} ---")
    print(f"  english_text:      {english_text}")
    print(f"  hindi_text (meta): {hindi_ref_text}")
    print(f"  gt_decoded:        {gt_decoded}")
    print(f"  generated:         {gen_decoded}")
    print(f"  gen == gt_meta:    {identical}")
    print(f"  gen == gt_decoded: {gen_decoded == gt_decoded}")
    print(f"  src_ids[:20]:      {src_ids[:20].tolist()}")
    print(f"  tgt_ids[:20]:      {tgt_ids[:20].tolist()}")
    print(f"  gen_ids[:20]:      {gen_ids[0][:20].tolist()}")

# 4. Cross-check: find same english_text in train and test
print("\n\n[4/5] Cross-checking train vs test overlap...")
print("=" * 70)

test_english = {ex['english_text'] for ex in test_data}
train_english = {ex['english_text'] for ex in train_data}
overlap = test_english & train_english
print(f"  Unique english texts in test:  {len(test_english)}")
print(f"  Unique english texts in train: {len(train_english)}")
print(f"  Overlapping english texts:     {len(overlap)}")

if overlap:
    # Pick one overlapping text and compare
    sample_text = sorted(overlap)[0]
    print(f"\n  Sample overlapping text: \"{sample_text}\"")

    train_ex = next(ex for ex in train_data if ex['english_text'] == sample_text)
    test_ex = next(ex for ex in test_data if ex['english_text'] == sample_text)

    print(f"    Train tgt_ids[:20]: {train_ex['tgt_ids'][:20].tolist()}")
    print(f"    Test  tgt_ids[:20]: {test_ex['tgt_ids'][:20].tolist()}")
    print(f"    Train hindi_text:   {train_ex['hindi_text']}")
    print(f"    Test  hindi_text:   {test_ex['hindi_text']}")
    print(f"    Train song_name:    {train_ex['song_name']}")
    print(f"    Test  song_name:    {test_ex['song_name']}")
    same_tgt = torch.equal(train_ex['tgt_ids'], test_ex['tgt_ids'])
    print(f"    tgt_ids identical:  {same_tgt}")
else:
    print("  No overlapping english texts found.")

# 5. Inspect model.generate() for label leakage
print("\n\n[5/5] Checking generate() call signature...")
print("=" * 70)
import inspect
gen_source = inspect.getsource(model.generate)
has_labels = "labels" in gen_source
print(f"  'labels' appears in generate() source: {has_labels}")
if has_labels:
    for line_no, line in enumerate(gen_source.split('\n'), 1):
        if 'labels' in line:
            print(f"    Line {line_no}: {line.strip()}")

print("\n" + "=" * 70)
print("DIAGNOSTIC 08 COMPLETE")
print("=" * 70)
