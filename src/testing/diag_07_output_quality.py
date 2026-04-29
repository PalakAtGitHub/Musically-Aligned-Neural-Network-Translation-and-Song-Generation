"""
Diagnostic test 07: qualitative output quality check on 10 test examples.

Loads best checkpoint, generates Hindi for 10 test examples, and prints
per-example quality metrics: syllable match, repetition ratio, English
leakage, and character overlap with ground truth.
"""

import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.mcnst_model import MCNST
from src.utils.syllable_utils import count_hindi_syllables

print("=" * 70)
print("DIAGNOSTIC 07: Output quality check (10 test examples)")
print("=" * 70)

# Load model
print("\nLoading MCNST + best checkpoint...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=16)
device = torch.device("cpu")
ckpt = torch.load("checkpoints/best_model.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"  Checkpoint epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
model.to(device)
model.eval()

tokenizer = model.tokenizer

# Load test data
data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)
print(f"  Test examples available: {len(data)}")
n = min(10, len(data))

# English detection: ASCII letters forming words of length >= 2
ENGLISH_WORD_RE = re.compile(r'[A-Za-z]{2,}')

print(f"\n{'=' * 70}")
print(f"Generating on {n} test examples...")
print(f"{'=' * 70}\n")

for i in range(n):
    ex = data[i]
    english_text = ex['english_text']
    hindi_ref = ex['hindi_text']
    num_notes = ex['num_notes']

    src_ids = ex['src_ids'].unsqueeze(0)
    melody = ex['melody_features'].unsqueeze(0)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=src_ids,
            melody_features=melody,
            max_length=50,
            num_beams=5,
        )
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    gen_text = model.postprocess_tgt(gen_text)[0]

    gen_syl = count_hindi_syllables(gen_text)
    syl_match = abs(gen_syl - num_notes) <= 2

    # Token-level repetition
    tokens = gen_text.split()
    n_total = len(tokens)
    n_unique = len(set(tokens))
    unique_ratio = f"{n_unique}/{n_total}" if n_total > 0 else "0/0"

    # English leakage
    english_words = ENGLISH_WORD_RE.findall(gen_text)
    contains_english = len(english_words) > 0

    print(f"--- Example {i+1} ---")
    print(f"  english:             {english_text}")
    print(f"  ground_truth_hi:     {hindi_ref}")
    print(f"  generated_hi:        {gen_text}")
    print(f"  gen_syl_count:       {gen_syl}")
    print(f"  target_notes:        {num_notes}")
    print(f"  syl_match:           {'✓' if syl_match else '✗'}")
    print(f"  gen_n_unique_tokens: {n_unique}")
    print(f"  gen_n_total_tokens:  {n_total}")
    print(f"  unique_ratio:        {unique_ratio}")
    print(f"  contains_english:    {contains_english}" +
          (f"  ({english_words})" if contains_english else ""))
    print()

print("=" * 70)
print("DIAGNOSTIC 07 COMPLETE")
print("=" * 70)
