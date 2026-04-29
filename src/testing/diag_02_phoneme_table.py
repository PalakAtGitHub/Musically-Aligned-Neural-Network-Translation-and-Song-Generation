"""
Diagnostic test 02: verify the precomputed phoneme table is usable.

What this script does:
  1. Load token_phoneme_table.pt.
  2. Print the three feature tensors' shapes, basic stats, and sample entries.
  3. Confirm the shape matches the LM head vocab (~122672).
  4. Spot-check a few specific Hindi tokens that we know the shape of
     (e.g., 'है', 'के', 'प्यार') to make sure they got sensible feature values.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer

print("=" * 70)
print("DIAGNOSTIC 02: Verify precomputed token phoneme table")
print("=" * 70)


# ----------------------------------------------------------------------
# 1. Load the table
# ----------------------------------------------------------------------
table_path = PROJECT_ROOT / "src" / "data" / "processed" / "token_phoneme_table.pt"
print(f"\n[1/3] Loading table from {table_path}")
table = torch.load(table_path, weights_only=False)

print(f"  Keys:            {list(table.keys())}")
print(f"  vocab_size:      {table['vocab_size']}")
print(f"  tokenizer_name:  {table['tokenizer_name']}")

for key in ['n_leading_consonants', 'ends_in_open_vowel', 'has_vowel']:
    t = table[key]
    print(f"  {key:25s} shape={tuple(t.shape)} "
          f"sum={float(t.sum()):.0f} mean={float(t.mean()):.3f} "
          f"max={float(t.max()):.0f}")


# ----------------------------------------------------------------------
# 2. Spot-check known tokens
# ----------------------------------------------------------------------
print("\n[2/3] Spot-checking known Hindi tokens...")
tok = AutoTokenizer.from_pretrained(table['tokenizer_name'], trust_remote_code=True)

# IndicTrans2 tokenizer expects 'src_lang tgt_lang <text>' format in INPUT
# (encoder / src) mode. For encoding raw Hindi we need TARGET mode.
try:
    tok._switch_to_target_mode()
except Exception:
    pass

# Encode some test strings and look up their token ids
# We want to see whether common function words and some clustered words
# get sensible features.
tests = [
    "है",      # common: should have vowel, open ending, few leading cons
    "के",      # common function word
    "प्यार",   # cluster + open vowel at end
    "स्त्री",  # heavy cluster
    "नमस्ते",  # multi-syllable
    "छोटा",    # open ending
    "नारी",    # light cluster, open ending
]

for text in tests:
    # IndicTrans2 tokenizer's public encode() path requires the language
    # prefix even in target mode, because internally the mode switch only
    # affects vocab lookup, not the text parser. Pre-format the input.
    prefixed = f"hin_Deva hin_Deva {text}"
    ids = tok.encode(prefixed, add_special_tokens=False)
    # Drop the src/tgt prefix tags
    parts = []
    for tid in ids:
        decoded = tok.decode([tid], skip_special_tokens=True)
        if not decoded.strip() or decoded.strip() in ('hin_Deva', 'eng_Latn'):
            continue
        n_lead = float(table['n_leading_consonants'][tid])
        open_v = float(table['ends_in_open_vowel'][tid])
        has_v  = float(table['has_vowel'][tid])
        parts.append((tid, decoded, n_lead, open_v, has_v))
    print(f"  '{text}' → tokens:")
    for tid, decoded, nl, ov, hv in parts:
        print(f"    [{tid:6d}] {decoded!r:12s} "
              f"leading={int(nl)} open={int(ov)} vowel={int(hv)}")


# ----------------------------------------------------------------------
# 3. Confirm shape matches what the LM head would produce
# ----------------------------------------------------------------------
print("\n[3/3] Sanity: is shape consistent with LM head vocab?")
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(table['tokenizer_name'], trust_remote_code=True)
expected = cfg.vocab_size
actual = table['n_leading_consonants'].shape[0]
if expected == actual:
    print(f"  ✓ Match: table size {actual} == LM head vocab {expected}")
else:
    print(f"  ✗ MISMATCH: table size {actual} vs LM head vocab {expected}")

print("\n" + "=" * 70)
print("DIAGNOSTIC 02 COMPLETE")
print("=" * 70)
