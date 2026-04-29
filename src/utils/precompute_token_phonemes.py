"""
Precompute Token Phonetic Table
================================

One-time script: for each token in the IndicTrans2 target (Hindi) vocabulary,
decode it, phonemize it with espeak-ng, and cache a small set of features:

  - n_leading_consonants:  count of leading consonant phonemes before the
                           first vowel (→ cluster-badness score)
  - ends_in_open_vowel:    1.0 if the decoded token ends in an open vowel
                           (aː, eː, oː, ɛː, ɔː, or unmarked a) (→ openness
                           score, paired with beat strength)
  - has_vowel:             1.0 if the decoded token contains any vowel
                           (redundant with existing vowel-mask but free to
                            have here for sanity-checking)
  - token_string:          the decoded string (for debugging only; dropped
                            from the saved tensor to keep the artifact small)

Artifact written:
  src/data/processed/token_phoneme_table.pt
    → dict with keys:
         vocab_size, tokenizer_name,
         n_leading_consonants:   [vocab] float tensor
         ends_in_open_vowel:     [vocab] float tensor
         has_vowel:              [vocab] float tensor

Run this ONCE per tokenizer. The loss loads the cached file at training
time — if the file doesn't exist, it falls back to the uniform-ones
approximation and prints a warning.

Usage:
    cd /path/to/musically-aligned-translation
    source venv/bin/activate
    export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
    python -m src.utils.precompute_token_phonemes

Takes ~10-15 minutes for a 50k-token vocab. Intermediate progress is
printed every 1000 tokens. If interrupted, re-run from scratch — there's
no resume logic (the computation is idempotent and not that slow).
"""

import os
import sys
import time
import torch
from pathlib import Path
from collections import Counter


# ============================================================================
# Configuration
# ============================================================================

TOKENIZER_NAME = "ai4bharat/indictrans2-en-indic-1B"
OUTPUT_DIR     = Path("src/data/processed")
OUTPUT_FILE    = OUTPUT_DIR / "token_phoneme_table.pt"

# IPA vowel set. Verified empirically against espeak-ng 1.52 Hindi output.
# Phonemizing each independent Hindi vowel plus several words gives these
# as the actual vowels espeak produces: a, e, i, o, u, ɔ, ə, ɛ, ɪ,
# ʊ, ʌ. The critical non-obvious entry is ʌ (U+028C), which espeak-ng
# uses for the unstressed Hindi schwa in polysyllabic words — for example
# कमल phonemizes to 'kʌməl'. An earlier version of this script inherited
# a vowel set from phoneme_utils.py that was missing ʌ, causing tokens
# starting with ʌ (roughly 10% of the vocab) to miscount leading consonants.
IPA_VOWELS = set('aɑæeɛiɪoɔuʊəʌɜɐ')

# Open vowels for the "ends in open vowel" score.
# Rationale: these are the vowels a singer can project loudly on a strong
# beat without modification. Closed vowels like ɪ, ʊ sound thin held loud.
# ʌ and ə are schwa-ish and don't count as "open" for projection.
OPEN_VOWEL_CHARS = set('aɑæeɛoɔ')

# Length marker — part of a long vowel, counts as vowel
LENGTH_MARKER = 'ː'


# ============================================================================
# Feature extraction from phonemized string
# ============================================================================

def analyze_phonemes(phoneme_string: str) -> dict:
    """
    Given an IPA-phonemized string, extract the three features we care about.

    Robust to weird espeak outputs: empty strings, stress markers (ˈˌ),
    punctuation, whitespace. These are all skipped.
    """
    if not phoneme_string:
        return {
            'n_leading_consonants': 0.0,
            'ends_in_open_vowel':   0.0,
            'has_vowel':            0.0,
        }

    # Strip stress markers and whitespace — they're not phonemes
    cleaned = ''.join(c for c in phoneme_string if c not in 'ˈˌ \t\n')
    if not cleaned:
        return {
            'n_leading_consonants': 0.0,
            'ends_in_open_vowel':   0.0,
            'has_vowel':            0.0,
        }

    # Leading consonants: count chars until first vowel
    n_leading = 0
    for c in cleaned:
        if c in IPA_VOWELS:
            break
        # Don't count punctuation as a consonant
        if c.isalpha() or ord(c) > 127:  # IPA symbols are mostly non-ASCII
            n_leading += 1

    # Has vowel anywhere
    has_vowel = any(c in IPA_VOWELS for c in cleaned)

    # Ends in open vowel: last char (possibly preceded by ː) is open vowel
    # Check last 2 chars to handle aː, eː, etc.
    ends_open = False
    if len(cleaned) >= 1:
        last = cleaned[-1]
        if last == LENGTH_MARKER and len(cleaned) >= 2:
            # Long vowel — check the vowel before the ː
            prev = cleaned[-2]
            ends_open = prev in OPEN_VOWEL_CHARS
        else:
            ends_open = last in OPEN_VOWEL_CHARS

    return {
        'n_leading_consonants': float(n_leading),
        'ends_in_open_vowel':   1.0 if ends_open else 0.0,
        'has_vowel':            1.0 if has_vowel else 0.0,
    }


# ============================================================================
# Main precompute loop
# ============================================================================

def main():
    # Ensure the espeak-ng library is visible to phonemizer.
    # This can be set in the shell too — checking here so the user gets a
    # clear error if they forgot.
    if 'PHONEMIZER_ESPEAK_LIBRARY' not in os.environ:
        # Try a sensible default for Homebrew on Apple Silicon
        default = "/opt/homebrew/lib/libespeak-ng.dylib"
        if Path(default).exists():
            os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = default
            print(f"  (auto-set PHONEMIZER_ESPEAK_LIBRARY={default})")
        else:
            print("⚠ PHONEMIZER_ESPEAK_LIBRARY is not set and the default "
                  "Homebrew path was not found. Set it manually:")
            print("    export PHONEMIZER_ESPEAK_LIBRARY=/path/to/libespeak-ng.dylib")
            sys.exit(1)

    # Imports after the env var is set, because phonemizer caches library
    # handles at import time on some systems.
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    from transformers import AutoTokenizer

    print("="*60)
    print("Token Phoneme Table Precomputation")
    print("="*60)
    print(f"Tokenizer: {TOKENIZER_NAME}")
    print(f"Output:    {OUTPUT_FILE}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True
    )
    # Switch to target-side vocab (Hindi) for IndicTrans2
    try:
        tokenizer._switch_to_target_mode()
    except Exception:
        pass

    # CRITICAL: we need to iterate over the FULL LM-head vocab size, not
    # len(tokenizer). IndicTrans2's LM head outputs logits of size ~122672
    # (all indic languages combined), but the Hindi tokenizer only reveals
    # ~32322 of those. The mask/features must be sized to match the logits,
    # otherwise the loss multiplication shape-mismatches. See also the
    # equivalent fix in src/training/loss.py::_build_vowel_mask.
    # We pull the full vocab_size from the model's config — cheaper than
    # loading the whole model just for this one number, so use a Config-only
    # import.
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
        vocab_size = cfg.vocab_size
    except Exception:
        # Fallback: use tokenizer length (produces a table too small, will
        # trigger the equivalent fallback in the loss — non-fatal)
        vocab_size = len(tokenizer)
        print(f"  ⚠ could not load config, using len(tokenizer)={vocab_size}")

    print(f"Tokenizer length: {len(tokenizer)}")
    print(f"LM head vocab:    {vocab_size}  (mask/features will be this size)")

    # ------------------------------------------------------------------
    # Pass 1: decode every token to a string. Some will be special
    # tokens, byte-level artifacts, or empty — those get zero features.
    # ------------------------------------------------------------------
    print("\n[1/3] Decoding every token...")
    decoded = []
    non_empty_ids = []
    for v in range(vocab_size):
        try:
            s = tokenizer.decode([v], skip_special_tokens=True)
        except Exception:
            s = ''
        decoded.append(s)
        if s.strip():
            non_empty_ids.append(v)

    print(f"  Tokens with non-empty decoded string: {len(non_empty_ids)} "
          f"({100*len(non_empty_ids)/vocab_size:.1f}%)")

    # ------------------------------------------------------------------
    # Pass 2: batch-phonemize the non-empty tokens.
    # phonemizer's espeak backend is MUCH faster when given a batch of
    # strings than one at a time (one espeak subprocess for the batch).
    # ------------------------------------------------------------------
    print("\n[2/3] Phonemizing tokens in batches...")
    # Create a persistent backend (avoid paying subprocess cost per batch)
    backend = EspeakBackend(
        language='hi',
        preserve_punctuation=False,
        with_stress=False,    # we don't need stress here — we're counting structure
    )

    BATCH_SIZE = 500
    phoneme_map = {}  # vocab_id -> phoneme_string

    t0 = time.time()
    for batch_start in range(0, len(non_empty_ids), BATCH_SIZE):
        batch_ids = non_empty_ids[batch_start:batch_start + BATCH_SIZE]
        batch_strs = [decoded[v] for v in batch_ids]

        try:
            batch_phons = backend.phonemize(batch_strs, strip=True)
        except Exception as e:
            # Fallback: do one at a time
            batch_phons = []
            for s in batch_strs:
                try:
                    p = backend.phonemize([s], strip=True)
                    batch_phons.append(p[0] if p else '')
                except Exception:
                    batch_phons.append('')

        for vid, phon in zip(batch_ids, batch_phons):
            phoneme_map[vid] = phon

        done = batch_start + len(batch_ids)
        pct  = 100 * done / len(non_empty_ids)
        el   = time.time() - t0
        eta  = el * (len(non_empty_ids) - done) / max(done, 1)
        print(f"  {done:6d}/{len(non_empty_ids)}  ({pct:5.1f}%)  "
              f"elapsed {el:5.0f}s  eta {eta:5.0f}s", flush=True)

    print(f"  Done in {time.time() - t0:.0f}s")

    # ------------------------------------------------------------------
    # Pass 3: derive the three feature tensors
    # ------------------------------------------------------------------
    print("\n[3/3] Extracting features...")
    n_leading  = torch.zeros(vocab_size)
    open_vowel = torch.zeros(vocab_size)
    has_vowel  = torch.zeros(vocab_size)

    for vid, phon in phoneme_map.items():
        feats = analyze_phonemes(phon)
        n_leading[vid]  = feats['n_leading_consonants']
        open_vowel[vid] = feats['ends_in_open_vowel']
        has_vowel[vid]  = feats['has_vowel']

    # ------------------------------------------------------------------
    # Sanity checks — print a few examples so we can eyeball the output
    # ------------------------------------------------------------------
    print("\nSanity-check samples:")
    print(f"  Tokens with vowel:        {int(has_vowel.sum())}/{vocab_size}")
    print(f"  Tokens ending in open V:  {int(open_vowel.sum())}/{vocab_size}")
    print(f"  Mean leading consonants:  {n_leading[non_empty_ids].mean():.2f}")
    print(f"  Max  leading consonants:  {int(n_leading.max())}")

    print("\n  Example tokens (decoded → phonemes → features):")
    examples = []
    # Show a spread: some high-cluster, some open-ending, some neither
    sorted_by_cluster = sorted(non_empty_ids, key=lambda v: -n_leading[v].item())
    for v in sorted_by_cluster[:5]:
        examples.append(v)
    open_ids = [v for v in non_empty_ids if open_vowel[v] > 0]
    for v in open_ids[:5]:
        if v not in examples:
            examples.append(v)

    for v in examples[:10]:
        print(f"    [{v:5d}] {decoded[v]!r:20s} → {phoneme_map.get(v, '')!r:25s} "
              f"leading={int(n_leading[v])} open={int(open_vowel[v])}")

    # ------------------------------------------------------------------
    # Save artifact
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        'vocab_size':             vocab_size,
        'tokenizer_name':         TOKENIZER_NAME,
        'n_leading_consonants':   n_leading,
        'ends_in_open_vowel':     open_vowel,
        'has_vowel':              has_vowel,
    }, OUTPUT_FILE)

    print(f"\n✓ Saved to {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
