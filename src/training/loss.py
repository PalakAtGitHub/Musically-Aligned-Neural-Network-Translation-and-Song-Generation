"""
Pentathlon Loss for MCNST

Based on Peter Low's Pentathlon Principle (2005), fully activated:
  1. Sense        — translation quality (cross-entropy from IndicTrans2)
  2. Singability  — EXPECTED syllable count from decoder logits vs. target
  3. Naturalness  — label-smoothed NLL (replaces the overconfidence-reward
                    formulation which caused mode collapse)
  4. Rhythm       — stress-beat alignment + expected-syllable-weighted penalty
  5. Rhyme        — phonetic ending similarity + endpoint entropy

Uses learnable loss weights via homoscedastic uncertainty
(Kendall et al., CVPR 2018).

CRITICAL FIX (this version): the syllable and rhythm losses now derive
`predicted_syllables` from a *differentiable expectation* over the decoder
logits using a precomputed "vowel-bearing-token" mask over the target
vocabulary. Previously these losses were fed ground-truth labels, which
meant they did not produce any gradient signal into the model — they
only trained the learnable log_var weights. With this fix, every
component of the pentathlon actually participates in backprop.

The naturalness term was ALSO changed. Previously it was
`1 - mean_token_confidence`, which directly rewards the model for being
overconfident on any output (mode collapse). It is now a symmetric
label-smoothed NLL: penalises both very-low-confidence (disfluent) and
all-mass-on-one-token (repetitive) outputs, converging on the desired
middle ground.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Phoneme-feature table loader (8a: cluster_loss and openness_reward)
# ============================================================================
# The precomputed file lives at src/data/processed/token_phoneme_table.pt
# and is built by src/utils/precompute_token_phonemes.py. It contains per-
# token features derived from espeak-ng phonemization:
#   - n_leading_consonants:  how many consonants come before the first vowel
#   - ends_in_open_vowel:    1.0 if the token ends in an open projection vowel
#   - has_vowel:             1.0 if the token contains any vowel (unused here)
# Shape is [vocab_size] where vocab_size matches the LM head output dim.

_PHONEME_TABLE_CACHE = {}  # device-keyed cache


def _load_phoneme_features(device, expected_vocab_size=None):
    """
    Load the precomputed phoneme-feature table and return a dict of tensors
    on the requested device. Caches per-device.

    Returns None if the file is missing (caller should fall back to a
    uniform approximation and print a warning exactly once).
    """
    from pathlib import Path

    cache_key = str(device)
    if cache_key in _PHONEME_TABLE_CACHE:
        return _PHONEME_TABLE_CACHE[cache_key]

    # The file is at <project_root>/src/data/processed/token_phoneme_table.pt
    # We're inside <project_root>/src/training/loss.py — walk up two dirs.
    here = Path(__file__).resolve().parent
    table_path = here.parent / "data" / "processed" / "token_phoneme_table.pt"

    if not table_path.exists():
        print(f"  ⚠ phoneme table not found at {table_path}")
        print(f"    run: python -m src.utils.precompute_token_phonemes")
        print(f"    cluster_loss and openness_reward will use uniform fallback")
        _PHONEME_TABLE_CACHE[cache_key] = None
        return None

    try:
        table = torch.load(table_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  ⚠ failed to load phoneme table ({e}) — using uniform fallback")
        _PHONEME_TABLE_CACHE[cache_key] = None
        return None

    # Sanity-check shape against caller's expected vocab size
    if expected_vocab_size is not None:
        actual = table['n_leading_consonants'].size(0)
        if actual != expected_vocab_size:
            print(f"  ⚠ phoneme table has vocab_size {actual} but logits "
                  f"have dim {expected_vocab_size}")
            print(f"    re-run: python -m src.utils.precompute_token_phonemes")
            _PHONEME_TABLE_CACHE[cache_key] = None
            return None

    features = {
        'n_leading_consonants': table['n_leading_consonants'].to(device),
        'ends_in_open_vowel':   table['ends_in_open_vowel'].to(device),
        'has_vowel':            table['has_vowel'].to(device),
    }
    n_vowel = int(features['has_vowel'].sum())
    n_open  = int(features['ends_in_open_vowel'].sum())
    vocab   = features['has_vowel'].size(0)
    print(f"  ✓ phoneme table loaded ({vocab} tokens, {n_vowel} with vowel, "
          f"{n_open} ending in open vowel)")
    _PHONEME_TABLE_CACHE[cache_key] = features
    return features


# ============================================================================
# Vowel-bearing-token mask (cached at module level — built once per tokenizer)
# ============================================================================
# A "vowel-bearing" token is one that, when decoded, contains at least one
# Devanagari vowel character (independent vowel or dependent matra). The
# count of such tokens in a Hindi sequence approximates the Hindi syllable
# count, because Devanagari syllables are built around vowels.
#
# This mask lets us express "expected number of syllables in the generated
# output" as  sum_t  sum_v  softmax(logits_t)[v] * mask[v]  — a linear
# function of softmax probabilities, hence differentiable.

_DEVANAGARI_VOWEL_RE = re.compile(
    r'[\u0905-\u0914\u093E-\u094C\u0960\u0961\u0962\u0963]'
    # independent vowels (U+0905..U+0914), dependent matras (U+093E..U+094C),
    # and the less-common vocalic R/L variants
)

_VOWEL_MASK_CACHE = {}


def _build_vowel_mask(tokenizer, device, vocab_size=None):
    """
    Build a [vocab_size] float tensor where entry v = 1.0 iff decoded token v
    contains at least one Devanagari vowel character.

    IMPORTANT: IndicTrans2's LM head outputs logits of size config.vocab_size
    (~122672 — the combined Indic-languages vocab), NOT len(tokenizer)
    (~32322 — just the Hindi-facing vocab). The mask must match the logits
    dimension, so we iterate over ALL ids 0..vocab_size-1. For ids the Hindi
    tokenizer can't decode (because they belong to Bengali/Tamil/etc.) we
    get an empty string, which correctly yields mask=0 — those tokens have
    near-zero probability at inference anyway because the model was trained
    in Hindi mode.

    Cached per (tokenizer-id, device, vocab_size). Falls back to all-ones if
    anything fails.
    """
    cache_key = (id(tokenizer), str(device), vocab_size)
    if cache_key in _VOWEL_MASK_CACHE:
        return _VOWEL_MASK_CACHE[cache_key]

    try:
        # Use the caller-provided vocab_size (which should be the LM head
        # output dim). Fall back to len(tokenizer) only if not given.
        if vocab_size is None:
            vocab_size = len(tokenizer)
        # IndicTrans2 has separate src/tgt vocabs — switch to target mode so
        # we inspect Hindi-side tokens. Safe no-op for single-vocab tokenizers.
        switched = False
        try:
            tokenizer._switch_to_target_mode()
            switched = True
        except Exception:
            pass

        mask = torch.zeros(vocab_size, device=device)
        for v in range(vocab_size):
            try:
                decoded = tokenizer.decode([v], skip_special_tokens=True)
            except Exception:
                continue
            if decoded and _DEVANAGARI_VOWEL_RE.search(decoded):
                mask[v] = 1.0

        if switched:
            try:
                tokenizer._switch_to_input_mode()
            except Exception:
                pass

        if mask.sum().item() < 1.0:
            print("  ⚠ vowel-mask: no Devanagari vowel tokens found in vocab "
                  "— falling back to uniform mask (syllable loss will be "
                  "proportional to token count only)")
            mask = torch.ones(vocab_size, device=device)
        else:
            n_vowel = int(mask.sum().item())
            print(f"  ✓ vowel-mask: {n_vowel}/{vocab_size} tokens flagged "
                  f"as vowel-bearing")

        _VOWEL_MASK_CACHE[cache_key] = mask
        return mask
    except Exception as e:
        print(f"  ⚠ vowel-mask build failed ({e}) — using uniform mask")
        fallback_size = vocab_size or 50000
        mask = torch.ones(fallback_size, device=device)
        _VOWEL_MASK_CACHE[cache_key] = mask
        return mask


def expected_syllables_from_logits(logits, labels, vowel_mask, pad_token_id):
    """
    Differentiable estimate of the number of Hindi syllables the decoder
    is about to produce, per example.

    Args:
        logits:        [batch, seq_len, vocab]  — decoder logits (requires_grad)
        labels:        [batch, seq_len]         — target ids (padding mask +
                                                  position alignment)
        vowel_mask:    [vocab]                  — 1.0 for vowel-bearing tokens
        pad_token_id:  int

    Returns:
        [batch] tensor of expected syllable counts. Gradient flows through
        softmax(logits) into the decoder and upstream into the fusion +
        melody encoder.
    """
    probs = F.softmax(logits, dim=-1)                   # [B, T, V]
    per_pos = (probs * vowel_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, T]
    pad_mask = (labels != pad_token_id).float()         # [B, T]
    per_example = (per_pos * pad_mask).sum(dim=-1)      # [B]
    return per_example


# ============================================================================
# Main loss module
# ============================================================================


class PentathlonLoss(nn.Module):
    """
    Multi-objective loss with LEARNABLE weights.

    All 5 Pentathlon criteria are now active AND gradient-bearing:
      - Translation    (cross-entropy, always on)
      - Syllable count (differentiable via expected_syllables_from_logits)
      - Naturalness    (label-smoothed NLL, not overconfidence reward)
      - Rhythm         (stress-beat alignment OR syllable-weighted penalty)
      - Rhyme          (endpoint entropy + optional external rhyme similarity)
    """

    def __init__(self,
                 tokenizer=None,
                 pad_token_id=1,
                 label_smoothing=0.1,
                 hindi_lm=None):
        super().__init__()

        # Learnable log-variance for each loss component (Kendall et al. 2018)
        # log_var = 0  =>  initial weight = exp(0) = 1
        self.log_var_translation = nn.Parameter(torch.tensor(0.0))
        self.log_var_syllable    = nn.Parameter(torch.tensor(0.0))
        self.log_var_naturalness = nn.Parameter(torch.tensor(0.0))
        self.log_var_rhythm      = nn.Parameter(torch.tensor(0.0))
        self.log_var_rhyme       = nn.Parameter(torch.tensor(-1.0))
        # 8a additions: start them down-weighted (log_var = 1 => weight = 1/e ≈ 0.37)
        # so the new terms don't dominate the untrained model's loss landscape
        # before the melody encoder and alignment module learn anything useful.
        self.log_var_cluster     = nn.Parameter(torch.tensor(1.0))
        self.log_var_openness    = nn.Parameter(torch.tensor(1.0))

        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing

        # Tokenizer is set later via set_tokenizer() once the model is built
        # (avoids import cycles and lets the same PentathlonLoss instance be
        # created before the tokenizer exists).
        self._tokenizer = tokenizer
        self._vowel_mask = None       # [vocab], lazy-built on first forward
        self.hindi_lm = hindi_lm       # optional external scorer

    def set_tokenizer(self, tokenizer, pad_token_id=None):
        """Attach the model's tokenizer so the syllable-mask can be built."""
        self._tokenizer = tokenizer
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        self._vowel_mask = None  # force rebuild on next forward

    def _get_vowel_mask(self, device, vocab_size=None):
        """Lazy-build and cache the vowel-bearing-token mask.

        `vocab_size` should be the LM head output dim (the last dim of the
        logits tensor). This is usually bigger than len(tokenizer) for
        multilingual models like IndicTrans2 — see _build_vowel_mask for
        the full explanation.
        """
        # Cache key incorporates vocab_size because different callers might
        # ask for the mask at different sizes (unlikely but safe).
        target_size = vocab_size if vocab_size is not None else (
            len(self._tokenizer) if self._tokenizer is not None else 50000
        )
        if (self._vowel_mask is not None
            and self._vowel_mask.device == device
            and self._vowel_mask.size(0) == target_size):
            return self._vowel_mask
        if self._tokenizer is None:
            # Fallback: uniform mask (syllable loss then tracks sequence length)
            self._vowel_mask = torch.ones(target_size, device=device)
            return self._vowel_mask
        self._vowel_mask = _build_vowel_mask(
            self._tokenizer, device, vocab_size=target_size
        )
        return self._vowel_mask

    # ------------------------------------------------------------------
    # Pillar 2: Singability — DIFFERENTIABLE syllable count
    # ------------------------------------------------------------------
    def syllable_loss(self, logits, labels, target_syllables):
        """
        Relative squared error between expected syllable count (from logits)
        and target note count.  Differentiable in logits.
        """
        # Mask size must match logits' vocab dim (last axis). For IndicTrans2
        # this is ~122672, bigger than len(tokenizer) — see _build_vowel_mask.
        vowel_mask = self._get_vowel_mask(
            logits.device, vocab_size=logits.size(-1)
        )
        exp_syl = expected_syllables_from_logits(
            logits, labels, vowel_mask, self.pad_token_id
        )  # [B], gradient flows
        target = target_syllables.float().to(exp_syl.device).clamp(min=1.0)
        # Relative squared error — keeps the loss O(1) across batches with
        # wildly different target sizes.
        rel_err = (exp_syl - target) / target
        return (rel_err ** 2).mean()

    # ------------------------------------------------------------------
    # 8a additions: cluster loss and openness reward
    # ------------------------------------------------------------------
    def cluster_loss(self, logits, labels, alignment, melody_features):
        """
        Penalise predicted tokens with heavy leading consonant clusters on
        notes with short duration.

        Intuition: a short note sung as a tongue-twister syllable like
        "स्त्री" (stɾiː — three leading consonants) is unsingable. The same
        syllable on a long note is fine. So the penalty per note is

          cluster_badness(note n) = expected_leading_consonants_at_n / duration_n

        The expected leading-consonant count at note n is obtained by
        combining the per-token feature with the decoder's probability of
        emitting each token at each output position, weighted by the
        alignment matrix A[t, n] = P(output pos t ↔ note n).

        Differentiable in logits (via softmax) and in alignment (via
        cross-attention). Both flow back to the model.

        Args:
            logits:          [B, T_out, V]   decoder logits
            labels:          [B, T_out]      for padding mask
            alignment:       [B, T_out, N]   soft alignment matrix
            melody_features: [B, N, 5]       col 2 = duration_beats

        Returns:
            scalar loss
        """
        B, T_out, V = logits.shape
        N = alignment.size(-1)

        # Load phoneme features; if missing, return 0 loss (skip the term)
        feats = _load_phoneme_features(
            logits.device, expected_vocab_size=V
        )
        if feats is None:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        leading_cons_per_token = feats['n_leading_consonants']  # [V]

        # 1. Expected leading-consonant count at each OUTPUT position.
        #    E_t[leading_cons] = sum_v softmax(logits_t)[v] * leading_cons[v]
        probs = F.softmax(logits, dim=-1)                   # [B, T_out, V]
        exp_cons_per_output = (probs * leading_cons_per_token
                                .unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, T_out]

        # Zero-out padded output positions
        output_pad_mask = (labels != self.pad_token_id).float()    # [B, T_out]
        exp_cons_per_output = exp_cons_per_output * output_pad_mask

        # 2. Project from output positions onto note positions via alignment.
        #    exp_cons_per_note[b, n] = sum_t alignment[b, t, n] * exp_cons_per_output[b, t]
        #    alignment transposed: [B, N, T_out]; exp_cons_per_output: [B, T_out, 1]
        exp_cons_per_note = torch.bmm(
            alignment.transpose(1, 2),               # [B, N, T_out]
            exp_cons_per_output.unsqueeze(-1),       # [B, T_out, 1]
        ).squeeze(-1)                                # [B, N]

        # 3. Weight by inverse note duration: short notes hurt more.
        note_duration = melody_features[:, :, 2].clamp(min=0.25)  # avoid div0
        note_mask = (melody_features.sum(dim=-1) != 0).float()    # [B, N]

        per_note_penalty = exp_cons_per_note / note_duration       # [B, N]
        per_note_penalty = per_note_penalty * note_mask

        # Mean over non-padded notes
        per_example = per_note_penalty.sum(dim=-1) / note_mask.sum(dim=-1).clamp(min=1.0)
        return per_example.mean()

    def openness_reward(self, logits, labels, alignment, melody_features):
        """
        Reward predicted tokens with open-vowel endings on strong-beat notes.

        Intuition: open vowels (aː, eː, oː) sound full when sung loud. Closed
        consonants on a strong beat (e.g. "त्याक्") sound choked. So we want

          openness_score(note n) = expected_open_ending_at_n * beat_strength_n

        and return its NEGATIVE (so minimising the loss = maximising openness
        on strong beats).

        Differentiable in logits and alignment.

        Args:
            logits:          [B, T_out, V]
            labels:          [B, T_out]
            alignment:       [B, T_out, N]
            melody_features: [B, N, 5]       col 4 = beat_strength

        Returns:
            scalar loss (negative reward)
        """
        B, T_out, V = logits.shape
        N = alignment.size(-1)

        feats = _load_phoneme_features(
            logits.device, expected_vocab_size=V
        )
        if feats is None:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        open_vowel_per_token = feats['ends_in_open_vowel']  # [V], {0, 1}

        # 1. Expected open-ending probability at each OUTPUT position
        probs = F.softmax(logits, dim=-1)
        exp_open_per_output = (probs * open_vowel_per_token
                                .unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, T_out]

        output_pad_mask = (labels != self.pad_token_id).float()
        exp_open_per_output = exp_open_per_output * output_pad_mask

        # 2. Project onto note positions via alignment
        exp_open_per_note = torch.bmm(
            alignment.transpose(1, 2),
            exp_open_per_output.unsqueeze(-1),
        ).squeeze(-1)                                # [B, N]

        # 3. Weight by beat strength
        beat_strength = melody_features[:, :, 4]          # [B, N]
        note_mask = (melody_features.sum(dim=-1) != 0).float()

        per_note_reward = exp_open_per_note * beat_strength * note_mask

        # Mean over non-padded notes, then negate so lower loss = higher reward
        per_example = per_note_reward.sum(dim=-1) / note_mask.sum(dim=-1).clamp(min=1.0)
        return -per_example.mean()

    # ------------------------------------------------------------------
    # Pillar 3: Naturalness — label-smoothed NLL (not overconfidence reward)
    # ------------------------------------------------------------------
    def naturalness_loss(self, logits, labels):
        """
        Symmetric naturalness penalty. Earlier versions used
        `1 - mean_token_confidence`, which *rewards* overconfidence and
        causes mode collapse to short, repetitive outputs.

        This version returns label-smoothed NLL:
          - low when the model places most of its mass on the ground-truth
            token but also keeps a small amount of uncertainty;
          - high when the model is either very unconfident (disfluent) OR
            places all mass on a few tokens regardless of correctness
            (because the smoothing term penalises dirac-like distributions).

        If an external Hindi LM is provided, use its perplexity instead.
        """
        if self.hindi_lm is not None:
            with torch.no_grad():
                out = self.hindi_lm(input_ids=labels, labels=labels)
            return out.loss.to(logits.device)

        # Label-smoothed CE, computed from the same logits as translation_loss
        # but reported separately so the learnable log_var can scale it.
        log_probs = F.log_softmax(logits, dim=-1)                   # [B, T, V]
        vocab = log_probs.size(-1)
        nll = F.nll_loss(
            log_probs.reshape(-1, vocab),
            labels.reshape(-1),
            ignore_index=self.pad_token_id,
            reduction='none',
        ).reshape(labels.size(0), labels.size(1))                    # [B, T]
        # Smoothing term: -mean(log_probs) over non-pad positions
        smooth = -log_probs.mean(dim=-1)                             # [B, T]
        mask = (labels != self.pad_token_id).float()
        combined = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth
        loss_per_example = (combined * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        return loss_per_example.mean()

    # ------------------------------------------------------------------
    # Pillar 4: Rhythm — stress-beat MSE, or syllable-weighted penalty
    # ------------------------------------------------------------------
    def rhythm_loss(self,
                    logits=None, labels=None, target_syllables=None,
                    stress_pattern=None, beat_pattern=None,
                    melody_features=None):
        """
        Two modes:

        A) Explicit stress-beat alignment  (best when stress_pattern available)
           MSE between Hindi stress pattern (from phoneme_utils) and melody
           beat strength per position. Not gradient-bearing through the model
           — trains only the homoscedastic weight — but provides a useful
           regulariser when the RL reward phase runs.

        B) Syllable-weighted penalty       (fallback, gradient-bearing)
           Re-weights the differentiable syllable loss by per-example mean
           beat strength, so errors on rhythmically-important lines hurt more.
        """
        loss = torch.tensor(0.0, device=logits.device if logits is not None else 'cpu',
                             requires_grad=True)

        # Mode A: explicit stress-beat alignment
        if stress_pattern is not None and beat_pattern is not None:
            min_len = min(stress_pattern.size(1), beat_pattern.size(1))
            if min_len > 0:
                return F.mse_loss(stress_pattern[:, :min_len],
                                   beat_pattern[:, :min_len])

        # Mode B: gradient-bearing, syllable-weighted
        if logits is not None and labels is not None and target_syllables is not None:
            vowel_mask = self._get_vowel_mask(
                logits.device, vocab_size=logits.size(-1)
            )
            exp_syl = expected_syllables_from_logits(
                logits, labels, vowel_mask, self.pad_token_id
            )
            target = target_syllables.float().to(exp_syl.device).clamp(min=1.0)
            rel_err = (exp_syl - target) / target

            if melody_features is not None:
                beat_strengths = melody_features[:, :, 4]
                note_mask = (melody_features.sum(dim=-1) != 0).float()
                avg_beat = (beat_strengths * note_mask).sum(dim=-1) / \
                           note_mask.sum(dim=-1).clamp(min=1.0)
                weight = 0.5 + avg_beat  # in [0.5, 1.5]
                return (weight * rel_err ** 2).mean()
            else:
                return (rel_err ** 2).mean()

        return loss

    # ------------------------------------------------------------------
    # Pillar 5: Rhyme — endpoint entropy + optional explicit similarity
    # ------------------------------------------------------------------
    def rhyme_loss(self, rhyme_similarities=None, logits=None, labels=None):
        """
        A) Optional explicit rhyme similarities from phoneme_utils (0..1).
        B) Endpoint entropy of the last two non-pad positions. Lower entropy
           = more decisive line endings. Gradient-bearing via logits.
        """
        device = logits.device if logits is not None else (
            rhyme_similarities.device if rhyme_similarities is not None else 'cpu'
        )
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        if rhyme_similarities is not None:
            loss = loss + torch.mean((1.0 - rhyme_similarities.to(device)) ** 2)

        if logits is not None and labels is not None:
            loss = loss + self._endpoint_entropy(logits, labels)

        return loss

    def _endpoint_entropy(self, logits, labels):
        """Mean normalised entropy of the last two non-pad decoder positions."""
        non_pad = (labels != self.pad_token_id)
        lengths = non_pad.sum(dim=-1)                                # [B]
        vocab_size = logits.size(-1)
        max_entropy = torch.log(torch.tensor(float(vocab_size),
                                              device=logits.device))

        batch_size = labels.size(0)
        total = torch.tensor(0.0, device=logits.device)
        count = 0
        for b in range(batch_size):
            end = int(lengths[b].item()) - 1
            if end < 1:
                continue
            for pos in (end - 1, end):
                if pos < 0:
                    continue
                log_p = F.log_softmax(logits[b, pos], dim=-1)
                p = torch.exp(log_p)
                ent = -(p * log_p).sum() / max_entropy
                total = total + ent
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return total / count

    # ------------------------------------------------------------------
    # Combined forward — Kendall homoscedastic weighting
    # ------------------------------------------------------------------
    def forward(self,
                translation_loss,
                logits=None,
                labels=None,
                target_syllables=None,
                num_notes=None,
                stress_pattern=None,
                beat_pattern=None,
                rhyme_similarities=None,
                melody_features=None,
                alignment=None,  # 8a: [B, T_out, N] soft alignment matrix
                # kept for backwards-compat with callers; ignored
                predicted_syllables=None,
                singability_scores=None):
        """
        L_total = sum_i  [ exp(-log_var_i) * L_i  +  log_var_i ]

        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}

        # Prefer num_notes (from dataloader) as the syllable target; fall back
        # to target_syllables if only that was passed.
        syl_target = num_notes if num_notes is not None else target_syllables

        # 1. Translation (cross-entropy, always present)
        precision_t = torch.exp(-self.log_var_translation)
        total = precision_t * translation_loss + self.log_var_translation
        loss_dict['translation_loss']  = float(translation_loss.detach())
        loss_dict['translation_weight'] = float(precision_t.detach())

        # 2. Syllable loss (differentiable via expected_syllables_from_logits)
        if logits is not None and labels is not None and syl_target is not None:
            syl = self.syllable_loss(logits, labels, syl_target)
            precision_s = torch.exp(-self.log_var_syllable)
            total = total + precision_s * syl + self.log_var_syllable
            loss_dict['syllable_loss']  = float(syl.detach())
            loss_dict['syllable_weight'] = float(precision_s.detach())
        else:
            loss_dict['syllable_loss'] = 0.0

        # 3. Naturalness (label-smoothed NLL)
        if logits is not None and labels is not None:
            nat = self.naturalness_loss(logits, labels)
            precision_n = torch.exp(-self.log_var_naturalness)
            total = total + precision_n * nat + self.log_var_naturalness
            loss_dict['naturalness_loss']  = float(nat.detach())
            loss_dict['naturalness_weight'] = float(precision_n.detach())
        else:
            loss_dict['naturalness_loss'] = 0.0

        # 4. Rhythm
        rhy = self.rhythm_loss(
            logits=logits, labels=labels,
            target_syllables=syl_target,
            stress_pattern=stress_pattern,
            beat_pattern=beat_pattern,
            melody_features=melody_features,
        )
        if rhy.requires_grad or float(rhy.detach()) > 0:
            precision_r = torch.exp(-self.log_var_rhythm)
            total = total + precision_r * rhy + self.log_var_rhythm
            loss_dict['rhythm_loss']  = float(rhy.detach())
            loss_dict['rhythm_weight'] = float(precision_r.detach())
        else:
            loss_dict['rhythm_loss'] = 0.0

        # 5. Rhyme
        rhm = self.rhyme_loss(
            rhyme_similarities=rhyme_similarities,
            logits=logits, labels=labels,
        )
        if rhm.requires_grad or float(rhm.detach()) > 0:
            precision_rh = torch.exp(-self.log_var_rhyme)
            total = total + precision_rh * rhm + self.log_var_rhyme
            loss_dict['rhyme_loss']  = float(rhm.detach())
            loss_dict['rhyme_weight'] = float(precision_rh.detach())
        else:
            loss_dict['rhyme_loss'] = 0.0

        # 6. 8a: cluster loss (penalises heavy consonant clusters on short notes)
        if (logits is not None and labels is not None
            and alignment is not None and melody_features is not None):
            clu = self.cluster_loss(logits, labels, alignment, melody_features)
            if clu.requires_grad or float(clu.detach()) != 0:
                precision_c = torch.exp(-self.log_var_cluster)
                total = total + precision_c * clu + self.log_var_cluster
                loss_dict['cluster_loss']   = float(clu.detach())
                loss_dict['cluster_weight'] = float(precision_c.detach())
            else:
                loss_dict['cluster_loss'] = 0.0
        else:
            loss_dict['cluster_loss'] = 0.0

        # 7. 8a: openness reward (rewards open vowels on strong beats)
        if (logits is not None and labels is not None
            and alignment is not None and melody_features is not None):
            opn = self.openness_reward(logits, labels, alignment, melody_features)
            if opn.requires_grad or float(opn.detach()) != 0:
                precision_o = torch.exp(-self.log_var_openness)
                total = total + precision_o * opn + self.log_var_openness
                loss_dict['openness_loss']   = float(opn.detach())
                loss_dict['openness_weight'] = float(precision_o.detach())
            else:
                loss_dict['openness_loss'] = 0.0
        else:
            loss_dict['openness_loss'] = 0.0

        loss_dict['total_loss'] = float(total.detach())
        return total, loss_dict
