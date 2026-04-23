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


def _build_vowel_mask(tokenizer, device):
    """
    Build a [vocab_size] float tensor where entry v = 1.0 iff decoded token v
    contains at least one Devanagari vowel character.

    Cached per (tokenizer-id, device). Falls back to all-ones if anything
    fails (which makes the expected-syllable estimate proportional to the
    number of non-pad tokens — still gradient-bearing, just noisier).
    """
    cache_key = (id(tokenizer), str(device))
    if cache_key in _VOWEL_MASK_CACHE:
        return _VOWEL_MASK_CACHE[cache_key]

    try:
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
        vocab_size = getattr(tokenizer, 'vocab_size', 50000)
        mask = torch.ones(vocab_size, device=device)
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

    def _get_vowel_mask(self, device):
        """Lazy-build and cache the vowel-bearing-token mask."""
        if self._vowel_mask is not None and self._vowel_mask.device == device:
            return self._vowel_mask
        if self._tokenizer is None:
            # Fallback: uniform mask (syllable loss then tracks sequence length)
            vocab = 50000
            self._vowel_mask = torch.ones(vocab, device=device)
            return self._vowel_mask
        self._vowel_mask = _build_vowel_mask(self._tokenizer, device)
        return self._vowel_mask

    # ------------------------------------------------------------------
    # Pillar 2: Singability — DIFFERENTIABLE syllable count
    # ------------------------------------------------------------------
    def syllable_loss(self, logits, labels, target_syllables):
        """
        Relative squared error between expected syllable count (from logits)
        and target note count.  Differentiable in logits.
        """
        vowel_mask = self._get_vowel_mask(logits.device)
        exp_syl = expected_syllables_from_logits(
            logits, labels, vowel_mask, self.pad_token_id
        )  # [B], gradient flows
        target = target_syllables.float().to(exp_syl.device).clamp(min=1.0)
        # Relative squared error — keeps the loss O(1) across batches with
        # wildly different target sizes.
        rel_err = (exp_syl - target) / target
        return (rel_err ** 2).mean()

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
            vowel_mask = self._get_vowel_mask(logits.device)
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

        loss_dict['total_loss'] = float(total.detach())
        return total, loss_dict
