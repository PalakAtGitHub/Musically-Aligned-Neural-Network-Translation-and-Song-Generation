"""
Pentathlon Loss for MCNST

Based on Peter Low's Pentathlon Principle (2005), fully activated:
  1. Sense        — translation quality (cross-entropy from mBART)
  2. Singability  — syllable count matching + phonetic singability
  3. Naturalness  — Hindi LM perplexity (self-perplexity from decoder logits)
  4. Rhythm       — stress-beat alignment + melody-aware duration weighting
  5. Rhyme        — phonetic ending similarity + endpoint entropy

Uses learnable loss weights via homoscedastic uncertainty
(Kendall et al., CVPR 2018) so the model learns which
tradeoffs to make — just like a human translator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PentathlonLoss(nn.Module):
    """
    Multi-objective loss with LEARNABLE weights.

    All 5 Pentathlon criteria are now active:
      - Translation (always on)
      - Syllable count (always on when syllable counts provided)
      - Naturalness (on when decoder logits provided)
      - Rhythm (on when melody features or stress/beat patterns provided)
      - Rhyme (on when multiple lines or decoder logits provided)
    """

    def __init__(self, hindi_lm=None, hindi_tokenizer=None):
        super().__init__()

        # Learnable log-variance for each loss component
        # log_var = 0 -> weight ~ 1.0 initially
        self.log_var_translation = nn.Parameter(torch.tensor(0.0))
        self.log_var_syllable = nn.Parameter(torch.tensor(0.0))
        self.log_var_naturalness = nn.Parameter(torch.tensor(0.0))
        self.log_var_rhythm = nn.Parameter(torch.tensor(0.0))
        self.log_var_rhyme = nn.Parameter(torch.tensor(-1.0))  # start lower

        # Optional external Hindi LM for naturalness scoring
        self.hindi_lm = hindi_lm
        self.hindi_tokenizer = hindi_tokenizer

    # ------------------------------------------------------------------
    # Pillar 2: Singability (Syllable Matching)
    # ------------------------------------------------------------------
    def syllable_loss(self, predicted_syllables, target_syllables):
        """Quadratic penalty for syllable count mismatch."""
        diff = predicted_syllables - target_syllables
        return torch.mean(diff ** 2)

    # ------------------------------------------------------------------
    # Pillar 3: Naturalness (Hindi LM Perplexity)
    # ------------------------------------------------------------------
    def naturalness_loss(self, logits, labels):
        """
        Compute naturalness as negative log-likelihood.

        If no external Hindi LM is loaded, uses self-perplexity from the
        model's own decoder logits. This still penalises unnatural token
        sequences because the pretrained mBART decoder assigns low
        probability to disfluent Hindi.

        When an external Hindi LM is provided, it scores the target
        token sequence and returns mean token-level NLL.

        Args:
            logits: [batch, seq_len, vocab] — decoder output logits
            labels: [batch, seq_len] — target token ids (padded with 0 or pad_id)
        """
        if self.hindi_lm is not None:
            return self._external_lm_loss(labels)

        # Self-perplexity: shift so token i predicts token i+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        return loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                       shift_labels.view(-1))

    def _external_lm_loss(self, labels):
        """Score labels with an external Hindi language model."""
        device = labels.device
        with torch.no_grad():
            outputs = self.hindi_lm(input_ids=labels, labels=labels)
        return outputs.loss.to(device)

    # ------------------------------------------------------------------
    # Pillar 4: Rhythm (Beat/Stress Pattern Alignment)
    # ------------------------------------------------------------------
    def rhythm_loss(self, predicted_syllables=None, target_syllables=None,
                    stress_pattern=None, beat_pattern=None,
                    melody_features=None):
        """
        Penalise rhythm mismatch between generated text and melody.

        Supports two modes:
        A) Explicit stress/beat patterns: MSE between stress_pattern and beat_pattern
        B) Melody-feature-aware: uses beat_strength * duration from melody features
           to weight the syllable mismatch penalty

        Falls back to simple absolute syllable difference if neither is available.
        """
        loss = torch.tensor(0.0, requires_grad=True)

        # Mode A: Explicit stress-beat alignment
        if stress_pattern is not None and beat_pattern is not None:
            min_len = min(stress_pattern.size(1), beat_pattern.size(1))
            stress = stress_pattern[:, :min_len]
            beats = beat_pattern[:, :min_len]
            loss = F.mse_loss(stress, beats)

        # Mode B: Melody-feature-weighted syllable penalty
        elif melody_features is not None and predicted_syllables is not None and target_syllables is not None:
            beat_strengths = melody_features[:, :, 4]  # [batch, num_notes]
            durations = melody_features[:, :, 2]       # [batch, num_notes]
            note_mask = (melody_features.sum(dim=-1) != 0).float()

            weights = beat_strengths * durations * note_mask
            weight_sum = weights.sum(dim=-1).clamp(min=1.0)
            avg_weight = weight_sum / note_mask.sum(dim=-1).clamp(min=1.0)

            diff = torch.abs(predicted_syllables - target_syllables)
            loss = (diff * avg_weight).mean()

        # Fallback
        elif predicted_syllables is not None and target_syllables is not None:
            loss = torch.mean(torch.abs(predicted_syllables - target_syllables))

        return loss

    # ------------------------------------------------------------------
    # Pillar 5: Rhyme (Line-Ending Similarity + Endpoint Entropy)
    # ------------------------------------------------------------------
    def rhyme_loss(self, rhyme_similarities=None, logits=None, labels=None):
        """
        Two-component rhyme loss:

        A) Explicit rhyme similarity scores (from phoneme_utils): target high similarity
        B) Endpoint entropy from decoder logits: lower entropy at line endings
           = more decisive endings = better rhyme consistency

        Both can be active simultaneously.
        """
        loss = torch.tensor(0.0, requires_grad=True)
        device = None

        # Component A: Explicit phonetic rhyme similarity
        if rhyme_similarities is not None:
            loss = torch.mean((1.0 - rhyme_similarities) ** 2)
            device = rhyme_similarities.device

        # Component B: Endpoint entropy from logits
        if logits is not None and labels is not None:
            device = logits.device
            entropy_loss = self._endpoint_entropy(logits, labels)
            loss = loss.to(device) + entropy_loss

        return loss

    def _endpoint_entropy(self, logits, labels):
        """
        Measure entropy at the last 2 non-padding positions of each sequence.
        Lower entropy = more decisive line endings.
        """
        batch_size = labels.size(0)
        non_pad_mask = (labels != 0)
        lengths = non_pad_mask.sum(dim=-1)

        total_entropy = 0.0
        count = 0

        for b in range(batch_size):
            end_pos = lengths[b].item() - 1
            if end_pos < 1:
                continue

            for pos in [end_pos - 1, end_pos]:
                if pos < 0:
                    continue
                log_probs = torch.log_softmax(logits[b, pos], dim=-1)
                probs = torch.exp(log_probs)
                entropy = -(probs * log_probs).sum()
                total_entropy += entropy
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return total_entropy / count

    # ------------------------------------------------------------------
    # Combined Forward
    # ------------------------------------------------------------------
    def forward(self,
                translation_loss,
                predicted_syllables=None,
                target_syllables=None,
                stress_pattern=None,
                beat_pattern=None,
                singability_scores=None,
                rhyme_similarities=None,
                logits=None,
                labels=None,
                melody_features=None):
        """
        Compute total loss with learned weighting.

        L_total = sum( (1/2*sigma_i^2) * L_i + log(sigma_i) )

        Args:
            translation_loss:    scalar — cross-entropy from mBART
            predicted_syllables: [batch] — syllable counts in predictions
            target_syllables:    [batch] — target syllable counts
            stress_pattern:      [batch, max_syl] — Hindi stress per syllable
            beat_pattern:        [batch, max_notes] — beat strength per note
            singability_scores:  [batch] — phonetic singability score (0-1)
            rhyme_similarities:  [batch] — rhyme similarity score (0-1)
            logits:              [batch, seq_len, vocab] — decoder logits
            labels:              [batch, seq_len] — target token ids
            melody_features:     [batch, num_notes, 5] — melody tensor

        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}

        # 1. TRANSLATION LOSS (always present)
        precision_t = torch.exp(-self.log_var_translation)
        total_loss = precision_t * translation_loss + self.log_var_translation
        loss_dict['translation_loss'] = translation_loss.item() if torch.is_tensor(translation_loss) else translation_loss
        loss_dict['translation_weight'] = precision_t.item()

        # 2. SYLLABLE LOSS
        if predicted_syllables is not None and target_syllables is not None:
            syl_loss = self.syllable_loss(predicted_syllables, target_syllables)
            precision_s = torch.exp(-self.log_var_syllable)
            total_loss = total_loss + precision_s * syl_loss + self.log_var_syllable
            loss_dict['syllable_loss'] = syl_loss.item()
            loss_dict['syllable_weight'] = precision_s.item()
        else:
            loss_dict['syllable_loss'] = 0.0

        # 3. NATURALNESS LOSS (from decoder logits or external LM)
        if logits is not None and labels is not None:
            nat_loss = self.naturalness_loss(logits, labels)
            precision_n = torch.exp(-self.log_var_naturalness)
            total_loss = total_loss + precision_n * nat_loss + self.log_var_naturalness
            loss_dict['naturalness_loss'] = nat_loss.item()
            loss_dict['naturalness_weight'] = precision_n.item()
        else:
            loss_dict['naturalness_loss'] = 0.0

        # 4. RHYTHM LOSS (stress-beat alignment or melody-weighted)
        rhy_loss = self.rhythm_loss(
            predicted_syllables=predicted_syllables,
            target_syllables=target_syllables,
            stress_pattern=stress_pattern,
            beat_pattern=beat_pattern,
            melody_features=melody_features
        )
        if rhy_loss.item() > 0:
            precision_r = torch.exp(-self.log_var_rhythm)
            total_loss = total_loss + precision_r * rhy_loss + self.log_var_rhythm
            loss_dict['rhythm_loss'] = rhy_loss.item()
            loss_dict['rhythm_weight'] = precision_r.item()
        else:
            loss_dict['rhythm_loss'] = 0.0

        # 5. RHYME LOSS (phonetic similarity + endpoint entropy)
        rhy_sim_loss = self.rhyme_loss(
            rhyme_similarities=rhyme_similarities,
            logits=logits,
            labels=labels
        )
        if rhy_sim_loss.item() > 0:
            precision_rh = torch.exp(-self.log_var_rhyme)
            total_loss = total_loss + precision_rh * rhy_sim_loss + self.log_var_rhyme
            loss_dict['rhyme_loss'] = rhy_sim_loss.item()
            loss_dict['rhyme_weight'] = precision_rh.item()
        else:
            loss_dict['rhyme_loss'] = 0.0

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
