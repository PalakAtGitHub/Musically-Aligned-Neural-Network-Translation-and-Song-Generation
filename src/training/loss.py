"""
Pentathlon Loss for MCNST

Based on Peter Low's Pentathlon Principle (2005), fully activated:
  1. Sense        — translation quality (cross-entropy from mBART)
  2. Singability  — syllable count matching + phonetic singability
  3. Naturalness  — vowel-to-consonant ratio for singing ease
  4. Rhythm       — stress-beat alignment
  5. Rhyme        — phonetic ending similarity across line pairs

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
      - Syllable count (always on)
      - Rhythm (on when melody features provided)
      - Singability (on when Hindi text provided)
      - Rhyme (on when multiple lines provided)
    """
    
    def __init__(self):
        super().__init__()
        
        # Learnable log-variance for each loss component
        # log_var = 0 → weight ≈ 1.0 initially
        self.log_var_translation = nn.Parameter(torch.tensor(0.0))
        self.log_var_syllable = nn.Parameter(torch.tensor(0.0))
        self.log_var_rhythm = nn.Parameter(torch.tensor(0.0))
        self.log_var_singability = nn.Parameter(torch.tensor(0.0))
        self.log_var_rhyme = nn.Parameter(torch.tensor(-1.0))  # start lower (less important initially)
    
    def syllable_loss(self, predicted_syllables, target_syllables):
        """Quadratic penalty for syllable count mismatch."""
        diff = predicted_syllables - target_syllables
        return torch.mean(diff ** 2)
    
    def rhythm_loss(self, stress_pattern, beat_pattern):
        """
        Penalize when stressed syllables land on weak beats.
        
        Args:
            stress_pattern: [batch, max_syl] — stress per syllable (0.0-1.0)
            beat_pattern:   [batch, max_notes] — beat_strength per note (0.0-1.0)
        
        Returns:
            scalar loss
        """
        # Align lengths (truncate to shorter)
        min_len = min(stress_pattern.size(1), beat_pattern.size(1))
        stress = stress_pattern[:, :min_len]
        beats = beat_pattern[:, :min_len]
        
        # MSE between stress and beat patterns
        return F.mse_loss(stress, beats)
    
    def singability_loss(self, singability_scores):
        """
        Penalize low singability (high consonant clusters, low vowel ratio).
        
        Args:
            singability_scores: [batch] — score from 0.0 (hard) to 1.0 (easy)
        
        Returns:
            scalar loss (lower singability = higher loss)
        """
        # Target: singability should be high (1.0)
        # Loss = (1 - score)^2
        return torch.mean((1.0 - singability_scores) ** 2)
    
    def rhyme_loss(self, rhyme_similarities):
        """
        Reward rhyme preservation.
        
        Args:
            rhyme_similarities: [batch] — rhyme similarity scores (0.0-1.0)
        
        Returns:
            scalar loss (low similarity = high loss)
        """
        # Target: high rhyme similarity (1.0)
        return torch.mean((1.0 - rhyme_similarities) ** 2)
    
    def forward(self,
                translation_loss,
                predicted_syllables=None,
                target_syllables=None,
                stress_pattern=None,
                beat_pattern=None,
                singability_scores=None,
                rhyme_similarities=None):
        """
        Compute total loss with learned weighting.
        
        L_total = Σ (1/2σ²_i) * L_i + log(σ_i)
        
        Args:
            translation_loss:    scalar — cross-entropy from mBART
            predicted_syllables: [batch] — syllable counts in predictions
            target_syllables:    [batch] — target syllable counts
            stress_pattern:      [batch, max_syl] — Hindi stress per syllable
            beat_pattern:        [batch, max_notes] — beat strength per note
            singability_scores:  [batch] — phonetic singability score (0-1)
            rhyme_similarities:  [batch] — rhyme similarity score (0-1)
        
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
        
        # 3. RHYTHM LOSS
        if stress_pattern is not None and beat_pattern is not None:
            rhy_loss = self.rhythm_loss(stress_pattern, beat_pattern)
            precision_r = torch.exp(-self.log_var_rhythm)
            total_loss = total_loss + precision_r * rhy_loss + self.log_var_rhythm
            loss_dict['rhythm_loss'] = rhy_loss.item()
            loss_dict['rhythm_weight'] = precision_r.item()
        else:
            loss_dict['rhythm_loss'] = 0.0
        
        # 4. SINGABILITY LOSS
        if singability_scores is not None:
            sing_loss = self.singability_loss(singability_scores)
            precision_sg = torch.exp(-self.log_var_singability)
            total_loss = total_loss + precision_sg * sing_loss + self.log_var_singability
            loss_dict['singability_loss'] = sing_loss.item()
            loss_dict['singability_weight'] = precision_sg.item()
        else:
            loss_dict['singability_loss'] = 0.0
        
        # 5. RHYME LOSS
        if rhyme_similarities is not None:
            rhy_sim_loss = self.rhyme_loss(rhyme_similarities)
            precision_rh = torch.exp(-self.log_var_rhyme)
            total_loss = total_loss + precision_rh * rhy_sim_loss + self.log_var_rhyme
            loss_dict['rhyme_loss'] = rhy_sim_loss.item()
            loss_dict['rhyme_weight'] = precision_rh.item()
        else:
            loss_dict['rhyme_loss'] = 0.0
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict