"""
Pentathlon Reward for RL Fine-Tuning

Scores generated Hindi translations against melody constraints.
Used as the reward signal in Minimum Risk Training — no gradients needed.

Each component returns a score in [0, 1] where 1 = perfect.
The combined reward is a weighted sum of all components.

This is the RL counterpart to PentathlonLoss.  The loss operates on
decoder logits (differentiable); this reward operates on decoded
Hindi strings (not differentiable) — which is precisely what makes
the RL phase valuable:  it lets the model train against criteria
(true syllable count, phonetic rhyme, consonant-cluster singability)
that cannot be expressed as differentiable loss terms.
"""

from src.utils.syllable_utils import count_hindi_syllables
from src.utils.phoneme_utils import (
    consonant_cluster_score,
    vowel_ratio,
    rhyme_similarity,
)


class PentathlonReward:
    """
    Score a generated Hindi translation on 4 singability criteria.

    Components (all in [0, 1], higher = better):
      1. Syllable match  —  1 - |hindi_syllables - num_notes| / num_notes
      2. Naturalness     —  consonant cluster friendliness + vowel ratio
      3. Rhythm          —  beat-weighted syllable deviation
      4. Rhyme           —  phonetic ending similarity to previous line
    """

    def __init__(self,
                 w_syllable=0.40,
                 w_naturalness=0.25,
                 w_rhythm=0.20,
                 w_rhyme=0.15):
        self.w_syl = w_syllable
        self.w_nat = w_naturalness
        self.w_rhy = w_rhythm
        self.w_rhm = w_rhyme

    # ------------------------------------------------------------------
    # Individual components
    # ------------------------------------------------------------------

    def syllable_reward(self, hindi_text: str, num_notes: int) -> float:
        """1 if syllable count matches num_notes exactly, decays linearly."""
        syl = count_hindi_syllables(hindi_text)
        if num_notes <= 0:
            return 1.0 if syl == 0 else 0.0
        return max(0.0, 1.0 - abs(syl - num_notes) / num_notes)

    def naturalness_reward(self, hindi_text: str) -> float:
        """
        Phonetic singability: fewer hard consonant clusters and more
        vowels make lyrics easier to sing.
        """
        if not hindi_text.strip():
            return 0.0
        cc = consonant_cluster_score(hindi_text)   # 0=harsh, 1=smooth
        vr = vowel_ratio(hindi_text)               # fraction of vowels
        return 0.6 * cc + 0.4 * min(vr * 2, 1.0)   # cap vowel contribution

    def rhythm_reward(self, hindi_text: str, num_notes: int,
                      melody_features=None) -> float:
        """
        Beat-weighted syllable deviation.

        If melody_features is provided, weighs the deviation by average
        beat_strength * duration so mismatches on strong beats hurt more.
        """
        syl = count_hindi_syllables(hindi_text)
        if num_notes <= 0:
            return 1.0

        raw_dev = abs(syl - num_notes) / num_notes

        if melody_features is not None and len(melody_features) > 0:
            # melody_features: [num_notes, 5]  col 4 = beat_strength, col 2 = duration
            beats = melody_features[:, 4]
            durs = melody_features[:, 2]
            importance = float((beats * durs).mean())
            raw_dev = raw_dev * (0.5 + importance)

        return max(0.0, 1.0 - raw_dev)

    def rhyme_reward(self, hindi_text: str, prev_line: str = None) -> float:
        """Phonetic ending similarity with the previous line."""
        if prev_line is None or not prev_line.strip() or not hindi_text.strip():
            return 0.5  # neutral when no comparison available
        return rhyme_similarity(hindi_text, prev_line)

    # ------------------------------------------------------------------
    # Combined reward
    # ------------------------------------------------------------------

    def score(self, hindi_text: str, num_notes: int,
              melody_features=None, prev_line: str = None) -> dict:
        """
        Compute full Pentathlon reward.

        Returns:
            dict with 'total' in [0,1] and per-component scores.
        """
        syl = self.syllable_reward(hindi_text, num_notes)
        nat = self.naturalness_reward(hindi_text)
        rhy = self.rhythm_reward(hindi_text, num_notes, melody_features)
        rhm = self.rhyme_reward(hindi_text, prev_line)

        total = (self.w_syl * syl +
                 self.w_nat * nat +
                 self.w_rhy * rhy +
                 self.w_rhm * rhm)

        return {
            'total': total,
            'syllable': syl,
            'naturalness': nat,
            'rhythm': rhy,
            'rhyme': rhm,
        }
