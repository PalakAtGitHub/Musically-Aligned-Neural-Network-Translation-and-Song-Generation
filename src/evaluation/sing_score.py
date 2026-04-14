"""
SingScore — Automated Singability Evaluation Metric

A novel composite metric for evaluating song translations.
No standard metric exists for "is this translation singable?" — 
this fills that gap.

Components:
  1. Syllable Accuracy  — syllable count vs note count
  2. Rhythm Alignment   — stressed syllables on strong beats
  3. Phonetic Fluency   — ease of articulation while singing
  4. Rhyme Preservation — original rhyme patterns maintained
  5. Semantic Fidelity  — meaning preservation (BERTScore)

Final SingScore = weighted combination of all 5.
"""

from typing import List, Dict, Optional
from src.utils.syllable_utils import count_hindi_syllables, count_english_syllables
from src.utils.phoneme_utils import (
    get_stress_pattern,
    consonant_cluster_score,
    vowel_ratio,
    rhyme_similarity
)


class SingScore:
    """
    Automated singability evaluation for song translations.
    
    Usage:
        scorer = SingScore()
        result = scorer.score_line(
            hindi_text="चमक चमक छोटा तारा",
            target_notes=7,
            beat_strengths=[1.0, 0.3, 0.3, 1.0, 0.3, 1.0, 0.3]
        )
        print(result['sing_score'])  # 0.0 - 1.0
    """
    
    def __init__(self,
                 weight_syllable=0.30,
                 weight_rhythm=0.25,
                 weight_fluency=0.20,
                 weight_rhyme=0.10,
                 weight_semantic=0.15,
                 syllable_tolerance=2):
        """
        Args:
            weight_*: Importance of each component (must sum to 1.0)
            syllable_tolerance: How many syllables off is still "perfect"
        """
        self.weights = {
            'syllable': weight_syllable,
            'rhythm': weight_rhythm,
            'fluency': weight_fluency,
            'rhyme': weight_rhyme,
            'semantic': weight_semantic,
        }
        self.tolerance = syllable_tolerance
    
    def syllable_score(self, hindi_text: str, target_notes: int) -> float:
        """
        Score: does the syllable count match the note count?
        
        1.0 = exact match or within tolerance
        0.0 = way off (>= 2x tolerance away)
        """
        syl_count = count_hindi_syllables(hindi_text)
        diff = abs(syl_count - target_notes)
        
        if diff <= self.tolerance:
            return 1.0
        elif diff <= self.tolerance * 2:
            # Linear decay
            return 1.0 - (diff - self.tolerance) / (self.tolerance * 2)
        else:
            return 0.0
    
    def rhythm_score(self, hindi_text: str, beat_strengths: List[float]) -> float:
        """
        Score: do stressed Hindi syllables land on strong beats?
        
        1.0 = perfect alignment
        0.0 = completely misaligned
        """
        stress = get_stress_pattern(hindi_text)
        
        if not stress or not beat_strengths:
            return 0.5  # neutral if we can't compute
        
        # Align lengths
        min_len = min(len(stress), len(beat_strengths))
        stress = stress[:min_len]
        beats = beat_strengths[:min_len]
        
        # Correlation between stress and beat strength
        # High stress on high beats = good
        if min_len == 0:
            return 0.5
        
        alignment = 0.0
        for s, b in zip(stress, beats):
            # Both high or both low = good alignment
            alignment += 1.0 - abs(s - b)
        
        return alignment / min_len
    
    def fluency_score(self, hindi_text: str) -> float:
        """
        Score: how easy is this text to sing?
        
        Combines consonant cluster analysis and vowel ratio.
        """
        cluster = consonant_cluster_score(hindi_text)
        v_ratio = vowel_ratio(hindi_text)
        
        # Ideal vowel ratio for singing ~0.45-0.55
        v_score = 1.0 - abs(v_ratio - 0.50) * 2
        v_score = max(0.0, min(1.0, v_score))
        
        # Weighted combination
        return 0.6 * cluster + 0.4 * v_score
    
    def rhyme_score(self, hindi_lines: List[str],
                     english_lines: Optional[List[str]] = None) -> float:
        """
        Score: are rhyme patterns preserved from English to Hindi?
        
        If English lines 1&2 rhyme, do Hindi lines 1&2 also rhyme?
        """
        if len(hindi_lines) < 2:
            return 1.0  # can't evaluate rhyme on single line
        
        # Check consecutive pairs
        scores = []
        for i in range(0, len(hindi_lines) - 1, 2):
            sim = rhyme_similarity(hindi_lines[i], hindi_lines[i + 1])
            scores.append(sim)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def semantic_score(self, hindi_text: str, english_text: str) -> float:
        """
        Score: is the meaning preserved?
        
        Uses BERTScore if available, falls back to simple token overlap.
        """
        try:
            from bert_score import score as bert_score
            P, R, F1 = bert_score(
                [hindi_text], [english_text],
                lang='hi',
                verbose=False
            )
            return F1.item()
        except ImportError:
            # Fallback: simple length ratio (crude proxy)
            hi_len = len(hindi_text.split())
            en_len = len(english_text.split())
            if en_len == 0:
                return 0.0
            ratio = min(hi_len, en_len) / max(hi_len, en_len)
            return ratio
    
    def score_line(self, hindi_text: str, target_notes: int,
                    beat_strengths: Optional[List[float]] = None,
                    english_text: Optional[str] = None) -> Dict:
        """
        Compute SingScore for a single translated line.
        
        Args:
            hindi_text:     Translated Hindi text
            target_notes:   Number of melody notes for this line
            beat_strengths: Beat strength for each note [0.0-1.0]
            english_text:   Original English text (for semantic score)
        
        Returns:
            Dict with individual scores and final SingScore
        """
        scores = {}
        
        # 1. Syllable Accuracy
        scores['syllable'] = self.syllable_score(hindi_text, target_notes)
        
        # 2. Rhythm Alignment
        if beat_strengths:
            scores['rhythm'] = self.rhythm_score(hindi_text, beat_strengths)
        else:
            scores['rhythm'] = 0.5  # neutral
        
        # 3. Phonetic Fluency
        scores['fluency'] = self.fluency_score(hindi_text)
        
        # 4. Rhyme (needs multiple lines — scored separately)
        scores['rhyme'] = 0.5  # neutral for single line
        
        # 5. Semantic Fidelity
        if english_text:
            scores['semantic'] = self.semantic_score(hindi_text, english_text)
        else:
            scores['semantic'] = 0.5
        
        # Weighted total
        sing_score = sum(
            self.weights[k] * scores[k] for k in self.weights
        )
        
        scores['sing_score'] = sing_score
        scores['hindi_syllables'] = count_hindi_syllables(hindi_text)
        scores['target_notes'] = target_notes
        
        return scores
    
    def score_song(self, hindi_lines: List[str],
                    english_lines: List[str],
                    notes_per_line: List[int],
                    beats_per_line: Optional[List[List[float]]] = None) -> Dict:
        """
        Compute SingScore for an entire song translation.
        
        Returns:
            Dict with per-line scores and overall SingScore
        """
        line_scores = []
        
        for i, (hi, en, notes) in enumerate(zip(hindi_lines, english_lines, notes_per_line)):
            beats = beats_per_line[i] if beats_per_line else None
            score = self.score_line(hi, notes, beats, en)
            line_scores.append(score)
        
        # Compute rhyme score across lines
        rhyme = self.rhyme_score(hindi_lines, english_lines)
        
        # Average per-line scores
        avg_scores = {}
        for key in ['syllable', 'rhythm', 'fluency', 'semantic']:
            avg_scores[key] = sum(s[key] for s in line_scores) / len(line_scores)
        avg_scores['rhyme'] = rhyme
        
        # Recompute weighted total with actual rhyme score
        avg_scores['sing_score'] = sum(
            self.weights[k] * avg_scores[k] for k in self.weights
        )
        
        return {
            'overall': avg_scores,
            'per_line': line_scores,
            'num_lines': len(hindi_lines)
        }


# ============================================================================
# PRETTY PRINTER
# ============================================================================

def print_sing_score(result: Dict, title: str = "SingScore Report"):
    """Pretty-print a SingScore result."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    if 'overall' in result:
        # Song-level result
        overall = result['overall']
        print(f"\n  Overall SingScore: {overall['sing_score']:.2f} / 1.00")
        print(f"  ─────────────────────────────────")
        print(f"  Syllable Accuracy:  {overall['syllable']:.2f}")
        print(f"  Rhythm Alignment:   {overall['rhythm']:.2f}")
        print(f"  Phonetic Fluency:   {overall['fluency']:.2f}")
        print(f"  Rhyme Preservation: {overall['rhyme']:.2f}")
        print(f"  Semantic Fidelity:  {overall['semantic']:.2f}")
        
        print(f"\n  Per-line breakdown:")
        for i, line in enumerate(result['per_line']):
            syl_match = "✓" if line['syllable'] >= 0.8 else "✗"
            print(f"    Line {i+1}: {line['sing_score']:.2f}  "
                  f"(syl: {line['hindi_syllables']}/{line['target_notes']} {syl_match})")
    else:
        # Single line result
        print(f"\n  SingScore: {result['sing_score']:.2f} / 1.00")
        print(f"  ─────────────────────────────────")
        print(f"  Syllable:  {result['syllable']:.2f}  ({result['hindi_syllables']}/{result['target_notes']} notes)")
        print(f"  Rhythm:    {result['rhythm']:.2f}")
        print(f"  Fluency:   {result['fluency']:.2f}")
        print(f"  Semantic:  {result['semantic']:.2f}")
    
    print(f"{'='*60}\n")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    scorer = SingScore()
    
    # Test single line
    result = scorer.score_line(
        hindi_text="चमक चमक छोटा तारा",
        target_notes=7,
        beat_strengths=[1.0, 0.3, 0.3, 1.0, 0.3, 1.0, 0.3],
        english_text="Twinkle twinkle little star"
    )
    print_sing_score(result, "Single Line Test")
    
    # Test full song
    song_result = scorer.score_song(
        hindi_lines=[
            "चमक चमक छोटा तारा",
            "कैसे अचरज मैं करूँ",
            "ऊपर आसमान में दूर",
            "चमके तू जैसे हीरा",
        ],
        english_lines=[
            "Twinkle twinkle little star",
            "How I wonder what you are",
            "Up above the world so high",
            "Like a diamond in the sky",
        ],
        notes_per_line=[7, 7, 7, 7],
        beats_per_line=[
            [1.0, 0.3, 0.3, 1.0, 0.3, 1.0, 0.3],
            [1.0, 0.3, 0.3, 1.0, 0.3, 1.0, 0.3],
            [1.0, 0.3, 0.3, 1.0, 0.3, 1.0, 0.3],
            [1.0, 0.3, 0.3, 1.0, 0.3, 1.0, 0.3],
        ]
    )
    print_sing_score(song_result, "Full Song Test")
