"""
Phoneme Utilities for Hindi

Provides:
  1. Hindi text → IPA phonemes (via espeak-ng)
  2. Stress pattern extraction
  3. Consonant cluster analysis (for singability scoring)
  4. Rhyme ending extraction

Dependencies:
  pip install phonemizer
  + espeak-ng must be installed on the system
    Windows: download from https://github.com/espeak-ng/espeak-ng/releases
    Linux:   sudo apt install espeak-ng
"""

import re
from typing import List, Tuple, Optional


# ============================================================================
# PHONEMIZER (with graceful fallback if espeak-ng not installed)
# ============================================================================

def phonemize_hindi(text: str) -> Optional[str]:
    """
    Convert Hindi text to IPA phonemes using espeak-ng.
    
    Args:
        text: Hindi text in Devanagari
    
    Returns:
        IPA string, or None if espeak-ng unavailable
    
    Example:
        >>> phonemize_hindi("चमक चमक")
        'tʃəmək tʃəmək'
    """
    try:
        from phonemizer import phonemize
        result = phonemize(
            text,
            language='hi',
            backend='espeak',
            strip=True,
            with_stress=True
        )
        return result.strip()
    except ImportError:
        # phonemizer not installed — use fallback
        return _fallback_phonemize(text)
    except Exception:
        # espeak-ng not installed or other error
        return _fallback_phonemize(text)


def _fallback_phonemize(text: str) -> str:
    """
    Rule-based fallback when espeak-ng is not available.
    Uses basic Devanagari → approximate phoneme mapping.
    Not as accurate as espeak, but functional for loss computation.
    """
    # Basic Devanagari consonant → phoneme map
    consonant_map = {
        'क': 'k', 'ख': 'kʰ', 'ग': 'g', 'घ': 'gʰ', 'ङ': 'ŋ',
        'च': 'tʃ', 'छ': 'tʃʰ', 'ज': 'dʒ', 'झ': 'dʒʰ', 'ञ': 'ɲ',
        'ट': 'ʈ', 'ठ': 'ʈʰ', 'ड': 'ɖ', 'ढ': 'ɖʰ', 'ण': 'ɳ',
        'त': 't̪', 'थ': 't̪ʰ', 'द': 'd̪', 'ध': 'd̪ʰ', 'न': 'n',
        'प': 'p', 'फ': 'pʰ', 'ब': 'b', 'भ': 'bʰ', 'म': 'm',
        'य': 'j', 'र': 'ɾ', 'ल': 'l', 'व': 'ʋ',
        'श': 'ʃ', 'ष': 'ʂ', 'स': 's', 'ह': 'ɦ',
        'ळ': 'ɭ',
    }
    
    # Vowel/matra → phoneme map
    vowel_map = {
        'अ': 'ə', 'आ': 'aː', 'इ': 'ɪ', 'ई': 'iː',
        'उ': 'ʊ', 'ऊ': 'uː', 'ए': 'eː', 'ऐ': 'ɛː',
        'ओ': 'oː', 'औ': 'ɔː', 'ऋ': 'ɾɪ',
        'ा': 'aː', 'ि': 'ɪ', 'ी': 'iː',
        'ु': 'ʊ', 'ू': 'uː', 'े': 'eː', 'ै': 'ɛː',
        'ो': 'oː', 'ौ': 'ɔː', 'ृ': 'ɾɪ',
    }
    
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        
        if char in consonant_map:
            result.append(consonant_map[char])
            # Check if next char is a halant (removes inherent vowel)
            if i + 1 < len(text) and text[i + 1] == '्':
                i += 2  # skip halant, no inherent vowel
                continue
            # Check if next char is a matra
            elif i + 1 < len(text) and text[i + 1] in vowel_map:
                result.append(vowel_map[text[i + 1]])
                i += 2
                continue
            else:
                # Inherent 'a' vowel (schwa)
                result.append('ə')
        elif char in vowel_map:
            result.append(vowel_map[char])
        elif char == ' ':
            result.append(' ')
        # Skip other characters (anusvara, visarga, etc.)
        
        i += 1
    
    return ''.join(result)


# ============================================================================
# STRESS PATTERN
# ============================================================================

# IPA vowels for analysis
IPA_VOWELS = set('aɑæeɛiɪoɔuʊəː')
IPA_LONG_MARKERS = {'ː'}  # long vowel marker


def get_stress_pattern(hindi_text: str) -> List[float]:
    """
    Estimate stress pattern for Hindi text.
    
    Hindi stress is primarily weight-based:
    - Heavy syllables (long vowel or closed) = stressed (1.0)
    - Light syllables (short vowel, open) = unstressed (0.0)
    - Medium = 0.5
    
    Args:
        hindi_text: Hindi text in Devanagari
    
    Returns:
        List of stress values [0.0, 1.0, 0.5, ...] per syllable
    """
    phonemes = phonemize_hindi(hindi_text)
    if not phonemes:
        return []
    
    # Split into syllable-like segments
    # A syllable roughly = consonant(s) + vowel (+ optional consonant)
    syllables = _split_into_syllables(phonemes)
    
    stress = []
    for syl in syllables:
        if 'ː' in syl:
            # Long vowel = heavy = stressed
            stress.append(1.0)
        elif _ends_with_consonant(syl) and _has_vowel(syl):
            # Closed syllable (CVC) = heavy = stressed
            stress.append(0.8)
        elif _has_vowel(syl):
            # Open syllable with short vowel = light = unstressed
            stress.append(0.3)
        else:
            stress.append(0.0)
    
    return stress


def _split_into_syllables(phonemes: str) -> List[str]:
    """Split IPA string into rough syllable chunks."""
    # Remove spaces first, then split on vowel boundaries
    phonemes = phonemes.replace(' ', ' _ ')
    
    syllables = []
    current = ''
    
    for char in phonemes:
        if char == '_':
            if current.strip():
                syllables.append(current.strip())
            current = ''
            continue
            
        current += char
        
        # After a vowel (possibly + length mark), we have a syllable boundary
        # unless followed by another vowel
        if char in IPA_VOWELS and char != 'ː':
            # Peek ahead — if next is 'ː', include it
            pass
        elif char == 'ː':
            # End of long vowel — syllable break after this
            syllables.append(current.strip())
            current = ''
    
    if current.strip():
        syllables.append(current.strip())
    
    # Merge very short fragments
    merged = []
    for s in syllables:
        if len(s) <= 1 and merged:
            merged[-1] += s
        else:
            merged.append(s)
    
    return merged if merged else ['']


def _has_vowel(text: str) -> bool:
    return any(c in IPA_VOWELS for c in text)


def _ends_with_consonant(text: str) -> bool:
    if not text:
        return False
    last = text[-1]
    return last not in IPA_VOWELS and last != 'ː'


# ============================================================================
# SINGABILITY ANALYSIS
# ============================================================================

def consonant_cluster_score(hindi_text: str) -> float:
    """
    Score how singable a Hindi text is based on consonant clusters.
    
    Singing is easier with alternating consonant-vowel (CV) patterns.
    Long consonant clusters (CCC+) are hard to sing.
    
    Returns:
        Score from 0.0 (very hard to sing) to 1.0 (very easy to sing)
    """
    phonemes = phonemize_hindi(hindi_text)
    if not phonemes:
        return 0.5  # neutral fallback
    
    # Count consonant clusters
    cluster_lengths = []
    current_cluster = 0
    
    for char in phonemes:
        if char in IPA_VOWELS or char == 'ː' or char == ' ':
            if current_cluster > 0:
                cluster_lengths.append(current_cluster)
            current_cluster = 0
        else:
            current_cluster += 1
    
    if current_cluster > 0:
        cluster_lengths.append(current_cluster)
    
    if not cluster_lengths:
        return 1.0  # all vowels = very singable
    
    # Penalty for long clusters
    # 1 consonant = fine (1.0)
    # 2 consonants = okay (0.8)
    # 3+ consonants = hard (0.3)
    penalties = []
    for length in cluster_lengths:
        if length <= 1:
            penalties.append(1.0)
        elif length == 2:
            penalties.append(0.8)
        elif length == 3:
            penalties.append(0.4)
        else:
            penalties.append(0.1)
    
    return sum(penalties) / len(penalties)


def vowel_ratio(hindi_text: str) -> float:
    """
    Compute vowel-to-total phoneme ratio.
    Higher ratio = easier to sing.
    
    Typical singing: 0.4-0.6
    Difficult to sing: < 0.3
    """
    phonemes = phonemize_hindi(hindi_text)
    if not phonemes:
        return 0.5
    
    # Remove spaces
    phonemes = phonemes.replace(' ', '')
    
    if len(phonemes) == 0:
        return 0.5
    
    vowel_count = sum(1 for c in phonemes if c in IPA_VOWELS)
    return vowel_count / len(phonemes)


# ============================================================================
# RHYME ANALYSIS
# ============================================================================

def get_rhyme_ending(hindi_text: str, num_phonemes: int = 3) -> str:
    """
    Extract the phonetic ending of a Hindi text for rhyme comparison.
    
    Args:
        hindi_text: Hindi text (single line)
        num_phonemes: How many ending phonemes to compare
    
    Returns:
        Last N phonemes as string
    
    Example:
        >>> get_rhyme_ending("छोटा तारा")
        'aːɾaː'  # ends with -ara
        >>> get_rhyme_ending("प्यारा")
        'aːɾaː'  # same ending = RHYMES!
    """
    phonemes = phonemize_hindi(hindi_text)
    if not phonemes:
        return ''
    
    # Remove trailing spaces
    phonemes = phonemes.rstrip()
    
    # Get last N characters (approximate phonemes)
    return phonemes[-num_phonemes:] if len(phonemes) >= num_phonemes else phonemes


def rhyme_similarity(line1: str, line2: str, num_phonemes: int = 3) -> float:
    """
    Compute rhyme similarity between two Hindi lines.
    
    Returns:
        0.0 (no rhyme) to 1.0 (perfect rhyme)
    """
    ending1 = get_rhyme_ending(line1, num_phonemes)
    ending2 = get_rhyme_ending(line2, num_phonemes)
    
    if not ending1 or not ending2:
        return 0.0
    
    # Count matching characters from the end
    matches = 0
    for c1, c2 in zip(reversed(ending1), reversed(ending2)):
        if c1 == c2:
            matches += 1
        else:
            break
    
    return matches / max(len(ending1), len(ending2))


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHONEME UTILITIES TEST")
    print("=" * 60)
    
    test_texts = [
        "चमक चमक छोटा तारा",
        "जाने दो",
        "नमस्ते",
        "प्यार",
        "जन्मदिन मुबारक",
    ]
    
    for text in test_texts:
        print(f"\n--- '{text}' ---")
        
        phonemes = phonemize_hindi(text)
        print(f"  Phonemes:     {phonemes}")
        
        stress = get_stress_pattern(text)
        print(f"  Stress:       {stress}")
        
        sing_score = consonant_cluster_score(text)
        print(f"  Singability:  {sing_score:.2f}")
        
        v_ratio = vowel_ratio(text)
        print(f"  Vowel ratio:  {v_ratio:.2f}")
        
        ending = get_rhyme_ending(text)
        print(f"  Rhyme ending: '{ending}'")
    
    # Test rhyme detection
    print(f"\n--- Rhyme Detection ---")
    pairs = [
        ("छोटा तारा", "सबसे प्यारा"),    # should rhyme (aːɾaː)
        ("छोटा तारा", "जाने दो"),          # should NOT rhyme
    ]
    
    for l1, l2 in pairs:
        sim = rhyme_similarity(l1, l2)
        print(f"  '{l1}' / '{l2}' → {sim:.2f} {'✓ Rhymes!' if sim > 0.5 else '✗ No rhyme'}")
