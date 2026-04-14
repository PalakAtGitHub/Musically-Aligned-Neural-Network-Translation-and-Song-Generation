"""
Syllable Counting Utilities for English and Hindi
"""

import re
from typing import List

# ============================================================================
# ENGLISH SYLLABLE COUNTER
# ============================================================================

class EnglishSyllableCounter:
    """
    Count syllables in English text using vowel-group method
    
    Method:
    1. Count vowel groups (consecutive vowels = 1 syllable)
    2. Handle silent 'e' at end of words
    3. Special cases: 'ion', 'ious', etc.
    """
    
    # Vowels
    VOWELS = 'aeiouy'
    
    # Common exceptions
    EXCEPTIONS = {
        'the': 1,
        'a': 1,
        'i': 1,
        'you': 1,
        'are': 1,
        'were': 1,
        'been': 1,
        'have': 1,
        'has': 1,
        'had': 1,
        'do': 1,
        'does': 1,
        'did': 1,
        'is': 1,
        'was': 1,
        'be': 1,
        'to': 1,
        'of': 1,
        'for': 1,
        'from': 1,
        'or': 1,
        'and': 1,
        'but': 1,
        'not': 1,
        'with': 1,
        'by': 1,
        'at': 1,
        'in': 1,
        'on': 1,
    }
    
    @staticmethod
    def count_word(word: str) -> int:
        """
        Count syllables in a single word
        
        Args:
            word: English word (lowercase)
        
        Returns:
            syllable_count: int
        """
        word = word.lower().strip()
        
        # Check exceptions first
        if word in EnglishSyllableCounter.EXCEPTIONS:
            return EnglishSyllableCounter.EXCEPTIONS[word]
        
        # Remove non-alphabetic characters
        word = re.sub(r'[^a-z]', '', word)
        
        if len(word) == 0:
            return 0
        
        # Count vowel groups
        syllables = 0
        previous_was_vowel = False
        
        for i, char in enumerate(word):
            is_vowel = char in EnglishSyllableCounter.VOWELS
            
            if is_vowel and not previous_was_vowel:
                syllables += 1
            
            previous_was_vowel = is_vowel
        
        # Handle silent 'e' at end
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        # Handle special endings
        if word.endswith('le') and len(word) > 2 and word[-3] not in EnglishSyllableCounter.VOWELS:
            syllables += 1  # Words like 'table', 'bottle'
        
        # Ensure at least 1 syllable
        return max(1, syllables)
    
    @staticmethod
    def count(text: str) -> int:
        """
        Count total syllables in text
        
        Args:
            text: English sentence or phrase
        
        Returns:
            total_syllables: int
        """
        # Split into words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Count syllables for each word
        total = sum(EnglishSyllableCounter.count_word(word) for word in words)
        
        return total


# ============================================================================
# HINDI SYLLABLE COUNTER
# ============================================================================

class HindiSyllableCounter:
    """
    Count syllables in Hindi (Devanagari) text
    
    Method:
    1. Identify independent vowels (स्वर)
    2. Identify consonants with matras (व्यंजन + मात्रा)
    3. Handle conjuncts (संयुक्त अक्षर)
    4. Handle halants (हलन्त)
    
    Key Rules:
    - Independent vowel = 1 syllable (अ, आ, इ, etc.)
    - Consonant + matra = 1 syllable (का, कि, कु, etc.)
    - Consonant alone (with inherent 'a') = 1 syllable (क = "ka")
    - Halant (्) removes inherent vowel (क् = no syllable)
    - Conjunct (क्त) = consonant cluster, check following vowel
    """
    
    # Unicode ranges
    INDEPENDENT_VOWELS = 'अआइईउऊऋॠऌॡएऐओऔ'  # 0x0904-0x0914
    CONSONANTS = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसह'  # 0x0915-0x0939
    MATRAS = 'ािीुूृॄॢॣेैोौ'  # 0x093E-0x094C (dependent vowel signs)
    HALANT = '्'  # 0x094D (virama)
    ANUSVARA = 'ंँ'  # 0x0902, 0x0901
    VISARGA = 'ः'  # 0x0903
    NUKTA = '़'  # 0x093C
    
    @staticmethod
    def count(text: str) -> int:
        """
        Count syllables in Hindi text
        
        Args:
            text: Hindi text in Devanagari script
        
        Returns:
            syllable_count: int
        """
        if not text:
            return 0
        
        # Remove punctuation and spaces
        text = re.sub(r'[^\u0900-\u097F]', '', text)
        
        syllables = 0
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Case 1: Independent vowel = 1 syllable
            if char in HindiSyllableCounter.INDEPENDENT_VOWELS:
                syllables += 1
                i += 1
            
            # Case 2: Consonant
            elif char in HindiSyllableCounter.CONSONANTS:
                # Look ahead for halant
                if i + 1 < len(text) and text[i + 1] == HindiSyllableCounter.HALANT:
                    # Consonant + halant = no syllable (part of conjunct)
                    i += 2
                    
                    # Skip nukta if present
                    if i < len(text) and text[i] == HindiSyllableCounter.NUKTA:
                        i += 1
                
                else:
                    # Consonant without halant = 1 syllable
                    syllables += 1
                    i += 1
                    
                    # Skip following matra (already counted)
                    if i < len(text) and text[i] in HindiSyllableCounter.MATRAS:
                        i += 1
                    
                    # Skip nukta
                    if i < len(text) and text[i] == HindiSyllableCounter.NUKTA:
                        i += 1
                    
                    # Skip anusvara/visarga (nasal marks don't add syllables)
                    if i < len(text) and text[i] in (HindiSyllableCounter.ANUSVARA + HindiSyllableCounter.VISARGA):
                        i += 1
            
            # Case 3: Standalone matra (shouldn't happen, but handle it)
            elif char in HindiSyllableCounter.MATRAS:
                syllables += 1
                i += 1
            
            # Case 4: Other (skip)
            else:
                i += 1
        
        return syllables


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def count_english_syllables(text: str) -> int:
    """
    Count syllables in English text
    
    Args:
        text: English sentence/phrase
    
    Returns:
        syllable_count: int
    
    Example:
        >>> count_english_syllables("Twinkle twinkle little star")
        7
    """
    return EnglishSyllableCounter.count(text)


def count_hindi_syllables(text: str) -> int:
    """
    Count syllables in Hindi text
    
    Args:
        text: Hindi text in Devanagari
    
    Returns:
        syllable_count: int
    
    Example:
        >>> count_hindi_syllables("चमक चमक छोटा तारा")
        7
    """
    return HindiSyllableCounter.count(text)


def check_syllable_alignment(english: str, hindi: str, num_notes: int, tolerance: int = 2) -> dict:
    """
    Check if syllable counts align between English, Hindi, and melody
    
    Args:
        english: English text
        hindi: Hindi text
        num_notes: Number of notes in melody
        tolerance: Acceptable deviation
    
    Returns:
        dict with alignment statistics
    """
    en_syl = count_english_syllables(english)
    hi_syl = count_hindi_syllables(hindi)
    
    # Check alignment
    hi_notes_diff = abs(hi_syl - num_notes)
    en_notes_diff = abs(en_syl - num_notes)
    
    is_aligned = hi_notes_diff <= tolerance
    
    return {
        'english_syllables': en_syl,
        'hindi_syllables': hi_syl,
        'num_notes': num_notes,
        'hindi_notes_diff': hi_notes_diff,
        'english_notes_diff': en_notes_diff,
        'is_aligned': is_aligned,
        'alignment_quality': 'Good' if is_aligned else 'Poor'
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SYLLABLE COUNTER TESTS")
    print("="*70)
    
    # Test English
    print("\n1. ENGLISH SYLLABLE COUNTING:")
    print("-" * 70)
    
    english_tests = [
        ("Twinkle twinkle little star", 7),
        ("How I wonder what you are", 6),
        ("Let it go", 3),
        ("Happy birthday to you", 6),
        ("The cat in the hat", 5),
        ("Beautiful", 3),
        ("Chocolate", 3),
        ("I", 1),
        ("You", 1),
        ("Hello", 2),
    ]
    
    en_passed = 0
    for text, expected in english_tests:
        result = count_english_syllables(text)
        status = "✓" if result == expected else "✗"
        if result == expected:
            en_passed += 1
        print(f"{status} '{text}': {result} (expected {expected})")
    
    print(f"\nEnglish: {en_passed}/{len(english_tests)} tests passed")
    
    # Test Hindi
    print("\n2. HINDI SYLLABLE COUNTING:")
    print("-" * 70)
    
    hindi_tests = [
        ("चमक चमक छोटा तारा", 7),  # Chamak chamak chhota tara
        ("जाने दो", 3),  # Jaane do
        ("जन्मदिन मुबारक", 6),  # Janmadin mubarak
        ("प्यार", 2),  # Pyaar (प् + यार)
        ("स्कूल", 2),  # School (स् + कूल)
        ("क्या", 2),  # Kya (क् + या)
        ("नमस्ते", 3),  # Namaste (न + म + स् + ते)
        ("अच्छा", 2),  # Achha (अ + च् + छा)
        ("दोस्त", 2),  # Dost (दो + स् + त)
        ("मैं", 1),  # Main
    ]
    
    hi_passed = 0
    for text, expected in hindi_tests:
        result = count_hindi_syllables(text)
        status = "✓" if result == expected else "✗"
        if result == expected:
            hi_passed += 1
        print(f"{status} '{text}': {result} (expected {expected})")
    
    print(f"\nHindi: {hi_passed}/{len(hindi_tests)} tests passed")
    
    # Test alignment checking
    print("\n3. SYLLABLE ALIGNMENT CHECKING:")
    print("-" * 70)
    
    alignment_tests = [
        ("Twinkle twinkle little star", "चमक चमक छोटा तारा", 7),
        ("Let it go", "जाने दो", 3),
        ("Happy birthday", "जन्मदिन मुबारक", 6),
    ]
    
    for en, hi, notes in alignment_tests:
        result = check_syllable_alignment(en, hi, notes)
        print(f"\nEnglish: '{en}'")
        print(f"Hindi:   '{hi}'")
        print(f"Notes:   {notes}")
        print(f"  EN syllables: {result['english_syllables']}")
        print(f"  HI syllables: {result['hindi_syllables']}")
        print(f"  Deviation:    {result['hindi_notes_diff']}")
        print(f"  Quality:      {result['alignment_quality']} {'✓' if result['is_aligned'] else '✗'}")
    
    print("\n" + "="*70)
    print(f"OVERALL: {en_passed + hi_passed}/{len(english_tests) + len(hindi_tests)} tests passed")
    print("="*70)