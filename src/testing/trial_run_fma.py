"""
Trial run script to test the full pipeline (Separation -> Extraction -> Constrained Translation)
using audio samples from the fma-medium dataset, validating Rhyme-Guided Beam Search.
"""

import os
import sys
import torch
import random
from pathlib import Path

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

from src.data.audio_separator import AudioSeparator
from src.data.audio_melody_extractor import AudioMelodyExtractor
from src.testing.test import load_trained_model
from transformers import MBart50TokenizerFast
from src.utils.syllable_utils import count_hindi_syllables
from src.utils.phoneme_utils import get_rhyme_ending, rhyme_similarity

def find_random_fma_mp3():
    """Finds a random .mp3 file in the src/data/fma_medium folder."""
    fma_dir = PROJECT_ROOT / "src" / "data" / "fma_medium"
    
    if not fma_dir.exists():
        print(f"Error: {fma_dir} not found. Ensure fma-medium dataset is downloaded.")
        return None
        
    mp3_files = list(fma_dir.rglob("*.mp3"))
    if not mp3_files:
        # FMA may not have extensions depending on how it was unzipped, or could be named .wav
        mp3_files = list(fma_dir.rglob("*.wav")) + list(fma_dir.rglob("*.flac"))
        
        # If still empty, grab any file inside subdirectories
        if not mp3_files:
            for subdir in fma_dir.iterdir():
                if subdir.is_dir():
                    for f in subdir.iterdir():
                        if f.is_file() and f.name != ".DS_Store":
                            mp3_files.append(f)
                            
    if not mp3_files:
        print("Error: No audio files found in fma_medium.")
        return None
        
    return random.choice(mp3_files)

def run_fma_trial():
    print("=" * 60)
    print("FMA Trial: Testing Rhyme-Guided Beam Search")
    print("=" * 60)
    
    # 1. Grab random audio file
    audio_path = find_random_fma_mp3()
    if not audio_path:
        return
        
    print(f"\n[1] Selected random FMA audio: {audio_path.relative_to(PROJECT_ROOT)}")
    
    # 2. Source Separation
    print("\n[2] Extracting Vocals...")
    separator = AudioSeparator()
    stems = separator.separate(str(audio_path))
    if not stems or not stems.get('vocals'):
        print("✗ Separation failed. Falling back to original audio for testing...")
        vocal_path = audio_path
    else:
        vocal_path = stems['vocals']
    
    # 3. Melody Extraction
    print("\n[3] Extracting Melody from Vocals...")
    extractor = AudioMelodyExtractor()
    # We will test two lines of lyrics
    num_lines = 2
    melody_features = extractor.extract_melody_features(str(vocal_path), num_lyric_lines=num_lines)
    
    if melody_features is None:
        print("✗ Melody extraction failed.")
        return
        
    print(f"  ✓ Extracted melody: {len(melody_features)} total frames")
    
    print("\n[4] Initializing Model for Translation...")
    try:
        model = load_trained_model("checkpoints/best_model.pt")
    except FileNotFoundError:
        print("  ⚠️ Checkpoint 'checkpoints/best_model.pt' not found. Overriding with an untrained MCNST model for trial mechanics...")
        from src.models.mcnst_model import MCNST
        model = MCNST(freeze_encoder=True, freeze_decoder_layers=10)
        model.eval()
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = "en_XX"
    forced_bos = tokenizer.lang_code_to_id["hi_IN"]
    
    # Let's say we have two english lines, and we want Line 2 to Rhyme with Line 1.
    lines = [
        "Twinkle twinkle little star",
        "How I wonder what you are"
    ]
    
    notes_per_line = len(melody_features) // num_lines
    previous_line_rhyme = None
    
    for i, line in enumerate(lines):
        start = i * notes_per_line
        end = len(melody_features) if i == len(lines) - 1 else start + notes_per_line
        line_melody = melody_features[start:end]
        num_notes = len(line_melody)
        
        print(f"\n--- Line {i+1}: '{line}' ---")
        print(f"Target Notes/Syllables: {num_notes}")
        
        src_ids = tokenizer(line, return_tensors="pt").input_ids
        melody_tensor = torch.tensor(line_melody, dtype=torch.float32).unsqueeze(0)
        
        if i == 0:
            # First line: Standard Constrained Generation
            print("  Generating Base Line (Standard Constrained)...")
            with torch.no_grad():
                gen_ids, gen_syl = model.generate_constrained(
                    input_ids=src_ids,
                    melody_features=melody_tensor,
                    target_syllables=num_notes,
                    num_beams=5
                )
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            # Store full Hindi text (not the phoneme string) — rhyme_similarity
            # calls get_rhyme_ending internally, so it needs the original text.
            previous_line_text = gen_text
            previous_line_rhyme = get_rhyme_ending(gen_text)  # display only

            print(f"  → TEXT:  {gen_text}")
            print(f"  → SYLS:  {gen_syl}")
            print(f"  → RHYME: {previous_line_rhyme}")

        else:
            # Subsequent lines: Generate standard vs rhyme-guided
            print("  A. Standard Constrained (No Rhyme Guidance):")
            with torch.no_grad():
                std_ids, std_syl = model.generate_constrained(
                    input_ids=src_ids,
                    melody_features=melody_tensor,
                    target_syllables=num_notes,
                    num_beams=5
                )
            std_text = tokenizer.decode(std_ids, skip_special_tokens=True)
            # Pass the original Hindi text, not the IPA ending string
            std_sim = rhyme_similarity(std_text, previous_line_text)

            print(f"     → TEXT:  {std_text}")
            print(f"     → MATCH: {std_sim:.2f} similarity to '{previous_line_rhyme}'")

            print("  B. Rhyme-Guided Constrained (Weighted):")
            with torch.no_grad():
                rhyme_ids, rhyme_syl = model.generate_constrained(
                    input_ids=src_ids,
                    melody_features=melody_tensor,
                    target_syllables=num_notes,
                    num_beams=5,
                    target_rhyme=previous_line_text,  # full Hindi text, not IPA
                    rhyme_weight=3.0  # Encourage rhyme heavily
                )
            rhyme_text = tokenizer.decode(rhyme_ids, skip_special_tokens=True)
            rhyme_sim = rhyme_similarity(rhyme_text, previous_line_text)

            print(f"     → TEXT:  {rhyme_text}")
            print(f"     → MATCH: {rhyme_sim:.2f} similarity to '{previous_line_rhyme}'")

    print("\n✓ Trial Complete!")

if __name__ == "__main__":
    run_fma_trial()
