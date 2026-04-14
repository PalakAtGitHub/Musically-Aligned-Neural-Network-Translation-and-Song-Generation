"""
MIDI Loader - Extract melody features from MIDI files
"""

import numpy as np
from music21 import converter, note, chord
from pathlib import Path

class MIDILoader:
    """
    Extract melody features from MIDI files
    """
    
    def __init__(self):
        pass
    
    def extract_melody_features(self, midi_path, num_lyric_lines=4):
        """
        Extract 5-dimensional features from MIDI
        
        Args:
            midi_path: Path to .mid file
            num_lyric_lines: Number of lyric lines (used to estimate verse length).
                             Default 4 (standard verse).
        
        Returns:
            features: [num_notes, 5] numpy array
                Columns: [pitch, pitch_class, duration, duration_bin, beat_strength]
        """
        try:
            # Parse MIDI
            score = converter.parse(str(midi_path))
            
            # Get all notes (flatten all parts)
            # Use flatten() as .flat is deprecated
            notes_structure = score.flatten().notes
            
            notes = []
            for element in notes_structure:
                if element.isNote:
                    notes.append(element)
                # Skip chords to match training data logic (keep simple melody line)
            
            if len(notes) == 0:
                print(f"Warning: No notes found in {midi_path}")
                return None
                
            # Trimming logic (from data_builder)
            # Estimate how many notes belong to one verse
            estimated_notes_per_line = 8  # reasonable average
            estimated_verse_notes = num_lyric_lines * estimated_notes_per_line
            
            if len(notes) > estimated_verse_notes * 2:
                # Likely multi-verse MIDI — only take first verse worth
                notes = notes[:estimated_verse_notes]
                print(f"  ℹ️  Trimmed to first ~{estimated_verse_notes} notes (from {len(notes) + estimated_verse_notes} total)")
            
            # Extract features
            features = []
            for n in notes:
                pitch = n.pitch.midi
                pitch_class = pitch % 12
                duration = n.duration.quarterLength
                duration_bin = 1 if duration >= 1.0 else 0
                beat_strength = n.beatStrength
                
                features.append([
                    pitch,
                    pitch_class,
                    duration,
                    duration_bin,
                    beat_strength
                ])
            
            return np.array(features, dtype=np.float32)
        
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None
    
    def segment_by_phrases(self, features, phrase_length=4):
        """
        Segment melody into phrases (optional - for future use)
        
        Args:
            features: [num_notes, 5]
            phrase_length: Notes per phrase
        
        Returns:
            List of phrase features
        """
        num_notes = len(features)
        phrases = []
        
        for i in range(0, num_notes, phrase_length):
            phrase = features[i:i+phrase_length]
            if len(phrase) > 0:
                phrases.append(phrase)
        
        return phrases


# Test
if __name__ == "__main__":
    loader = MIDILoader()
    
    # Test on a sample file
    test_file = "data/raw_midis/twinkle_twinkle.mid"
    
    if Path(test_file).exists():
        features = loader.extract_melody_features(test_file)
        
        if features is not None:
            print(f"✓ Loaded {test_file}")
            print(f"  Num notes: {len(features)}")
            print(f"  Feature shape: {features.shape}")
            print(f"\n  First 3 notes:")
            print(features[:3])
    else:
        print(f"✗ File not found: {test_file}")
        print("  Place MIDI files in data/raw_midis/")



# lyrics → data/song_manifest.json
#               {
#                 "songs": [
#                   {"name": "...", "midi_file": "song1.mid", "lines": [...]}
#                 ]
#               }