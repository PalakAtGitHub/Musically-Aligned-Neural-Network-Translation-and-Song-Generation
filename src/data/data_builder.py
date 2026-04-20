"""
Dataset Builder - Creates actual training data from MIDI + lyrics
"""

import json
import torch
from pathlib import Path
from typing import List, Dict
from transformers import MBart50TokenizerFast
from music21 import converter
import numpy as np

class DatasetBuilder:
    """
    Converts raw MIDI files + lyrics → training-ready tensors
    """
    
    def __init__(self, 
                 raw_midi_dir="src/data/raw_midis",
                 processed_dir="src/data/processed"):
        
        self.raw_midi_dir = Path(raw_midi_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt"
        )
        
        # Initialize syllable counters (you already have these)
        from src.utils.syllable_utils import count_english_syllables, count_hindi_syllables
        self.count_en_syl = count_english_syllables
        self.count_hi_syl = count_hindi_syllables
        
        # Initialize MIDI loader
        from src.data.midi_loader import MIDILoader
        self.midi_loader = MIDILoader()
    
    def extract_melody_features(self, midi_path: Path, num_lyric_lines: int = 4) -> np.ndarray:
        """
        DEPRECATED: Use self.midi_loader.extract_melody_features instead
        Wrapper kept for compatibility if needed, but logic delegates to MIDILoader
        """
        return self.midi_loader.extract_melody_features(midi_path, num_lyric_lines)
    
    def create_training_example(self, 
                                english_text: str,
                                hindi_text: str,
                                melody_features: np.ndarray,
                                song_name: str) -> Dict:
        """
        Create single training example
        """
        # Tokenize English
        self.tokenizer.src_lang = "en_XX"
        en_tokens = self.tokenizer(
            english_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=128
        ).input_ids[0]
        
        # Tokenize Hindi
        self.tokenizer.src_lang = "hi_IN"
        hi_tokens = self.tokenizer(
            hindi_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=128
        ).input_ids[0]
        
        # Count syllables
        en_syllables = self.count_en_syl(english_text)
        hi_syllables = self.count_hi_syl(hindi_text)
        num_notes = len(melody_features)
        
        # Quality check
        syllable_diff = abs(hi_syllables - num_notes)
        if syllable_diff > 3:
            print(f"⚠️  Warning: {song_name} - Syllable mismatch")
            print(f"   Hindi syllables: {hi_syllables}, Notes: {num_notes}")
        
        return {
            'src_ids': en_tokens,
            'tgt_ids': hi_tokens,
            'melody_features': torch.tensor(melody_features, dtype=torch.float32),
            'src_syllables': en_syllables,
            'tgt_syllables': hi_syllables,
            'num_notes': num_notes,
            'song_name': song_name,
            'english_text': english_text,
            'hindi_text': hindi_text
        }
    
    def build_dataset_from_manifest(self, manifest_path: str):
        """
        Build dataset from manifest JSON file
        
        Manifest format:
        {
            "songs": [
                {
                    "name": "Twinkle Twinkle Little Star",
                    "midi_file": "twinkle_twinkle.mid",
                    "lines": [
                        {
                            "english": "Twinkle twinkle little star",
                            "hindi": "चमक चमक छोटा तारा"
                        },
                        ...
                    ]
                }
            ]
        }
        """
        print(f"\n{'='*60}")
        print(f"Building dataset from: {manifest_path}")
        print(f"{'='*60}\n")
        
        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        all_examples = []
        song_stats = []
        
        for song_data in manifest['songs']:
            song_name = song_data['name']
            midi_file = song_data['midi_file']
            
            print(f"Processing: {song_name}")
            
            # Extract melody
            midi_path = self.raw_midi_dir / midi_file
            if not midi_path.exists():
                print(f"  ✗ MIDI file not found: {midi_path}")
                continue
            
            melody_features = self.extract_melody_features(midi_path, num_lyric_lines=len(song_data['lines']))
            if melody_features is None:
                print(f"  ✗ Failed to extract melody")
                continue
            
            print(f"  ✓ Extracted {len(melody_features)} notes")

            # Segment melody across lyric lines
            # Use notes_per_line from manifest if available, otherwise compute evenly
            num_lines_total = len(song_data['lines'])
            notes_per_line = song_data.get('notes_per_line', len(melody_features) // max(num_lines_total, 1))
            
            # Process each line with its corresponding melody segment
            num_lines = 0
            for line_idx, line_data in enumerate(song_data['lines']):
                # Assign a slice of melody notes to this line
                start_note = line_idx * notes_per_line
                if line_idx == num_lines_total - 1:
                    # Last line gets remaining notes
                    end_note = len(melody_features)
                else:
                    end_note = start_note + notes_per_line
                
                line_melody = melody_features[start_note:end_note]
                
                # Skip if no notes for this line
                if len(line_melody) == 0:
                    print(f"  ⚠️  No notes for line {line_idx}, using full melody")
                    line_melody = melody_features
                
                example = self.create_training_example(
                    english_text=line_data['english'],
                    hindi_text=line_data['hindi'],
                    melody_features=line_melody,
                    song_name=song_name
                )
                all_examples.append(example)
                num_lines += 1
            
            print(f"  ✓ Created {num_lines} training examples")
            print(f"    Notes per line: ~{notes_per_line}\n")
            
            song_stats.append({
                'name': song_name,
                'num_lines': num_lines,
                'num_notes': len(melody_features),
                'notes_per_line': notes_per_line
            })
        
        # Save dataset
        output_path = self.processed_dir / "training_data.pt"
        torch.save(all_examples, output_path)
        
        # Save stats
        stats_path = self.processed_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'total_examples': len(all_examples),
                'total_songs': len(song_stats),
                'songs': song_stats
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ Dataset built successfully!")
        print(f"   Total examples: {len(all_examples)}")
        print(f"   Saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return all_examples


if __name__ == "__main__":
    # Build dataset
    builder = DatasetBuilder()
    
    # You'll create this manifest file next
    dataset = builder.build_dataset_from_manifest("src/data/song_manifest.json")