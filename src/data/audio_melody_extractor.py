"""
Audio Melody Extractor - Extract melody features from vocal audio

Phase 2 of the audio pipeline:
  Input:  Isolated vocals (from AudioSeparator)
  Output: [num_notes, 5] feature array — SAME format as MIDILoader

This replaces MIDILoader when working with real audio instead of MIDI files.
The rest of the pipeline (MCNST model, training, etc.) doesn't need to change
because the output format is identical.

Features extracted per note:
  [0] pitch        — MIDI note number (60 = C4)
  [1] pitch_class  — pitch % 12 (0-11, key-independent)
  [2] duration     — note length in beats
  [3] duration_bin — 0 (short) or 1 (long, >= 1 beat)
  [4] beat_strength — rhythmic emphasis (0.0 to 1.0)

Install: pip install librosa soundfile
"""

import numpy as np
from pathlib import Path


def hz_to_midi(freq_hz):
    """Convert frequency in Hz to MIDI note number."""
    if freq_hz <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq_hz / 440.0)))


class AudioMelodyExtractor:
    """
    Extract melody features from vocal audio using pitch tracking
    and onset detection.
    
    Produces the SAME [num_notes, 5] format as MIDILoader,
    so the MCNST model doesn't need any changes.
    """
    
    def __init__(self, sr=22050):
        """
        Args:
            sr: Sample rate to use (22050 is librosa's default)
        """
        self.sr = sr
    
    def extract_melody_features(self, audio_path, num_lyric_lines=4):
        """
        Extract melody features from vocal audio.
        
        Args:
            audio_path: Path to vocal audio file (WAV)
            num_lyric_lines: Expected number of lyric lines
                             (used for trimming, same as MIDILoader)
        
        Returns:
            features: [num_notes, 5] numpy array
                Same format as MIDILoader output.
            None on failure.
        """
        try:
            import librosa
        except ImportError:
            print("✗ librosa not installed. Run: pip install librosa")
            return None
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            print(f"✗ File not found: {audio_path}")
            return None
        
        print(f"  Loading audio: {audio_path.name}")
        
        # 1. Load audio
        y, sr = librosa.load(str(audio_path), sr=self.sr)
        duration_sec = len(y) / sr
        print(f"  Duration: {duration_sec:.1f}s")
        
        # 2. Estimate tempo and beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        print(f"  Tempo: {tempo:.0f} BPM")
        
        # 3. Pitch tracking (pYIN — probabilistic YIN)
        # Returns F0 (fundamental frequency) per frame
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),   # ~65 Hz (low male voice)
            fmax=librosa.note_to_hz('C6'),   # ~1047 Hz (high female voice)
            sr=sr
        )
        
        # Time axis for pitch frames
        pitch_times = librosa.times_like(f0, sr=sr)
        
        # 4. Onset detection (where notes start)
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, 
            backtrack=True,  # snap to nearest energy minimum
            units='frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        if len(onset_times) == 0:
            print("  ⚠️  No onsets detected")
            return None
        
        print(f"  Detected {len(onset_times)} note onsets")
        
        # 5. Build note features
        # For each onset, find the pitch and duration
        features = []
        seconds_per_beat = 60.0 / tempo
        
        for i, onset in enumerate(onset_times):
            # Note end = next onset or end of audio
            if i + 1 < len(onset_times):
                note_end = onset_times[i + 1]
            else:
                note_end = duration_sec
            
            duration_sec_note = note_end - onset
            
            # Skip very short segments (likely artifacts)
            if duration_sec_note < 0.05:
                continue
            
            # Get pitch for this note (median F0 during the note)
            mask = (pitch_times >= onset) & (pitch_times < note_end)
            note_f0 = f0[mask]
            
            # Filter out unvoiced frames (NaN)
            voiced_f0 = note_f0[~np.isnan(note_f0)]
            
            if len(voiced_f0) == 0:
                continue  # skip unvoiced segments (rests, breaths)
            
            # Median pitch for stability
            median_f0 = np.median(voiced_f0)
            midi_pitch = hz_to_midi(median_f0)
            
            if midi_pitch < 20 or midi_pitch > 108:
                continue  # outside reasonable singing range
            
            # Duration in beats
            duration_beats = duration_sec_note / seconds_per_beat
            
            # Duration bin (same as MIDILoader: >= 1 beat = long)
            duration_bin = 1 if duration_beats >= 1.0 else 0
            
            # Beat strength: how close is this onset to a beat?
            if len(beat_times) > 0:
                distances = np.abs(beat_times - onset)
                min_dist = np.min(distances)
                # Closer to a beat = higher strength
                # Normalize: within 0.1s of beat = strong (1.0)
                beat_strength = max(0.0, 1.0 - (min_dist / seconds_per_beat))
            else:
                beat_strength = 0.5  # fallback
            
            features.append([
                midi_pitch,
                midi_pitch % 12,
                duration_beats,
                duration_bin,
                beat_strength
            ])
        
        if len(features) == 0:
            print("  ⚠️  No valid notes extracted")
            return None
        
        features = np.array(features, dtype=np.float32)
        
        # Trimming logic (same as MIDILoader)
        estimated_notes_per_line = 8
        estimated_verse_notes = num_lyric_lines * estimated_notes_per_line
        
        if len(features) > estimated_verse_notes * 2:
            original_count = len(features)
            features = features[:estimated_verse_notes]
            print(f"  ℹ️  Trimmed to first ~{estimated_verse_notes} notes (from {original_count})")
        
        print(f"  ✓ Extracted {len(features)} notes")
        print(f"    Pitch range: {int(features[:,0].min())}-{int(features[:,0].max())} (MIDI)")
        print(f"    Avg duration: {features[:,2].mean():.2f} beats")
        
        return features


# Test
if __name__ == "__main__":
    import sys
    
    extractor = AudioMelodyExtractor()
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Try to find separated vocals
        separated_dir = Path("src/data/separated")
        if separated_dir.exists():
            vocal_files = list(separated_dir.glob("*/vocals.wav"))
            if vocal_files:
                audio_file = str(vocal_files[0])
                print(f"Using: {audio_file}")
            else:
                print("No separated vocals found.")
                print("Run audio_separator.py first, or provide a path:")
                print("  python audio_melody_extractor.py <vocals.wav>")
                sys.exit(1)
        else:
            print("Usage: python audio_melody_extractor.py <vocals.wav>")
            sys.exit(1)
    
    features = extractor.extract_melody_features(audio_file)
    
    if features is not None:
        print(f"\n✓ Feature shape: {features.shape}")
        print(f"  Same format as MIDILoader → can be fed directly to MCNST!")
        print(f"\n  First 3 notes:")
        print(f"  [pitch, pitch_class, duration, dur_bin, beat_str]")
        for row in features[:3]:
            print(f"  {row}")
