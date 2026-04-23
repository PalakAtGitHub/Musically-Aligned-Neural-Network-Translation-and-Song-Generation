"""
Audio Melody Extractor - Extract melody features from vocal audio

Phase 2 of the audio pipeline:
  Input:  Isolated vocals (from AudioSeparator)
  Output: [num_notes, 5] feature array — SAME format as MIDILoader

Primary backend: Basic-Pitch (Spotify, neural pitch tracker)
  - Neural network trained for polyphonic pitch detection
  - Handles singer vibrato and glides as single sustained notes
  - Returns clean note events with accurate onset/offset times
  - No over-segmentation artifacts that plagued librosa pyin+onset

Fallback backend: librosa pyin + onset detection
  - Used only if basic-pitch is not installed
  - Prone to over-detecting notes in sustained vocal phrases

Features extracted per note:
  [0] pitch        — MIDI note number (60 = C4)
  [1] pitch_class  — pitch % 12 (0-11, key-independent)
  [2] duration     — note length in beats
  [3] duration_bin — 0 (short) or 1 (long, >= 1 beat)
  [4] beat_strength — rhythmic emphasis (0.0 to 1.0)
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
    Extract melody features from vocal audio.

    Uses Basic-Pitch (neural) as primary backend to accurately detect
    notes in singing vocals, avoiding the over-segmentation caused by
    librosa onset detection on vibrato and gliding phrases.

    Produces the SAME [num_notes, 5] format as MIDILoader.
    """

    def __init__(self, sr=22050):
        self.sr = sr

    def extract_melody_features(self, audio_path, num_lyric_lines=4):
        """
        Extract melody features from vocal audio.

        Args:
            audio_path:      Path to vocal audio file (WAV)
            num_lyric_lines: Expected number of lyric lines (used for trimming)

        Returns:
            features: [num_notes, 5] numpy array, or None on failure.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            print(f"✗ File not found: {audio_path}")
            return None

        print(f"  Loading audio: {audio_path.name}")

        try:
            import librosa
            y, sr = librosa.load(str(audio_path), sr=self.sr)
        except ImportError:
            print("✗ librosa not installed. Run: pip install librosa")
            return None

        duration_sec = len(y) / sr
        print(f"  Duration: {duration_sec:.1f}s")

        # Tempo and beat times (shared by both backends)
        import librosa
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        seconds_per_beat = 60.0 / max(tempo, 1.0)
        print(f"  Tempo: {tempo:.0f} BPM")

        # Try Basic-Pitch first
        features = self._extract_basic_pitch(
            audio_path, y, sr, beat_times, seconds_per_beat
        )

        if features is None:
            print("  ⚠  Basic-Pitch failed or unavailable, falling back to librosa")
            features = self._extract_librosa(
                y, sr, beat_times, seconds_per_beat, duration_sec
            )

        if features is None or len(features) == 0:
            print("  ⚠  No valid notes extracted")
            return None

        # Trim to a sensible number of notes for the lyric lines
        estimated_notes_per_line = 8
        max_notes = num_lyric_lines * estimated_notes_per_line * 2
        if len(features) > max_notes:
            original_count = len(features)
            features = features[:max_notes]
            print(f"  ℹ  Trimmed {original_count} → {len(features)} notes")

        features = np.array(features, dtype=np.float32)
        print(f"  ✓ Extracted {len(features)} notes")
        print(f"    Pitch range: {int(features[:, 0].min())}-{int(features[:, 0].max())} (MIDI)")
        print(f"    Avg duration: {features[:, 2].mean():.2f} beats")

        return features

    # ------------------------------------------------------------------
    # Backend 1: Basic-Pitch (neural, primary)
    # ------------------------------------------------------------------

    def _extract_basic_pitch(self, audio_path, y, sr, beat_times, seconds_per_beat):
        """
        Use Spotify's Basic-Pitch neural model to extract notes.

        Basic-Pitch returns note events as (start_s, end_s, midi_pitch, confidence).
        Each event corresponds to one sung note, with vibrato and glides
        correctly merged into single sustained notes.
        """
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except ImportError:
            return None

        try:
            print("  [Basic-Pitch] running neural pitch tracker...")
            model_output, midi_data, note_events = predict(
                str(audio_path),
                ICASSP_2022_MODEL_PATH,
                onset_threshold=0.5,
                frame_threshold=0.3,
                minimum_note_length=58,   # ms — ignore very short artifacts
                minimum_frequency=65.0,   # Hz ~C2 (low male vocal)
                maximum_frequency=1047.0, # Hz ~C6 (high female vocal)
                melodia_trick=True,       # keep only most prominent pitch per frame
            )
        except Exception as e:
            print(f"  [Basic-Pitch] error: {e}")
            return None

        if note_events is None or len(note_events) == 0:
            print("  [Basic-Pitch] no notes detected")
            return None

        print(f"  [Basic-Pitch] detected {len(note_events)} note events")

        features = []
        for start_s, end_s, midi_pitch, amplitude, _ in note_events:
            if midi_pitch < 20 or midi_pitch > 108:
                continue

            duration_s = max(end_s - start_s, 0.01)
            duration_beats = duration_s / seconds_per_beat
            duration_bin = 1 if duration_beats >= 1.0 else 0

            beat_strength = self._beat_strength(start_s, beat_times, seconds_per_beat)

            features.append([
                float(midi_pitch),
                float(midi_pitch % 12),
                duration_beats,
                float(duration_bin),
                beat_strength,
            ])

        return features if features else None

    # ------------------------------------------------------------------
    # Backend 2: librosa pyin + onset (fallback)
    # ------------------------------------------------------------------

    def _extract_librosa(self, y, sr, beat_times, seconds_per_beat, duration_sec):
        """
        Fallback: librosa pyin pitch tracking + onset detection.

        Note: prone to over-segmenting sustained vocal phrases.
        Only used when Basic-Pitch is unavailable.
        """
        import librosa

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=sr
        )
        pitch_times = librosa.times_like(f0, sr=sr)

        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, backtrack=True, units='frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        if len(onset_times) == 0:
            return None

        print(f"  [librosa] detected {len(onset_times)} onsets")

        features = []
        for i, onset in enumerate(onset_times):
            note_end = onset_times[i + 1] if i + 1 < len(onset_times) else duration_sec
            duration_s = note_end - onset
            if duration_s < 0.05:
                continue

            mask = (pitch_times >= onset) & (pitch_times < note_end)
            voiced_f0 = f0[mask]
            voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
            if len(voiced_f0) == 0:
                continue

            midi_pitch = hz_to_midi(np.median(voiced_f0))
            if midi_pitch < 20 or midi_pitch > 108:
                continue

            duration_beats = duration_s / seconds_per_beat
            duration_bin = 1 if duration_beats >= 1.0 else 0
            beat_strength = self._beat_strength(onset, beat_times, seconds_per_beat)

            features.append([
                float(midi_pitch),
                float(midi_pitch % 12),
                duration_beats,
                float(duration_bin),
                beat_strength,
            ])

        return features if features else None

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    def _beat_strength(self, onset_s, beat_times, seconds_per_beat):
        """Fraction [0,1] reflecting how close onset_s is to a beat."""
        if len(beat_times) == 0:
            return 0.5
        distances = np.abs(beat_times - onset_s)
        min_dist = float(np.min(distances))
        return max(0.0, 1.0 - (min_dist / max(seconds_per_beat, 1e-6)))


# ============================================================================
# CLI smoke test
# ============================================================================

if __name__ == "__main__":
    import sys

    extractor = AudioMelodyExtractor()

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        separated_dir = Path("src/data/separated")
        if separated_dir.exists():
            vocal_files = list(separated_dir.glob("*/vocals.wav"))
            if vocal_files:
                audio_file = str(vocal_files[0])
                print(f"Using: {audio_file}")
            else:
                print("No separated vocals found. Run audio_separator.py first.")
                sys.exit(1)
        else:
            print("Usage: python audio_melody_extractor.py <vocals.wav>")
            sys.exit(1)

    features = extractor.extract_melody_features(audio_file)
    if features is not None:
        print(f"\n✓ Feature shape: {features.shape}")
        print(f"  [pitch, pitch_class, duration, dur_bin, beat_str]")
        for row in features[:5]:
            print(f"  {row}")
