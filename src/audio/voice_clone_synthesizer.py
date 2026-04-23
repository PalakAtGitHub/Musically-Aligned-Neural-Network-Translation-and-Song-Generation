"""
Voice Clone Synthesizer — Zero-shot voice cloning for Hindi song translation

Uses XTTS v2 (Coqui TTS) to synthesize Hindi vocals that sound like the
original singer, using only a short reference clip — no RVC training required.

Full pipeline (called from src/main.py voice_clone command):
  Phase 1: Demucs  → vocals.wav + instrumental.wav
  Phase 2: Librosa → melody features (note count, tempo, durations)
  Phase 3: MCNST   → English → Hindi text
  Phase 4: XTTS v2 → Hindi audio in original singer's voice (this file)
  Phase 5: Mixer   → Hindi vocals time-stretched + original instrumental

Install:
  pip install TTS soundfile librosa

XTTS v2 model will be auto-downloaded on first run (~1.8 GB).
"""

import numpy as np
from pathlib import Path
from typing import List, Optional


class VoiceCloneSynthesizer:
    """
    Zero-shot voice cloning TTS using XTTS v2.

    One reference audio clip → clone the voice for any text, in any language.
    No training. No RVC model. Works out of the box.
    """

    XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    SAMPLE_RATE = 24000  # XTTS v2 output sample rate

    def __init__(self, output_dir: str = "src/data/synthesized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tts = None

    # ------------------------------------------------------------------
    # Model loading — XTTS v2 (requires transformers < 4.33)
    # ------------------------------------------------------------------

    def _load_xtts(self) -> bool:
        """
        Load XTTS v2 for zero-shot voice cloning.

        KNOWN LIMITATION: XTTS v2 requires transformers < 4.33 (BeamSearchScorer
        was removed in later versions). If MCNST is in the same venv (which uses
        newer transformers for mBART), XTTS cannot load.
        Solution: run XTTS in a separate Python ≥3.10 venv:
          python3.10 -m venv xtts_env && pip install TTS transformers==4.27
        """
        if self._tts is not None:
            return True
        try:
            import os
            os.environ['COQUI_TOS_AGREED'] = '1'
            from TTS.api import TTS
            print("  Loading XTTS v2...")
            self._tts = TTS(model_name=self.XTTS_MODEL, progress_bar=True)
            print("  ✓ XTTS v2 loaded (zero-shot voice cloning active)")
            return True
        except ImportError:
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core synthesis — with fallback chain
    # ------------------------------------------------------------------

    def synthesize_line(self,
                         text: str,
                         reference_vocal: str,
                         output_name: str,
                         target_duration: Optional[float] = None) -> Optional[Path]:
        """
        Synthesize one Hindi line.

        Tries in order:
          1. XTTS v2       — zero-shot voice cloning (best, needs separate venv)
          2. gTTS          — Google TTS, clear Hindi, no voice cloning, needs internet
          3. espeak-ng     — offline, robotic, always works

        Args:
            text:              Hindi text in Devanagari
            reference_vocal:   Vocalist reference audio (used by XTTS only)
            output_name:       Output filename stem (no extension)
            target_duration:   Time-stretch output to this many seconds (optional)

        Returns:
            Path to output WAV, or None on failure.
        """
        output_path = self.output_dir / f"{output_name}.wav"

        # Try 1: XTTS v2 (zero-shot voice cloning)
        if self._load_xtts():
            try:
                self._tts.tts_to_file(
                    text=text,
                    language="hi",
                    speaker_wav=str(reference_vocal),
                    file_path=str(output_path)
                )
                print(f"  ✓ [XTTS] {output_path.name}")
                if target_duration and output_path.exists():
                    output_path = self._time_stretch(output_path, target_duration, output_name)
                return output_path
            except Exception as e:
                print(f"  ⚠ XTTS failed ({e}), falling back to gTTS")

        # Try 2: edge-tts (Microsoft Neural TTS — high quality Hindi, needs internet)
        edge_path = self._synthesize_edge_tts(text, output_name)
        if edge_path:
            if target_duration:
                edge_path = self._time_stretch(edge_path, target_duration, output_name)
            return edge_path

        # Try 3: gTTS (Google TTS — Hindi, no voice cloning)
        gtts_path = self._synthesize_gtts(text, output_name)
        if gtts_path:
            if target_duration:
                gtts_path = self._time_stretch(gtts_path, target_duration, output_name)
            return gtts_path

        # Try 4: espeak-ng (offline fallback)
        return self._synthesize_espeak(text, output_path)

    def _synthesize_edge_tts(self, text: str, output_name: str) -> Optional[Path]:
        """
        Microsoft Edge TTS — high-quality Neural Hindi TTS, needs internet.
        Uses hi-IN-SwaraNeural (female) voice. No package conflicts.
        Install: pip install edge-tts
        """
        try:
            import edge_tts
            import asyncio
            import librosa
            import soundfile as sf

            mp3_path = self.output_dir / f"{output_name}_tmp.mp3"

            async def _do_synthesize():
                communicate = edge_tts.Communicate(text, "hi-IN-SwaraNeural")
                await communicate.save(str(mp3_path))

            asyncio.run(_do_synthesize())

            y, sr = librosa.load(str(mp3_path), sr=self.SAMPLE_RATE)
            wav_path = self.output_dir / f"{output_name}.wav"
            sf.write(str(wav_path), y, sr)
            mp3_path.unlink(missing_ok=True)

            print(f"  ✓ [edge-tts] {wav_path.name}")
            return wav_path
        except ImportError:
            print("  ⚠ edge-tts not installed (pip install edge-tts), falling back to gTTS")
            return None
        except Exception as e:
            print(f"  ⚠ edge-tts failed ({e}), falling back to gTTS")
            return None

    def _synthesize_gtts(self, text: str, output_name: str) -> Optional[Path]:
        """
        Google TTS — clear Hindi pronunciation, needs internet.
        Saves to MP3 first, then converts to WAV via librosa (no ffmpeg needed).
        """
        try:
            from gtts import gTTS
            import librosa
            import soundfile as sf

            # Save MP3 to a temp file (librosa can load MP3 directly)
            mp3_path = self.output_dir / f"{output_name}_tmp.mp3"
            tts = gTTS(text=text, lang='hi', slow=False)
            tts.save(str(mp3_path))

            # Load with librosa (uses audioread backend, no ffmpeg needed for MP3)
            y, sr = librosa.load(str(mp3_path), sr=self.SAMPLE_RATE)
            wav_path = self.output_dir / f"{output_name}.wav"
            sf.write(str(wav_path), y, sr)
            mp3_path.unlink(missing_ok=True)

            print(f"  ✓ [gTTS] {wav_path.name}")
            return wav_path
        except Exception as e:
            print(f"  ⚠ gTTS failed ({e})")
            return None

    def _synthesize_espeak(self, text: str, output_path: Path) -> Optional[Path]:
        """espeak-ng — offline, robotic but always works."""
        import subprocess
        for cmd_name in ["espeak-ng", "espeak"]:
            try:
                result = subprocess.run(
                    [cmd_name, "-v", "hi", "-w", str(output_path), text],
                    capture_output=True, timeout=30
                )
                if result.returncode == 0 and output_path.exists():
                    print(f"  ✓ [espeak] {output_path.name}")
                    return output_path
            except FileNotFoundError:
                continue
        print("  ✗ No TTS backend available (XTTS/gTTS/espeak all failed)")
        return None

    def synthesize_song(self,
                         hindi_lines: List[str],
                         reference_vocal: str,
                         output_name: str = "hindi_vocals",
                         target_durations: Optional[List[float]] = None) -> Optional[Path]:
        """
        Synthesize all lines and concatenate into a single vocal track.

        Args:
            hindi_lines:       List of translated Hindi lines
            reference_vocal:   Path to original vocal audio for voice cloning
            output_name:       Output filename stem
            target_durations:  Target duration per line in seconds (from melody).
                               If None, uses natural TTS timing.

        Returns:
            Path to combined WAV file.
        """
        print(f"\n  Synthesizing {len(hindi_lines)} line(s) with voice cloning...")
        print(f"  Reference voice: {Path(reference_vocal).name}")

        line_paths = []
        for i, line in enumerate(hindi_lines):
            target_dur = (target_durations[i]
                          if target_durations and i < len(target_durations)
                          else None)
            path = self.synthesize_line(
                text=line,
                reference_vocal=reference_vocal,
                output_name=f"{output_name}_line{i}",
                target_duration=target_dur
            )
            if path:
                line_paths.append(path)
            else:
                print(f"  ✗ Line {i+1} synthesis failed, skipping")

        if not line_paths:
            print("  ✗ All lines failed — no output generated")
            return None

        output_path = self.output_dir / f"{output_name}.wav"
        self._concatenate(line_paths, output_path)
        return output_path

    # ------------------------------------------------------------------
    # Audio utilities
    # ------------------------------------------------------------------

    def _time_stretch(self, audio_path: Path, target_duration: float,
                       output_name: str) -> Path:
        """
        Time-stretch audio to match target_duration using librosa.
        Preserves pitch (unlike simple resampling).
        """
        try:
            import librosa
            import soundfile as sf

            y, sr = librosa.load(str(audio_path), sr=None)
            current_duration = len(y) / sr

            if current_duration <= 0 or target_duration <= 0:
                return audio_path

            rate = current_duration / target_duration   # stretch ratio
            # Clamp to avoid extreme distortion
            rate = max(0.5, min(2.0, rate))

            stretched = librosa.effects.time_stretch(y, rate=rate)
            stretched_path = self.output_dir / f"{output_name}_stretched.wav"
            sf.write(str(stretched_path), stretched, sr)

            actual = len(stretched) / sr
            print(f"    ↔ Time-stretched {current_duration:.1f}s → {actual:.1f}s "
                  f"(target: {target_duration:.1f}s, rate: {rate:.2f}x)")
            return stretched_path

        except ImportError:
            print("    ⚠ librosa/soundfile needed for time-stretching — skipping")
            return audio_path
        except Exception as e:
            print(f"    ⚠ Time-stretch failed ({e}) — using original timing")
            return audio_path

    def _concatenate(self, wav_paths: List[Path], output_path: Path,
                      silence_between: float = 0.25):
        """Concatenate WAV files with a short silence between lines."""
        try:
            import soundfile as sf

            all_audio = []
            sr = self.SAMPLE_RATE

            for path in wav_paths:
                y, file_sr = sf.read(str(path))
                if file_sr != sr:
                    # Resample to unified rate
                    import librosa
                    y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                all_audio.append(y)
                # Short silence between lines
                all_audio.append(np.zeros(int(sr * silence_between)))

            combined = np.concatenate(all_audio)
            sf.write(str(output_path), combined, sr)
            print(f"  ✓ Combined {len(wav_paths)} lines → {output_path.name}")

        except ImportError:
            print("  ✗ soundfile not installed — run: pip install soundfile")
        except Exception as e:
            print(f"  ✗ Concatenation error: {e}")


# ============================================================================
# Timing helper — estimate melody line durations from features
# ============================================================================

def estimate_line_durations(melody_features: np.ndarray,
                              notes_per_line: int,
                              tempo_bpm: float = 120.0) -> List[float]:
    """
    Estimate how long each lyric line should take based on melody note durations.

    Args:
        melody_features:  [num_notes, 5] array (col 2 = duration in beats)
        notes_per_line:   How many notes belong to each line
        tempo_bpm:        Tempo in beats per minute

    Returns:
        List of durations in seconds, one per line.
    """
    seconds_per_beat = 60.0 / tempo_bpm
    durations = []
    num_notes = len(melody_features)
    i = 0
    while i < num_notes:
        end = min(i + notes_per_line, num_notes)
        line_notes = melody_features[i:end, 2]  # duration column (beats)
        line_seconds = float(line_notes.sum()) * seconds_per_beat
        durations.append(max(1.0, line_seconds))  # at least 1 second per line
        i = end
    return durations


# ============================================================================
# CLI smoke test
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    synth = VoiceCloneSynthesizer()

    if len(sys.argv) < 3:
        print("Usage: python voice_clone_synthesizer.py <reference_vocal.wav> <hindi_text>")
        print("Example: python voice_clone_synthesizer.py vocals.wav 'चमक चमक छोटा तारा'")
        sys.exit(1)

    reference = sys.argv[1]
    text = sys.argv[2]

    result = synth.synthesize_line(
        text=text,
        reference_vocal=reference,
        output_name="clone_test"
    )

    if result:
        print(f"\n✓ Output: {result}")
    else:
        print("\n✗ Synthesis failed")
