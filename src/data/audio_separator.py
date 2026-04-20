"""
Audio Separator - Split songs into vocals and instrumentals using Demucs

Phase 1 of the audio pipeline:
  Input:  Any song audio file (MP3, WAV, FLAC)
  Output: vocals.wav + instrumental.wav

After Demucs separation, vocals are post-processed to remove:
  1. Reverb / delay tails  — via spectral noise reduction (noisereduce)
  2. Backing vocals        — via mid-side extraction (lead vocal is mono/
                             centered; backing vocals + reverb live in the
                             stereo Side channel, which we discard)

This produces clean, dry lead vocals suitable for RVC training and
voice-clone synthesis without echo hallucination artefacts.

Install: pip install demucs noisereduce
"""

import subprocess
import shutil
import numpy as np
from pathlib import Path


class AudioSeparator:
    """
    Separate a song into vocal and instrumental stems using Demucs,
    then clean the vocals (de-reverb + backing-vocal reduction).
    """

    def __init__(self, output_dir="src/data/separated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Vocal post-processing
    # ------------------------------------------------------------------

    def clean_vocals(self, vocals_path: Path, sr: int = 44100) -> np.ndarray:
        """
        Return a cleaned mono vocal array from a (possibly stereo) vocal WAV.

        Two-stage process:
          Stage 1 — Mid-side extraction
            Lead vocals are recorded mono and panned centre; backing vocals
            and reverb returns are spread across the stereo field.
            The Mid channel  (L+R)/2  retains the lead vocal while
            suppressing the Side channel  (L-R)/2  which carries backing
            vocals, room reverb, and stereo FX.

          Stage 2 — Spectral de-reverb (noisereduce)
            Uses a stationary noise estimate taken from low-energy frames
            (the reverb tail / ambience) to subtract residual room sound.

        Args:
            vocals_path: Path to vocals.wav (stereo or mono)
            sr:          Sample rate of the file

        Returns:
            Cleaned mono float32 numpy array, or None on failure.
        """
        try:
            import soundfile as sf
            audio, file_sr = sf.read(str(vocals_path))
        except Exception as e:
            print(f"  ⚠ clean_vocals: could not read {vocals_path}: {e}")
            return None

        # ── Stage 1: Mid-side backing-vocal reduction ──────────────────
        if audio.ndim == 2 and audio.shape[1] == 2:
            left, right = audio[:, 0], audio[:, 1]
            mid  = (left + right) / 2.0   # lead vocal energy concentrated here
            side = (left - right) / 2.0   # backing vocals + stereo reverb

            # Soft suppression: keep Mid, subtract a fraction of |Side|
            # A factor of 0.8 removes most backing without clipping the lead.
            side_mag = np.abs(side)
            vocal_mono = mid - 0.8 * np.sign(mid) * side_mag
            vocal_mono = np.clip(vocal_mono, -1.0, 1.0)
            print("  ✓ Mid-side extraction applied (backing vocal reduction)")
        else:
            # Already mono
            vocal_mono = audio if audio.ndim == 1 else audio[:, 0]

        # ── Stage 2: Spectral de-reverb (noisereduce) ─────────────────
        try:
            import noisereduce as nr
            # Use only low-energy frames as the reverb/noise profile.
            # time_mask_smooth_ms targets reverb tails specifically.
            cleaned = nr.reduce_noise(
                y=vocal_mono,
                sr=file_sr,
                stationary=False,       # non-stationary catches reverb decay
                prop_decrease=0.85,     # aggressively remove reverb
                time_mask_smooth_ms=100,
                freq_mask_smooth_hz=500,
            )
            print("  ✓ Spectral de-reverb applied (noisereduce)")
            return cleaned.astype(np.float32)
        except ImportError:
            print("  ⚠ noisereduce not installed — skipping de-reverb step")
            return vocal_mono.astype(np.float32)
        except Exception as e:
            print(f"  ⚠ De-reverb failed ({e}) — using mid-side result only")
            return vocal_mono.astype(np.float32)
    
    def separate(self, audio_path, model="htdemucs", device="auto"):
        """
        Separate audio into vocals and instrumentals.

        Args:
            audio_path: Path to audio file (MP3, WAV, FLAC)
            model: Demucs model to use
                   - "htdemucs"    : best quality (default)
                   - "htdemucs_ft" : fine-tuned, even better but slower
                   - "mdx_extra"   : good alternative
            device: "auto" (detect), "mps", "cuda", or "cpu"

        Returns:
            dict with paths: {'vocals': Path, 'instrumental': Path, 'song_name': str}
            or None on failure
        """
        import torch
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            print(f"✗ File not found: {audio_path}")
            return None
        
        song_name = audio_path.stem
        print(f"\n{'='*60}")
        print(f"Separating: {audio_path.name}")
        print(f"Model: {model}")
        print(f"{'='*60}\n")
        
        try:
            import sys
            # Run Demucs
            # --two-stems=vocals : only split into vocals + everything else
            # -o : output directory
            # --device cpu : use CPU (safe for 2GB GPU)
            cmd = [
                sys.executable, "-m", "demucs",
                "--two-stems=vocals",
                "-o", str(self.output_dir),
                "--device", device,
                "-n", model,
                str(audio_path)
            ]
            
            print("Running Demucs (this may take 1-2 minutes)...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 min timeout
            )
            
            if result.returncode != 0:
                print(f"✗ Demucs failed:\n{result.stderr}")
                return None
            
            # Demucs outputs to: output_dir/model_name/song_name/vocals.wav
            demucs_output = self.output_dir / model / song_name
            vocals_path = demucs_output / "vocals.wav"
            no_vocals_path = demucs_output / "no_vocals.wav"
            
            if not vocals_path.exists():
                print(f"✗ Expected output not found: {vocals_path}")
                return None
            
            # Copy to cleaner location
            clean_dir = self.output_dir / song_name
            clean_dir.mkdir(exist_ok=True)

            raw_vocals      = clean_dir / "vocals_raw.wav"
            clean_vocals    = clean_dir / "vocals.wav"
            clean_instrumental = clean_dir / "instrumental.wav"

            shutil.copy2(vocals_path, raw_vocals)
            shutil.copy2(no_vocals_path, clean_instrumental)

            print(f"\n  Post-processing vocals (de-reverb + backing vocal reduction)...")
            import soundfile as sf
            _, file_sr = sf.read(str(raw_vocals), always_2d=False)
            cleaned = self.clean_vocals(raw_vocals, sr=file_sr)
            if cleaned is not None:
                sf.write(str(clean_vocals), cleaned, file_sr)
                print(f"  ✓ Clean vocals saved: {clean_vocals}")
            else:
                # Fallback: use raw Demucs output
                shutil.copy2(raw_vocals, clean_vocals)
                print(f"  ⚠ Vocal cleaning failed — using raw Demucs output")

            print(f"\n✓ Separation complete!")
            print(f"  Vocals (clean): {clean_vocals}")
            print(f"  Vocals (raw):   {raw_vocals}")
            print(f"  Instrumental:   {clean_instrumental}")

            return {
                'vocals': clean_vocals,
                'instrumental': clean_instrumental,
                'song_name': song_name
            }
        
        except FileNotFoundError:
            print("✗ Demucs not installed. Run: pip install demucs")
            return None
        except subprocess.TimeoutExpired:
            print("✗ Demucs timed out (>10 minutes)")
            return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def get_separated(self, song_name, require_clean: bool = True):
        """
        Get paths to previously separated stems.

        Args:
            song_name:     Stem name of the song file.
            require_clean: If True (default), only return cached stems that
                           went through vocal cleaning (vocals_raw.wav present).
                           Set to False to accept old Demucs-only outputs.

        Returns:
            dict with paths or None if not found / not cleaned.
        """
        clean_dir = self.output_dir / song_name
        vocals = clean_dir / "vocals.wav"
        instrumental = clean_dir / "instrumental.wav"

        if not (vocals.exists() and instrumental.exists()):
            return None

        # If the caller wants cleaned vocals, check that cleaning was applied.
        # Cleaned runs produce vocals_raw.wav alongside vocals.wav.
        if require_clean and not (clean_dir / "vocals_raw.wav").exists():
            return None  # old uncleaned cache — caller should re-separate

        return {
            'vocals': vocals,
            'instrumental': instrumental,
            'song_name': song_name
        }


# Test
if __name__ == "__main__":
    import sys
    
    separator = AudioSeparator()
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Default test — look for any audio file in data/raw_audio/
        raw_dir = Path("src/data/raw_audio")
        if raw_dir.exists():
            audio_files = list(raw_dir.glob("*.mp3")) + list(raw_dir.glob("*.wav"))
            if audio_files:
                audio_file = str(audio_files[0])
            else:
                print("No audio files found in src/data/raw_audio/")
                print("Usage: python audio_separator.py <path_to_song.mp3>")
                sys.exit(1)
        else:
            print("Create src/data/raw_audio/ and place a song file there.")
            print("Usage: python audio_separator.py <path_to_song.mp3>")
            sys.exit(1)
    
    result = separator.separate(audio_file)
    
    if result:
        print(f"\n✓ Ready for Phase 2!")
        print(f"  Next: extract melody from {result['vocals']}")
