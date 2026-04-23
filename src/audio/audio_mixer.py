"""
Audio Mixer — Combine converted vocals with original instrumental

Final stage of the audio pipeline:
  Input:  Hindi vocals (from VoiceConverter) + Instrumental (from Demucs)
  Output: Complete translated song

Processing chain applied before mixing:
  Vocals → high-pass (remove rumble) → compressor (even out dynamics)
         → subtle reverb (match instrumental space) → gain makeup
  Instrumental → sidechain ducking when vocals are present
  Sum → soft-knee limiter to avoid clipping

Requires: pip install pedalboard  (Spotify's open-source DSP library).
Falls back to raw sum if pedalboard is unavailable (original behaviour).
"""

import numpy as np
from pathlib import Path
from typing import Optional


class AudioMixer:
    """
    Mix vocals and instrumental into a final song.
    """
    
    def __init__(self, output_dir="src/data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def mix(self, vocals_path: str, instrumental_path: str,
            output_name: str = "final_song",
            vocal_db: float = 0.0,
            instrumental_db: float = 0.0,
            enable_dsp: bool = True,
            duck_db: float = 6.0,
            duck_attack_ms: float = 20.0,
            duck_release_ms: float = 250.0,
            reverb_wet: float = 0.15) -> Optional[Path]:
        """
        Mix vocals and instrumental together.

        RMS-normalizes both stems so the instrumental is clearly audible,
        then applies optional dB offsets. Default: equal loudness.

        If enable_dsp is True and pedalboard is installed, additionally
        runs a vocal production chain (high-pass, compressor, subtle
        reverb, makeup gain) and sidechain-ducks the instrumental when
        vocals are present. Falls through to raw sum if pedalboard is
        missing.

        Args:
            vocals_path:         Path to vocal audio (WAV)
            instrumental_path:   Path to instrumental audio (WAV)
            output_name:         Name for output file
            vocal_db:            dB boost/cut for vocals (0 = match RMS target)
            instrumental_db:     dB boost/cut for instrumental (0 = target)
            enable_dsp:          Apply vocal production chain + ducking.
            duck_db:             dB to pull the instrumental down when
                                 vocals are active (typical: 3–8 dB).
            duck_attack_ms:      How fast the duck clamps on vocal onset.
            duck_release_ms:     How fast the duck releases between phrases.
            reverb_wet:          0..1. Subtle reverb on vocals so they sit
                                 in the same space as the Demucs instrumental.

        Returns:
            Path to final mixed audio
        """
        try:
            import soundfile as sf
            import librosa
        except ImportError:
            print("✗ soundfile/librosa not installed.")
            return None

        print(f"\n{'='*60}")
        print(f"Mixing audio")
        print(f"{'='*60}")
        print(f"  Vocals:       {vocals_path}")
        print(f"  Instrumental: {instrumental_path}")

        try:
            # Load both audio files
            vocals, vocals_sr = sf.read(str(vocals_path))
            instrumental, inst_sr = sf.read(str(instrumental_path))

            # Ensure mono
            if vocals.ndim > 1:
                vocals = vocals.mean(axis=1)
            if instrumental.ndim > 1:
                instrumental = instrumental.mean(axis=1)

            # Resample to common rate using librosa (high quality)
            target_sr = max(vocals_sr, inst_sr)
            if vocals_sr != target_sr:
                vocals = librosa.resample(vocals, orig_sr=vocals_sr, target_sr=target_sr)
            if inst_sr != target_sr:
                instrumental = librosa.resample(instrumental, orig_sr=inst_sr, target_sr=target_sr)

            # Match lengths (pad shorter with silence)
            max_len = max(len(vocals), len(instrumental))
            if len(vocals) < max_len:
                vocals = np.pad(vocals, (0, max_len - len(vocals)))
            if len(instrumental) < max_len:
                instrumental = np.pad(instrumental, (0, max_len - len(instrumental)))

            # RMS-normalize both to a common target, then apply dB offsets
            target_rms = 0.08
            voc_rms = np.sqrt(np.mean(vocals ** 2)) + 1e-8
            inst_rms = np.sqrt(np.mean(instrumental ** 2)) + 1e-8

            vocals = vocals * (target_rms / voc_rms) * (10 ** (vocal_db / 20))
            instrumental = instrumental * (target_rms / inst_rms) * (10 ** (instrumental_db / 20))

            print(f"  Vocal RMS (orig):       {voc_rms:.4f} → normalized to {target_rms}")
            print(f"  Instrumental RMS (orig): {inst_rms:.4f} → normalized to {target_rms}")

            # Optional DSP chain: vocal production + sidechain ducking.
            if enable_dsp:
                vocals, instrumental = self._apply_dsp(
                    vocals.astype(np.float32),
                    instrumental.astype(np.float32),
                    target_sr,
                    duck_db=duck_db,
                    duck_attack_ms=duck_attack_ms,
                    duck_release_ms=duck_release_ms,
                    reverb_wet=reverb_wet,
                )

            # Mix
            mixed = vocals + instrumental

            # Normalize to prevent clipping
            peak = np.max(np.abs(mixed))
            if peak > 0.95:
                mixed = mixed * (0.95 / peak)

            # Save
            output_path = self.output_dir / f"{output_name}.wav"
            sf.write(str(output_path), mixed, target_sr)

            duration = len(mixed) / target_sr
            print(f"\n  ✓ Mixed successfully!")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Output: {output_path}")

            return output_path

        except Exception as e:
            print(f"  ✗ Mixing error: {e}")
            return None

    # ------------------------------------------------------------------
    # DSP chain (pedalboard + numpy sidechain ducking)
    # ------------------------------------------------------------------

    def _apply_dsp(self, vocals, instrumental, sr,
                    duck_db=6.0, duck_attack_ms=20.0,
                    duck_release_ms=250.0, reverb_wet=0.15):
        """Apply vocal processing + sidechain ducking.

        Returns (processed_vocals, processed_instrumental).
        Falls back to identity if pedalboard is unavailable.
        """
        try:
            from pedalboard import (
                Pedalboard, HighpassFilter, Compressor, Reverb, Gain,
            )
        except ImportError:
            print("  ⚠ pedalboard not installed (pip install pedalboard) — "
                  "skipping DSP, using raw sum")
            return vocals, instrumental

        vocal_board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80.0),
            Compressor(threshold_db=-16.0, ratio=3.0,
                        attack_ms=5.0, release_ms=120.0),
            Reverb(room_size=0.25, damping=0.5,
                    wet_level=reverb_wet, dry_level=1.0 - reverb_wet,
                    width=0.8),
            Gain(gain_db=2.0),
        ])
        vocals_proc = vocal_board(vocals, sr)
        print("  ✓ Vocals: HP80 → Comp(3:1, -16dB) → Reverb(0.25) → +2dB")

        # Sidechain ducking driven by vocal envelope
        envelope = self._amplitude_envelope(
            vocals_proc, sr,
            attack_ms=duck_attack_ms,
            release_ms=duck_release_ms,
        )
        env_max = float(np.max(envelope)) + 1e-8
        env_norm = envelope / env_max
        duck_linear = 10.0 ** (-duck_db / 20.0)
        gain_curve = 1.0 - env_norm * (1.0 - duck_linear)
        inst_proc = instrumental * gain_curve
        print(f"  ✓ Instrumental: sidechain ducked −{duck_db:.1f}dB on vocal")

        return vocals_proc, inst_proc

    @staticmethod
    def _amplitude_envelope(signal, sr, attack_ms=20.0, release_ms=250.0):
        """Classic attack/release envelope follower (one-pole IIR).

        Drives sidechain ducking so the instrumental pulls down quickly
        on vocal onsets and recovers smoothly between phrases.
        """
        abs_sig = np.abs(signal)
        attack = np.exp(-1.0 / (sr * attack_ms / 1000.0))
        release = np.exp(-1.0 / (sr * release_ms / 1000.0))
        env = np.zeros_like(abs_sig)
        prev = 0.0
        for i in range(len(abs_sig)):
            x = abs_sig[i]
            coef = attack if x > prev else release
            prev = coef * prev + (1.0 - coef) * x
            env[i] = prev
        return env


# Test
if __name__ == "__main__":
    mixer = AudioMixer()
    
    # Check for separated stems
    separated_dir = Path("src/data/separated")
    if separated_dir.exists():
        songs = list(separated_dir.iterdir())
        if songs:
            song_dir = songs[0]
            vocals = song_dir / "vocals.wav"
            instrumental = song_dir / "instrumental.wav"
            
            if vocals.exists() and instrumental.exists():
                result = mixer.mix(
                    str(vocals),
                    str(instrumental),
                    output_name=f"{song_dir.name}_mixed"
                )
            else:
                print("No separated stems found. Run audio_separator.py first.")
        else:
            print("No songs in data/separated/")
    else:
        print("Run audio separation first: python -m src.main separate <song.mp3>")
