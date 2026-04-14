"""
Audio Mixer — Combine converted vocals with original instrumental

Final stage of the audio pipeline:
  Input:  Hindi vocals (from VoiceConverter) + Instrumental (from Demucs)
  Output: Complete translated song
"""

import numpy as np
from pathlib import Path
from typing import Optional


class AudioMixer:
    """
    Mix vocals and instrumental into a final song.
    """
    
    def __init__(self, output_dir="data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def mix(self, vocals_path: str, instrumental_path: str,
            output_name: str = "final_song",
            vocal_volume: float = 1.2,
            instrumental_volume: float = 0.8) -> Optional[Path]:
        """
        Mix vocals and instrumental together.
        
        Args:
            vocals_path: Path to vocal audio (WAV)
            instrumental_path: Path to instrumental audio (WAV)
            output_name: Name for output file
            vocal_volume: Volume multiplier for vocals (>1 = louder)
            instrumental_volume: Volume multiplier for instrumental
        
        Returns:
            Path to final mixed audio
        """
        try:
            import soundfile as sf
        except ImportError:
            print("✗ soundfile not installed. Run: pip install soundfile")
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
            
            # Ensure same sample rate
            target_sr = max(vocals_sr, inst_sr)
            
            if vocals_sr != target_sr:
                vocals = self._resample(vocals, vocals_sr, target_sr)
            if inst_sr != target_sr:
                instrumental = self._resample(instrumental, inst_sr, target_sr)
            
            # Ensure mono (1D arrays)
            if vocals.ndim > 1:
                vocals = vocals.mean(axis=1)
            if instrumental.ndim > 1:
                instrumental = instrumental.mean(axis=1)
            
            # Match lengths (pad shorter with silence)
            max_len = max(len(vocals), len(instrumental))
            
            if len(vocals) < max_len:
                vocals = np.pad(vocals, (0, max_len - len(vocals)))
            if len(instrumental) < max_len:
                instrumental = np.pad(instrumental, (0, max_len - len(instrumental)))
            
            # Apply volume adjustments
            vocals = vocals * vocal_volume
            instrumental = instrumental * instrumental_volume
            
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
    
    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Simple resampling via linear interpolation."""
        ratio = to_sr / from_sr
        new_length = int(len(audio) * ratio)
        
        if audio.ndim == 1:
            return np.interp(
                np.linspace(0, len(audio) - 1, new_length),
                np.arange(len(audio)),
                audio
            )
        else:
            # Multi-channel
            result = np.zeros((new_length, audio.shape[1]))
            for ch in range(audio.shape[1]):
                result[:, ch] = np.interp(
                    np.linspace(0, len(audio) - 1, new_length),
                    np.arange(len(audio)),
                    audio[:, ch]
                )
            return result


# Test
if __name__ == "__main__":
    mixer = AudioMixer()
    
    # Check for separated stems
    separated_dir = Path("data/separated")
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
