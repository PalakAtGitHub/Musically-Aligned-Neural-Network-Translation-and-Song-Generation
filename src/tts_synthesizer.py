"""
Hindi TTS Synthesizer for Song Translation

Takes translated Hindi lyrics + melody timing and produces
spoken/sung audio.

Two modes:
  1. espeak-ng  — fast, CPU-only, robotic but functional
  2. Coqui TTS  — neural, much more natural, needs ~2GB RAM

The output is "raw" Hindi vocals that will later be converted
to the original singer's voice via RVC (voice_converter.py).

Install:
  Mode 1: pip install pyttsx3  (or just use espeak-ng CLI)
  Mode 2: pip install TTS
"""

import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List


class TTSSynthesizer:
    """
    Synthesize Hindi speech/singing from translated lyrics.
    """
    
    def __init__(self, output_dir="data/synthesized", mode="espeak"):
        """
        Args:
            output_dir: Where to save generated audio
            mode: "espeak" (CPU, fast) or "coqui" (neural, natural)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
    
    def synthesize(self, hindi_text: str, output_name: str = "output",
                    speed: int = 130, pitch: int = 50) -> Optional[Path]:
        """
        Synthesize Hindi text to audio.
        
        Args:
            hindi_text: Hindi text in Devanagari
            output_name: Name for output file (without extension)
            speed: Words per minute (lower = slower, more song-like)
            pitch: Pitch (0-99, 50 = normal)
        
        Returns:
            Path to generated WAV file, or None
        """
        output_path = self.output_dir / f"{output_name}.wav"
        
        if self.mode == "espeak":
            return self._synthesize_espeak(hindi_text, output_path, speed, pitch)
        elif self.mode == "coqui":
            return self._synthesize_coqui(hindi_text, output_path)
        else:
            print(f"✗ Unknown mode: {self.mode}")
            return None
    
    def synthesize_song(self, hindi_lines: List[str], 
                         durations_per_line: Optional[List[float]] = None,
                         output_name: str = "song") -> Optional[Path]:
        """
        Synthesize multiple lines into a single audio file.
        
        Args:
            hindi_lines: List of Hindi text lines
            durations_per_line: Target duration in seconds per line
            output_name: Name for output file
        
        Returns:
            Path to combined WAV file
        """
        line_paths = []
        
        for i, line in enumerate(hindi_lines):
            # Adjust speed based on target duration if available
            speed = 130  # default
            if durations_per_line and i < len(durations_per_line):
                # Rough: more syllables in less time = faster
                from src.utils.syllable_utils import count_hindi_syllables
                num_syl = count_hindi_syllables(line)
                if durations_per_line[i] > 0 and num_syl > 0:
                    # Approximate syllables per second → words per minute
                    syl_per_sec = num_syl / durations_per_line[i]
                    speed = max(80, min(200, int(syl_per_sec * 60)))
            
            path = self.synthesize(line, f"{output_name}_line{i}", speed=speed)
            if path:
                line_paths.append(path)
        
        if not line_paths:
            return None
        
        # Concatenate all lines
        output_path = self.output_dir / f"{output_name}.wav"
        self._concatenate_wav(line_paths, output_path)
        
        return output_path
    
    def _synthesize_espeak(self, text: str, output_path: Path,
                            speed: int = 130, pitch: int = 50) -> Optional[Path]:
        """Use espeak-ng for synthesis (CPU-only, fast)."""
        try:
            cmd = [
                "espeak-ng",
                "-v", "hi",           # Hindi voice
                "-s", str(speed),     # Speed (words per minute)
                "-p", str(pitch),     # Pitch
                "-w", str(output_path),  # Write to WAV
                text
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                # Try without -ng suffix (some Windows installs)
                cmd[0] = "espeak"
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and output_path.exists():
                print(f"  ✓ Synthesized: {output_path.name}")
                return output_path
            else:
                print(f"  ✗ espeak failed: {result.stderr}")
                return None
                
        except FileNotFoundError:
            print("  ✗ espeak-ng not installed.")
            print("    Windows: download from https://github.com/espeak-ng/espeak-ng/releases")
            print("    Linux: sudo apt install espeak-ng")
            return None
        except Exception as e:
            print(f"  ✗ TTS error: {e}")
            return None
    
    def _synthesize_coqui(self, text: str, output_path: Path) -> Optional[Path]:
        """Use Coqui TTS for synthesis (neural, natural sounding)."""
        try:
            from TTS.api import TTS
            
            # Use a multilingual model that supports Hindi
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            
            tts.tts_to_file(
                text=text,
                language="hi",
                file_path=str(output_path)
            )
            
            print(f"  ✓ Synthesized (Coqui): {output_path.name}")
            return output_path
            
        except ImportError:
            print("  ✗ Coqui TTS not installed. Run: pip install TTS")
            return None
        except Exception as e:
            print(f"  ✗ Coqui TTS error: {e}")
            # Fallback to espeak
            print("  → Falling back to espeak-ng...")
            return self._synthesize_espeak(text, output_path)
    
    def _concatenate_wav(self, wav_paths: List[Path], output_path: Path):
        """Concatenate multiple WAV files into one."""
        try:
            import soundfile as sf
            
            all_audio = []
            sample_rate = None
            
            for path in wav_paths:
                audio, sr = sf.read(str(path))
                if sample_rate is None:
                    sample_rate = sr
                
                # Resample if needed
                if sr != sample_rate:
                    # Simple resampling via interpolation
                    ratio = sample_rate / sr
                    new_length = int(len(audio) * ratio)
                    audio = np.interp(
                        np.linspace(0, len(audio) - 1, new_length),
                        np.arange(len(audio)),
                        audio
                    )
                
                all_audio.append(audio)
                
                # Add small silence between lines (0.3s)
                silence = np.zeros(int(sample_rate * 0.3))
                all_audio.append(silence)
            
            combined = np.concatenate(all_audio)
            sf.write(str(output_path), combined, sample_rate)
            print(f"  ✓ Combined {len(wav_paths)} lines → {output_path.name}")
            
        except ImportError:
            print("  ✗ soundfile not installed. Run: pip install soundfile")
        except Exception as e:
            print(f"  ✗ Concatenation error: {e}")


# Test
if __name__ == "__main__":
    synth = TTSSynthesizer(mode="espeak")
    
    result = synth.synthesize(
        "चमक चमक छोटा तारा",
        output_name="test_tts",
        speed=110,  # slower = more song-like
        pitch=55
    )
    
    if result:
        print(f"\n✓ Generated: {result}")
    else:
        print("\n✗ Synthesis failed — see errors above")
