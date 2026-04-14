"""
Audio Separator - Split songs into vocals and instrumentals using Demucs

Phase 1 of the audio pipeline:
  Input:  Any song audio file (MP3, WAV, FLAC)
  Output: vocals.wav + instrumental.wav

Uses Meta's Demucs (Hybrid Transformer) for state-of-the-art
source separation.

Install: pip install demucs
"""

import subprocess
import shutil
from pathlib import Path


class AudioSeparator:
    """
    Separate a song into vocal and instrumental stems using Demucs.
    """
    
    def __init__(self, output_dir="data/separated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def separate(self, audio_path, model="htdemucs"):
        """
        Separate audio into vocals and instrumentals.
        
        Args:
            audio_path: Path to audio file (MP3, WAV, FLAC)
            model: Demucs model to use
                   - "htdemucs"    : best quality (default)
                   - "htdemucs_ft" : fine-tuned, even better but slower
                   - "mdx_extra"   : good alternative
        
        Returns:
            dict with paths: {'vocals': Path, 'instrumental': Path, 'song_name': str}
            or None on failure
        """
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
            # Run Demucs
            # --two-stems=vocals : only split into vocals + everything else
            # -o : output directory
            # --device cpu : use CPU (safe for 2GB GPU)
            cmd = [
                "python", "-m", "demucs",
                "--two-stems=vocals",
                "-o", str(self.output_dir),
                "--device", "cpu",
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
            
            clean_vocals = clean_dir / "vocals.wav"
            clean_instrumental = clean_dir / "instrumental.wav"
            
            shutil.copy2(vocals_path, clean_vocals)
            shutil.copy2(no_vocals_path, clean_instrumental)
            
            print(f"✓ Separation complete!")
            print(f"  Vocals:       {clean_vocals}")
            print(f"  Instrumental: {clean_instrumental}")
            
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
    
    def get_separated(self, song_name):
        """
        Get paths to previously separated stems.
        
        Returns:
            dict with paths or None if not found
        """
        clean_dir = self.output_dir / song_name
        vocals = clean_dir / "vocals.wav"
        instrumental = clean_dir / "instrumental.wav"
        
        if vocals.exists() and instrumental.exists():
            return {
                'vocals': vocals,
                'instrumental': instrumental,
                'song_name': song_name
            }
        return None


# Test
if __name__ == "__main__":
    import sys
    
    separator = AudioSeparator()
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Default test — look for any audio file in data/raw_audio/
        raw_dir = Path("data/raw_audio")
        if raw_dir.exists():
            audio_files = list(raw_dir.glob("*.mp3")) + list(raw_dir.glob("*.wav"))
            if audio_files:
                audio_file = str(audio_files[0])
            else:
                print("No audio files found in data/raw_audio/")
                print("Usage: python audio_separator.py <path_to_song.mp3>")
                sys.exit(1)
        else:
            print("Create data/raw_audio/ and place a song file there.")
            print("Usage: python audio_separator.py <path_to_song.mp3>")
            sys.exit(1)
    
    result = separator.separate(audio_file)
    
    if result:
        print(f"\n✓ Ready for Phase 2!")
        print(f"  Next: extract melody from {result['vocals']}")
