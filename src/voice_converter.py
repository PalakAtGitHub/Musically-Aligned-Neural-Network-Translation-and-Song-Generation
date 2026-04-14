"""
Voice Converter — Convert synthesized vocals to original singer's voice

Uses RVC (Retrieval-based Voice Conversion) to transfer the timbre
of the original singer onto the synthesized Hindi vocals.

Pipeline:
  1. Extract speaker embedding from original vocals (reference)
  2. Extract content features from synthesized Hindi vocals (HuBERT)
  3. Generate audio with original speaker's timbre

Install:
  pip install rvc-python
  (or use manual RVC integration — see https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
"""

import subprocess
from pathlib import Path
from typing import Optional


class VoiceConverter:
    """
    Convert any voice to sound like a target speaker using RVC.
    """
    
    def __init__(self, output_dir="data/converted"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rvc = None
    
    def _load_rvc(self):
        """Lazy-load RVC to avoid import errors if not installed."""
        if self._rvc is not None:
            return True
        
        try:
            from rvc_python.infer import RVCInference
            self._rvc = RVCInference(device="cpu")
            return True
        except ImportError:
            print("  ✗ rvc-python not installed. Run: pip install rvc-python")
            return False
        except Exception as e:
            print(f"  ✗ RVC init error: {e}")
            return False
    
    def train_voice_model(self, reference_vocals_path: str,
                           model_name: str = "singer",
                           epochs: int = 100) -> Optional[Path]:
        """
        Train an RVC model on the original singer's voice.
        
        This only needs to be done ONCE per singer.
        Uses the isolated vocals from Demucs (Phase 1) as training data.
        
        Args:
            reference_vocals_path: Path to clean vocal audio (from Demucs)
            model_name: Name for the voice model
            epochs: Training epochs (100 = ~10 min on GPU, ~1 hr on CPU)
        
        Returns:
            Path to trained model file (.pth)
        """
        print(f"\n{'='*60}")
        print(f"Training RVC voice model: {model_name}")
        print(f"Reference: {reference_vocals_path}")
        print(f"{'='*60}\n")
        
        if not self._load_rvc():
            return None
        
        try:
            model_path = self.output_dir / f"{model_name}.pth"
            
            # RVC training pipeline
            self._rvc.train(
                dataset_path=str(reference_vocals_path),
                model_name=model_name,
                epochs=epochs
            )
            
            print(f"  ✓ Voice model trained: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"  ✗ Training error: {e}")
            print("  → For manual training, use the RVC WebUI:")
            print("    https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI")
            return None
    
    def convert(self, source_audio: str, model_path: str,
                 output_name: str = "converted",
                 pitch_shift: int = 0) -> Optional[Path]:
        """
        Convert synthesized vocals to target singer's voice.
        
        Args:
            source_audio: Path to Hindi TTS output (any voice)
            model_path: Path to trained RVC model (.pth)
            output_name: Name for output file
            pitch_shift: Semitones to shift (0 = same, +12 = octave up)
        
        Returns:
            Path to converted audio
        """
        print(f"  Converting voice: {Path(source_audio).name}")
        
        if not self._load_rvc():
            return self._fallback_convert(source_audio, output_name)
        
        output_path = self.output_dir / f"{output_name}.wav"
        
        try:
            self._rvc.load_model(str(model_path))
            self._rvc.infer(
                input_path=str(source_audio),
                output_path=str(output_path),
                f0_up_key=pitch_shift
            )
            
            print(f"  ✓ Voice converted: {output_path.name}")
            return output_path
            
        except Exception as e:
            print(f"  ✗ Conversion error: {e}")
            return self._fallback_convert(source_audio, output_name)
    
    def _fallback_convert(self, source_audio: str, output_name: str) -> Optional[Path]:
        """
        Fallback: just copy the source audio (no voice conversion).
        This lets the pipeline work end-to-end even without RVC.
        """
        import shutil
        
        output_path = self.output_dir / f"{output_name}.wav"
        
        try:
            shutil.copy2(source_audio, output_path)
            print(f"  ⚠️  RVC unavailable — using original TTS voice")
            print(f"     Output: {output_path}")
            return output_path
        except Exception as e:
            print(f"  ✗ Fallback copy error: {e}")
            return None


# Test
if __name__ == "__main__":
    converter = VoiceConverter()
    
    # Check if RVC is available
    if converter._load_rvc():
        print("✓ RVC is available!")
    else:
        print("⚠️  RVC not installed — voice conversion will be skipped")
        print("   The pipeline will still work, just with the TTS voice.")
        print("   Install later: pip install rvc-python")
