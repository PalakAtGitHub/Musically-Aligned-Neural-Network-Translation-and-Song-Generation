"""
FMA Dataset Builder — Creates training data from FMA-medium audio files.

Correct pipeline per song:
  1. Demucs htdemucs    → vocals.wav  (source separation, cached)
  2. Whisper base       → English lyric segments  (ASR on vocals)
  3. IndicTrans2 MT    → Hindi translation per lyric line
  4. AudioMelodyExtractor → [num_notes, 5] features from vocals
  5. Melody split       → one note-slice per lyric line
  6. Tokenize (IndicTrans2) → src_ids + tgt_ids
  7. Save example       → same format as DatasetBuilder

Song-level train/test split:
  - MP3s shuffled with a fixed seed for reproducibility.
  - 10% of songs → fma_test_data.pt   (held out, never seen during training)
  - 90% of songs → fma_train_data.pt  (used by Trainer)

Skip conditions (a song is skipped if):
  - Demucs fails / times out
  - Whisper returns < MIN_SEGMENTS lyric segments  (likely instrumental)
  - Each segment has < MIN_WORDS words             (noise / very sparse vocals)
  - Melody extraction returns no notes

Resume support:
  - Processed song IDs are logged to processed_dir/fma_progress.json
  - Re-running the builder skips already-processed songs automatically.

Install:
  pip install openai-whisper demucs librosa soundfile

Usage:
  cd musically-aligned-translation/
  python -m src.data.fma_data_builder              # up to 200 songs
  python -m src.data.fma_data_builder --max 500    # up to 500 songs
"""

import argparse
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.data.audio_melody_extractor import AudioMelodyExtractor
from src.data.audio_separator import AudioSeparator
from src.utils.syllable_utils import count_english_syllables, count_hindi_syllables
from src.utils.phoneme_utils import get_stress_pattern

# Minimum quality thresholds for vocal content
MIN_SEGMENTS = 2    # at least this many Whisper lyric segments
MIN_WORDS = 3       # minimum words per segment to keep it


class FMADatasetBuilder:
    """
    Build MCNST training/test datasets from FMA-medium MP3 files.

    Each processed song contributes one training example per lyric segment —
    the English lyric is real (from Whisper ASR), the Hindi translation is
    from IndicTrans2, and the melody features come from the same separated
    vocals track.
    """

    TRANSLATION_MODEL = "ai4bharat/indictrans2-en-indic-1B"

    def __init__(self,
                 fma_dir: str = "src/data/fma_medium",
                 separated_dir: str = "src/data/separated",
                 processed_dir: str = "src/data/processed",
                 seed: int = 42,
                 whisper_model: str = "base",
                 device: str = "auto"):
        self.fma_dir = Path(fma_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.whisper_model_name = whisper_model

        # Auto-detect best available device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        # Whisper uses sparse tensors not supported on MPS — always run on CPU
        self.whisper_device = "cpu" if device == "mps" else device
        print(f"Device: {self.device}  (Whisper: {self.whisper_device})")

        if not self.fma_dir.exists():
            raise FileNotFoundError(f"FMA directory not found: {self.fma_dir}")

        self.IT2_MODEL = self.TRANSLATION_MODEL  # same model used for labels
        print(f"Loading IndicTrans2 tokenizer  ({self.IT2_MODEL})...")
        self.it2_tokenizer = AutoTokenizer.from_pretrained(
            self.IT2_MODEL, trust_remote_code=True
        )

        print(f"Loading translation model  ({self.TRANSLATION_MODEL})...")
        # IndicTrans2 serves as BOTH the label generator and the downstream
        # MCNST backbone. Using the same model for both means training labels
        # and the student model share a ceiling, but it is a vastly better
        # teacher than MarianMT opus-mt-en-hi on lyric-like text.
        self.mt_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.IT2_MODEL, trust_remote_code=True
        )
        self.mt_model.to(self.device)
        self.mt_model.eval()
        # Translation tokenizer is the SAME tokenizer as it2_tokenizer —
        # alias for readability below.
        self.mt_tokenizer = self.it2_tokenizer

        # Lazy-load Whisper (large download — only on first _transcribe call)
        self._whisper = None

        self.separator = AudioSeparator(output_dir=separated_dir)
        self.extractor = AudioMelodyExtractor()

        # Progress log — lets us resume interrupted runs
        self._progress_path = self.processed_dir / "fma_progress.json"
        self._progress = self._load_progress()

    # ------------------------------------------------------------------
    # Progress / resume helpers
    # ------------------------------------------------------------------

    def _load_progress(self) -> Dict:
        if self._progress_path.exists():
            with open(self._progress_path) as f:
                return json.load(f)
        return {"done": [], "failed": []}

    def _save_progress(self):
        with open(self._progress_path, "w") as f:
            json.dump(self._progress, f, indent=2)

    def _mark_done(self, song_id: str):
        if song_id not in self._progress["done"]:
            self._progress["done"].append(song_id)
        self._save_progress()

    def _mark_failed(self, song_id: str, reason: str):
        entry = {"id": song_id, "reason": reason}
        if not any(e["id"] == song_id for e in self._progress["failed"]):
            self._progress["failed"].append(entry)
        self._save_progress()

    # ------------------------------------------------------------------
    # Phase 1: Demucs source separation (cached)
    # ------------------------------------------------------------------

    def _separate(self, mp3_path: Path) -> Optional[Path]:
        """Separate MP3 → vocals.wav using Demucs. Returns vocals path or None."""
        song_id = mp3_path.stem

        # Check cache first (AudioSeparator caches under separated_dir/<song_id>/)
        cached = self.separator.get_separated(song_id)
        if cached:
            return cached["vocals"]

        print(f"    [Demucs] separating {mp3_path.name}...")
        result = self.separator.separate(str(mp3_path), device=self.device)
        if result is None:
            return None
        return result["vocals"]

    # ------------------------------------------------------------------
    # Phase 2: Whisper ASR — English lyric segments
    # ------------------------------------------------------------------

    def _load_whisper(self):
        if self._whisper is not None:
            return
        try:
            import whisper
            print(f"\nLoading Whisper '{self.whisper_model_name}' model...")
            self._whisper = whisper.load_model(self.whisper_model_name, device=self.whisper_device)
            print("  ✓ Whisper loaded")
        except ImportError:
            raise RuntimeError(
                "openai-whisper not installed. Run: pip install openai-whisper"
            )

    def _transcribe(self, vocals_path: Path) -> List[str]:
        """
        Run Whisper ASR on vocals.wav → list of English lyric segment texts.

        Returns [] if the track appears instrumental (too few / too short segments).
        """
        self._load_whisper()
        import whisper

        print(f"    [Whisper] transcribing {vocals_path.name}...")
        result = self._whisper.transcribe(
            str(vocals_path),
            language="en",
            fp16=False,
            verbose=False
        )

        segments = result.get("segments", [])
        lines = []
        for seg in segments:
            text = seg["text"].strip()
            words = text.split()
            if len(words) >= MIN_WORDS:
                lines.append(text)

        print(f"    [Whisper] → {len(lines)} usable lyric segments "
              f"(of {len(segments)} total)")
        return lines

    # ------------------------------------------------------------------
    # Phase 3: English → Hindi translation (MarianMT)
    # ------------------------------------------------------------------

    def _translate(self, english_lines: List[str]) -> List[str]:
        """Translate English lyric lines to Hindi using IndicTrans2.

        IndicTrans2 input format is 'src_lang tgt_lang <text>'. The target
        tokenizer vocab is separate, so we switch modes to decode properly.
        """
        if not english_lines:
            return []

        # Prefix with language tags (IndicTrans2 convention)
        tagged = [f"eng_Latn hin_Deva {line}" for line in english_lines]

        inputs = self.mt_tokenizer(
            tagged,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            translated_ids = self.mt_model.generate(
                **inputs,
                num_beams=5,
                max_length=128,
                use_cache=True,
            )

        # Decode using target-side vocabulary
        try:
            self.mt_tokenizer._switch_to_target_mode()
        except Exception:
            pass
        hindi_lines = self.mt_tokenizer.batch_decode(
            translated_ids, skip_special_tokens=True
        )
        try:
            self.mt_tokenizer._switch_to_input_mode()
        except Exception:
            pass
        return hindi_lines

    # ------------------------------------------------------------------
    # Phase 4: Melody extraction
    # ------------------------------------------------------------------

    def _extract_melody(self, vocals_path: Path, num_lines: int) -> Optional[np.ndarray]:
        """Extract melody features from vocals. num_lines guides note trimming."""
        return self.extractor.extract_melody_features(
            str(vocals_path), num_lyric_lines=num_lines
        )

    # ------------------------------------------------------------------
    # Phase 5: Melody split — one slice per lyric line
    # ------------------------------------------------------------------

    def _split_melody(self,
                       features: np.ndarray,
                       num_lines: int) -> List[np.ndarray]:
        """
        Divide melody note array evenly into num_lines slices.
        Each slice contains the notes that correspond to one lyric line.
        """
        total = len(features)
        notes_per_line = max(1, total // num_lines)
        slices = []
        for i in range(num_lines):
            start = i * notes_per_line
            end = start + notes_per_line if i < num_lines - 1 else total
            slices.append(features[start:end])
        return slices

    # ------------------------------------------------------------------
    # Phase 6: Tokenize + build training example dict
    # ------------------------------------------------------------------

    def _make_example(self,
                       english: str,
                       hindi: str,
                       melody_slice: np.ndarray,
                       song_name: str) -> Dict:
        """Create one training example in the same format as DatasetBuilder."""
        # IndicTrans2 source tokenization: "eng_Latn hin_Deva <text>"
        src_text = f"eng_Latn hin_Deva {english}"
        src_ids = self.it2_tokenizer(
            src_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=128
        ).input_ids[0]

        # IndicTrans2 target tokenization: switch to target SPM/vocab
        self.it2_tokenizer._switch_to_target_mode()
        tgt_ids = self.it2_tokenizer(
            hindi,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=128
        ).input_ids[0]
        self.it2_tokenizer._switch_to_input_mode()

        en_syl = count_english_syllables(english)
        hi_syl = count_hindi_syllables(hindi)
        num_notes = len(melody_slice)

        stress = get_stress_pattern(hindi)
        stress_tensor = (
            torch.tensor(stress, dtype=torch.float32)
            if stress
            else torch.zeros(max(1, hi_syl), dtype=torch.float32)
        )

        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "melody_features": torch.tensor(melody_slice, dtype=torch.float32),
            "src_syllables": en_syl,
            "tgt_syllables": hi_syl,
            "num_notes": num_notes,
            "song_name": song_name,
            "english_text": english,
            "hindi_text": hindi,
            "stress_pattern": stress_tensor,
        }

    # ------------------------------------------------------------------
    # Per-song processing
    # ------------------------------------------------------------------

    def _process_mp3(self, mp3_path: Path) -> Tuple[List[Dict], Optional[str]]:
        """
        Run the full pipeline for one FMA MP3.

        Returns:
            (examples, None)          — success, list of training examples
            ([],       "skip reason") — song skipped (instrumental / no vocals / error)
        """
        song_id = mp3_path.stem

        # Phase 1: Demucs
        vocals_path = self._separate(mp3_path)
        if vocals_path is None:
            return [], "demucs_failed"

        # Phase 2: Whisper ASR
        try:
            english_lines = self._transcribe(vocals_path)
        except Exception as e:
            return [], f"whisper_error:{e}"

        if len(english_lines) < MIN_SEGMENTS:
            return [], f"too_few_segments:{len(english_lines)}"

        # Phase 3: Translate
        try:
            hindi_lines = self._translate(english_lines)
        except Exception as e:
            return [], f"translation_error:{e}"

        if len(hindi_lines) != len(english_lines):
            hindi_lines = hindi_lines[:len(english_lines)]

        # Phase 4: Melody extraction
        melody = self._extract_melody(vocals_path, num_lines=len(english_lines))
        if melody is None or len(melody) == 0:
            return [], "no_melody"

        # Phase 5: Split melody per lyric line
        melody_slices = self._split_melody(melody, len(english_lines))

        # Phase 6: Build examples
        examples = []
        for i, (en, hi, mel_slice) in enumerate(
                zip(english_lines, hindi_lines, melody_slices)):
            if len(mel_slice) == 0:
                continue
            example = self._make_example(en, hi, mel_slice, f"{song_id}_line{i}")
            examples.append(example)

        return examples, None

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _process_batch(self, mp3_paths: List[Path], label: str) -> List[Dict]:
        """Process a list of MP3 files, returning all training examples."""
        all_examples = []
        already_done = set(self._progress["done"])
        already_failed = {e["id"] for e in self._progress["failed"]}

        for i, mp3_path in enumerate(mp3_paths):
            song_id = mp3_path.stem
            prefix = f"  [{label}] {i+1}/{len(mp3_paths)}  {mp3_path.name}"

            if song_id in already_done:
                print(f"{prefix}  (already processed, skipping)")
                continue
            if song_id in already_failed:
                print(f"{prefix}  (previously failed, skipping)")
                continue

            print(f"\n{prefix}")
            try:
                examples, skip_reason = self._process_mp3(mp3_path)
            except Exception as exc:
                skip_reason = f"exception:{exc}"
                examples = []

            if skip_reason:
                print(f"    ✗ skipped — {skip_reason}")
                self._mark_failed(song_id, skip_reason)
            else:
                print(f"    ✓ {len(examples)} example(s) created")
                all_examples.extend(examples)
                self._mark_done(song_id)

        return all_examples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_mp3s(self, max_songs: int = 200) -> List[Path]:
        """Return a reproducibly-shuffled list of up to max_songs MP3 paths."""
        all_mp3s = sorted(self.fma_dir.rglob("*.mp3"))
        rng = random.Random(self.seed)
        rng.shuffle(all_mp3s)
        selected = all_mp3s[:max_songs]
        print(f"Found {len(all_mp3s):,} MP3s. Using {len(selected)} "
              f"(seed={self.seed}).")
        return selected

    def build(self, max_songs: int = 200, test_ratio: float = 0.10):
        """
        Process up to max_songs FMA files and save train/test .pt files.

        Args:
            max_songs:   Maximum number of songs to attempt.
            test_ratio:  Fraction of songs held out for testing (default 10%).
        """
        mp3_paths = self.collect_mp3s(max_songs)

        # Song-level split BEFORE any processing so test set is truly held out
        split_idx = max(1, int(len(mp3_paths) * (1 - test_ratio)))
        train_paths = mp3_paths[:split_idx]
        test_paths = mp3_paths[split_idx:]

        train_out = self.processed_dir / "fma_train_data.pt"
        test_out = self.processed_dir / "fma_test_data.pt"

        # Load existing processed examples so resumed runs don't lose prior work
        def _load_existing(path):
            if path.exists():
                try:
                    return torch.load(path, weights_only=False)
                except Exception:
                    return []
            return []

        def _song_id(ex):
            return ex['song_name'].split('_line')[0]

        existing_train = _load_existing(train_out)
        existing_test = _load_existing(test_out)
        existing_train_ids = {_song_id(ex) for ex in existing_train}
        existing_test_ids = {_song_id(ex) for ex in existing_test}
        print(f"Loaded {len(existing_train)} existing train / "
              f"{len(existing_test)} existing test examples from disk.")

        print(f"\nSong split: {len(train_paths)} train | {len(test_paths)} test "
              f"({100*test_ratio:.0f}% held out)\n")
        print("Note: Demucs takes ~1-2 min/song; Whisper 'base' ~20s/song on CPU.\n"
              "      This run can be interrupted and resumed — already-done songs\n"
              f"      are logged in {self._progress_path}.\n")

        new_train = self._process_batch(train_paths, label="train")
        new_test = self._process_batch(test_paths, label="test")

        # Merge: keep existing examples for songs not re-processed this run,
        # replace with new examples if the song was re-processed.
        new_train_ids = {_song_id(ex) for ex in new_train}
        new_test_ids = {_song_id(ex) for ex in new_test}

        train_examples = (
            [ex for ex in existing_train if _song_id(ex) not in new_train_ids]
            + new_train
        )
        test_examples = (
            [ex for ex in existing_test if _song_id(ex) not in new_test_ids]
            + new_test
        )

        # Always save so incremental progress is preserved
        torch.save(train_examples, train_out)
        torch.save(test_examples, test_out)

        failed_count = len(self._progress["failed"])
        print(f"\n{'='*60}")
        print("Dataset saved!")
        print(f"  Train : {len(train_examples)} examples "
              f"({len(new_train)} new + {len(existing_train) - len(existing_train_ids & new_train_ids)} kept) → {train_out}")
        print(f"  Test  : {len(test_examples)} examples "
              f"({len(new_test)} new + {len(existing_test) - len(existing_test_ids & new_test_ids)} kept) → {test_out}")
        print(f"  Skipped/failed songs: {failed_count} "
              f"(see {self._progress_path})")
        print(f"{'='*60}\n")

        return train_examples, test_examples


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import os

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {PROJECT_ROOT}\n")

    parser = argparse.ArgumentParser(
        description="Build FMA training/test datasets for MCNST "
                    "(uses Demucs + Whisper + IndicTrans2)"
    )
    parser.add_argument(
        "--max", type=int, default=200,
        help="Maximum number of FMA songs to process (default: 200)"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.10,
        help="Fraction of songs held out for testing (default: 0.10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible song selection (default: 42)"
    )
    parser.add_argument(
        "--whisper", type=str, default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base). 'tiny' is fastest."
    )
    parser.add_argument(
        "--fma-dir", type=str, default="src/data/fma_medium",
        help="Path to FMA medium directory (default: src/data/fma_medium)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device for Demucs, Whisper, MarianMT (default: auto-detect)"
    )
    args = parser.parse_args()

    builder = FMADatasetBuilder(
        fma_dir=args.fma_dir,
        seed=args.seed,
        whisper_model=args.whisper,
        device=args.device
    )
    builder.build(max_songs=args.max, test_ratio=args.test_ratio)
