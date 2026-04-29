"""
Disney Song Curation Pipeline
==============================
Given a catalog of Hindi/English Disney song YouTube URLs, this script:

1. Downloads audio via yt-dlp (runs locally with your YT auth)
2. Separates vocals with Demucs (htdemucs)
3. Transcribes Hindi vocals with Whisper (hindi lyrics are REAL, not MT)
4. Transcribes English vocals with Whisper (english lyrics are REAL too)
5. Extracts melody features from English vocals via Basic-Pitch
6. Aligns Hindi/English lines by timestamp overlap
7. Tokenizes with IndicTrans2 (same tokenizer as MCNST training)
8. Packages into the SAME dict format as fma_train_data.pt
9. Flags alignment problems for manual review

WHY THIS DATA IS BETTER THAN FMA:
  - FMA uses machine-translated Hindi (IndicTrans2 output as labels)
  - Disney songs have REAL human-translated Hindi lyrics sung by pros
  - The Hindi was crafted to fit the same melody → natural singability
  - This is ground-truth musically-aligned translation data

Usage:
  cd musically-aligned-translation/
  python -m data_gathering.disney_song_curator
  python -m data_gathering.disney_song_curator --catalog path/to/catalog.json
  python -m data_gathering.disney_song_curator --max 10 --device mps

Requirements:
  pip install yt-dlp openai-whisper demucs basic-pitch librosa transformers
"""

import argparse
import json
import re
import subprocess
import tempfile
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Data classes for structured tracking
# ---------------------------------------------------------------------------

@dataclass
class AlignmentFlag:
    """A single alignment problem flagged for manual review."""
    song_name: str
    line_index: int
    flag_type: str          # e.g. "line_count_mismatch", "syllable_ratio", "no_melody"
    severity: str           # "warning" or "error"
    message: str
    english_text: str = ""
    hindi_text: str = ""

@dataclass
class SongResult:
    """Result of processing one song pair."""
    song_name: str
    status: str             # "success", "partial", "failed"
    num_examples: int = 0
    flags: List[AlignmentFlag] = field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# YouTube Audio Downloader
# ---------------------------------------------------------------------------

class YouTubeDownloader:
    """Download audio from YouTube using yt-dlp."""

    def __init__(self, output_dir: str = "data_gathering/downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, name: str, lang_tag: str) -> Optional[Path]:
        """
        Download audio from a YouTube URL.

        Args:
            url:      YouTube URL
            name:     Song identifier (e.g. "let_it_go")
            lang_tag: "hindi" or "english"

        Returns:
            Path to downloaded WAV file, or None on failure.
        """
        out_path = self.output_dir / f"{name}_{lang_tag}.wav"

        if out_path.exists():
            print(f"    [yt-dlp] cached: {out_path.name}")
            return out_path

        print(f"    [yt-dlp] downloading {lang_tag} version: {url}")
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--output", str(out_path.with_suffix(".%(ext)s")),
            "--no-playlist",
            "--quiet",
            url,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"    [yt-dlp] FAILED: {result.stderr[:200]}")
                return None
        except FileNotFoundError:
            print("    [yt-dlp] ERROR: yt-dlp not found. Install: pip install yt-dlp")
            return None
        except subprocess.TimeoutExpired:
            print("    [yt-dlp] ERROR: download timed out (5 min)")
            return None

        # yt-dlp may produce the file with a slightly different name
        if out_path.exists():
            return out_path

        # Check for common yt-dlp output patterns
        for candidate in self.output_dir.glob(f"{name}_{lang_tag}.*"):
            if candidate.suffix in (".wav", ".mp3", ".m4a", ".opus"):
                if candidate.suffix != ".wav":
                    wav_path = self._convert_to_wav(candidate)
                    if wav_path:
                        return wav_path
                return candidate

        print(f"    [yt-dlp] file not found after download")
        return None

    def _convert_to_wav(self, audio_path: Path) -> Optional[Path]:
        """Convert non-WAV audio to WAV using ffmpeg."""
        wav_path = audio_path.with_suffix(".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(audio_path), "-ar", "22050",
                 "-ac", "1", str(wav_path), "-y", "-loglevel", "quiet"],
                check=True, timeout=120
            )
            return wav_path
        except Exception as e:
            print(f"    [ffmpeg] conversion failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Vocal Separator (wraps Demucs)
# ---------------------------------------------------------------------------

class VocalSeparator:
    """Separate vocals from full mix using Demucs."""

    def __init__(self, output_dir: str = "data_gathering/separated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def separate(self, audio_path: Path, song_name: str,
                 lang_tag: str, device: str = "cpu") -> Optional[Path]:
        """
        Run Demucs htdemucs on audio → return path to vocals.wav.

        Caches results: if vocals already exist, returns cached path.
        """
        cache_dir = self.output_dir / f"{song_name}_{lang_tag}"
        vocals_path = cache_dir / "vocals.wav"

        if vocals_path.exists():
            print(f"    [Demucs] cached: {vocals_path.name}")
            return vocals_path

        print(f"    [Demucs] separating {audio_path.name}...")
        cmd = [
            "python", "-m", "demucs",
            "--two-stems", "vocals",
            "-n", "htdemucs",
            "-d", device,
            "-o", str(self.output_dir),
            str(audio_path),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                print(f"    [Demucs] FAILED: {result.stderr[:300]}")
                return None
        except subprocess.TimeoutExpired:
            print("    [Demucs] timed out (10 min)")
            return None

        # Demucs outputs to: output_dir/htdemucs/<stem_name>/vocals.wav
        demucs_out = self.output_dir / "htdemucs" / audio_path.stem / "vocals.wav"
        if demucs_out.exists():
            # Move to our cache structure
            cache_dir.mkdir(parents=True, exist_ok=True)
            demucs_out.rename(vocals_path)
            return vocals_path

        # Search for the vocals file in demucs output
        for vf in self.output_dir.rglob("vocals.wav"):
            if audio_path.stem in str(vf):
                cache_dir.mkdir(parents=True, exist_ok=True)
                vf.rename(vocals_path)
                return vocals_path

        print(f"    [Demucs] vocals file not found in output")
        return None


# ---------------------------------------------------------------------------
# Whisper Transcriber (with timestamps for alignment)
# ---------------------------------------------------------------------------

class WhisperTranscriber:
    """Transcribe vocals using OpenAI Whisper, returning timed segments."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        # Whisper uses sparse tensors not supported on MPS
        self.device = "cpu" if device == "mps" else device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import whisper
            print(f"  Loading Whisper '{self.model_size}'...")
            self._model = whisper.load_model(self.model_size, device=self.device)
        except ImportError:
            raise RuntimeError("openai-whisper not installed. pip install openai-whisper")

    def transcribe(self, vocals_path: Path, language: str) -> List[Dict]:
        """
        Transcribe vocals → list of segments with timestamps.

        Each segment: {
            "text": str,
            "start": float (seconds),
            "end": float (seconds)
        }
        """
        self._load()
        import whisper

        print(f"    [Whisper] transcribing ({language})...")

        # For Hindi: use an initial_prompt in Devanagari to bias Whisper
        # away from Urdu (Nastaliq) script — the two are spoken identically
        # but Whisper base often picks the wrong script without a hint.
        transcribe_kwargs = dict(
            language=language[:2],   # "hi" or "en"
            fp16=False,
            verbose=False,
            word_timestamps=False,
        )
        if language.startswith("hi"):
            transcribe_kwargs["initial_prompt"] = (
                "यह एक हिंदी गाना है। गीत के बोल देवनागरी लिपि में हैं।"
            )

        result = self._model.transcribe(str(vocals_path), **transcribe_kwargs)

        # Words from the initial_prompt that Whisper may echo back
        prompt_keywords = {"गाना", "गीत", "बोल", "देवनागरी", "लिपि", "हिंदी"}

        segments = []
        for seg_idx, seg in enumerate(result.get("segments", [])):
            text = seg["text"].strip()
            if len(text) < 2:
                continue

            # Skip prompt-echo: Whisper sometimes regurgitates the
            # initial_prompt as its first segment (hallucination).
            # Detect by checking if early segments contain prompt keywords.
            if language.startswith("hi") and seg_idx < 2:
                if any(kw in text for kw in prompt_keywords):
                    print(f"    [Whisper] SKIP (prompt echo): {text[:60]}")
                    continue

            # Reject segments that came out in Urdu script (U+0600-U+06FF)
            # instead of Devanagari (U+0900-U+097F)
            if language.startswith("hi"):
                urdu_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
                deva_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
                if urdu_chars > deva_chars:
                    print(f"    [Whisper] SKIP (Urdu script): {text[:60]}")
                    continue

                # Reject hallucinations: Latin words in Hindi transcription
                # (e.g. "the", "sphere") indicate Whisper confusion.
                latin_words = re.findall(r'[A-Za-z]{2,}', text)
                if latin_words:
                    print(f"    [Whisper] SKIP (Latin in Hindi): {text[:60]}")
                    continue

            # Skip consecutive duplicate segments — Whisper sometimes repeats
            # the same text multiple times when the audio is quiet/ambiguous.
            if segments and text == segments[-1]["text"]:
                print(f"    [Whisper] SKIP (duplicate): {text[:60]}")
                continue

            segments.append({
                "text": text,
                "start": seg["start"],
                "end": seg["end"],
            })

        print(f"    [Whisper] → {len(segments)} segments")
        return segments


# ---------------------------------------------------------------------------
# Melody Extractor (Basic-Pitch on English vocals)
# ---------------------------------------------------------------------------

class MelodyExtractor:
    """
    Extract melody features from English vocals via Basic-Pitch.

    Output format: [num_notes, 5] matching fma_data_builder exactly:
      [pitch, pitch_class, duration_beats, duration_bin, beat_strength]
    """

    def __init__(self, sr: int = 22050):
        self.sr = sr

    def extract(self, vocals_path: Path) -> Optional[np.ndarray]:
        """Extract melody features → [N, 5] array or None."""
        try:
            import librosa
            y, sr = librosa.load(str(vocals_path), sr=self.sr)
        except ImportError:
            print("    [melody] librosa not installed")
            return None

        duration_sec = len(y) / sr
        print(f"    [melody] audio duration: {duration_sec:.1f}s")

        import librosa
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        seconds_per_beat = 60.0 / max(tempo, 1.0)
        print(f"    [melody] tempo: {tempo:.0f} BPM")

        features = self._extract_basic_pitch(vocals_path, beat_times, seconds_per_beat)
        if features is None:
            print("    [melody] Basic-Pitch failed, trying librosa fallback")
            features = self._extract_librosa(y, sr, beat_times, seconds_per_beat, duration_sec)

        if features is None or len(features) == 0:
            return None

        features = np.array(features, dtype=np.float32)
        print(f"    [melody] ✓ {len(features)} notes extracted")
        return features

    def _extract_basic_pitch(self, audio_path, beat_times, spb):
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except ImportError:
            return None

        try:
            print("    [Basic-Pitch] running neural pitch tracker...")
            _, _, note_events = predict(
                str(audio_path), ICASSP_2022_MODEL_PATH,
                onset_threshold=0.5, frame_threshold=0.3,
                minimum_note_length=58, minimum_frequency=65.0,
                maximum_frequency=1047.0, melodia_trick=True,
            )
        except Exception as e:
            print(f"    [Basic-Pitch] error: {e}")
            return None

        if not note_events:
            return None

        features = []
        for start_s, end_s, midi_pitch, amplitude, _ in note_events:
            if midi_pitch < 20 or midi_pitch > 108:
                continue
            dur_s = max(end_s - start_s, 0.01)
            dur_beats = dur_s / spb
            dur_bin = 1.0 if dur_beats >= 1.0 else 0.0
            bs = self._beat_strength(start_s, beat_times, spb)
            features.append([float(midi_pitch), float(midi_pitch % 12),
                             dur_beats, dur_bin, bs])
        return features or None

    def _extract_librosa(self, y, sr, beat_times, spb, duration_sec):
        import librosa
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'), sr=sr
        )
        pitch_times = librosa.times_like(f0, sr=sr)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        if len(onset_times) == 0:
            return None

        features = []
        for i, onset in enumerate(onset_times):
            note_end = onset_times[i + 1] if i + 1 < len(onset_times) else duration_sec
            dur_s = note_end - onset
            if dur_s < 0.05:
                continue
            mask = (pitch_times >= onset) & (pitch_times < note_end)
            voiced_f0 = f0[mask]
            voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
            if len(voiced_f0) == 0:
                continue
            midi_pitch = int(round(69 + 12 * np.log2(np.median(voiced_f0) / 440.0)))
            if midi_pitch < 20 or midi_pitch > 108:
                continue
            dur_beats = dur_s / spb
            dur_bin = 1.0 if dur_beats >= 1.0 else 0.0
            bs = self._beat_strength(onset, beat_times, spb)
            features.append([float(midi_pitch), float(midi_pitch % 12),
                             dur_beats, dur_bin, bs])
        return features or None

    @staticmethod
    def _beat_strength(onset_s, beat_times, spb):
        if len(beat_times) == 0:
            return 0.5
        min_dist = float(np.min(np.abs(beat_times - onset_s)))
        return max(0.0, 1.0 - (min_dist / max(spb, 1e-6)))


# ---------------------------------------------------------------------------
# Timestamp-based Line Aligner
# ---------------------------------------------------------------------------

class LineAligner:
    """
    Align Hindi and English lyric segments using timestamp overlap.

    Disney songs share the same melody → the Hindi version is sung at
    roughly the same timestamps as the English. We use Whisper's segment
    timestamps to pair them up via maximum temporal overlap.
    """

    def align(self, hindi_segs: List[Dict], english_segs: List[Dict],
              song_name: str) -> Tuple[List[Tuple[Dict, Dict]], List[AlignmentFlag]]:
        """
        Pair Hindi segments with English segments by timestamp overlap.

        Returns:
            pairs: list of (hindi_seg, english_seg) tuples
            flags: alignment issues found
        """
        flags = []

        if not hindi_segs or not english_segs:
            flags.append(AlignmentFlag(
                song_name=song_name, line_index=-1,
                flag_type="empty_transcription", severity="error",
                message=f"Hindi: {len(hindi_segs)} segs, English: {len(english_segs)} segs"
            ))
            return [], flags

        # Flag large line count discrepancy
        ratio = len(hindi_segs) / len(english_segs) if english_segs else 0
        if ratio < 0.5 or ratio > 2.0:
            flags.append(AlignmentFlag(
                song_name=song_name, line_index=-1,
                flag_type="line_count_mismatch", severity="warning",
                message=f"Hindi has {len(hindi_segs)} segs vs English {len(english_segs)} "
                        f"(ratio {ratio:.2f}) — may need manual alignment"
            ))

        # Greedy alignment by maximum temporal overlap
        pairs = []
        used_english = set()

        for hi_idx, hi_seg in enumerate(hindi_segs):
            best_overlap = 0.0
            best_en_idx = -1

            for en_idx, en_seg in enumerate(english_segs):
                if en_idx in used_english:
                    continue
                overlap = self._overlap(hi_seg, en_seg)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_en_idx = en_idx

            if best_en_idx >= 0 and best_overlap > 0.3:
                pairs.append((hi_seg, english_segs[best_en_idx]))
                used_english.add(best_en_idx)
            else:
                flags.append(AlignmentFlag(
                    song_name=song_name, line_index=hi_idx,
                    flag_type="no_english_match", severity="warning",
                    message=f"Hindi line {hi_idx} has no good English match "
                            f"(best overlap: {best_overlap:.2f}s)",
                    hindi_text=hi_seg["text"]
                ))

        # Flag unmatched English lines
        unmatched_en = len(english_segs) - len(used_english)
        if unmatched_en > len(english_segs) * 0.3:
            flags.append(AlignmentFlag(
                song_name=song_name, line_index=-1,
                flag_type="many_unmatched_english", severity="warning",
                message=f"{unmatched_en}/{len(english_segs)} English lines unmatched"
            ))

        return pairs, flags

    @staticmethod
    def _overlap(seg_a: Dict, seg_b: Dict) -> float:
        """Compute temporal overlap in seconds between two segments."""
        start = max(seg_a["start"], seg_b["start"])
        end = min(seg_a["end"], seg_b["end"])
        return max(0.0, end - start)


# ---------------------------------------------------------------------------
# Example Builder — formats into fma_train_data.pt structure
# ---------------------------------------------------------------------------

class ExampleBuilder:
    """
    Build training examples in the exact same format as FMADatasetBuilder.

    Output dict per example:
        src_ids          : LongTensor   — IndicTrans2 tokenized English
        tgt_ids          : LongTensor   — IndicTrans2 tokenized Hindi
        melody_features  : FloatTensor  — [num_notes, 5]
        src_syllables    : int
        tgt_syllables    : int
        num_notes        : int
        song_name        : str
        english_text     : str
        hindi_text       : str
        stress_pattern   : FloatTensor
        source           : str          — "disney" (to distinguish from FMA)
    """

    IT2_MODEL = "ai4bharat/indictrans2-en-indic-1B"

    def __init__(self, device: str = "cpu"):
        from transformers import AutoTokenizer
        print(f"  Loading IndicTrans2 tokenizer ({self.IT2_MODEL})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.IT2_MODEL, trust_remote_code=True
        )
        self.device = device

    def build(self, english: str, hindi: str, melody_slice: np.ndarray,
              song_name: str) -> Dict:
        """Create one training example matching fma_train_data.pt format."""
        from src.utils.syllable_utils import count_english_syllables, count_hindi_syllables
        from src.utils.phoneme_utils import get_stress_pattern

        # Source tokenization (IndicTrans2 format)
        src_text = f"eng_Latn hin_Deva {english}"
        src_ids = self.tokenizer(
            src_text, return_tensors="pt", padding=False,
            truncation=True, max_length=128
        ).input_ids[0]

        # Target tokenization — use SPM directly to avoid _src_tokenize
        # dispatch issue in transformers >=5.x
        self.tokenizer._switch_to_target_mode()
        tgt_pieces = self.tokenizer.spm.EncodeAsPieces(hindi)
        tgt_token_ids = [self.tokenizer.encoder.get(p, self.tokenizer.unk_token_id) for p in tgt_pieces]
        bos = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token) if self.tokenizer.bos_token else None
        eos = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token) if self.tokenizer.eos_token else None
        if bos is not None:
            tgt_token_ids = [bos] + tgt_token_ids
        if eos is not None:
            tgt_token_ids = tgt_token_ids + [eos]
        tgt_ids = torch.tensor(tgt_token_ids[:128], dtype=torch.long)
        self.tokenizer._switch_to_input_mode()

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
            "source": "disney",
        }


# ---------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ---------------------------------------------------------------------------

class DisneySongCurator:
    """
    End-to-end pipeline: catalog.json → training examples + QA report.
    """

    def __init__(self,
                 catalog_path: str = "data_gathering/disney_song_catalog.json",
                 output_dir: str = "data_gathering/output",
                 download_dir: str = "data_gathering/downloads",
                 separated_dir: str = "data_gathering/separated",
                 whisper_model: str = "base",
                 device: str = "auto"):

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        print(f"Device: {self.device}")

        self.catalog_path = Path(catalog_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.downloader = YouTubeDownloader(download_dir)
        self.separator = VocalSeparator(separated_dir)
        self.transcriber = WhisperTranscriber(whisper_model, device)
        self.melody_extractor = MelodyExtractor()
        self.aligner = LineAligner()
        self.example_builder = None  # lazy-init (heavy model load)

        # Progress tracking
        self.progress_path = self.output_dir / "disney_progress.json"
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict:
        if self.progress_path.exists():
            with open(self.progress_path) as f:
                return json.load(f)
        return {"done": [], "failed": []}

    def _save_progress(self):
        with open(self.progress_path, "w") as f:
            json.dump(self.progress, f, indent=2)

    def _load_catalog(self) -> List[Dict]:
        """Load song catalog JSON."""
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog not found: {self.catalog_path}\n"
                f"Create one with the format in disney_song_catalog.json"
            )
        with open(self.catalog_path) as f:
            catalog = json.load(f)
        return catalog.get("songs", catalog) if isinstance(catalog, dict) else catalog

    def _init_example_builder(self):
        """Lazy-load the tokenizer (only when we actually have data to process)."""
        if self.example_builder is None:
            self.example_builder = ExampleBuilder(self.device)

    def process_song(self, entry: Dict) -> SongResult:
        """
        Process one song entry from the catalog.

        Entry format:
            {
                "name": "let_it_go",
                "title": "Let It Go (Frozen)",
                "hindi_url": "https://youtube.com/watch?v=...",
                "english_url": "https://youtube.com/watch?v=...",
                "notes": "optional human notes"
            }
        """
        name = entry["name"]
        result = SongResult(song_name=name, status="failed")

        print(f"\n{'='*60}")
        print(f"  Processing: {entry.get('title', name)}")
        print(f"{'='*60}")

        # Step 1: Download both versions
        print("\n  Step 1: Download audio")
        hindi_audio = self.downloader.download(entry["hindi_url"], name, "hindi")
        english_audio = self.downloader.download(entry["english_url"], name, "english")

        if hindi_audio is None or english_audio is None:
            result.error = "download_failed"
            return result

        # Step 2: Separate vocals
        print("\n  Step 2: Separate vocals")
        hindi_vocals = self.separator.separate(hindi_audio, name, "hindi", self.device)
        english_vocals = self.separator.separate(english_audio, name, "english", self.device)

        if hindi_vocals is None or english_vocals is None:
            result.error = "separation_failed"
            return result

        # Step 3: Transcribe both
        print("\n  Step 3: Transcribe")
        hindi_segs = self.transcriber.transcribe(hindi_vocals, "hindi")
        english_segs = self.transcriber.transcribe(english_vocals, "english")

        if not hindi_segs:
            result.error = "hindi_transcription_empty"
            return result
        if not english_segs:
            result.error = "english_transcription_empty"
            return result

        # Step 4: Extract melody from English vocals
        print("\n  Step 4: Extract melody (from English vocals)")
        melody_features = self.melody_extractor.extract(english_vocals)
        if melody_features is None:
            result.flags.append(AlignmentFlag(
                song_name=name, line_index=-1,
                flag_type="no_melody", severity="error",
                message="Melody extraction failed entirely"
            ))
            result.error = "melody_extraction_failed"
            return result

        # Step 5: Align Hindi/English lines by timestamp
        print("\n  Step 5: Align lines by timestamp")
        pairs, align_flags = self.aligner.align(hindi_segs, english_segs, name)
        result.flags.extend(align_flags)

        if not pairs:
            result.error = "no_aligned_pairs"
            return result

        # Step 6: Split melody across aligned pairs
        print(f"\n  Step 6: Build {len(pairs)} training examples")
        self._init_example_builder()

        notes_per_line = max(1, len(melody_features) // len(pairs))
        examples = []

        for i, (hi_seg, en_seg) in enumerate(pairs):
            # Slice melody for this line
            start = i * notes_per_line
            end = start + notes_per_line if i < len(pairs) - 1 else len(melody_features)
            mel_slice = melody_features[start:end]

            if len(mel_slice) == 0:
                result.flags.append(AlignmentFlag(
                    song_name=name, line_index=i,
                    flag_type="empty_melody_slice", severity="warning",
                    message=f"Line {i} got 0 melody notes",
                    english_text=en_seg["text"],
                    hindi_text=hi_seg["text"]
                ))
                continue

            # Quality checks
            self._check_quality(name, i, en_seg["text"], hi_seg["text"],
                                mel_slice, result.flags)

            example = self.example_builder.build(
                english=en_seg["text"],
                hindi=hi_seg["text"],
                melody_slice=mel_slice,
                song_name=f"{name}_line{i}"
            )
            examples.append(example)

        result.num_examples = len(examples)
        result.status = "success" if examples else "failed"
        if examples and result.flags:
            result.status = "partial"

        return result, examples

    def _check_quality(self, song_name, line_idx, english, hindi,
                       melody_slice, flags):
        """Run quality checks on a single aligned line."""
        from src.utils.syllable_utils import count_english_syllables, count_hindi_syllables

        en_syl = count_english_syllables(english)
        hi_syl = count_hindi_syllables(hindi)
        num_notes = len(melody_slice)

        # Syllable ratio check — if Hindi has 3x more or fewer syllables,
        # the translation probably doesn't fit the melody
        if en_syl > 0:
            syl_ratio = hi_syl / en_syl
            if syl_ratio > 2.5 or syl_ratio < 0.3:
                flags.append(AlignmentFlag(
                    song_name=song_name, line_index=line_idx,
                    flag_type="syllable_ratio", severity="warning",
                    message=f"Syllable ratio {syl_ratio:.2f} "
                            f"(en={en_syl}, hi={hi_syl}) — may not be singable",
                    english_text=english, hindi_text=hindi
                ))

        # Melody coverage — notes vs syllables
        if hi_syl > 0 and num_notes > 0:
            note_ratio = num_notes / hi_syl
            if note_ratio < 0.3 or note_ratio > 5.0:
                flags.append(AlignmentFlag(
                    song_name=song_name, line_index=line_idx,
                    flag_type="melody_coverage", severity="warning",
                    message=f"Notes/syllables ratio {note_ratio:.2f} "
                            f"(notes={num_notes}, hi_syl={hi_syl})",
                    english_text=english, hindi_text=hindi
                ))


    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    def run(self, max_songs: int = None, test_ratio: float = 0.15):
        """
        Process all songs in the catalog.

        Args:
            max_songs:   Limit number of songs (None = all)
            test_ratio:  Fraction held out for testing
        """
        catalog = self._load_catalog()
        if max_songs:
            catalog = catalog[:max_songs]

        print(f"\nCatalog: {len(catalog)} songs")
        print(f"Test split: {test_ratio*100:.0f}%\n")

        all_examples = []
        all_results = []
        already_done = set(self.progress["done"])

        for entry in catalog:
            name = entry["name"]
            if name in already_done:
                print(f"\n  [skip] {name} already processed")
                continue

            try:
                result_tuple = self.process_song(entry)
                if isinstance(result_tuple, tuple):
                    result, examples = result_tuple
                else:
                    result = result_tuple
                    examples = []
            except Exception as e:
                result = SongResult(song_name=name, status="failed",
                                    error=f"exception: {e}")
                examples = []
                import traceback
                traceback.print_exc()

            all_results.append(result)

            if result.status in ("success", "partial"):
                all_examples.extend(examples)
                self.progress["done"].append(name)
            else:
                self.progress["failed"].append({
                    "name": name, "error": result.error
                })

            self._save_progress()

        # Load any existing Disney examples from prior runs
        train_path = self.output_dir / "disney_train_data.pt"
        test_path = self.output_dir / "disney_test_data.pt"

        existing = []
        if train_path.exists():
            existing.extend(torch.load(train_path, weights_only=False))
        if test_path.exists():
            existing.extend(torch.load(test_path, weights_only=False))

        # Merge (deduplicate by song_name)
        seen = {ex["song_name"] for ex in all_examples}
        for ex in existing:
            if ex["song_name"] not in seen:
                all_examples.append(ex)
                seen.add(ex["song_name"])

        # Train/test split
        import random
        random.seed(42)
        random.shuffle(all_examples)
        split = max(1, int(len(all_examples) * (1 - test_ratio)))
        train_examples = all_examples[:split]
        test_examples = all_examples[split:]

        # Save
        if train_examples:
            torch.save(train_examples, train_path)
        if test_examples:
            torch.save(test_examples, test_path)

        # Save QA report
        self._save_qa_report(all_results)

        # Summary
        total_flags = sum(len(r.flags) for r in all_results)
        print(f"\n{'='*60}")
        print(f"  CURATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Songs processed : {len(all_results)}")
        print(f"  Successful      : {sum(1 for r in all_results if r.status == 'success')}")
        print(f"  Partial (flagged): {sum(1 for r in all_results if r.status == 'partial')}")
        print(f"  Failed          : {sum(1 for r in all_results if r.status == 'failed')}")
        print(f"  Total examples  : {len(all_examples)}")
        print(f"    Train         : {len(train_examples)} → {train_path}")
        print(f"    Test          : {len(test_examples)} → {test_path}")
        print(f"  QA flags        : {total_flags}")
        print(f"  Report          : {self.output_dir / 'qa_report.json'}")
        print(f"{'='*60}\n")

        return all_examples, all_results

    def _save_qa_report(self, results: List[SongResult]):
        """Save a JSON report of all flags for manual review."""
        report = []
        for r in results:
            entry = {
                "song_name": r.song_name,
                "status": r.status,
                "num_examples": r.num_examples,
                "error": r.error,
                "flags": [asdict(f) for f in r.flags],
            }
            report.append(entry)

        report_path = self.output_dir / "qa_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  QA report saved → {report_path}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import os

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {PROJECT_ROOT}\n")

    parser = argparse.ArgumentParser(
        description="Disney Song Curation Pipeline — build musically-aligned "
                    "translation data from Hindi/English Disney song pairs"
    )
    parser.add_argument(
        "--catalog", type=str,
        default="data_gathering/disney_song_catalog.json",
        help="Path to song catalog JSON"
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Max songs to process (default: all)"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Fraction held out for testing (default: 0.15)"
    )
    parser.add_argument(
        "--whisper", type=str, default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Compute device (default: auto-detect)"
    )
    args = parser.parse_args()

    curator = DisneySongCurator(
        catalog_path=args.catalog,
        whisper_model=args.whisper,
        device=args.device,
    )
    curator.run(max_songs=args.max, test_ratio=args.test_ratio)
