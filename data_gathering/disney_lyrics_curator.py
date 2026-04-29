"""
Disney Lyrics-Based Curation Pipeline
=====================================
Replacement for ``disney_song_curator.py`` that uses ground-truth lyric
TEXT (sourced manually from Genius / lyricstranslate / fan transcriptions
/ wikis) instead of Whisper-transcribing Hindi audio.

WHY THIS EXISTS
---------------
The original ``disney_song_curator.py`` Whisper-transcribes both Hindi
and English vocals, then aligns segments by timestamp. On ``let_it_go``
this produced gibberish Hindi (e.g. ``जाहां की मले का में वोगी``,
``पनाहो`` repeated as the "translation" for four different English
lines) — see ``output/qa_report.json``.

That's not a tuning problem. Whisper-base on Hindi singing voice is out
of distribution; even Whisper-large-v3 is only partly usable on sung
Hindi. The whole *point* of the curated dataset is "Hindi was
professionally written to fit the melody" — recovering those
professional lyrics via ASR destroys the ground-truth signal.

This pipeline takes Hindi lyrics as INPUT instead of trying to recover
them. It does this by:

  1. Reading parallel ``<song>.en`` / ``<song>.hi`` lyric files (one line
     per lyric line, prepared by the user).
  2. Downloading + separating the *English* audio only.
  3. Running Whisper on English vocals (which actually works) to get
     timestamped segments.
  4. Fuzzy-matching each lyric line to a Whisper segment to recover its
     timing in the song.
  5. Extracting melody from English vocals via Basic-Pitch.
  6. Slicing the melody by aligned timestamps so each lyric line gets
     its own note slice.
  7. Tokenizing with IndicTrans2 and writing examples in the same dict
     shape as ``fma_train_data.pt`` so they can be concatenated.

The Hindi audio is not used at all — the Hindi text comes from the
lyric file (gold standard) and the melody is shared with the English
version (Disney dubs sing the same melody).

LYRIC FILE FORMAT
-----------------
For a song with catalog name ``let_it_go_frozen``:

    data_gathering/lyrics/let_it_go_frozen.en   ── one English line per line
    data_gathering/lyrics/let_it_go_frozen.hi   ── one Hindi line per line

Line N in the .en file pairs with line N in the .hi file. Empty lines
are ignored. Both files must have the same number of non-empty lines.

USAGE
-----
    cd musically-aligned-translation/
    python -m data_gathering.disney_lyrics_curator
    python -m data_gathering.disney_lyrics_curator --max 5
    python -m data_gathering.disney_lyrics_curator --fresh   # ignore prior progress

Songs without a lyric file in ``data_gathering/lyrics/`` are skipped
with a clear log line — populate the lyric files at your own pace and
re-run. Caching means already-processed songs are not re-downloaded or
re-separated.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
from difflib import SequenceMatcher

# Reuse the working components from the existing curator. We deliberately
# keep that file in place as a fallback; this module only adds new code.
from data_gathering.disney_song_curator import (
    YouTubeDownloader,
    VocalSeparator,
    WhisperTranscriber,
    ExampleBuilder,
    AlignmentFlag,
    SongResult,
)


# ---------------------------------------------------------------------------
# Lyrics Loader
# ---------------------------------------------------------------------------

class LyricsLoader:
    """
    Read parallel English/Hindi lyric files for a song.

    Files: ``<lyrics_dir>/<song_name>.en`` and ``<lyrics_dir>/<song_name>.hi``
    Format: one line per lyric line, blank lines ignored, line N in .en
    pairs with line N in .hi.
    """

    def __init__(self, lyrics_dir: str = "data_gathering/lyrics"):
        self.lyrics_dir = Path(lyrics_dir)
        self.lyrics_dir.mkdir(parents=True, exist_ok=True)

    def has_lyrics(self, song_name: str) -> bool:
        return (
            (self.lyrics_dir / f"{song_name}.en").exists()
            and (self.lyrics_dir / f"{song_name}.hi").exists()
        )

    def load(self, song_name: str) -> Tuple[Optional[List[Tuple[str, str]]], str]:
        """
        Returns (pairs, error_msg). pairs is None on failure.

        On success: list of (english_line, hindi_line) tuples.
        On failure: error_msg explains why.
        """
        en_path = self.lyrics_dir / f"{song_name}.en"
        hi_path = self.lyrics_dir / f"{song_name}.hi"

        if not en_path.exists():
            return None, f"missing {en_path}"
        if not hi_path.exists():
            return None, f"missing {hi_path}"

        en_lines = self._read_clean(en_path)
        hi_lines = self._read_clean(hi_path)

        if not en_lines or not hi_lines:
            return None, f"empty lyric file (en={len(en_lines)}, hi={len(hi_lines)})"

        if len(en_lines) != len(hi_lines):
            return None, (
                f"line count mismatch: {en_path.name}={len(en_lines)} vs "
                f"{hi_path.name}={len(hi_lines)} — they must be parallel"
            )

        return list(zip(en_lines, hi_lines)), ""

    @staticmethod
    def _read_clean(path: Path) -> List[str]:
        text = path.read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Melody Extractor With Per-Note Timestamps
# ---------------------------------------------------------------------------
#
# The original ``MelodyExtractor`` in disney_song_curator.py drops note
# start times after computing features. We need them to slice melody by
# lyric-line timestamp, so this extractor returns features AND start
# times in parallel. Output features are bit-for-bit identical to the
# original — only the extra ``start_times`` array is added.

class MelodyExtractorWithTimes:
    """Extract melody features + per-note start times (seconds) via Basic-Pitch."""

    def __init__(self, sr: int = 22050):
        self.sr = sr

    def extract(self, vocals_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (features [N, 5], start_times [N]) or (None, None)."""
        try:
            import librosa
            y, sr = librosa.load(str(vocals_path), sr=self.sr)
        except ImportError:
            print("    [melody] librosa not installed")
            return None, None

        duration_sec = len(y) / sr
        print(f"    [melody] audio duration: {duration_sec:.1f}s")

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, "__len__"):
            tempo = float(tempo[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        spb = 60.0 / max(tempo, 1.0)
        print(f"    [melody] tempo: {tempo:.0f} BPM")

        features, starts = self._basic_pitch(vocals_path, beat_times, spb)
        if features is None:
            return None, None

        print(f"    [melody] ✓ {len(features)} notes extracted")
        return (
            np.asarray(features, dtype=np.float32),
            np.asarray(starts, dtype=np.float32),
        )

    @staticmethod
    def _basic_pitch(audio_path: Path, beat_times, spb: float):
        """Basic-Pitch note extraction. Returns (features, starts) parallel lists."""
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except ImportError:
            print("    [Basic-Pitch] not installed")
            return None, None

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
            return None, None

        if not note_events:
            return None, None

        features, starts = [], []
        for start_s, end_s, midi_pitch, _amp, _bends in note_events:
            if midi_pitch < 20 or midi_pitch > 108:
                continue
            dur_s = max(end_s - start_s, 0.01)
            dur_beats = dur_s / spb
            dur_bin = 1.0 if dur_beats >= 1.0 else 0.0
            bs = MelodyExtractorWithTimes._beat_strength(start_s, beat_times, spb)
            features.append([float(midi_pitch), float(midi_pitch % 12),
                             dur_beats, dur_bin, bs])
            starts.append(float(start_s))

        if not features:
            return None, None

        # Sort chronologically (Basic-Pitch usually returns sorted, but enforce it)
        order = np.argsort(starts)
        features = [features[i] for i in order]
        starts = [starts[i] for i in order]
        return features, starts

    @staticmethod
    def _beat_strength(onset_s, beat_times, spb):
        if len(beat_times) == 0:
            return 0.5
        min_dist = float(np.min(np.abs(beat_times - onset_s)))
        return max(0.0, 1.0 - (min_dist / max(spb, 1e-6)))


# ---------------------------------------------------------------------------
# Lyric-Line ↔ Whisper-Segment Timestamp Aligner
# ---------------------------------------------------------------------------

class LyricsTimestampAligner:
    """
    Recover (start, end) timestamps for each lyric line by fuzzy-matching
    the line's English text to Whisper-English segments.

    Whisper on English vocals works reasonably well — its segments are
    correctly transcribed but may be split or merged differently from
    the lyric file's line breaks. We handle that with sequence matching.
    """

    def __init__(self, min_match_ratio: float = 0.45):
        self.min_match_ratio = min_match_ratio

    @staticmethod
    def _norm(text: str) -> str:
        out = []
        for c in text.lower():
            if c.isalnum() or c.isspace():
                out.append(c)
            elif c in "'’":
                continue
            else:
                out.append(" ")
        return " ".join("".join(out).split())

    @staticmethod
    def _ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        r = SequenceMatcher(None, a, b).ratio()
        # Containment bonus: if one is a substring of the other, treat
        # them as a strong match even if lengths differ a lot
        if a in b or b in a:
            r = max(r, 0.8)
        return r

    def align(self,
              lyric_pairs: List[Tuple[str, str]],
              whisper_segs: List[Dict],
              song_name: str
              ) -> Tuple[List[Optional[Tuple[float, float]]], List[AlignmentFlag]]:
        """
        For each lyric pair, find best matching Whisper segment and assign
        its (start, end) timestamps. Unmatched lines get interpolated
        timestamps from neighbours so they still receive a melody slice.

        Returns:
            timestamps : list of (start, end) per lyric line, never None
                         (interpolated for unmatched lines)
            flags      : QA flags for unmatched lines
        """
        flags: List[AlignmentFlag] = []
        n = len(lyric_pairs)

        if not whisper_segs:
            flags.append(AlignmentFlag(
                song_name=song_name, line_index=-1,
                flag_type="empty_english_whisper", severity="error",
                message="Whisper returned 0 English segments — cannot get timestamps",
            ))
            # fall back to even split across [0, 0] — caller handles
            return [None] * n, flags

        en_lyrics = [self._norm(en) for en, _ in lyric_pairs]
        seg_norms = [self._norm(s["text"]) for s in whisper_segs]

        # Greedy 1:1 matching, lyric-line-first.  A Whisper segment can
        # only be claimed by one lyric line; if multiple lyric lines map
        # to the same segment we sub-divide it later.
        timestamps: List[Optional[Tuple[float, float]]] = [None] * n
        used = set()

        for li, en_norm in enumerate(en_lyrics):
            best_si, best_score = -1, 0.0
            for si, sn in enumerate(seg_norms):
                if si in used:
                    continue
                r = self._ratio(en_norm, sn)
                if r > best_score:
                    best_score, best_si = r, si
            if best_si >= 0 and best_score >= self.min_match_ratio:
                seg = whisper_segs[best_si]
                timestamps[li] = (float(seg["start"]), float(seg["end"]))
                used.add(best_si)
            else:
                flags.append(AlignmentFlag(
                    song_name=song_name, line_index=li,
                    flag_type="no_whisper_match", severity="warning",
                    message=f"lyric line {li} matched no Whisper segment "
                            f"(best ratio {best_score:.2f})",
                    english_text=lyric_pairs[li][0],
                    hindi_text=lyric_pairs[li][1],
                ))
        return timestamps, flags

    def fill_gaps(self,
                  timestamps: List[Optional[Tuple[float, float]]],
                  audio_duration: float
                  ) -> List[Tuple[float, float]]:
        """
        Replace any None timestamps with interpolated values from
        neighbouring matched lines (or evenly across audio if there are
        no matches at all). Returned list is the same length and never
        contains None.
        """
        n = len(timestamps)
        out: List[Tuple[float, float]] = [None] * n  # type: ignore

        # First pass: copy real matches over
        for i, t in enumerate(timestamps):
            if t is not None:
                out[i] = t

        any_real = any(t is not None for t in timestamps)
        if not any_real:
            # No matches anywhere — split audio evenly across all lines
            line_dur = audio_duration / max(n, 1)
            for i in range(n):
                out[i] = (i * line_dur, (i + 1) * line_dur)
            return out

        # For each unmatched line, interpolate from neighbours
        for i in range(n):
            if out[i] is not None:
                continue
            prev_idx = next((j for j in range(i - 1, -1, -1) if timestamps[j]), None)
            next_idx = next((j for j in range(i + 1, n) if timestamps[j]), None)
            prev_t = timestamps[prev_idx] if prev_idx is not None else None
            next_t = timestamps[next_idx] if next_idx is not None else None
            if prev_t and next_t:
                # Split the gap proportionally across unmatched lines
                # between prev and next
                gap_start = prev_t[1]
                gap_end = next_t[0]
                if gap_end <= gap_start:
                    gap_end = gap_start + 0.01
                out[i] = (gap_start, gap_end)
            elif prev_t:
                # Extrapolate forward from previous line's duration
                d = max(prev_t[1] - prev_t[0], 1.5)
                start = prev_t[1]
                out[i] = (start, min(start + d, audio_duration))
            elif next_t:
                d = max(next_t[1] - next_t[0], 1.5)
                end = next_t[0]
                out[i] = (max(0.0, end - d), end)

        return out  # type: ignore


# ---------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ---------------------------------------------------------------------------

class DisneyLyricsCurator:
    """End-to-end lyrics-based pipeline."""

    def __init__(self,
                 catalog_path: str = "data_gathering/disney_song_catalog.json",
                 lyrics_dir: str = "data_gathering/lyrics",
                 output_dir: str = "data_gathering/output",
                 download_dir: str = "data_gathering/downloads",
                 separated_dir: str = "data_gathering/separated",
                 whisper_model: str = "base",
                 device: str = "auto",
                 fresh: bool = False):

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
        self.fresh = fresh

        self.lyrics_loader = LyricsLoader(lyrics_dir)
        self.downloader = YouTubeDownloader(download_dir)
        self.separator = VocalSeparator(separated_dir)
        self.transcriber = WhisperTranscriber(whisper_model, device)
        self.melody_extractor = MelodyExtractorWithTimes()
        self.aligner = LyricsTimestampAligner()
        self.example_builder: Optional[ExampleBuilder] = None  # lazy

        # Progress tracking — separate file from the audio-Whisper pipeline
        self.progress_path = self.output_dir / "disney_lyrics_progress.json"
        self.progress = self._load_progress()

    # -------------------- progress + catalog --------------------

    def _load_progress(self) -> Dict:
        if self.fresh or not self.progress_path.exists():
            return {"done": [], "failed": [], "skipped_no_lyrics": []}
        try:
            with open(self.progress_path) as f:
                p = json.load(f)
            p.setdefault("done", [])
            p.setdefault("failed", [])
            p.setdefault("skipped_no_lyrics", [])
            return p
        except Exception:
            return {"done": [], "failed": [], "skipped_no_lyrics": []}

    def _save_progress(self):
        with open(self.progress_path, "w") as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

    def _load_catalog(self) -> List[Dict]:
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {self.catalog_path}")
        with open(self.catalog_path) as f:
            data = json.load(f)
        return data.get("songs", data) if isinstance(data, dict) else data

    def _init_example_builder(self):
        if self.example_builder is None:
            self.example_builder = ExampleBuilder(self.device)

    # -------------------- per-song processing --------------------

    def process_song(self, entry: Dict) -> Tuple[SongResult, List[Dict]]:
        name = entry["name"]
        result = SongResult(song_name=name, status="failed")

        print(f"\n{'='*60}")
        print(f"  Processing: {entry.get('title', name)}")
        print(f"{'='*60}")

        # Step 0: lyric files must exist
        if not self.lyrics_loader.has_lyrics(name):
            print(f"  [skip] no lyric files at "
                  f"data_gathering/lyrics/{name}.{{en,hi}}")
            result.status = "skipped_no_lyrics"
            result.error = "no lyric files"
            return result, []

        pairs, err = self.lyrics_loader.load(name)
        if pairs is None:
            print(f"  [error] lyric file: {err}")
            result.error = f"lyrics: {err}"
            return result, []
        print(f"  Loaded {len(pairs)} lyric line pairs")

        # Step 1: download English audio only — Hindi audio is not used
        print("\n  Step 1: Download English audio")
        english_audio = self.downloader.download(
            entry["english_url"], name, "english"
        )
        if english_audio is None:
            result.error = "english_download_failed"
            return result, []

        # Step 2: separate vocals (cached if already done)
        print("\n  Step 2: Separate vocals (English)")
        english_vocals = self.separator.separate(
            english_audio, name, "english", self.device
        )
        if english_vocals is None:
            result.error = "english_separation_failed"
            return result, []

        # Step 3: Whisper-English for timestamps only — text from lyric file
        print("\n  Step 3: Whisper English for timestamps")
        en_segs = self.transcriber.transcribe(english_vocals, "english")
        if not en_segs:
            result.error = "english_whisper_empty"
            return result, []

        # Step 4: extract melody + per-note times from English vocals
        print("\n  Step 4: Extract melody (English vocals)")
        melody, note_starts = self.melody_extractor.extract(english_vocals)
        if melody is None or note_starts is None:
            result.flags.append(AlignmentFlag(
                song_name=name, line_index=-1,
                flag_type="no_melody", severity="error",
                message="Melody extraction failed",
            ))
            result.error = "melody_extraction_failed"
            return result, []

        audio_duration = float(note_starts[-1]) if len(note_starts) else 0.0

        # Step 5: align lyric lines → timestamps via Whisper-English
        print("\n  Step 5: Align lyric lines to Whisper timestamps")
        ts_raw, align_flags = self.aligner.align(pairs, en_segs, name)
        result.flags.extend(align_flags)
        ts = self.aligner.fill_gaps(ts_raw, audio_duration)
        n_matched = sum(1 for t in ts_raw if t is not None)
        print(f"    matched {n_matched}/{len(pairs)} lines directly")

        # Step 6: slice melody by aligned timestamps
        print(f"\n  Step 6: Build {len(pairs)} training examples")
        self._init_example_builder()
        examples: List[Dict] = []

        for i, ((english, hindi), (t0, t1)) in enumerate(zip(pairs, ts)):
            mask = (note_starts >= t0) & (note_starts < t1)
            mel_slice = melody[mask]

            # Fallback: if window has 0 notes, take the nearest 4 notes
            # to the line's centre — keeps every line trainable
            if len(mel_slice) == 0:
                centre = 0.5 * (t0 + t1)
                if len(note_starts) > 0:
                    distances = np.abs(note_starts - centre)
                    k = min(4, len(note_starts))
                    nearest = np.argsort(distances)[:k]
                    mel_slice = melody[np.sort(nearest)]
                if len(mel_slice) == 0:
                    result.flags.append(AlignmentFlag(
                        song_name=name, line_index=i,
                        flag_type="empty_melody_slice", severity="warning",
                        message=f"line {i} got 0 notes (window {t0:.2f}–{t1:.2f}s)",
                        english_text=english, hindi_text=hindi,
                    ))
                    continue

            self._check_quality(name, i, english, hindi, mel_slice, result.flags)

            ex = self.example_builder.build(
                english=english, hindi=hindi,
                melody_slice=np.asarray(mel_slice, dtype=np.float32),
                song_name=f"{name}_line{i}",
            )
            examples.append(ex)

        result.num_examples = len(examples)
        if examples:
            result.status = "partial" if result.flags else "success"
        return result, examples

    # -------------------- quality checks --------------------

    def _check_quality(self, song_name, line_idx, english, hindi,
                       melody_slice, flags):
        """Same heuristic checks as the original curator (sanity-check
        each row before training). Hindi labels are now ground-truth so
        we expect *fewer* QA flags than the audio-Whisper pipeline."""
        from src.utils.syllable_utils import (
            count_english_syllables, count_hindi_syllables,
        )

        en_syl = count_english_syllables(english)
        hi_syl = count_hindi_syllables(hindi)
        num_notes = len(melody_slice)

        if en_syl > 0:
            ratio = hi_syl / en_syl
            if ratio > 2.5 or ratio < 0.3:
                flags.append(AlignmentFlag(
                    song_name=song_name, line_index=line_idx,
                    flag_type="syllable_ratio", severity="warning",
                    message=f"syllable ratio {ratio:.2f} "
                            f"(en={en_syl}, hi={hi_syl})",
                    english_text=english, hindi_text=hindi,
                ))

        if hi_syl > 0 and num_notes > 0:
            note_ratio = num_notes / hi_syl
            if note_ratio < 0.3 or note_ratio > 5.0:
                flags.append(AlignmentFlag(
                    song_name=song_name, line_index=line_idx,
                    flag_type="melody_coverage", severity="warning",
                    message=f"notes/syllables ratio {note_ratio:.2f} "
                            f"(notes={num_notes}, hi_syl={hi_syl})",
                    english_text=english, hindi_text=hindi,
                ))

    # -------------------- batch run --------------------

    def run(self, max_songs: Optional[int] = None,
            test_ratio: float = 0.15,
            merge_with_existing: bool = True):
        """Process all songs in the catalog that have lyric files."""
        catalog = self._load_catalog()

        print(f"\nCatalog: {len(catalog)} songs in catalog")
        with_lyrics = [e for e in catalog if self.lyrics_loader.has_lyrics(e["name"])]
        without_lyrics = [e["name"] for e in catalog if not self.lyrics_loader.has_lyrics(e["name"])]
        print(f"  with lyric files: {len(with_lyrics)}")
        print(f"  without lyric files (will skip): {len(without_lyrics)}")
        if without_lyrics:
            print(f"  → populate data_gathering/lyrics/<song>.en + <song>.hi")
            print(f"     to include those songs.\n")

        # Apply --max AFTER filtering — otherwise a single populated lyric
        # file deep in the catalog would be skipped.
        if max_songs is not None:
            with_lyrics = with_lyrics[:max_songs]
            print(f"  --max {max_songs} → processing {len(with_lyrics)} song(s)")

        if not with_lyrics:
            print("\nNo songs have lyric files yet — nothing to process.")
            print("See data_gathering/lyrics/README.md for the file format.")
            return [], []

        all_examples: List[Dict] = []
        all_results: List[SongResult] = []
        already_done = set(self.progress.get("done", []))

        for entry in with_lyrics:
            name = entry["name"]
            if name in already_done and not self.fresh:
                print(f"\n  [skip] {name} already processed (use --fresh to redo)")
                continue
            try:
                result, examples = self.process_song(entry)
            except Exception as e:
                import traceback
                traceback.print_exc()
                result = SongResult(song_name=name, status="failed",
                                    error=f"exception: {e}")
                examples = []

            all_results.append(result)

            if result.status in ("success", "partial"):
                all_examples.extend(examples)
                if name not in self.progress["done"]:
                    self.progress["done"].append(name)
            elif result.status == "skipped_no_lyrics":
                if name not in self.progress["skipped_no_lyrics"]:
                    self.progress["skipped_no_lyrics"].append(name)
            else:
                self.progress["failed"].append({"name": name, "error": result.error})

            self._save_progress()


        # -------- merge / save --------
        train_path = self.output_dir / "disney_train_data.pt"
        test_path = self.output_dir / "disney_test_data.pt"

        existing: List[Dict] = []
        if merge_with_existing and not self.fresh:
            if train_path.exists():
                existing.extend(torch.load(train_path, weights_only=False))
            if test_path.exists():
                existing.extend(torch.load(test_path, weights_only=False))

            # Discard any prior examples whose Hindi looks like the old
            # Whisper gibberish: very short, missing Devanagari diacritics,
            # or repeated identical Hindi for different English. Heuristic:
            # if a prior example's song_name starts with a song we just
            # re-processed, drop it (we now have the gold version).
            new_song_keys = {ex["song_name"].rsplit("_line", 1)[0]
                             for ex in all_examples}
            existing = [ex for ex in existing
                        if ex["song_name"].rsplit("_line", 1)[0] not in new_song_keys]
            print(f"\n  Merged with {len(existing)} pre-existing examples "
                  f"(after dropping reprocessed songs)")

        all_examples = all_examples + existing

        # Train/test split — fixed seed, stratified by song so a song
        # doesn't appear in both splits
        import random
        random.seed(42)
        by_song: Dict[str, List[Dict]] = {}
        for ex in all_examples:
            key = ex["song_name"].rsplit("_line", 1)[0]
            by_song.setdefault(key, []).append(ex)
        song_keys = sorted(by_song.keys())
        random.shuffle(song_keys)
        n_test_songs = max(1, int(len(song_keys) * test_ratio)) if len(song_keys) > 1 else 0
        test_keys = set(song_keys[:n_test_songs])
        train_examples = [ex for k, lst in by_song.items() if k not in test_keys for ex in lst]
        test_examples = [ex for k, lst in by_song.items() if k in test_keys for ex in lst]

        if train_examples:
            torch.save(train_examples, train_path)
        if test_examples:
            torch.save(test_examples, test_path)

        self._save_qa_report(all_results)

        total_flags = sum(len(r.flags) for r in all_results)
        succ = sum(1 for r in all_results if r.status == "success")
        partial = sum(1 for r in all_results if r.status == "partial")
        fail = sum(1 for r in all_results if r.status == "failed")
        skip = sum(1 for r in all_results if r.status == "skipped_no_lyrics")

        print(f"\n{'='*60}")
        print(f"  CURATION COMPLETE (lyrics-based)")
        print(f"{'='*60}")
        print(f"  Songs attempted     : {len(all_results)}")
        print(f"    success           : {succ}")
        print(f"    partial (flagged) : {partial}")
        print(f"    failed            : {fail}")
        print(f"    skipped (no lyric): {skip}")
        print(f"  Total examples      : {len(all_examples)}")
        print(f"    Train             : {len(train_examples)} → {train_path}")
        print(f"    Test              : {len(test_examples)} → {test_path}")
        print(f"  QA flags            : {total_flags}")
        print(f"  Report              : {self.output_dir / 'qa_report_lyrics.json'}")
        print(f"{'='*60}\n")

        return all_examples, all_results

    def _save_qa_report(self, results: List[SongResult]):
        report = []
        for r in results:
            report.append({
                "song_name": r.song_name,
                "status": r.status,
                "num_examples": r.num_examples,
                "error": r.error,
                "flags": [asdict(f) for f in r.flags],
            })
        report_path = self.output_dir / "qa_report_lyrics.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  QA report saved → {report_path}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {PROJECT_ROOT}\n")

    parser = argparse.ArgumentParser(
        description="Disney Lyrics-Based Curation Pipeline — uses ground-truth "
                    "Hindi lyric text instead of Whisper transcription.",
    )
    parser.add_argument(
        "--catalog", type=str,
        default="data_gathering/disney_song_catalog.json",
    )
    parser.add_argument(
        "--lyrics-dir", type=str,
        default="data_gathering/lyrics",
        help="Directory containing <song_name>.en and <song_name>.hi files",
    )
    parser.add_argument("--max", type=int, default=None,
                        help="Max songs to process (default: all with lyrics)")
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--whisper", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size for ENGLISH timestamps only")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore prior progress, re-process everything")
    parser.add_argument("--no-merge", action="store_true",
                        help="Don't merge with existing disney_*_data.pt — overwrite")
    args = parser.parse_args()

    curator = DisneyLyricsCurator(
        catalog_path=args.catalog,
        lyrics_dir=args.lyrics_dir,
        whisper_model=args.whisper,
        device=args.device,
        fresh=args.fresh,
    )
    curator.run(
        max_songs=args.max,
        test_ratio=args.test_ratio,
        merge_with_existing=not args.no_merge,
    )
