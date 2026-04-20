"""
Batch Hindi Audio Synthesis

For every FMA song that has been processed (separated + transcribed),
runs the full inference pipeline and writes a Hindi audio file:

  src/data/separated/<id>/vocals.wav        (already exists)
  src/data/separated/<id>/instrumental.wav  (already exists)
  src/data/processed/fma_train_data.pt      (lyrics + melody already built)

  → src/data/output/<id>_hindi_final.wav    (NEW — Hindi song)
  → src/data/output/<id>_hindi_vocals.wav   (NEW — Hindi vocals only)

The model (checkpoints/best_model.pt) must be trained before running this.

Usage:
  cd musically-aligned-translation/
  python -m src.testing.batch_synthesize              # all processed songs
  python -m src.testing.batch_synthesize --max 10     # first 10 only
  python -m src.testing.batch_synthesize --song 085822
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

from src.testing.test import load_trained_model
from src.audio.voice_clone_synthesizer import VoiceCloneSynthesizer, estimate_line_durations
from src.audio.audio_mixer import AudioMixer
from src.data.audio_melody_extractor import AudioMelodyExtractor
from src.utils.syllable_utils import count_hindi_syllables
from src.evaluation.sing_score import SingScore, print_sing_score
from transformers import MBart50TokenizerFast


PROGRESS_PATH = PROJECT_ROOT / "src/data/output/synthesis_progress.json"


def load_synthesis_progress() -> Dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"done": [], "failed": []}


def save_synthesis_progress(progress: Dict):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def group_examples_by_song(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Group training examples by song ID, preserving line order."""
    songs = {}
    for ex in data:
        song_id = ex["song_name"].split("_line")[0]
        songs.setdefault(song_id, []).append(ex)
    # Sort lines within each song by line index
    for song_id in songs:
        songs[song_id].sort(key=lambda x: x["song_name"])
    return songs


def translate_lines(
    model,
    tokenizer,
    examples: List[Dict],
    melody_features: np.ndarray,
) -> List[Dict]:
    """
    Run MCNST inference for each lyric line of a song.

    Returns list of dicts: {english, hindi, syllables, target_notes}
    """
    forced_bos = tokenizer.lang_code_to_id["hi_IN"]
    tokenizer.src_lang = "hi_IN"

    total_notes = len(melody_features)
    num_lines = len(examples)
    notes_per_line = max(1, total_notes // num_lines)

    results = []
    for i, ex in enumerate(examples):
        start = i * notes_per_line
        end = total_notes if i == num_lines - 1 else start + notes_per_line
        line_melody = melody_features[start:end]

        src_ids = ex["src_ids"].unsqueeze(0)
        melody_t = torch.tensor(line_melody, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=src_ids,
                melody_features=melody_t,
                max_length=60,
                num_beams=5,
                forced_bos_token_id=forced_bos,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        hindi = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        syl = count_hindi_syllables(hindi)
        target = len(line_melody)
        match = "✓" if abs(syl - target) <= 2 else "≈"

        print(f"    {match} [{i+1}] EN: {ex['english_text']}")
        print(f"         HI: {hindi}  ({syl} syl / {target} notes)")

        results.append({
            "english": ex["english_text"],
            "hindi": hindi,
            "syllables": syl,
            "target_notes": target,
        })

    return results


def synthesize_song(
    song_id: str,
    examples: List[Dict],
    model,
    tokenizer,
    synth: VoiceCloneSynthesizer,
    mixer: AudioMixer,
    extractor: AudioMelodyExtractor,
) -> Optional[Path]:
    """
    Run the full pipeline for one song:
      1. Load separated stems (must already exist)
      2. Re-extract melody from clean vocals
      3. MCNST translate
      4. TTS synthesize Hindi vocals
      5. Mix with instrumental
      6. SingScore evaluation

    Returns path to final Hindi song, or None on failure.
    """
    sep_dir = PROJECT_ROOT / "src/data/separated" / song_id
    vocals_path = sep_dir / "vocals.wav"
    instrumental_path = sep_dir / "instrumental.wav"

    if not vocals_path.exists() or not instrumental_path.exists():
        print(f"  ✗ Separated stems not found for {song_id} — skipping")
        return None

    print(f"\n  [2] Extracting melody from clean vocals...")
    melody_features = extractor.extract_melody_features(
        str(vocals_path), num_lyric_lines=len(examples)
    )
    if melody_features is None or len(melody_features) == 0:
        print(f"  ✗ Melody extraction failed")
        return None

    # Estimate BPM and per-line durations for time-stretching
    avg_beat_dur = float(melody_features[:, 2].mean())
    bpm = max(60.0, min(180.0, 60.0 / max(avg_beat_dur, 0.25) / 2))
    notes_per_line = max(1, len(melody_features) // len(examples))
    target_durations = estimate_line_durations(melody_features, notes_per_line, bpm)

    print(f"  [3] Translating {len(examples)} line(s) with MCNST...")
    results = translate_lines(model, tokenizer, examples, melody_features)
    if not results:
        print(f"  ✗ Translation produced no output")
        return None

    hindi_lines = [r["hindi"] for r in results]
    english_lines = [r["english"] for r in results]
    notes_list = [r["target_notes"] for r in results]

    print(f"  [4] Synthesizing Hindi vocals...")
    hindi_vocal_path = synth.synthesize_song(
        hindi_lines=hindi_lines,
        reference_vocal=str(vocals_path),
        output_name=f"{song_id}_hindi_vocals",
        target_durations=target_durations,
    )
    if hindi_vocal_path is None:
        print(f"  ✗ TTS synthesis failed")
        return None

    print(f"  [5] Mixing with instrumental...")
    final_path = mixer.mix(
        vocals_path=str(hindi_vocal_path),
        instrumental_path=str(instrumental_path),
        output_name=f"{song_id}_hindi_final",
    )

    # SingScore
    scorer = SingScore()
    score_result = scorer.score_song(hindi_lines, english_lines, notes_list)
    overall = score_result["overall"]["sing_score"]
    syl_acc = score_result["overall"]["syllable"]
    print(f"  SingScore: {overall:.2f}  (syllable: {syl_acc:.2f})")

    return final_path


def run_batch(song_ids: List[str], songs_map: Dict[str, List[Dict]]):
    progress = load_synthesis_progress()
    already_done = set(progress["done"])
    already_failed = {e["id"] for e in progress["failed"]}

    # Load model + tokenizer once
    print("Loading model and tokenizer...")
    model = load_trained_model("checkpoints/best_model.pt")
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )

    synth = VoiceCloneSynthesizer()
    mixer = AudioMixer()
    extractor = AudioMelodyExtractor()

    total = len(song_ids)
    succeeded = 0
    skipped = 0

    for i, song_id in enumerate(song_ids):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] Song: {song_id}")
        print(f"{'='*60}")

        if song_id in already_done:
            print("  Already synthesized — skipping")
            skipped += 1
            continue
        if song_id in already_failed:
            print("  Previously failed — skipping")
            skipped += 1
            continue

        examples = songs_map.get(song_id, [])
        if not examples:
            print("  No training examples found — skipping")
            progress["failed"].append({"id": song_id, "reason": "no_examples"})
            save_synthesis_progress(progress)
            continue

        print(f"  {len(examples)} lyric line(s)")
        try:
            final_path = synthesize_song(
                song_id, examples, model, tokenizer, synth, mixer, extractor
            )
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            final_path = None

        if final_path:
            print(f"\n  ✓ Saved: {final_path}")
            progress["done"].append(song_id)
            succeeded += 1
        else:
            progress["failed"].append({"id": song_id, "reason": "synthesis_failed"})

        save_synthesis_progress(progress)

    print(f"\n{'='*60}")
    print(f"Batch complete: {succeeded} synthesized, {skipped} skipped, "
          f"{total - succeeded - skipped} failed")
    print(f"Output: src/data/output/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-synthesize Hindi audio for all processed FMA songs"
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Max number of songs to process (default: all)"
    )
    parser.add_argument(
        "--song", type=str, default=None,
        help="Process a single song by ID (e.g. --song 085822)"
    )
    parser.add_argument(
        "--data", type=str,
        default="src/data/processed/fma_train_data.pt",
        help="Path to training data .pt file"
    )
    args = parser.parse_args()

    print("Loading training data...")
    data = torch.load(args.data, weights_only=False)
    songs_map = group_examples_by_song(data)
    print(f"Found {len(songs_map)} unique songs in {args.data}")

    if args.song:
        song_ids = [args.song]
    else:
        song_ids = sorted(songs_map.keys())
        if args.max:
            song_ids = song_ids[:args.max]

    run_batch(song_ids, songs_map)
