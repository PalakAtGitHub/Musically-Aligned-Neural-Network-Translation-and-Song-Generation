"""
Main entry point for MCNST project

Commands:
  build_fma       — Build dataset from FMA-medium audio (Demucs+Whisper+MarianMT)
  train           — Supervised training on FMA data
  test            — Test trained model on examples
  evaluate        — Full evaluation (SingScore + BERTScore + BLEU + syllable)
  separate        — Separate a song into vocals + instrumental
  extract         — Extract melody features from vocals
  translate       — Translate English lyrics to Hindi (text only)
  generate_vocals — Generate separate Hindi vocal audio files from lyrics
  voice_clone     — Full pipeline: separate + translate + synthesize + mix
"""

import argparse
from pathlib import Path


def build_fma_dataset(max_songs: int = 200, test_ratio: float = 0.10,
                      seed: int = 42, whisper_model: str = "base"):
    """Build train/test datasets from FMA-medium audio files."""
    from src.data.fma_data_builder import FMADatasetBuilder

    builder = FMADatasetBuilder(seed=seed, whisper_model=whisper_model)
    builder.build(max_songs=max_songs, test_ratio=test_ratio)


def train(data_path: str = "src/data/processed/fma_train_data.pt",
          test_data_path: str = None):
    """Supervised training on FMA data."""
    from src.training.train import Trainer

    trainer = Trainer(
        data_path=data_path,
        test_data_path=test_data_path,
        batch_size=4,
        num_epochs=10
    )
    trainer.train()


def test():
    """Test trained model."""
    from src.testing.test import load_trained_model, test_on_examples

    model = load_trained_model("checkpoints/best_model.pt")
    test_on_examples(model, num_examples=5)


def separate(audio_path):
    """Phase 1: Separate song into vocals + instrumental."""
    from src.data.audio_separator import AudioSeparator

    separator = AudioSeparator()
    result = separator.separate(audio_path)
    if result:
        print(f"\n✓ Separation complete!")
        print(f"  Vocals:       {result['vocals']}")
        print(f"  Instrumental: {result['instrumental']}")
    return result


def extract(audio_path, num_lines=4):
    """Phase 2: Extract melody features from vocal audio."""
    from src.data.audio_melody_extractor import AudioMelodyExtractor
    import numpy as np

    extractor = AudioMelodyExtractor()
    features = extractor.extract_melody_features(audio_path, num_lyric_lines=num_lines)

    if features is not None:
        save_path = Path("src/data/processed") / f"{Path(audio_path).stem}_melody.npy"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, features)
        print(f"  Saved: {save_path}  shape: {features.shape}")
    return features


def _translate_line(model, line, melody_slice):
    """Tokenize, generate, decode one English line → Hindi with melody fusion."""
    import torch

    tokenizer = model.tokenizer
    preprocessed = model.preprocess_src(line)
    src_ids = tokenizer(preprocessed[0], return_tensors="pt").input_ids
    melody_tensor = torch.tensor(melody_slice, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=src_ids, melody_features=melody_tensor,
            max_length=50, num_beams=5,
        )

    hindi_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return model.postprocess_tgt(hindi_text)[0]


def translate(audio_path, english_lyrics, num_lines=4):
    """Separate + extract melody + translate (text only, no audio synthesis)."""
    from src.data.audio_separator import AudioSeparator
    from src.data.audio_melody_extractor import AudioMelodyExtractor
    from src.testing.test import load_trained_model
    from src.utils.syllable_utils import count_hindi_syllables

    print("=" * 60)
    print("MCNST Translation")
    print("=" * 60)

    separator = AudioSeparator()
    stems = separator.separate(audio_path)
    if stems is None:
        return None

    extractor = AudioMelodyExtractor()
    melody_features = extractor.extract_melody_features(
        str(stems['vocals']), num_lyric_lines=num_lines
    )
    if melody_features is None:
        return None

    model = load_trained_model("checkpoints/best_model.pt")

    lines = [l.strip() for l in english_lyrics.strip().split('\n') if l.strip()]
    notes_per_line = len(melody_features) // max(len(lines), 1)
    results = []

    for i, line in enumerate(lines):
        start = i * notes_per_line
        end = len(melody_features) if i == len(lines) - 1 else start + notes_per_line
        line_melody = melody_features[start:end]

        hindi_text = _translate_line(model, line, line_melody)
        syl = count_hindi_syllables(hindi_text)
        match = "✓" if abs(syl - len(line_melody)) <= 2 else "✗"
        results.append({'english': line, 'hindi': hindi_text,
                        'syllables': syl, 'target_notes': len(line_melody)})
        print(f"  {match} EN: {line}")
        print(f"    HI: {hindi_text}  ({syl} syl, target: {len(line_melody)})")

    # Save
    out_dir = Path("src/data/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(audio_path).stem}_translation.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"EN: {r['english']}\nHI: {r['hindi']}\n\n")
    print(f"\n  Saved: {out_file}")
    return results


def generate_vocals(audio_path, english_lyrics, num_lines=4):
    """Generate Hindi vocal audio files and a mixed (vocals + instrumental) version.

    Pipeline: Demucs → melody extraction → MCNST translation → TTS per line.
    Outputs two final files:
      1. {song}_hindi_vocals.wav  — combined Hindi vocals only
      2. {song}_hindi_mixed.wav   — Hindi vocals mixed with instrumental
    Plus per-line WAVs in src/data/hindi_vocals/{song}/.
    """
    from src.data.audio_separator import AudioSeparator
    from src.data.audio_melody_extractor import AudioMelodyExtractor
    from src.audio.voice_clone_synthesizer import VoiceCloneSynthesizer
    from src.audio.audio_mixer import AudioMixer
    from src.testing.test import load_trained_model
    from src.utils.syllable_utils import count_hindi_syllables

    print("=" * 60)
    print("Hindi Vocal Generation")
    print("=" * 60)

    song_name = Path(audio_path).stem
    output_dir = Path("src/data/hindi_vocals") / song_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Separate vocals
    print("\n[1] Separating vocals...")
    separator = AudioSeparator()
    cached = separator.get_separated(song_name)
    if cached:
        stems = cached
    else:
        stems = separator.separate(audio_path)
    if stems is None:
        print("  Separation failed")
        return None

    # Phase 2: Extract melody
    print("\n[2] Extracting melody...")
    extractor = AudioMelodyExtractor()
    melody_features = extractor.extract_melody_features(
        str(stems['vocals']), num_lyric_lines=num_lines
    )
    if melody_features is None:
        print("  Melody extraction failed")
        return None

    total_notes = len(melody_features)
    notes_per_line = max(1, total_notes // num_lines)

    # Phase 3: Translate each line
    print("\n[3] Translating lyrics (MCNST + IndicTrans2)...")
    model = load_trained_model("checkpoints/best_model.pt")

    lines = [l.strip() for l in english_lyrics.strip().split('\n') if l.strip()]
    hindi_lines = []

    for i, line in enumerate(lines):
        start = i * notes_per_line
        end = total_notes if i == len(lines) - 1 else start + notes_per_line
        line_melody = melody_features[start:end]

        hindi_text = _translate_line(model, line, line_melody)
        hindi_syl = count_hindi_syllables(hindi_text)
        print(f"  [{i+1}] EN: {line}")
        print(f"       HI: {hindi_text}  ({hindi_syl} syl, target: {len(line_melody)} notes)")
        hindi_lines.append(hindi_text)

    # Phase 4: Generate separate audio for each Hindi line
    print(f"\n[4] Generating Hindi vocal audio files → {output_dir}/")
    synth = VoiceCloneSynthesizer(output_dir=str(output_dir))
    vocal_paths = []

    for i, hindi_text in enumerate(hindi_lines):
        path = synth.synthesize_line(
            text=hindi_text,
            reference_vocal=str(stems['vocals']),
            output_name=f"line_{i+1:02d}",
        )
        if path:
            vocal_paths.append(path)
            print(f"    ✓ {path.name}")
        else:
            print(f"    ✗ Line {i+1} failed")

    # Phase 5: Combine all line WAVs into a single Hindi vocals file
    combined_vocals_path = None
    if vocal_paths:
        print(f"\n[5] Combining lines into single vocal track...")
        combined_vocals_path = synth.synthesize_song(
            hindi_lines=hindi_lines,
            reference_vocal=str(stems['vocals']),
            output_name=f"{song_name}_hindi_vocals",
        )

    # Phase 6: Mix Hindi vocals with instrumental
    mixed_path = None
    if combined_vocals_path and stems.get('instrumental'):
        print(f"\n[6] Mixing Hindi vocals with instrumental...")
        mixer = AudioMixer(output_dir=str(output_dir))
        mixed_path = mixer.mix(
            vocals_path=str(combined_vocals_path),
            instrumental_path=str(stems['instrumental']),
            output_name=f"{song_name}_hindi_mixed",
        )

    # Save translation text alongside audio
    txt_path = output_dir / "lyrics.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, (en, hi) in enumerate(zip(lines, hindi_lines)):
            audio_file = f"line_{i+1:02d}.wav"
            f.write(f"[{audio_file}]\n  EN: {en}\n  HI: {hi}\n\n")

    print(f"\n{'='*60}")
    print(f"✓ Hindi audio generation complete!")
    print(f"  Per-line WAVs:    {output_dir}/line_*.wav")
    if combined_vocals_path:
        print(f"  Hindi vocals:     {combined_vocals_path}")
    if mixed_path:
        print(f"  Vocals + music:   {mixed_path}")
    print(f"  Lyrics file:      {txt_path}")
    print(f"{'='*60}")

    return {'vocal_paths': vocal_paths, 'hindi_lines': hindi_lines,
            'combined_vocals': str(combined_vocals_path) if combined_vocals_path else None,
            'mixed': str(mixed_path) if mixed_path else None,
            'output_dir': str(output_dir)}


def voice_clone(audio_path, english_lyrics, num_lines=4):
    """Full pipeline: Demucs + MCNST + XTTS v2 + Mixer → Hindi song."""
    from src.data.audio_separator import AudioSeparator
    from src.data.audio_melody_extractor import AudioMelodyExtractor
    from src.audio.voice_clone_synthesizer import VoiceCloneSynthesizer, estimate_line_durations
    from src.audio.audio_mixer import AudioMixer
    from src.testing.test import load_trained_model
    from src.utils.syllable_utils import count_hindi_syllables

    print("=" * 60)
    print("MCNST Voice Cloning Pipeline")
    print("=" * 60)

    song_name = Path(audio_path).stem

    # Phase 1: Source Separation
    print("\n[1] Separating vocals from instrumental (Demucs)...")
    separator = AudioSeparator()
    cached = separator.get_separated(song_name)
    if cached:
        print(f"  ✓ Using cached separation: {song_name}")
        stems = cached
    else:
        stems = separator.separate(audio_path)
    if stems is None:
        print("  ✗ Separation failed")
        return None

    vocal_path = stems['vocals']
    instrumental_path = stems['instrumental']

    # Phase 2: Melody extraction (Basic-Pitch)
    print("\n[2] Extracting melody features...")
    extractor = AudioMelodyExtractor()
    melody_features = extractor.extract_melody_features(
        str(vocal_path), num_lyric_lines=num_lines
    )
    if melody_features is None:
        print("  ✗ Melody extraction failed")
        return None

    total_notes = len(melody_features)
    notes_per_line = max(1, total_notes // num_lines)
    avg_beat_dur = float(melody_features[:, 2].mean())
    estimated_bpm = max(60.0, min(180.0, 60.0 / max(avg_beat_dur, 0.25) / 2))
    target_durations = estimate_line_durations(melody_features, notes_per_line, estimated_bpm)

    # Phase 3: MCNST Translation (IndicTrans2)
    print("\n[3] Translating lyrics (MCNST + IndicTrans2)...")
    model = load_trained_model("checkpoints/best_model.pt")

    lines = [l.strip() for l in english_lyrics.strip().split('\n') if l.strip()]
    hindi_lines = []

    for i, line in enumerate(lines):
        start = i * notes_per_line
        end = total_notes if i == len(lines) - 1 else start + notes_per_line
        line_melody = melody_features[start:end]

        hindi_text = _translate_line(model, line, line_melody)
        hindi_syl = count_hindi_syllables(hindi_text)
        match = "✓" if abs(hindi_syl - len(line_melody)) <= 2 else "≈"
        print(f"  {match} [{i+1}] EN: {line}")
        print(f"       HI: {hindi_text}  ({hindi_syl} syl, target: {len(line_melody)} notes)")
        hindi_lines.append(hindi_text)

    # Phase 4: Voice Clone TTS
    print("\n[4] Synthesizing Hindi vocals (XTTS v2)...")
    synth = VoiceCloneSynthesizer()
    hindi_vocal_path = synth.synthesize_song(
        hindi_lines=hindi_lines, reference_vocal=str(vocal_path),
        output_name=f"{song_name}_hindi_vocals", target_durations=target_durations
    )
    if hindi_vocal_path is None:
        print("  ✗ Voice synthesis failed")
        return None

    # Phase 5: Mix
    print("\n[5] Mixing Hindi vocals with instrumental...")
    mixer = AudioMixer()
    final_path = mixer.mix(
        vocals_path=str(hindi_vocal_path),
        instrumental_path=str(instrumental_path),
        output_name=f"{song_name}_hindi_final"
    )

    # Save translation
    out_dir = Path("src/data/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{song_name}_translation.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for en, hi in zip(lines, hindi_lines):
            f.write(f"EN: {en}\nHI: {hi}\n\n")

    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete!")
    if final_path:
        print(f"  Final song:   {final_path}")
    print(f"  Hindi vocals: {hindi_vocal_path}")
    print(f"  Instrumental: {instrumental_path}")
    print(f"  Translation:  {out_file}")
    print(f"{'='*60}")

    return {'final': final_path, 'hindi_vocals': hindi_vocal_path,
            'instrumental': instrumental_path,
            'translations': list(zip(lines, hindi_lines))}


def evaluate():
    """Full evaluation: SingScore + BERTScore + BLEU + syllable accuracy."""
    import torch
    from src.evaluation.sing_score import SingScore, print_sing_score
    from src.evaluation.metrics import compute_bert_score
    from src.testing.test import load_trained_model
    from src.utils.syllable_utils import count_hindi_syllables

    model = load_trained_model("checkpoints/best_model.pt")
    tokenizer = model.tokenizer
    data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)
    scorer = SingScore()

    hindi_lines, english_lines, notes_list, references = [], [], [], []

    for ex in data[:10]:
        src_ids = ex['src_ids'].unsqueeze(0)
        melody = ex['melody_features'].unsqueeze(0)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=src_ids, melody_features=melody,
                max_length=50, num_beams=5,
            )

        hindi_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        hindi_text = model.postprocess_tgt(hindi_text)[0]
        hindi_lines.append(hindi_text)
        english_lines.append(ex['english_text'])
        notes_list.append(ex['num_notes'])
        references.append(ex['hindi_text'])

    result = scorer.score_song(hindi_lines, english_lines, notes_list)
    print_sing_score(result, "MCNST Model Evaluation")

    bert_scores = compute_bert_score(hindi_lines, references, lang="hi")
    print(f"\n{'='*60}")
    print("BERTScore (Hindi)")
    print(f"{'='*60}")
    print(f"  Precision: {bert_scores['precision']:.4f}")
    print(f"  Recall:    {bert_scores['recall']:.4f}")
    print(f"  F1:        {bert_scores['f1']:.4f}")

    try:
        from sacrebleu import corpus_bleu
        bleu = corpus_bleu(hindi_lines, [references])
        print(f"\nBLEU Score:  {bleu.score:.2f}")
    except ImportError:
        print("\n  Warning: sacrebleu not installed")

    syl_errors = [abs(count_hindi_syllables(h) - n)
                  for h, n in zip(hindi_lines, notes_list)]
    syl_acc = sum(1 for e in syl_errors if e <= 2) / max(len(syl_errors), 1)
    avg_err = sum(syl_errors) / max(len(syl_errors), 1)
    print(f"\nSyllable Accuracy (±2): {syl_acc*100:.1f}%")
    print(f"Avg Syllable Error:     {avg_err:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import os
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    os.chdir(PROJECT_ROOT)

    parser = argparse.ArgumentParser(
        description="MCNST — Musically Constrained Song Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main build_fma --max 500
  python -m src.main train
  python -m src.main test
  python -m src.main evaluate
  python -m src.main generate_vocals song.mp3 --lyrics "line1\\nline2"
  python -m src.main voice_clone song.mp3 --lyrics "line1\\nline2"
        """
    )

    parser.add_argument('command',
                        choices=['build_fma', 'train', 'test',
                                 'separate', 'extract', 'translate',
                                 'generate_vocals', 'voice_clone', 'evaluate'])
    parser.add_argument('audio', nargs='?', help='Path to audio file')
    parser.add_argument('--lyrics', type=str, help='English lyrics (newline-separated)')
    parser.add_argument('--lines', type=int, default=4, help='Number of lyric lines')
    # build_fma
    parser.add_argument('--max', type=int, default=500)
    parser.add_argument('--test-ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--whisper', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'])
    # train
    parser.add_argument('--data', type=str,
                        default="src/data/processed/fma_train_data.pt")
    parser.add_argument('--test-data', type=str, default=None)

    args = parser.parse_args()

    if args.command == 'build_fma':
        build_fma_dataset(max_songs=args.max, test_ratio=args.test_ratio,
                          seed=args.seed, whisper_model=args.whisper)
    elif args.command == 'train':
        train(data_path=args.data, test_data_path=args.test_data)
    elif args.command == 'test':
        test()
    elif args.command == 'separate':
        if not args.audio:
            print("Error: provide audio path")
        else:
            separate(args.audio)
    elif args.command == 'extract':
        if not args.audio:
            print("Error: provide vocal audio path")
        else:
            extract(args.audio, num_lines=args.lines)
    elif args.command == 'translate':
        if not args.audio or not args.lyrics:
            print("Error: provide audio + lyrics")
        else:
            translate(args.audio, args.lyrics, num_lines=args.lines)
    elif args.command == 'generate_vocals':
        if not args.audio or not args.lyrics:
            print("Error: provide audio + lyrics")
        else:
            generate_vocals(args.audio, args.lyrics, num_lines=args.lines)
    elif args.command == 'voice_clone':
        if not args.audio or not args.lyrics:
            print("Error: provide audio + lyrics")
        else:
            voice_clone(args.audio, args.lyrics, num_lines=args.lines)
    elif args.command == 'evaluate':
        evaluate()
