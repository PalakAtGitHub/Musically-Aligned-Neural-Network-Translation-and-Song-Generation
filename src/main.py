"""
Main entry point for MCNST project

Commands:
  build     — Build dataset from MIDI + manifest
  train     — Train MCNST model
  test      — Test trained model
  separate  — Separate a song into vocals + instrumental (Phase 1)
  extract   — Extract melody from vocals (Phase 2)
  translate — Full pipeline: separate → extract → translate (Phase 1-3)
"""

import argparse
from pathlib import Path


def build_dataset():
    """Build dataset from raw MIDI + manifest."""
    from src.data.data_builder import DatasetBuilder

    builder = DatasetBuilder()
    builder.build_dataset_from_manifest("src/data/song_manifest.json")


def build_fma_dataset(max_songs: int = 200, test_ratio: float = 0.10,
                      seed: int = 42, whisper_model: str = "base"):
    """Build train/test datasets from FMA-medium audio files.

    Pipeline per song: Demucs → Whisper ASR → MarianMT translation →
    AudioMelodyExtractor → melody split → mBART tokenization → .pt file.
    """
    from src.data.fma_data_builder import FMADatasetBuilder

    builder = FMADatasetBuilder(seed=seed, whisper_model=whisper_model)
    builder.build(max_songs=max_songs, test_ratio=test_ratio)


def train(data_path: str = "src/data/processed/fma_train_data.pt",
          test_data_path: str = None):
    """Train MCNST model."""
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
        print(f"\n✓ Phase 1 complete!")
        print(f"  Next step: python -m src.main extract {result['vocals']}")
    return result


def extract(audio_path, num_lines=4):
    """Phase 2: Extract melody features from vocal audio."""
    from src.data.audio_melody_extractor import AudioMelodyExtractor
    import numpy as np
    
    extractor = AudioMelodyExtractor()
    features = extractor.extract_melody_features(audio_path, num_lyric_lines=num_lines)
    
    if features is not None:
        # Save features for later use
        save_path = Path("src/data/processed") / f"{Path(audio_path).stem}_melody.npy"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, features)
        print(f"  Saved melody features to: {save_path}")
        print(f"\n✓ Phase 2 complete!")
        print(f"  Features shape: {features.shape}")
    return features


def translate(audio_path, english_lyrics, num_lines=4):
    """Full pipeline: separate → extract → translate (Phases 1-3)."""
    import torch
    import numpy as np
    from src.data.audio_separator import AudioSeparator
    from src.data.audio_melody_extractor import AudioMelodyExtractor
    from src.testing.test import load_trained_model
    from transformers import MBart50TokenizerFast
    from src.utils.syllable_utils import count_hindi_syllables
    
    print("=" * 60)
    print("MCNST Full Audio Pipeline")
    print("=" * 60)
    
    # Phase 1: Separate
    print("\n--- Phase 1: Source Separation ---")
    separator = AudioSeparator()
    stems = separator.separate(audio_path)
    if stems is None:
        return None
    
    # Phase 2: Extract melody
    print("\n--- Phase 2: Melody Extraction ---")
    extractor = AudioMelodyExtractor()
    melody_features = extractor.extract_melody_features(
        str(stems['vocals']),
        num_lyric_lines=num_lines
    )
    if melody_features is None:
        return None
    
    # Phase 3: Translate with MCNST
    print("\n--- Phase 3: Melody-Constrained Translation ---")
    model = load_trained_model("checkpoints/best_model.pt")
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    
    # Split lyrics into lines
    lines = [l.strip() for l in english_lyrics.strip().split('\n') if l.strip()]
    
    # Segment melody across lines
    notes_per_line = len(melody_features) // max(len(lines), 1)
    
    results = []
    print(f"\nTranslating {len(lines)} lines...\n")
    
    for i, line in enumerate(lines):
        # Get melody segment for this line
        start = i * notes_per_line
        end = len(melody_features) if i == len(lines) - 1 else start + notes_per_line
        line_melody = melody_features[start:end]
        
        # Tokenize
        tokenizer.src_lang = "en_XX"
        src_ids = tokenizer(line, return_tensors="pt").input_ids
        melody_tensor = torch.tensor(line_melody, dtype=torch.float32).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=src_ids,
                melody_features=melody_tensor,
                max_length=50,
                num_beams=5,
                forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
            )
        
        hindi_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        hindi_syl = count_hindi_syllables(hindi_text)
        num_notes = len(line_melody)
        match = "✓" if abs(hindi_syl - num_notes) <= 2 else "✗"
        
        results.append({
            'english': line,
            'hindi': hindi_text,
            'syllables': hindi_syl,
            'target_notes': num_notes
        })
        
        print(f"  {match} EN: {line}")
        print(f"    HI: {hindi_text}  ({hindi_syl} syl, target: {num_notes})")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Translation complete!")
    print(f"  Instrumental saved at: {stems['instrumental']}")
    print(f"  Next: Phase 4 (DiffSinger) → synthesize Hindi singing")
    print(f"  Next: Phase 5 (RVC) → clone original singer's voice")
    print(f"  Next: Phase 6 → mix vocals + instrumental")
    print(f"{'='*60}")
    
    return {
        'stems': stems,
        'melody_features': melody_features,
        'translations': results
    }

def voice_clone(audio_path, english_lyrics, num_lines=4):
    """
    Full voice-cloning pipeline (Option A — easier, no RVC training):

      Phase 1: Demucs     → vocals.wav + instrumental.wav
      Phase 2: Librosa    → melody features + tempo estimate
      Phase 3: MCNST      → English lyrics → Hindi text
      Phase 4: XTTS v2    → Hindi audio in original singer's voice (zero-shot)
      Phase 5: Mixer      → time-stretched Hindi vocals + original instrumental
                          → final_song.wav

    Args:
        audio_path:     Path to any song audio file (MP3/WAV/FLAC)
        english_lyrics: English lyrics as a newline-separated string
        num_lines:      Number of lyric lines
    """
    import torch
    from src.data.audio_separator import AudioSeparator
    from src.data.audio_melody_extractor import AudioMelodyExtractor
    from src.audio.voice_clone_synthesizer import VoiceCloneSynthesizer, estimate_line_durations
    from src.audio.audio_mixer import AudioMixer
    from src.testing.test import load_trained_model
    from src.utils.syllable_utils import count_hindi_syllables
    from transformers import MBart50TokenizerFast

    print("=" * 60)
    print("MCNST Voice Cloning Pipeline")
    print("=" * 60)

    song_name = Path(audio_path).stem

    # ── Phase 1: Source Separation ──────────────────────────────────
    print("\n[1] Separating vocals from instrumental (Demucs)...")
    separator = AudioSeparator()

    # Check cache first
    cached = separator.get_separated(song_name)
    if cached:
        print(f"  ✓ Using cached separation: {song_name}")
        stems = cached
    else:
        stems = separator.separate(audio_path)

    if stems is None:
        print("  ✗ Separation failed. Cannot proceed — reference vocal needed for cloning.")
        return None

    vocal_path = stems['vocals']
    instrumental_path = stems['instrumental']

    # ── Phase 2: Melody extraction ───────────────────────────────────
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
    print(f"  {total_notes} notes extracted, ~{notes_per_line} per line")

    # Estimate tempo from melody features (duration col = beats per note)
    avg_beat_dur = float(melody_features[:, 2].mean())
    # Rough: if avg note is 0.5 beats and melody flows, ~120 BPM is typical
    estimated_bpm = 60.0 / max(avg_beat_dur, 0.25) / 2
    estimated_bpm = max(60.0, min(180.0, estimated_bpm))
    print(f"  Estimated tempo: {estimated_bpm:.0f} BPM")

    # Per-line target durations (so Hindi vocals fit the melody timing)
    target_durations = estimate_line_durations(melody_features, notes_per_line, estimated_bpm)

    # ── Phase 3: MCNST Translation ────────────────────────────────────
    print("\n[3] Translating lyrics (MCNST)...")
    model = load_trained_model("checkpoints/best_model.pt")
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    tokenizer.src_lang = "en_XX"
    forced_bos = tokenizer.lang_code_to_id["hi_IN"]

    lines = [l.strip() for l in english_lyrics.strip().split('\n') if l.strip()]
    hindi_lines = []

    for i, line in enumerate(lines):
        start = i * notes_per_line
        end = total_notes if i == len(lines) - 1 else start + notes_per_line
        line_melody = melody_features[start:end]

        src_ids = tokenizer(line, return_tensors="pt").input_ids
        melody_tensor = torch.tensor(line_melody, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=src_ids,
                melody_features=melody_tensor,
                max_length=50,
                num_beams=5,
                forced_bos_token_id=forced_bos,
                repetition_penalty=1.3,      # penalise repeated tokens
                no_repeat_ngram_size=3,      # forbid repeated 3-grams
                early_stopping=True
            )

        hindi_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        hindi_syl = count_hindi_syllables(hindi_text)
        target_notes = len(line_melody)
        match = "✓" if abs(hindi_syl - target_notes) <= 2 else "≈"

        print(f"  {match} [{i+1}] EN: {line}")
        print(f"       HI: {hindi_text}  ({hindi_syl} syl, target: {target_notes} notes)")
        hindi_lines.append(hindi_text)

    # ── Phase 4: Voice Clone TTS ──────────────────────────────────────
    print("\n[4] Synthesizing Hindi vocals in original voice (XTTS v2)...")
    synth = VoiceCloneSynthesizer()
    hindi_vocal_path = synth.synthesize_song(
        hindi_lines=hindi_lines,
        reference_vocal=str(vocal_path),
        output_name=f"{song_name}_hindi_vocals",
        target_durations=target_durations
    )

    if hindi_vocal_path is None:
        print("  ✗ Voice synthesis failed")
        return None

    # ── Phase 5: Mix with instrumental ───────────────────────────────
    print("\n[5] Mixing Hindi vocals with original instrumental...")
    mixer = AudioMixer()
    final_path = mixer.mix(
        vocals_path=str(hindi_vocal_path),
        instrumental_path=str(instrumental_path),
        output_name=f"{song_name}_hindi_final"
    )

    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete!")
    if final_path:
        print(f"  Final song:   {final_path}")
    print(f"  Hindi vocals: {hindi_vocal_path}")
    print(f"  Instrumental: {instrumental_path}")
    print(f"{'='*60}")

    return {
        'final': final_path,
        'hindi_vocals': hindi_vocal_path,
        'instrumental': instrumental_path,
        'translations': list(zip(lines, hindi_lines)),
    }


def synthesize_audio(audio_path, lyrics, voice_model=None, num_lines=4):
    """Full pipeline: translate + synthesize + voice convert + mix."""
    if not audio_path or not lyrics:
        print("Error: provide audio + lyrics.")
        print('Example: python -m src.main synthesize song.mp3 --lyrics "line1\\nline2"')
        return
    
    from src.data.audio_separator import AudioSeparator
    from src.data.audio_melody_extractor import AudioMelodyExtractor
    from src.audio.tts_synthesizer import TTSSynthesizer
    from src.audio.voice_converter import VoiceConverter
    from src.audio.audio_mixer import AudioMixer
    from src.testing.test import load_trained_model
    from transformers import MBart50TokenizerFast
    from src.utils.syllable_utils import count_hindi_syllables
    import torch
    
    print("=" * 60)
    print("MCNST Full Audio Synthesis Pipeline")
    print("=" * 60)
    
    # Phase 1: Separate
    print("\n--- Phase 1: Source Separation ---")
    separator = AudioSeparator()
    stems = separator.separate(audio_path)
    if stems is None:
        print("✗ Separation failed. Trying without separation...")
        stems = {'vocals': audio_path, 'instrumental': None, 'song_name': Path(audio_path).stem}
    
    # Phase 2: Extract melody
    print("\n--- Phase 2: Melody Extraction ---")
    extractor = AudioMelodyExtractor()
    melody_features = extractor.extract_melody_features(
        str(stems['vocals']), num_lyric_lines=num_lines
    )
    if melody_features is None:
        print("✗ Melody extraction failed")
        return
    
    # Phase 3: Translate
    print("\n--- Phase 3: Translation ---")
    model = load_trained_model("checkpoints/best_model.pt")
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    
    lines = [l.strip() for l in lyrics.strip().split('\n') if l.strip()]
    notes_per_line = len(melody_features) // max(len(lines), 1)
    
    hindi_lines = []
    for i, line in enumerate(lines):
        start = i * notes_per_line
        end = len(melody_features) if i == len(lines) - 1 else start + notes_per_line
        line_melody = melody_features[start:end]
        
        tokenizer.src_lang = "en_XX"
        src_ids = tokenizer(line, return_tensors="pt").input_ids
        melody_tensor = torch.tensor(line_melody, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=src_ids,
                melody_features=melody_tensor,
                max_length=50, num_beams=5,
                forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
            )
        
        hindi_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        hindi_lines.append(hindi_text)
        print(f"  {line} → {hindi_text}")
    
    # Phase 4: TTS
    print("\n--- Phase 4: Text-to-Speech ---")
    tts = TTSSynthesizer(mode="espeak")
    tts_path = tts.synthesize_song(hindi_lines, output_name=stems['song_name'] + "_hindi")
    
    if tts_path is None:
        print("✗ TTS failed")
        return
    
    # Phase 5: Voice Conversion
    print("\n--- Phase 5: Voice Conversion ---")
    converter = VoiceConverter()
    if voice_model:
        converted_path = converter.convert(str(tts_path), voice_model,
                                           output_name=stems['song_name'] + "_converted")
    else:
        print("  ⚠️  No voice model provided — skipping voice conversion")
        converted_path = tts_path
    
    # Phase 6: Mix
    if stems.get('instrumental') and Path(str(stems['instrumental'])).exists():
        print("\n--- Phase 6: Audio Mixing ---")
        mixer = AudioMixer()
        final_path = mixer.mix(
            str(converted_path), str(stems['instrumental']),
            output_name=stems['song_name'] + "_final"
        )
    else:
        final_path = converted_path
        print("\n  ⚠️  No instrumental track — output is vocals only")
    
    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete!")
    print(f"  Final output: {final_path}")
    print(f"{'='*60}")


def evaluate():
    """Run SingScore + BERTScore + BLEU evaluation on trained model."""
    import torch
    from src.evaluation.sing_score import SingScore, print_sing_score
    from src.evaluation.metrics import compute_bert_score
    from src.testing.test import load_trained_model
    from transformers import MBart50TokenizerFast
    from src.utils.syllable_utils import count_hindi_syllables

    model = load_trained_model("checkpoints/best_model.pt")
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    data = torch.load("src/data/processed/fma_train_data.pt", weights_only=False)
    scorer = SingScore()

    hindi_lines, english_lines, notes_list, references = [], [], [], []

    for ex in data[:10]:
        src_ids = ex['src_ids'].unsqueeze(0)
        melody = ex['melody_features'].unsqueeze(0)

        tokenizer.src_lang = "hi_IN"
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=src_ids, melody_features=melody,
                max_length=50, num_beams=5,
                forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
            )

        hindi_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        hindi_lines.append(hindi_text)
        english_lines.append(ex['english_text'])
        notes_list.append(ex['num_notes'])
        references.append(ex['hindi_text'])

    # SingScore
    result = scorer.score_song(hindi_lines, english_lines, notes_list)
    print_sing_score(result, "MCNST Model Evaluation")

    # BERTScore
    bert_scores = compute_bert_score(hindi_lines, references, lang="hi")
    print(f"\n{'='*60}")
    print("BERTScore (Hindi)")
    print(f"{'='*60}")
    print(f"  Precision: {bert_scores['precision']:.4f}")
    print(f"  Recall:    {bert_scores['recall']:.4f}")
    print(f"  F1:        {bert_scores['f1']:.4f}")

    # BLEU
    try:
        from sacrebleu import corpus_bleu
        bleu = corpus_bleu(hindi_lines, [references])
        print(f"\nBLEU Score:  {bleu.score:.2f}")
    except ImportError:
        print("\n  Warning: sacrebleu not installed, skipping BLEU score")

    # Syllable accuracy
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
  python -m src.main build
  python -m src.main build_fma --max 200 --test-ratio 0.10
  python -m src.main train --data src/data/processed/fma_train_data.pt \\
                           --test-data src/data/processed/fma_test_data.pt
  python -m src.main test
  python -m src.main separate song.mp3
  python -m src.main extract data/separated/song/vocals.wav
  python -m src.main translate song.mp3 --lyrics "Twinkle twinkle little star"
  python -m src.main voice_clone song.mp3 --lyrics "Twinkle twinkle little star\\nHow I wonder what you are"
        """
    )

    parser.add_argument('command',
                        choices=['build', 'build_fma', 'train', 'test', 'separate',
                                 'extract', 'translate', 'voice_clone',
                                 'synthesize', 'evaluate'])
    parser.add_argument('audio', nargs='?', help='Path to audio file (for separate/extract/translate)')
    parser.add_argument('--lyrics', type=str, help='English lyrics (for translate/synthesize)')
    parser.add_argument('--lines', type=int, default=4, help='Number of lyric lines (default: 4)')
    parser.add_argument('--voice-model', type=str, help='Path to RVC voice model (for synthesize)')
    # build_fma options
    parser.add_argument('--max', type=int, default=200,
                        help='Max FMA songs to process for build_fma (default: 200)')
    parser.add_argument('--test-ratio', type=float, default=0.10,
                        help='Fraction of songs held out for testing (default: 0.10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for build_fma (default: 42)')
    parser.add_argument('--whisper', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size for ASR in build_fma (default: base)')
    # train options
    parser.add_argument('--data', type=str,
                        default="src/data/processed/fma_train_data.pt",
                        help='Training data path (default: fma_train_data.pt)')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Held-out test data path (optional)')

    args = parser.parse_args()

    if args.command == 'build':
        build_dataset()
    elif args.command == 'build_fma':
        build_fma_dataset(max_songs=args.max, test_ratio=args.test_ratio,
                          seed=args.seed, whisper_model=args.whisper)
    elif args.command == 'train':
        train(data_path=args.data, test_data_path=args.test_data)
    elif args.command == 'test':
        test()
    elif args.command == 'separate':
        if not args.audio:
            print("Error: provide audio path. Example: python -m src.main separate song.mp3")
        else:
            separate(args.audio)
    elif args.command == 'extract':
        if not args.audio:
            print("Error: provide vocal audio path. Example: python -m src.main extract vocals.wav")
        else:
            extract(args.audio, num_lines=args.lines)
    elif args.command == 'translate':
        if not args.audio or not args.lyrics:
            print("Error: provide audio + lyrics.")
            print('Example: python -m src.main translate song.mp3 --lyrics "line1\\nline2"')
        else:
            translate(args.audio, args.lyrics, num_lines=args.lines)
    elif args.command == 'voice_clone':
        if not args.audio or not args.lyrics:
            print("Error: provide audio + lyrics.")
            print('Example: python -m src.main voice_clone song.mp3 --lyrics "Twinkle twinkle little star\\nHow I wonder what you are"')
        else:
            voice_clone(args.audio, args.lyrics, num_lines=args.lines)
    elif args.command == 'synthesize':
        synthesize_audio(args.audio, args.lyrics, args.voice_model, args.lines)
    elif args.command == 'evaluate':
        evaluate()