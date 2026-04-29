"""
Generate Hindi audio files from the trained MCNST model.

Uses the trained checkpoint to translate English test examples to Hindi,
then synthesizes Hindi audio using the VoiceCloneSynthesizer fallback chain.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.mcnst_model import MCNST
from src.audio.voice_clone_synthesizer import VoiceCloneSynthesizer
from src.utils.syllable_utils import count_hindi_syllables

print("=" * 70)
print("HINDI AUDIO GENERATION FROM TRAINED MCNST")
print("=" * 70)

# 1. Load model
print("\n[1/4] Loading trained model...")
model = MCNST(freeze_encoder=True, freeze_decoder_layers=16)
device = torch.device("cpu")
ckpt = torch.load("checkpoints/best_model.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()
tokenizer = model.tokenizer
print(f"  Loaded epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

# 2. Load test examples
print("\n[2/4] Loading test data...")
data = torch.load("src/data/processed/fma_test_data.pt", weights_only=False)

# Pick 5 diverse examples
indices = [0, 1, 4, 8, 9]
examples = [data[i] for i in indices]

# 3. Generate Hindi translations
print("\n[3/4] Generating Hindi translations...")
output_dir = Path("src/data/hindi_audio_output")
output_dir.mkdir(parents=True, exist_ok=True)

translations = []
for i, ex in enumerate(examples):
    english = ex['english_text']
    hindi_ref = ex['hindi_text']
    num_notes = ex['num_notes']

    src_ids = ex['src_ids'].unsqueeze(0)
    melody = ex['melody_features'].unsqueeze(0)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=src_ids,
            melody_features=melody,
            max_length=50,
            num_beams=5,
        )
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    gen_text = model.postprocess_tgt(gen_text)[0]
    gen_syl = count_hindi_syllables(gen_text)

    translations.append({
        'english': english,
        'hindi_ref': hindi_ref,
        'hindi_gen': gen_text,
        'gen_syl': gen_syl,
        'num_notes': num_notes,
    })

    match = "✓" if abs(gen_syl - num_notes) <= 2 else "✗"
    print(f"  [{i+1}] EN: {english}")
    print(f"       HI: {gen_text}  ({gen_syl} syl, target: {num_notes} notes) {match}")

# 4. Synthesize Hindi audio
print("\n[4/4] Synthesizing Hindi audio...")

# Use one of the separated vocals as reference for voice cloning
ref_vocal = "data/separated/145515/vocals.wav"
if not Path(ref_vocal).exists():
    ref_vocal = "data/separated/073174/vocals.wav"
if not Path(ref_vocal).exists():
    ref_vocal = None
    print("  No reference vocal found — voice cloning won't work, using TTS fallback")

synth = VoiceCloneSynthesizer(output_dir=str(output_dir))

generated_files = []
for i, t in enumerate(translations):
    print(f"\n  Line {i+1}: \"{t['hindi_gen']}\"")
    path = synth.synthesize_line(
        text=t['hindi_gen'],
        reference_vocal=ref_vocal if ref_vocal else "",
        output_name=f"hindi_line_{i+1:02d}",
    )
    if path:
        generated_files.append(path)
        print(f"    -> {path}")
    else:
        print(f"    -> FAILED")

# Save lyrics file
lyrics_path = output_dir / "translations.txt"
with open(lyrics_path, "w", encoding="utf-8") as f:
    for i, t in enumerate(translations):
        f.write(f"Line {i+1}:\n")
        f.write(f"  EN: {t['english']}\n")
        f.write(f"  HI (generated): {t['hindi_gen']}  ({t['gen_syl']} syl)\n")
        f.write(f"  HI (reference): {t['hindi_ref']}\n")
        f.write(f"  Target notes:   {t['num_notes']}\n\n")

print(f"\n{'=' * 70}")
print(f"RESULTS:")
print(f"  Translations saved: {lyrics_path}")
print(f"  Audio files generated: {len(generated_files)}/{len(translations)}")
for f in generated_files:
    print(f"    {f}")
print(f"{'=' * 70}")
