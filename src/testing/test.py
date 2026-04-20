"""
Test the trained MCNST model (Two-Stage Architecture)

Supports three inference modes:
  1. Standard generation (melody-fused, single pass)
  2. Two-stage generation (draft → adapt)
  3. Constrained generation (syllable-aware beam search)
"""

import torch
from transformers import MBart50TokenizerFast
from src.models.mcnst_model import MCNST
from src.utils.syllable_utils import count_hindi_syllables, count_english_syllables
from src.data.midi_loader import MIDILoader


def load_trained_model(checkpoint_path="checkpoints/best_model.pt"):
    """Load the trained model."""
    print(f"Loading model from {checkpoint_path}...")
    
    model = MCNST(freeze_encoder=True, freeze_decoder_layers=10)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Handle old checkpoints that may have renamed loss parameters
    state = checkpoint['model_state_dict']
    if 'loss_fn.log_var_singability' in state and 'loss_fn.log_var_naturalness' not in state:
        state['loss_fn.log_var_naturalness'] = state.pop('loss_fn.log_var_singability')
    model.load_state_dict(state, strict=False)
    model.eval()
    
    print("✓ Model loaded successfully!")
    return model


def test_on_examples(model, num_examples=5):
    """Test model on training examples using all three generation modes."""
    
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    
    data = torch.load("src/data/processed/fma_train_data.pt", weights_only=False)
    
    print(f"\n{'='*80}")
    print(f"Testing on {min(num_examples, len(data))} examples")
    print(f"{'='*80}\n")
    
    for i, example in enumerate(data[:num_examples]):
        print(f"Example {i+1}:")
        print("-" * 80)
        
        english_text = example['english_text']
        hindi_reference = example['hindi_text']
        num_notes = example['num_notes']
        
        print(f"English:   {english_text}")
        print(f"Reference: {hindi_reference}")
        print(f"Notes:     {num_notes}")
        
        src_ids = example['src_ids'].unsqueeze(0)
        melody = example['melody_features'].unsqueeze(0)
        
        tokenizer.src_lang = "hi_IN"
        forced_bos = tokenizer.lang_code_to_id["hi_IN"]
        
        with torch.no_grad():
            # --- Mode 1: Standard Generation ---
            gen_ids = model.generate(
                input_ids=src_ids,
                melody_features=melody,
                max_length=50,
                num_beams=5,
                forced_bos_token_id=forced_bos
            )
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            gen_syl = count_hindi_syllables(gen_text)
            
            # --- Mode 2: Two-Stage Generation ---
            two_stage_ids = model.generate_two_stage(
                input_ids=src_ids,
                melody_features=melody,
                max_length=50,
                num_beams=5,
                forced_bos_token_id=forced_bos
            )
            two_stage_text = tokenizer.decode(two_stage_ids[0], skip_special_tokens=True)
            two_stage_syl = count_hindi_syllables(two_stage_text)
        
        ref_syl = count_hindi_syllables(hindi_reference)
        
        print(f"\n  [Standard]   {gen_text}  ({gen_syl} syl)")
        print(f"  [Two-Stage]  {two_stage_text}  ({two_stage_syl} syl)")
        print(f"  [Reference]  {hindi_reference}  ({ref_syl} syl)")
        print(f"  [Target]     {num_notes} notes")
        
        std_match = abs(gen_syl - num_notes) <= 2
        ts_match = abs(two_stage_syl - num_notes) <= 2
        print(f"\n  Standard syllable match:  {'✓' if std_match else '✗'}")
        print(f"  Two-stage syllable match: {'✓' if ts_match else '✗'}")
        print()


def test_new_song(model, english_line, midi_path, use_constrained=True):
    """Test on a completely new song (not in training data)."""
    
    print(f"\n{'='*80}")
    print(f"Testing on NEW song (unseen)")
    print(f"{'='*80}\n")
    
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    
    loader = MIDILoader()
    melody_features = loader.extract_melody_features(midi_path, num_lyric_lines=1)
    
    if melody_features is None:
        print("✗ Failed to load MIDI")
        return
    
    num_notes = len(melody_features)
    print(f"English: {english_line}")
    print(f"Notes:   {num_notes}")
    
    tokenizer.src_lang = "en_XX"
    src_ids = tokenizer(english_line, return_tensors="pt").input_ids
    melody_tensor = torch.tensor(melody_features, dtype=torch.float32).unsqueeze(0)
    
    forced_bos = tokenizer.lang_code_to_id["hi_IN"]
    
    with torch.no_grad():
        # Standard generation
        gen_ids = model.generate(
            input_ids=src_ids,
            melody_features=melody_tensor,
            max_length=50,
            num_beams=5,
            forced_bos_token_id=forced_bos
        )
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        gen_syl = count_hindi_syllables(gen_text)
        
        print(f"\n[Standard]  {gen_text}  ({gen_syl} syl, target: {num_notes})")
        
        # Constrained generation (if requested)
        if use_constrained:
            print("\nRunning constrained beam search (may take a moment)...")
            constrained_tokens, constrained_syl = model.generate_constrained(
                input_ids=src_ids,
                melody_features=melody_tensor,
                target_syllables=num_notes,
                num_beams=3
            )
            constrained_text = tokenizer.decode(constrained_tokens, skip_special_tokens=True)
            
            print(f"[Constrained] {constrained_text}  ({constrained_syl} syl, target: {num_notes})")
    
    std_match = abs(gen_syl - num_notes) <= 2
    print(f"\nStandard result:     {'✓ Singable!' if std_match else '✗ Needs adjustment'}")
    if use_constrained:
        cst_match = abs(constrained_syl - num_notes) <= 2
        print(f"Constrained result:  {'✓ Singable!' if cst_match else '✗ Needs adjustment'}")


if __name__ == "__main__":
    import os
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {PROJECT_ROOT}")
    
    # Load model
    model = load_trained_model()
    
    # Test on training examples (Standard + Two-Stage)
    test_on_examples(model, num_examples=5)
    
    # Test on unseen song (Standard + Constrained)
    test_midi = "src/data/raw_midis/you-are-the-sunshine-of-my-life-lead-sheet-with-lyrics.mid"
    if Path(test_midi).exists():
        test_new_song(
            model,
            english_line="You are the sunshine of my life",
            midi_path=test_midi,
            use_constrained=True
        )
    else:
        print(f"\nSkipping new song test: {test_midi} not found")