"""
Evaluation metrics for MCNST
"""

import torch
from src.utils.syllable_utils import count_hindi_syllables


def syllable_accuracy(predictions, targets):
    """
    Calculate syllable matching accuracy.
    
    Args:
        predictions: List of generated Hindi texts
        targets: List of target syllable counts
    
    Returns:
        accuracy: Percentage of predictions within ±2 syllables
    """
    if len(predictions) == 0:
        return 0.0
    
    correct = 0
    for pred_text, target_count in zip(predictions, targets):
        pred_count = count_hindi_syllables(pred_text)
        if abs(pred_count - target_count) <= 2:
            correct += 1
    
    return correct / len(predictions)


def evaluate_model(model, test_data, tokenizer):
    """
    Compute automatic evaluation metrics.
    
    Returns dict with BLEU, Syllable Accuracy, Avg Syllable Error.
    """
    predictions = []
    references = []
    syllable_errors = []
    
    model.eval()
    
    for example in test_data:
        with torch.no_grad():
            src_ids = example['src_ids'].unsqueeze(0)
            melody = example['melody_features'].unsqueeze(0)
            
            tokenizer.src_lang = "hi_IN"
            generated_ids = model.generate(
                input_ids=src_ids,
                melody_features=melody,
                max_length=50,
                num_beams=5,
                forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
            )
        
        pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        ref_text = example['hindi_text']
        
        predictions.append(pred_text)
        references.append(ref_text)
        
        pred_syl = count_hindi_syllables(pred_text)
        target_syl = example['num_notes']
        syllable_errors.append(abs(pred_syl - target_syl))
    
    # BLEU score (optional dependency)
    bleu_score = 0.0
    try:
        from sacrebleu import corpus_bleu
        bleu = corpus_bleu(predictions, [references])
        bleu_score = bleu.score
    except ImportError:
        print("  ⚠️  sacrebleu not installed, skipping BLEU score")
    
    syl_acc = sum(1 for e in syllable_errors if e <= 2) / max(len(syllable_errors), 1)
    avg_syl_error = sum(syllable_errors) / max(len(syllable_errors), 1)
    
    return {
        'BLEU': bleu_score,
        'Syllable_Accuracy': syl_acc,
        'Avg_Syllable_Error': avg_syl_error,
        'Predictions': predictions,
        'References': references
    }


if __name__ == "__main__":
    import os
    from pathlib import Path
    from transformers import MBart50TokenizerFast
    from src.models.mcnst_model import MCNST
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    os.chdir(PROJECT_ROOT)
    
    # Load model
    model = MCNST(freeze_encoder=True, freeze_decoder_layers=10)
    checkpoint = torch.load("checkpoints/best_model.pt", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    
    # Load test data
    test_data = torch.load("data/processed/training_data.pt")
    
    # Evaluate
    results = evaluate_model(model, test_data, tokenizer)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"BLEU Score:          {results['BLEU']:.2f}")
    print(f"Syllable Accuracy:   {results['Syllable_Accuracy']*100:.1f}%")
    print(f"Avg Syllable Error:  {results['Avg_Syllable_Error']:.2f}")
    print("="*60)
    
    print("\nSample Predictions:")
    for i in range(min(5, len(results['Predictions']))):
        print(f"\n{i+1}. Reference: {results['References'][i]}")
        print(f"   Predicted: {results['Predictions'][i]}")