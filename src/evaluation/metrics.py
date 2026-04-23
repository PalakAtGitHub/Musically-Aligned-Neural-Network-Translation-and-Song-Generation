"""
Evaluation Metrics for MCNST

Computes:
- BLEU Score (translation quality via sacrebleu)
- BERTScore (semantic similarity via bert-score)
- Syllable Accuracy (singability)
- Average Syllable Error
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
        accuracy: Percentage of predictions within +/-2 syllables
    """
    correct = 0
    for pred, target in zip(predictions, targets):
        pred_syllables = count_hindi_syllables(pred)
        if abs(pred_syllables - target) <= 2:
            correct += 1
    return correct / max(len(predictions), 1)


def compute_bert_score(predictions, references, lang="hi"):
    """Compute BERTScore for Hindi translations."""
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(
            predictions, references,
            lang=lang,
            verbose=False
        )
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    except ImportError:
        print("  Warning: bert-score not installed, skipping BERTScore")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


def evaluate_model(model, test_data, tokenizer=None):
    """
    Compute all automatic metrics on test data.

    Args:
        model: Trained MCNST model
        test_data: List of example dicts (from training_data.pt)
        tokenizer: Tokenizer instance (defaults to model.tokenizer)

    Returns:
        dict with BLEU, BERTScore (P/R/F1), Syllable_Accuracy,
        Avg_Syllable_Error, Predictions, References
    """
    if tokenizer is None:
        tokenizer = model.tokenizer

    predictions = []
    references = []
    syllable_errors = []

    model.eval()

    for example in test_data:
        with torch.no_grad():
            src_ids = example['src_ids'].unsqueeze(0)
            melody = example['melody_features'].unsqueeze(0)

            generated_ids = model.generate(
                input_ids=src_ids,
                melody_features=melody,
                max_length=50,
                num_beams=5,
            )

        pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        pred_text = model.postprocess_tgt(pred_text)[0]
        ref_text = example['hindi_text']

        predictions.append(pred_text)
        references.append(ref_text)

        pred_syl = count_hindi_syllables(pred_text)
        target_syl = example['num_notes']
        syllable_errors.append(abs(pred_syl - target_syl))

    # BLEU
    bleu_score = 0.0
    try:
        from sacrebleu import corpus_bleu
        bleu = corpus_bleu(predictions, [references])
        bleu_score = bleu.score
    except ImportError:
        print("  Warning: sacrebleu not installed, skipping BLEU score")

    # BERTScore
    bert_scores = compute_bert_score(predictions, references, lang="hi")

    # Syllable metrics
    syl_acc = sum(1 for e in syllable_errors if e <= 2) / max(len(syllable_errors), 1)
    avg_syl_error = sum(syllable_errors) / max(len(syllable_errors), 1)

    return {
        'BLEU': bleu_score,
        'BERTScore_P': bert_scores['precision'],
        'BERTScore_R': bert_scores['recall'],
        'BERTScore_F1': bert_scores['f1'],
        'Syllable_Accuracy': syl_acc,
        'Avg_Syllable_Error': avg_syl_error,
        'Predictions': predictions,
        'References': references
    }


if __name__ == "__main__":
    import os
    from pathlib import Path
    from src.models.mcnst_model import MCNST

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    os.chdir(PROJECT_ROOT)

    # Load model
    model = MCNST(freeze_encoder=True, freeze_decoder_layers=10)
    checkpoint = torch.load("checkpoints/best_model.pt", map_location='cpu',
                            weights_only=False)
    state = checkpoint['model_state_dict']
    if 'loss_fn.log_var_singability' in state and 'loss_fn.log_var_naturalness' not in state:
        state['loss_fn.log_var_naturalness'] = state.pop('loss_fn.log_var_singability')
    model.load_state_dict(state, strict=False)
    model.eval()

    tokenizer = model.tokenizer

    test_data = torch.load("src/data/processed/training_data.pt", weights_only=False)

    results = evaluate_model(model, test_data, tokenizer)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"BLEU Score:          {results['BLEU']:.2f}")
    print(f"BERTScore F1:        {results['BERTScore_F1']:.4f}")
    print(f"BERTScore Precision: {results['BERTScore_P']:.4f}")
    print(f"BERTScore Recall:    {results['BERTScore_R']:.4f}")
    print(f"Syllable Accuracy:   {results['Syllable_Accuracy']*100:.1f}%")
    print(f"Avg Syllable Error:  {results['Avg_Syllable_Error']:.2f}")
    print("=" * 60)

    print("\nSample Predictions:")
    for i in range(min(5, len(results['Predictions']))):
        print(f"\n{i+1}. Reference: {results['References'][i]}")
        print(f"   Predicted: {results['Predictions'][i]}")
