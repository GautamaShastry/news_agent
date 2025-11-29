#!/usr/bin/env python3
"""
Evaluate TRUST Agents Predictions

Computes accuracy, F1, and other metrics for fact-checking predictions.
Handles the label mismatch between 3-class predictions (true/false/uncertain)
and binary gold labels (true/false).
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def load_predictions(path: Path):
    """Load predictions from JSONL file."""
    predictions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def evaluate(predictions, uncertain_as: str = "false"):
    """
    Evaluate predictions against gold labels.
    
    Args:
        predictions: List of prediction dicts with 'gold' and 'pred' keys
        uncertain_as: How to treat 'uncertain' predictions ('true', 'false', or 'drop')
        
    Returns:
        Dictionary with evaluation metrics
    """
    gold_labels = []
    pred_labels = []
    
    # Count prediction distribution
    pred_dist = Counter()
    gold_dist = Counter()
    
    dropped = 0
    
    for p in predictions:
        gold = p.get('gold', 'unknown').lower()
        pred = p.get('pred', 'uncertain').lower()
        
        # Skip invalid gold labels
        if gold not in ['true', 'false']:
            continue
        
        gold_dist[gold] += 1
        pred_dist[pred] += 1
        
        # Handle uncertain predictions
        if pred == 'uncertain' or pred == 'error':
            if uncertain_as == 'drop':
                dropped += 1
                continue
            else:
                pred = uncertain_as
        
        # Normalize predictions
        if pred in ['supported', 'true']:
            pred = 'true'
        elif pred in ['contradicted', 'refuted', 'false']:
            pred = 'false'
        else:
            pred = uncertain_as  # Fallback
        
        gold_labels.append(gold)
        pred_labels.append(pred)
    
    if not gold_labels:
        return {"error": "No valid predictions to evaluate"}
    
    # Compute metrics
    accuracy = accuracy_score(gold_labels, pred_labels)
    f1_binary = f1_score(gold_labels, pred_labels, pos_label='true', average='binary')
    f1_macro = f1_score(gold_labels, pred_labels, average='macro')
    
    # Confusion matrix
    cm = confusion_matrix(gold_labels, pred_labels, labels=['true', 'false'])
    
    return {
        "total_predictions": len(predictions),
        "evaluated": len(gold_labels),
        "dropped": dropped,
        "accuracy": round(accuracy, 4),
        "f1_binary": round(f1_binary, 4),
        "f1_macro": round(f1_macro, 4),
        "gold_distribution": dict(gold_dist),
        "pred_distribution": dict(pred_dist),
        "confusion_matrix": {
            "true_true": int(cm[0][0]),
            "true_false": int(cm[0][1]),
            "false_true": int(cm[1][0]),
            "false_false": int(cm[1][1])
        },
        "uncertain_handling": uncertain_as
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRUST predictions")
    parser.add_argument("--preds", required=True, help="Path to predictions JSONL")
    parser.add_argument("--uncertain-as", choices=["true", "false", "drop"], default="false",
                       help="How to handle 'uncertain' predictions (default: false)")
    parser.add_argument("--output", help="Save metrics to JSON file")
    
    args = parser.parse_args()
    
    # Load predictions
    preds_path = Path(args.preds)
    if not preds_path.exists():
        print(f"Error: Predictions file not found: {preds_path}")
        exit(1)
    
    predictions = load_predictions(preds_path)
    print(f"Loaded {len(predictions)} predictions from {preds_path}")
    
    # Evaluate with different uncertain handling
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Main evaluation
    results = evaluate(predictions, uncertain_as=args.uncertain_as)
    
    print(f"\nTotal predictions: {results['total_predictions']}")
    print(f"Evaluated: {results['evaluated']}")
    print(f"Dropped (uncertain): {results['dropped']}")
    
    print(f"\n--- Metrics (uncertain → {args.uncertain_as}) ---")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"F1 (binary): {results['f1_binary']:.4f}")
    print(f"F1 (macro): {results['f1_macro']:.4f}")
    
    print(f"\n--- Distribution ---")
    print(f"Gold: {results['gold_distribution']}")
    print(f"Pred: {results['pred_distribution']}")
    
    print(f"\n--- Confusion Matrix ---")
    cm = results['confusion_matrix']
    print(f"              Pred True  Pred False")
    print(f"Gold True     {cm['true_true']:>8}  {cm['true_false']:>10}")
    print(f"Gold False    {cm['false_true']:>8}  {cm['false_false']:>10}")
    
    # Also show results with different uncertain handling
    print("\n" + "="*60)
    print("COMPARISON: Different Uncertain Handling")
    print("="*60)
    
    for handling in ["true", "false", "drop"]:
        r = evaluate(predictions, uncertain_as=handling)
        marker = " ← current" if handling == args.uncertain_as else ""
        print(f"  uncertain → {handling:5}: Accuracy={r['accuracy']:.2%}, F1={r['f1_macro']:.4f}{marker}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Metrics saved to {output_path}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
