#!/usr/bin/env python3
"""
Compare All Methods

Compares BERT, RoBERTa, LLM baseline, and TRUST Agents.
"""

import argparse
import json
from pathlib import Path


def load_metrics(path: Path):
    """Load metrics from JSON file."""
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def create_comparison_table(dataset: str, output: str):
    """Create comparison table of all methods."""
    
    methods = {}
    
    # BERT
    bert_metrics = load_metrics(Path(f"outputs/hf/bert_{dataset}_metrics.json"))
    if bert_metrics:
        methods['BERT'] = {
            'accuracy': bert_metrics.get('accuracy', 0),
            'f1': bert_metrics.get('f1', 0),
            'type': 'baseline'
        }
    
    # RoBERTa
    roberta_metrics = load_metrics(Path(f"outputs/hf/roberta_{dataset}_metrics.json"))
    if roberta_metrics:
        methods['RoBERTa'] = {
            'accuracy': roberta_metrics.get('accuracy', 0),
            'f1': roberta_metrics.get('f1', 0),
            'type': 'baseline'
        }
    
    # LLM (GPT-4.1-mini)
    llm_metrics = load_metrics(Path(f"outputs/llm_baseline/metrics_{dataset}.json"))
    if llm_metrics:
        methods['GPT-4.1-mini'] = {
            'accuracy': llm_metrics.get('accuracy', 0),
            'f1': llm_metrics.get('f1', 0),
            'type': 'baseline'
        }
    
    # TRUST Agents
    trust_metrics = load_metrics(Path(f"outputs/trust_agents/metrics_{dataset}.json"))
    if trust_metrics:
        methods['TRUST Agents'] = {
            'accuracy': trust_metrics.get('accuracy', 0),
            'f1': trust_metrics.get('f1', 0),
            'precision': trust_metrics.get('precision', 0),
            'recall': trust_metrics.get('recall', 0),
            'type': 'multi-agent',
            'confidence_mean': trust_metrics.get('confidence_stats', {}).get('mean', 0),
            'avg_claims': trust_metrics.get('claim_stats', {}).get('mean_claims_per_text', 0)
        }
    
    # Create comparison
    comparison = {
        'dataset': dataset,
        'methods': methods,
        'best_accuracy': max((m['accuracy'] for m in methods.values()), default=0),
        'best_f1': max((m['f1'] for m in methods.values()), default=0)
    }
    
    # Find best performers
    for method, metrics in methods.items():
        if metrics['accuracy'] == comparison['best_accuracy']:
            metrics['best_accuracy'] = True
        if metrics['f1'] == comparison['best_f1']:
            metrics['best_f1'] = True
    
    return comparison


def print_comparison(comparison):
    """Pretty print comparison."""
    print("\n" + "="*80)
    print("METHOD COMPARISON - {}".format(comparison['dataset'].upper()))
    print("="*80)
    
    print("\n{:<20} {:>12} {:>12} {:>10}".format("Method", "Accuracy", "F1 Score", "Type"))
    print("-"*80)
    
    for method, metrics in comparison['methods'].items():
        acc_str = f"{metrics['accuracy']:.4f}"
        f1_str = f"{metrics['f1']:.4f}"
        
        # Add star for best performance
        if metrics.get('best_accuracy'):
            acc_str += " ★"
        if metrics.get('best_f1'):
            f1_str += " ★"
        
        print("{:<20} {:>12} {:>12} {:>10}".format(
            method, acc_str, f1_str, metrics['type']
        ))
    
    # TRUST Agents additional metrics
    if 'TRUST Agents' in comparison['methods']:
        trust = comparison['methods']['TRUST Agents']
        print("\n" + "-"*80)
        print("TRUST Agents Additional Metrics:")
        print(f"  Precision:         {trust.get('precision', 0):.4f}")
        print(f"  Recall:            {trust.get('recall', 0):.4f}")
        print(f"  Avg Confidence:    {trust.get('confidence_mean', 0):.4f}")
        print(f"  Avg Claims/Text:   {trust.get('avg_claims', 0):.2f}")
    
    print("\n" + "="*80)
    print("★ = Best performance")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all methods")
    parser.add_argument("--dataset", choices=["liar", "fnn"], required=True)
    parser.add_argument("--out", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Create comparison
    comparison = create_comparison_table(args.dataset, args.out)
    
    # Print comparison
    print_comparison(comparison)
    
    # Save to file
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {output_path}")