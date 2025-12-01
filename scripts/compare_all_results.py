#!/usr/bin/env python3
"""
Compare All Results - Baselines vs Agent Pipelines

Compares results from:
1. Fine-tuned models (BERT, RoBERTa)
2. LLM baseline (GPT)
3. Original TRUST pipeline
4. Research pipeline (LoCal + Delphi)

Generates a comprehensive comparison table and saves to JSON.
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

try:
    from sklearn.metrics import accuracy_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_jsonl(path: Path):
    """Load predictions from JSONL file."""
    if not path.exists():
        return None
    predictions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def load_json(path: Path):
    """Load metrics from JSON file."""
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_predictions(predictions, uncertain_as="false"):
    """Evaluate predictions with uncertain handling."""
    if not predictions:
        return None
    
    gold_labels = []
    pred_labels = []
    pred_dist = Counter()
    
    for p in predictions:
        gold = p.get('gold', 'unknown').lower()
        pred = p.get('pred', 'uncertain').lower()
        
        if gold not in ['true', 'false']:
            continue
        
        pred_dist[pred] += 1
        
        # Handle uncertain
        if pred in ['uncertain', 'error']:
            pred = uncertain_as
        
        # Normalize
        if pred in ['supported', 'true']:
            pred = 'true'
        elif pred in ['contradicted', 'refuted', 'false']:
            pred = 'false'
        else:
            pred = uncertain_as
        
        gold_labels.append(gold)
        pred_labels.append(pred)
    
    if not gold_labels or not HAS_SKLEARN:
        return None
    
    accuracy = accuracy_score(gold_labels, pred_labels)
    f1 = f1_score(gold_labels, pred_labels, average='macro')
    
    uncertain_count = pred_dist.get('uncertain', 0) + pred_dist.get('error', 0)
    uncertain_rate = uncertain_count / len(predictions) if predictions else 0
    
    return {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1, 4),
        "total": len(predictions),
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_rate, 4),
        "pred_distribution": dict(pred_dist)
    }


def get_baseline_results(base_dir: Path, results_dir: Path = None):
    """Get results from baseline models."""
    results = {}
    
    # First, try to load from results/baselines_liar.json (pre-computed baselines)
    if results_dir:
        baselines_file = results_dir / "baselines_liar.json"
        if baselines_file.exists():
            data = load_json(baselines_file)
            if data and "baselines" in data:
                baselines = data["baselines"]
                
                # BERT
                if "bert" in baselines:
                    results["BERT (fine-tuned)"] = {
                        "accuracy": baselines["bert"].get("accuracy", 0),
                        "f1_macro": baselines["bert"].get("f1", 0),
                        "type": "fine-tuned",
                        "source": str(baselines_file)
                    }
                
                # RoBERTa
                if "roberta" in baselines:
                    results["RoBERTa (fine-tuned)"] = {
                        "accuracy": baselines["roberta"].get("accuracy", 0),
                        "f1_macro": baselines["roberta"].get("f1", 0),
                        "type": "fine-tuned",
                        "source": str(baselines_file)
                    }
                
                # GPT baseline
                if "gpt4.1-nano" in baselines:
                    results["LLM Baseline (GPT-4.1-nano)"] = {
                        "accuracy": baselines["gpt4.1-nano"].get("accuracy", 0),
                        "f1_macro": baselines["gpt4.1-nano"].get("f1", 0),
                        "type": "llm-baseline",
                        "source": str(baselines_file)
                    }
                
                if results:
                    return results
    
    # Fallback: try to load from outputs directory
    # BERT
    bert_metrics = base_dir / "hf" / "bert_liar_metrics.json"
    if bert_metrics.exists():
        data = load_json(bert_metrics)
        if data:
            results["BERT (fine-tuned)"] = {
                "accuracy": data.get("accuracy", 0),
                "f1_macro": data.get("f1", 0),
                "type": "fine-tuned",
                "source": str(bert_metrics)
            }
    
    # RoBERTa
    roberta_metrics = base_dir / "hf" / "roberta_liar_metrics.json"
    if roberta_metrics.exists():
        data = load_json(roberta_metrics)
        if data:
            results["RoBERTa (fine-tuned)"] = {
                "accuracy": data.get("accuracy", 0),
                "f1_macro": data.get("f1", 0),
                "type": "fine-tuned",
                "source": str(roberta_metrics)
            }
    
    # LLM baseline
    llm_metrics = base_dir / "llm_baseline" / "metrics_liar.json"
    if llm_metrics.exists():
        data = load_json(llm_metrics)
        if data:
            results["LLM Baseline (GPT)"] = {
                "accuracy": data.get("accuracy", 0),
                "f1_macro": data.get("f1", 0),
                "type": "llm-baseline",
                "source": str(llm_metrics)
            }
    
    return results


def get_pipeline_results(base_dir: Path):
    """Get results from agent pipelines."""
    results = {}
    
    # Original TRUST pipeline
    trust_preds = base_dir / "trust_agents" / "liar_predictions.jsonl"
    if trust_preds.exists():
        predictions = load_jsonl(trust_preds)
        if predictions:
            for uncertain_as in ["false", "true"]:
                metrics = evaluate_predictions(predictions, uncertain_as)
                if metrics:
                    suffix = f" (uncertain→{uncertain_as})"
                    results[f"Original Pipeline{suffix}"] = {
                        **metrics,
                        "type": "agent-pipeline",
                        "source": str(trust_preds)
                    }
    
    # Research pipeline
    research_preds = base_dir / "trust_research" / "liar_predictions.jsonl"
    if research_preds.exists():
        predictions = load_jsonl(research_preds)
        if predictions:
            for uncertain_as in ["false", "true"]:
                metrics = evaluate_predictions(predictions, uncertain_as)
                if metrics:
                    suffix = f" (uncertain→{uncertain_as})"
                    results[f"Research Pipeline{suffix}"] = {
                        **metrics,
                        "type": "research-pipeline",
                        "source": str(research_preds)
                    }
    
    return results


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON: All Methods")
    print("=" * 80)
    
    # Sort by accuracy
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get('accuracy', 0),
        reverse=True
    )
    
    # Header
    print(f"\n{'Method':<45} {'Accuracy':>10} {'F1 (macro)':>12} {'Uncertain %':>12}")
    print("-" * 80)
    
    # Results
    for name, metrics in sorted_results:
        acc = metrics.get('accuracy', 0)
        f1 = metrics.get('f1_macro', 0)
        unc = metrics.get('uncertain_rate', 0)
        
        acc_str = f"{acc:.2%}" if acc else "N/A"
        f1_str = f"{f1:.4f}" if f1 else "N/A"
        unc_str = f"{unc:.1%}" if unc else "-"
        
        print(f"{name:<45} {acc_str:>10} {f1_str:>12} {unc_str:>12}")
    
    print("-" * 80)


def print_detailed_analysis(all_results):
    """Print detailed analysis focusing on research insights."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY METHOD TYPE")
    print("=" * 80)
    
    # Group by type
    fine_tuned = {k: v for k, v in all_results.items() if v.get('type') == 'fine-tuned'}
    llm_baseline = {k: v for k, v in all_results.items() if v.get('type') == 'llm-baseline'}
    agent_pipelines = {k: v for k, v in all_results.items() if 'pipeline' in v.get('type', '')}
    
    # Fine-tuned models
    if fine_tuned:
        print("\n--- Fine-tuned Models (Baselines) ---")
        for name, metrics in fine_tuned.items():
            print(f"  {name}")
            print(f"    Accuracy: {metrics['accuracy']:.2%}")
            print(f"    F1: {metrics['f1_macro']:.4f}")
    
    # LLM baseline
    if llm_baseline:
        print("\n--- LLM Baseline ---")
        for name, metrics in llm_baseline.items():
            print(f"  {name}")
            print(f"    Accuracy: {metrics['accuracy']:.2%}")
            print(f"    F1: {metrics['f1_macro']:.4f}")
    
    # Agent pipelines
    if agent_pipelines:
        print("\n--- Agent Pipelines (Experimental) ---")
        for name, metrics in agent_pipelines.items():
            unc_rate = metrics.get('uncertain_rate', 0)
            print(f"\n  {name}")
            print(f"    Accuracy: {metrics['accuracy']:.2%}")
            print(f"    F1: {metrics['f1_macro']:.4f}")
            print(f"    Uncertain rate: {unc_rate:.1%}")
            if 'pred_distribution' in metrics:
                print(f"    Prediction distribution: {metrics['pred_distribution']}")
    
    # Research observations
    print("\n" + "=" * 80)
    print("RESEARCH OBSERVATIONS")
    print("=" * 80)
    
    if agent_pipelines:
        # Calculate average uncertain rate
        unc_rates = [m.get('uncertain_rate', 0) for m in agent_pipelines.values()]
        avg_unc = sum(unc_rates) / len(unc_rates) if unc_rates else 0
        
        print(f"\n1. Agent Pipeline Uncertainty:")
        print(f"   Average uncertain rate: {avg_unc:.1%}")
        print(f"   This indicates the model's conservative behavior with")
        print(f"   multi-step reasoning tasks using GPT-4.1-mini.")
        
        print(f"\n2. Three-Class vs Binary Mismatch:")
        print(f"   - Pipeline outputs: true / false / uncertain")
        print(f"   - Gold labels: true / false only")
        print(f"   - 'Uncertain' predictions are penalized in accuracy calculation")
        
        print(f"\n3. Model Constraints:")
        print(f"   - GPT-4.1-mini has token limits requiring prompt truncation")
        print(f"   - Smaller models default to 'uncertain' when not confident")
        print(f"   - More capable models (GPT-4o) would likely reduce uncertainty")
        
        print(f"\n4. Architecture Value:")
        print(f"   - The decompose → verify → aggregate approach is sound")
        print(f"   - High uncertainty shows model recognizes its limitations")
        print(f"   - Could be valuable for human-in-the-loop systems")


def main():
    parser = argparse.ArgumentParser(description="Compare all fact-checking results")
    parser.add_argument("--outputs-dir", default="outputs", help="Outputs directory")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output", default="results/comparison_all.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    results_dir = Path(args.results_dir)
    
    print("=" * 80)
    print("TRUST Agents - Results Comparison")
    print("=" * 80)
    print(f"\nOutputs directory: {outputs_dir}")
    print(f"Results directory: {results_dir}")
    
    # Collect all results
    all_results = {}
    
    # Get baseline results
    print("\n--- Loading Baseline Results ---")
    baseline_results = get_baseline_results(outputs_dir, results_dir)
    for name, metrics in baseline_results.items():
        print(f"  ✓ {name}: {metrics['accuracy']:.2%}")
        all_results[name] = metrics
    
    if not baseline_results:
        print("  No baseline results found. Run baselines first:")
        print("    make hf.bert && make eval.bert")
        print("    make hf.roberta && make eval.roberta")
        print("    make llm.run && make llm.eval")
    
    # Get pipeline results
    print("\n--- Loading Pipeline Results ---")
    pipeline_results = get_pipeline_results(outputs_dir)
    for name, metrics in pipeline_results.items():
        print(f"  ✓ {name}: {metrics['accuracy']:.2%} (uncertain: {metrics.get('uncertain_rate', 0):.1%})")
        all_results[name] = metrics
    
    if not pipeline_results:
        print("  No pipeline results found. Run pipelines first:")
        print("    python scripts/run_trust_agents.py --dataset liar --split val --limit 200 --skip-evidence")
        print("    python scripts/run_trust_research.py --dataset liar --split val --limit 200 --skip-evidence")
    
    # Check if we have any results
    if not all_results:
        print("\n✗ No results found. Please run the baselines and pipelines first.")
        return
    
    # Print comparison table
    print_comparison_table(all_results)
    
    # Print detailed analysis
    print_detailed_analysis(all_results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "summary": {
            "total_methods": len(all_results),
            "best_method": max(all_results.items(), key=lambda x: x[1].get('accuracy', 0))[0] if all_results else None,
            "best_accuracy": max(r.get('accuracy', 0) for r in all_results.values()) if all_results else 0
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
