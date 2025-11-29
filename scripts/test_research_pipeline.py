#!/usr/bin/env python3
"""
Test Research TRUST Pipeline

Demonstrates the LoCal + Delphi + MEGA-RAG pipeline on example claims.
"""

import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TRUST_agents.orchestrator_research import run_research_pipeline_sync

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

# Suppress noisy logs
for noisy in ["httpcore", "httpx", "openai", "langchain"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


def test_simple_claim():
    """Test on a simple claim"""
    print("\n" + "="*70)
    print("TEST 1: Simple Claim")
    print("="*70)
    
    text = "Joe Biden won the 2020 presidential election."
    
    result = run_research_pipeline_sync(
        text=text,
        skip_evidence=True,  # Skip evidence for quick test
        use_delphi_jury=False  # Use single verifier for speed
    )
    
    print(f"\nText: {text}")
    print(f"Atomic Claims: {len(result['atomic_verdicts'])}")
    for v in result['atomic_verdicts']:
        print(f"  - {v['claim']}")
    print(f"Logic: {result['decomposed_claim']['logic_structure']}")
    print(f"Final: {result['logic_aggregation']['verdict']} ({result['logic_aggregation']['confidence']:.2f})")


def test_complex_claim():
    """Test on a complex claim with logical structure"""
    print("\n" + "="*70)
    print("TEST 2: Complex Claim (AND logic)")
    print("="*70)
    
    text = "Biden won the 2020 election and became president in January 2021."
    
    result = run_research_pipeline_sync(
        text=text,
        skip_evidence=True,
        use_delphi_jury=False
    )
    
    print(f"\nText: {text}")
    print(f"Atomic Claims: {len(result['atomic_verdicts'])}")
    for i, v in enumerate(result['atomic_verdicts'], 1):
        print(f"  C{i}: {v['claim']} → {v['verdict']}")
    print(f"Logic: {result['decomposed_claim']['logic_structure']}")
    print(f"Final: {result['logic_aggregation']['verdict']} ({result['logic_aggregation']['confidence']:.2f})")


def test_causal_claim():
    """Test on a causal claim"""
    print("\n" + "="*70)
    print("TEST 3: Causal Claim")
    print("="*70)
    
    text = "After the policy passed in 2020, unemployment decreased by 5%."
    
    result = run_research_pipeline_sync(
        text=text,
        skip_evidence=True,
        use_delphi_jury=False
    )
    
    print(f"\nText: {text}")
    print(f"Atomic Claims: {len(result['atomic_verdicts'])}")
    for v in result['atomic_verdicts']:
        print(f"  - {v['claim']}")
    
    print(f"\nCausal Edges: {len(result['decomposed_claim']['causal_edges'])}")
    for edge in result['decomposed_claim']['causal_edges']:
        print(f"  {edge['cause']} → {edge['effect']}")
    
    print(f"\nComplexity: {result['decomposed_claim']['complexity_score']:.2f}")
    print(f"Final: {result['logic_aggregation']['verdict']} ({result['logic_aggregation']['confidence']:.2f})")


def test_delphi_jury():
    """Test Delphi jury on a controversial claim"""
    print("\n" + "="*70)
    print("TEST 4: Delphi Jury (Multi-Agent)")
    print("="*70)
    
    text = "Climate change is caused by human activity."
    
    result = run_research_pipeline_sync(
        text=text,
        skip_evidence=True,
        use_delphi_jury=True  # Enable multi-agent jury
    )
    
    print(f"\nText: {text}")
    print(f"Atomic Claims: {len(result['atomic_verdicts'])}")
    
    # Show jury verdicts
    for i, v in enumerate(result['atomic_verdicts'], 1):
        print(f"\nAtomic Claim {i}: {v['claim']}")
        if 'jury_verdicts' in v:
            print("  Jury Verdicts:")
            for jv in v['jury_verdicts']:
                print(f"    - {jv['persona']}: {jv['verdict']} ({jv['confidence']:.2f})")
            print(f"  Trust Scores: {[f'{t:.2f}' for t in v['trust_scores']]}")
            print(f"  Weighted Votes: {v['weighted_votes']}")
        print(f"  Final: {v['verdict']} ({v['confidence']:.2f})")
    
    print(f"\nMetadata:")
    for k, v in result['metadata'].items():
        print(f"  {k}: {v}")


def test_all():
    """Run all tests"""
    test_simple_claim()
    test_complex_claim()
    test_causal_claim()
    test_delphi_jury()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Run with evidence: --skip-evidence=False")
    print("2. Test on LIAR dataset: python scripts/run_trust_research.py")
    print("3. Compare with baseline: python scripts/compare_research_baseline.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test research pipeline")
    parser.add_argument("--test", choices=["simple", "complex", "causal", "delphi", "all"], 
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "simple":
        test_simple_claim()
    elif args.test == "complex":
        test_complex_claim()
    elif args.test == "causal":
        test_causal_claim()
    elif args.test == "delphi":
        test_delphi_jury()
    else:
        test_all()
