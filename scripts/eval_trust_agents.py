#!/usr/bin/env python3
"""
Run TRUST Agents on Dataset

Runs the complete TRUST pipeline (Claim Extraction → Evidence Retrieval → Verification → Explanation)
on a dataset and saves predictions for evaluation.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Configure logging
os.environ["PYTHONUNBUFFERED"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)

# Suppress noisy logs
for noisy in ["httpcore", "httpx", "openai", "langchain", "langgraph", "asyncio", "sentence_transformers"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Import TRUST orchestrator
from TRUST_agents.orchestrator import TRUSTOrchestrator


def process_dataset(
    dataset_path: Path,
    output_path: Path,
    limit: int = None,
    top_k_evidence: int = 5,
    skip_evidence: bool = False
):
    """
    Process dataset through TRUST pipeline.
    
    Args:
        dataset_path: Path to input dataset (JSONL)
        output_path: Path to save predictions
        limit: Maximum number of examples to process
        top_k_evidence: Number of evidence passages per claim
        skip_evidence: Skip evidence retrieval (for testing without index)
    """
    logger.info("="*70)
    logger.info("TRUST AGENTS - Dataset Processing")
    logger.info("="*70)
    logger.info(f"Input: {dataset_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Limit: {limit if limit else 'None (process all)'}")
    logger.info(f"Top-k evidence: {top_k_evidence}")
    logger.info(f"Skip evidence: {skip_evidence}")
    logger.info("="*70)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("✗ Error: OPENAI_API_KEY not set in .env file")
    
    # Initialize orchestrator
    orchestrator = TRUSTOrchestrator(
        index_dir="retrieval_index",
        top_k_evidence=top_k_evidence
    )
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    examples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and len(examples) >= limit:
                break
            examples.append(json.loads(line))
    
    logger.info(f"Loaded {len(examples)} examples")
    
    # Process each example
    results = []
    successful = 0
    failed = 0
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for i, example in enumerate(tqdm(examples, desc="Processing")):
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"Example {i+1}/{len(examples)}: {example['id']}")
                logger.info(f"{'='*70}")
                
                # Run TRUST pipeline
                result = orchestrator.process_text(
                    text=example['text'],
                    skip_evidence=skip_evidence
                )
                
                # Extract prediction from first result (if any claims found)
                if result.results:
                    # Use first claim's verdict as overall prediction
                    first_result = result.results[0]
                    prediction = {
                        "id": example['id'],
                        "text": example['text'],
                        "gold": example.get('label', 'unknown'),
                        "pred": first_result.get('verdict', 'uncertain'),
                        "confidence": first_result.get('confidence', 0.0),
                        "claims": result.claims,
                        "num_claims": len(result.claims),
                        "all_verdicts": [r.get('verdict') for r in result.results],
                        "summary": result.summary
                    }
                else:
                    # No claims found
                    prediction = {
                        "id": example['id'],
                        "text": example['text'],
                        "gold": example.get('label', 'unknown'),
                        "pred": "uncertain",
                        "confidence": 0.0,
                        "claims": [],
                        "num_claims": 0,
                        "error": "No claims extracted"
                    }
                
                # Save prediction
                out_file.write(json.dumps(prediction, ensure_ascii=False) + '\n')
                out_file.flush()
                
                results.append(prediction)
                successful += 1
                
                logger.info(f"✓ Processed: pred={prediction['pred']}, gold={prediction['gold']}, claims={prediction['num_claims']}")
                
            except Exception as e:
                logger.error(f"✗ Error processing example {i+1}: {e}", exc_info=True)
                
                # Save error record
                error_record = {
                    "id": example['id'],
                    "text": example['text'],
                    "gold": example.get('label', 'unknown'),
                    "pred": "error",
                    "confidence": 0.0,
                    "error": str(e)
                }
                out_file.write(json.dumps(error_record, ensure_ascii=False) + '\n')
                out_file.flush()
                
                failed += 1
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(f"Total examples: {len(examples)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/len(examples)*100:.1f}%")
    logger.info(f"\n✓ Predictions saved to: {output_path.absolute()}")
    logger.info("="*70)
    
    return successful, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TRUST agents on dataset")
    parser.add_argument("--dataset", choices=["liar", "fnn"], required=True)
    parser.add_argument("--proc-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--top-k", type=int, default=5, help="Evidence passages per claim")
    parser.add_argument("--skip-evidence", action="store_true", help="Skip evidence retrieval (for testing)")
    parser.add_argument("--out", default=None, help="Output file path")
    
    args = parser.parse_args()
    
    # Input/output paths
    dataset_file = f"{args.dataset}_{args.split}.jsonl"
    input_path = Path(args.proc_dir) / dataset_file
    output_path = Path(args.out or f"outputs/trust_agents/{args.dataset}_predictions.jsonl")
    
    if not input_path.exists():
        print(f"✗ Error: Dataset not found: {input_path}")
        print(f"\nRun data preparation first:")
        print(f"  make data.{args.dataset}")
        exit(1)
    
    # Check for evidence index if not skipping
    if not args.skip_evidence:
        index_dir = Path("retrieval_index")
        index_meta = index_dir / "index_meta.json"
        if not index_meta.exists():
            print(f"✗ Error: Evidence index not found at {index_dir}")
            print(f"\nCreate and index evidence corpus first:")
            print(f"  1. python scripts/prepare_evidence_corpus.py --dataset {input_path}")
            print(f"  2. python scripts/index_evidence.py --corpus data/evidence_corpus")
            print(f"\nOr run with --skip-evidence to test without retrieval")
            exit(1)
    
    # Process dataset
    process_dataset(
        dataset_path=input_path,
        output_path=output_path,
        limit=args.limit,
        top_k_evidence=args.top_k,
        skip_evidence=args.skip_evidence
    )