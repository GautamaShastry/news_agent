#!/usr/bin/env python3
"""
Prepare Evidence Corpus for TRUST Agents

Creates a simple evidence corpus from the training data for the retrieval agent.
For a production system, you'd want to use Wikipedia, fact-checking databases, etc.
"""

import json
import argparse
from pathlib import Path


def create_evidence_docs(dataset_path: Path, output_dir: Path, max_docs: int = 1000):
    """
    Create evidence documents from dataset.
    
    For demo purposes, we use true statements from the dataset as evidence.
    In production, use Wikipedia, Snopes, PolitiFact, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating evidence corpus from {dataset_path}...")
    
    # Read dataset
    docs = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(docs) >= max_docs:
                break
            data = json.loads(line)
            # Use true statements as evidence
            if data.get('label') == 'true':
                docs.append(data['text'])
    
    print(f"Collected {len(docs)} evidence documents")
    
    # Save as individual text files (better for retrieval)
    for i, doc in enumerate(docs):
        doc_file = output_dir / f"evidence_{i:04d}.txt"
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(doc)
    
    print(f"✓ Saved {len(docs)} documents to {output_dir}")
    print(f"\nTo use with TRUST agents:")
    print(f"  - Evidence corpus: {output_dir.absolute()}")
    print(f"  - Index with: python scripts/index_evidence.py --corpus {output_dir}")
    
    return len(docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare evidence corpus for TRUST agents")
    parser.add_argument("--dataset", required=True, help="Path to training dataset (e.g., data/processed/liar_train.jsonl)")
    parser.add_argument("--output", default="data/evidence_corpus", help="Output directory for evidence documents")
    parser.add_argument("--max-docs", type=int, default=1000, help="Maximum documents to include")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    
    if not dataset_path.exists():
        print(f"✗ Error: Dataset not found: {dataset_path}")
        print(f"\nRun data preparation first:")
        print(f"  make data.liar")
        exit(1)
    
    create_evidence_docs(dataset_path, output_dir, args.max_docs)