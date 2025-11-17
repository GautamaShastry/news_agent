#!/usr/bin/env python3
"""
Index Evidence Corpus for TRUST Agents

Builds BM25 and FAISS indices for the evidence retrieval agent.
"""

import argparse
import logging
from pathlib import Path
import sys 

ROOT = Path(__file__).resolve().parent.parent  # NEWS_AGENT directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from TRUST_agents.agents.retrieval_agent_core import RetrievalAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def index_corpus(corpus_path: str, index_dir: str = "retrieval_index", chunk_size: int = 160):
    """
    Index a corpus for evidence retrieval.
    
    Args:
        corpus_path: Path to directory containing evidence documents
        index_dir: Output directory for indices
        chunk_size: Maximum words per chunk
    """
    logger.info(f"Indexing corpus: {corpus_path}")
    logger.info(f"Index directory: {index_dir}")
    logger.info(f"Chunk size: {chunk_size} words")
    
    # Create retrieval agent
    agent = RetrievalAgent(
        index_dir=index_dir,
        dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
        faiss_index_type="Flat",
        device="cpu"
    )
    
    # Index the corpus
    logger.info("Building indices (BM25 + FAISS)...")
    agent.index_corpus(
        corpus_path=corpus_path,
        chunk_max_words=chunk_size,
        chunk_overlap=20,
        batch_size=64,
        rebuild_dense=True
    )
    
    # Save indices
    logger.info("Saving indices...")
    agent.save_index()
    
    logger.info(f"✓ Successfully indexed {len(agent.passages)} passages")
    logger.info(f"✓ Index saved to: {Path(index_dir).absolute()}")
    
    # Test search
    logger.info("\n" + "="*60)
    logger.info("Testing retrieval with sample query...")
    logger.info("="*60)
    
    test_query = "climate change global warming"
    results = agent.retrieve(test_query, top_k=3)
    
    logger.info(f"\nQuery: {test_query}")
    logger.info(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        logger.info(f"Result {i}:")
        logger.info(f"  Text: {result['text'][:150]}...")
        logger.info(f"  Score: {result['hybrid_score']:.4f}")
        logger.info(f"  Source: {result['title']}")
        logger.info("")
    
    return len(agent.passages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index evidence corpus for TRUST agents")
    parser.add_argument("--corpus", required=True, help="Path to evidence corpus directory")
    parser.add_argument("--index-dir", default="retrieval_index", help="Output directory for indices")
    parser.add_argument("--chunk-size", type=int, default=160, help="Maximum words per chunk")
    
    args = parser.parse_args()
    
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"✗ Error: Corpus directory not found: {corpus_path}")
        print(f"\nCreate evidence corpus first:")
        print(f"  python scripts/prepare_evidence_corpus.py --dataset data/processed/liar_train.jsonl")
        exit(1)
    
    # Check for documents
    docs = list(corpus_path.glob("*.txt")) + list(corpus_path.glob("*.pdf"))
    if not docs:
        print(f"✗ Error: No .txt or .pdf files found in {corpus_path}")
        exit(1)
    
    print(f"Found {len(docs)} documents to index")
    
    # Index the corpus
    index_corpus(str(corpus_path), args.index_dir, args.chunk_size)