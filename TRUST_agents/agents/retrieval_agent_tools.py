# -*- coding: utf-8 -*-

"""
Retrieval Agent Tools - Tools for evidence retrieval using hybrid search.

Tools used by the Evidence Retriever ReAct Agent:
- Search Tool: Hybrid BM25 + Dense retrieval
- Index Tool: Build or load document index
- Get Passage Tool: Retrieve specific passage by ID
"""

import os
import json
import logging
from typing import Optional

from langchain_core.tools import tool
from TRUST_agents.agents.retrieval_agent_core import RetrievalAgent

logger = logging.getLogger("TRUST_agents.agents.retrieval_agent_tools")
logger.propagate = True

# Global retrieval agent instance
_global_retrieval_agent: Optional[RetrievalAgent] = None


def get_retrieval_agent(index_dir: str = "retrieval_index") -> RetrievalAgent:
    """Get or create the global retrieval agent instance."""
    global _global_retrieval_agent
    if _global_retrieval_agent is None:
        _global_retrieval_agent = RetrievalAgent(
            index_dir=index_dir,
            dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
            faiss_index_type="Flat",
            device="cpu"
        )
        # Try to load existing index
        try:
            _global_retrieval_agent.load_index()
            logger.info("Loaded existing retrieval index")
        except FileNotFoundError:
            logger.warning("No existing index found. You need to index documents first.")
    return _global_retrieval_agent


@tool()
async def search_evidence_tool(query: str, top_k: int = 5) -> str:
    """
    Search for evidence passages relevant to a query using hybrid retrieval.
    
    Uses BM25 (keyword-based) and dense semantic search (embeddings) to find
    the most relevant passages from the indexed document corpus.
    
    Args:
        query: The search query or claim to find evidence for
        top_k: Number of top results to return (default: 5)
        
    Returns:
        JSON string with search results including passages, scores, and metadata
    """
    logger.info(f"[DEBUG] search_evidence_tool called with query: {query[:100]}...")
    
    try:
        agent = get_retrieval_agent()
        
        if not agent.passages:
            error_msg = "No documents indexed. Please index documents first using index_documents_tool."
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "results": []})
        
        # Perform hybrid retrieval
        results = agent.retrieve(
            query=query,
            top_k=top_k,
            bm25_weight=0.6,  # 60% BM25, 40% dense
            candidate_k=50,
            rerank_with_cross=False,
            mmr=True,  # Use MMR for diversity
            mmr_diversity=0.7
        )
        
        # Format results
        formatted_results = []
        for i, r in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "passage_id": r["passage_id"],
                "text": r["text"],
                "source": r["metadata"].get("source_path", "unknown"),
                "title": r["title"],
                "hybrid_score": round(r["hybrid_score"], 4),
                "bm25_score": round(r["bm25_score"], 4) if r["bm25_score"] else 0,
                "dense_score": round(r["dense_score"], 4) if r["dense_score"] else 0
            })
        
        logger.info(f"search_evidence_tool completed: {len(formatted_results)} results found")
        
        result = {
            "query": query,
            "num_results": len(formatted_results),
            "results": formatted_results
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error during evidence search: {e}")
        return json.dumps({"error": str(e), "results": []})


@tool()
async def index_documents_tool(corpus_path: str, chunk_size: int = 160) -> str:
    """
    Index a corpus of documents for retrieval.
    
    Processes documents (PDF, HTML, TXT, MD) from a directory or file,
    chunks them, and builds BM25 and dense (FAISS) indices.
    
    Args:
        corpus_path: Path to directory containing documents or single document file
        chunk_size: Maximum words per chunk (default: 160)
        
    Returns:
        JSON string with indexing status and statistics
    """
    logger.info(f"[DEBUG] index_documents_tool called with path: {corpus_path}")
    
    try:
        agent = get_retrieval_agent()
        
        # Index the corpus
        agent.index_corpus(
            corpus_path=corpus_path,
            chunk_max_words=chunk_size,
            chunk_overlap=20,
            batch_size=64,
            rebuild_dense=True
        )
        
        # Save the index
        agent.save_index()
        
        stats = {
            "success": True,
            "num_passages": len(agent.passages),
            "corpus_path": corpus_path,
            "chunk_size": chunk_size,
            "index_dir": str(agent.index_dir)
        }
        
        logger.info(f"index_documents_tool completed: {stats['num_passages']} passages indexed")
        
        return json.dumps(stats, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error during document indexing: {e}")
        return json.dumps({"success": False, "error": str(e)})


@tool()
async def get_passage_tool(passage_id: str) -> str:
    """
    Retrieve a specific passage by its ID.
    
    Args:
        passage_id: The unique identifier of the passage
        
    Returns:
        JSON string with passage details or error message
    """
    logger.info(f"[DEBUG] get_passage_tool called with passage_id: {passage_id}")
    
    try:
        agent = get_retrieval_agent()
        
        # Find passage by ID
        passage = next((p for p in agent.passages if p.id == passage_id), None)
        
        if passage is None:
            return json.dumps({"error": f"Passage not found: {passage_id}"})
        
        result = {
            "passage_id": passage.id,
            "doc_id": passage.doc_id,
            "title": passage.title,
            "text": passage.text,
            "metadata": passage.metadata
        }
        
        logger.info(f"get_passage_tool completed: passage found")
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error retrieving passage: {e}")
        return json.dumps({"error": str(e)})


@tool()
async def list_indexed_documents_tool() -> str:
    """
    List all indexed documents and their statistics.
    
    Returns:
        JSON string with list of indexed documents and passage counts
    """
    logger.info(f"[DEBUG] list_indexed_documents_tool called")
    
    try:
        agent = get_retrieval_agent()
        
        if not agent.passages:
            return json.dumps({
                "num_documents": 0,
                "num_passages": 0,
                "documents": []
            })
        
        # Group passages by document
        doc_stats = {}
        for passage in agent.passages:
            doc_id = passage.doc_id
            if doc_id not in doc_stats:
                doc_stats[doc_id] = {
                    "doc_id": doc_id,
                    "title": passage.title,
                    "num_passages": 0,
                    "source_path": passage.metadata.get("source_path", "unknown")
                }
            doc_stats[doc_id]["num_passages"] += 1
        
        result = {
            "num_documents": len(doc_stats),
            "num_passages": len(agent.passages),
            "documents": list(doc_stats.values())
        }
        
        logger.info(f"list_indexed_documents_tool completed: {result['num_documents']} documents")
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return json.dumps({"error": str(e)})