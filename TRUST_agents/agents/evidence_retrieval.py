# -*- coding: utf-8 -*-

"""
Evidence Retrieval Agent - ReAct Agent with hybrid search tools.

Uses BM25 + Dense retrieval (FAISS) to find evidence passages for fact-checking.
- No logging reconfiguration here. Only use module-scoped logger.
- Prints full, untruncated message contents for debugging.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Tools (async @tool functions)
from TRUST_agents.agents.retrieval_agent_tools import (
    search_evidence_tool,
    index_documents_tool,
    get_passage_tool,
    list_indexed_documents_tool,
)

load_dotenv()
logger = logging.getLogger("EvidenceRetriever.Agent")


async def run_evidence_retrieval_agent(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve evidence passages relevant to a query using hybrid search.
    
    Args:
        query: The search query or claim to find evidence for
        top_k: Number of top results to return (default: 5)
    
    Returns:
        List of evidence passages with metadata and scores
    """
    # Model
    model_name = os.getenv("MODEL", "gpt-4.1-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        openai_api_key=api_key,
    )

    logger.info("[AGENT] Evidence Retrieval Agent initialized")

    # Agent prompt
    agent_prompt = f"""
You are an Evidence Retrieval agent. Your task is to find relevant evidence passages for the given query or claim.

Query: {query}

You have access to the following tools:
- search_evidence_tool: Search for evidence passages using hybrid retrieval (BM25 + semantic search)
- get_passage_tool: Get details of a specific passage by ID
- list_indexed_documents_tool: List all indexed documents

Your GOAL:
Use search_evidence_tool to find the top {top_k} most relevant evidence passages for the query.

Process:
1. Use search_evidence_tool with the query and top_k={top_k}
2. If needed, examine specific passages using get_passage_tool
3. Return the search results

After retrieving evidence, return JSON: {{"evidence": [list of passages with text, source, and scores]}}.
""".strip()

    tools = [search_evidence_tool, get_passage_tool, list_indexed_documents_tool]
    logger.info("[AGENT] Loaded %d tools.", len(tools))

    # Create ReAct agent
    agent = create_react_agent(model, tools)
    logger.info("[AGENT] LangGraph agent initialized. Invoking agent...")

    # Run agent
    result: dict[str, Any] = await agent.ainvoke(
        {
            "messages": [{"role": "user", "content": agent_prompt}],
        }
    )

    # Debug: print full message contents (no truncation)
    msgs = result.get("messages", [])
    logger.debug("[AGENT] Received %d messages", len(msgs))

    for i, msg in enumerate(msgs):
        content = getattr(msg, "content", "")
        # normalize parts → string
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        elif not isinstance(content, str):
            content = str(content)
        logger.debug("Message %d | %s | len=%d\n%s\n", i, type(msg).__name__, len(content), content)

    logger.info("[AGENT] Agent execution completed. Processing results...")

    # Extract evidence from final AI message
    if msgs:
        content = getattr(msgs[-1], "content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        elif not isinstance(content, str):
            content = str(content)

        try:
            if "{" in content and "evidence" in content:
                json_match = re.search(r'\{[^{}]*"evidence"[^{}]*\[[^\]]*\][^{}]*\}', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    evidence = parsed.get("evidence", [])
                    if isinstance(evidence, list):
                        logger.info("[AGENT] Successfully extracted %d evidence passages", len(evidence))
                        return evidence
        except Exception as e:
            logger.warning("[AGENT] Could not parse evidence JSON: %s", e)

    logger.warning("[AGENT] No evidence found")
    return []


def run_evidence_retrieval_agent_sync(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Synchronous wrapper for run_evidence_retrieval_agent."""
    import asyncio
    return asyncio.run(run_evidence_retrieval_agent(query, top_k))


async def index_documents_async(corpus_path: str, chunk_size: int = 160) -> Dict[str, Any]:
    """
    Index documents for retrieval.
    
    Args:
        corpus_path: Path to directory or file to index
        chunk_size: Maximum words per chunk
    
    Returns:
        Dictionary with indexing statistics
    """
    # Model
    model_name = os.getenv("MODEL", "gpt-4.1-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        openai_api_key=api_key,
    )

    agent_prompt = f"""
You are an indexing agent. Index the documents at the given path.

Corpus path: {corpus_path}
Chunk size: {chunk_size}

Use index_documents_tool to index the documents and report the results.
""".strip()

    tools = [index_documents_tool]
    agent = create_react_agent(model, tools)
    
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": agent_prompt}]}
    )
    
    msgs = result.get("messages", [])
    if msgs:
        content = getattr(msgs[-1], "content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        
        try:
            if "{" in content:
                json_match = re.search(r'\{[^{}]*"success"[^{}]*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
        except Exception:
            pass
    
    return {"success": False, "error": "Failed to parse indexing result"}


def index_documents_sync(corpus_path: str, chunk_size: int = 160) -> Dict[str, Any]:
    """Synchronous wrapper for index_documents_async."""
    import asyncio
    return asyncio.run(index_documents_async(corpus_path, chunk_size))