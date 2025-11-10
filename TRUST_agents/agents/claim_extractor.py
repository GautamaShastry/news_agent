# -*- coding: utf-8 -*-

"""
Claim Extractor Agent - ReAct Agent with NLP tools.

Uses Named Entity Recognition, Dependency Parsing, and LLM to extract factual claims.
- No logging reconfiguration here. Only use module-scoped logger.
- Prints full, untruncated message contents for debugging.
"""

import os
import re
import json
import logging
from typing import List, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Tools (async @tool functions)
from TRUST_agents.agents.claim_extractor_tools import (
    ner_claim_extraction_tool,
    dependency_claim_extraction_tool,
    llm_claim_extraction_tool,
)

load_dotenv()
logger = logging.getLogger("ClaimExtractor.Agent")


async def run_claim_extractor_agent(text: str) -> List[str]:
    """
    Extract claims from text using NER, dependency parsing, and LLM tools.
    Returns a list of claim strings.
    """
    # Model
    model_name = os.getenv("MODEL", "gpt-4.1-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        openai_api_key=api_key,
    )

    # Initialize state (informational; not mutated programmatically)
    state = {"text": text, "ner_done": False, "dependency_done": False, "llm_done": False}
    logger.info("[AGENT] Initial state initialized")
    logger.info("[DEBUG] State : %s", state)

    # System/user prompt for the ReAct loop
    agent_prompt = f"""
You are a Claim Extractor agent. Identify factual claims or key assertions within text using the tools provided.

Text: {text}

State: {json.dumps(state, indent=2)}

You have access to three tools:
- ner_claim_extraction_tool: Extract claims using Named Entity Recognition
- dependency_claim_extraction_tool: Extract claims using dependency parsing
- llm_claim_extraction_tool: Extract claims using LLM zero-shot reasoning

Your GOAL:
Extract all factual claims from the text using all available tools.

**Before taking ANY action, you must always carefully examine the current state dictionary.**
- If the required output for a step is already present in the state, you must explicitly explain that you are skipping that step and move to the next required action.
- Only perform a tool call if the corresponding state entry is missing or False.
- For each action, reason step by step and provide a detailed explanation of why you are calling or skipping each tool.
- Always re-check state before each action.
- After calling a tool, update the corresponding state flag to True.
- After updating the state, show the updated state in your response.

When all tools have been used, combine all results, remove duplicates, and return JSON: {{"claims": ["claim1", "claim2", ...]}}.
""".strip()

    tools = [ner_claim_extraction_tool, dependency_claim_extraction_tool, llm_claim_extraction_tool]
    logger.info("[AGENT] Loaded %d tools.", len(tools))

    # Create ReAct agent
    agent = create_react_agent(model, tools)
    logger.info("[AGENT] LangGraph agent initialized. Invoking agent...")

    # Run agent
    result: dict[str, Any] = await agent.ainvoke(
        {
            "messages": [{"role": "user", "content": agent_prompt}],
            **state,
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

    # Extract claims from final AI message
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
            if "{" in content and "claims" in content:
                json_match = re.search(r'\{[^{}]*"claims"[^{}]*\[[^\]]*\][^{}]*\}', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    claims = parsed.get("claims", [])
                    if isinstance(claims, list):
                        cleaned = [str(c).strip() for c in claims if c]
                        logger.info("[AGENT] Successfully extracted %d claims", len(cleaned))
                        return cleaned
        except Exception as e:
            logger.warning("[AGENT] Could not parse claims JSON: %s", e)

    logger.warning("[AGENT] No claims found")
    return []


def run_claim_extractor_agent_sync(text: str) -> List[str]:
    """Synchronous wrapper for run_claim_extractor_agent."""
    import asyncio
    return asyncio.run(run_claim_extractor_agent(text))