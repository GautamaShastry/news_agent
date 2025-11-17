# -*- coding: utf-8 -*-

"""
Verifier Agent - ReAct Agent for claim verification.

Uses evidence passages to verify claims and generate verdicts.
- Compares claims against evidence
- Aggregates multiple evidence sources
- Generates confidence-calibrated verdicts
"""

import os
import re
import json
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from TRUST_agents.agents.verifier_tools import (
    compare_claim_evidence_tool,
    aggregate_evidence_tool,
    generate_verdict_tool,
    confidence_calibration_tool,
)

load_dotenv()
logger = logging.getLogger("Verifier.Agent")


async def run_verifier_agent(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verify a claim against evidence passages.
    
    Args:
        claim: The claim to verify
        evidence: List of evidence passages with text and scores
    
    Returns:
        Dictionary with verdict, confidence, and reasoning
    """
    # Model
    model_name = os.getenv("MODEL", "gpt-4.1-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        openai_api_key=api_key,
    )

    logger.info("[AGENT] Verifier Agent initialized")

    # Format evidence for prompt
    evidence_summary = "\n\n".join([
        f"Evidence {i+1} (score: {ev.get('hybrid_score', ev.get('score', 0.5)):.3f}):\n{ev.get('text', str(ev))[:300]}..."
        for i, ev in enumerate(evidence[:5])
    ])
    
    evidence_json = json.dumps(evidence[:5], ensure_ascii=False)

    # Agent prompt
    agent_prompt = f"""
You are a Verifier agent. Your task is to verify the claim against the provided evidence.

Claim: {claim}

Evidence Passages:
{evidence_summary}

You have access to these tools:
- compare_claim_evidence_tool: Compare claim against individual evidence passages
- aggregate_evidence_tool: Aggregate multiple evidence assessments
- generate_verdict_tool: Generate final verdict based on aggregated assessment
- confidence_calibration_tool: Calibrate confidence based on evidence quality

Your PROCESS:
1. Use aggregate_evidence_tool with the evidence JSON: {evidence_json[:500]}...
2. Use generate_verdict_tool with the aggregated assessment
3. Return the final verdict

After verification, return JSON: {{"verdict": "true|false|uncertain", "confidence": 0.0-1.0, "reasoning": "explanation"}}.
""".strip()

    tools = [
        compare_claim_evidence_tool,
        aggregate_evidence_tool,
        generate_verdict_tool,
        confidence_calibration_tool,
    ]
    logger.info("[AGENT] Loaded %d tools.", len(tools))

    # Create ReAct agent
    agent = create_react_agent(model, tools)
    logger.info("[AGENT] LangGraph agent initialized. Invoking agent...")

    # Run agent
    result: dict[str, Any] = await agent.ainvoke(
        {"messages": [{"role": "user", "content": agent_prompt}]}
    )

    # Debug: print full message contents
    msgs = result.get("messages", [])
    logger.debug("[AGENT] Received %d messages", len(msgs))

    for i, msg in enumerate(msgs):
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        elif not isinstance(content, str):
            content = str(content)
        logger.debug("Message %d | %s | len=%d\n%s\n", i, type(msg).__name__, len(content), content)

    logger.info("[AGENT] Agent execution completed. Processing results...")

    # Extract verdict from final AI message
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
            # Try to extract JSON with verdict
            if "{" in content:
                json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    logger.info("[AGENT] Successfully extracted verdict: %s", parsed.get("verdict"))
                    return parsed
        except Exception as e:
            logger.warning("[AGENT] Could not parse verdict JSON: %s", e)

    logger.warning("[AGENT] No verdict found, returning uncertain")
    return {
        "claim": claim,
        "verdict": "uncertain",
        "confidence": 0.3,
        "label": "uncertain",
        "reasoning": "Unable to determine verdict from available evidence"
    }


def run_verifier_agent_sync(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Synchronous wrapper for run_verifier_agent."""
    import asyncio
    return asyncio.run(run_verifier_agent(claim, evidence))