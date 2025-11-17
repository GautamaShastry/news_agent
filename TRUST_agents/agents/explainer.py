
"""
Explainer Agent - ReAct Agent for generating explanations.

Creates natural language explanations of verification results with evidence citations.
- Summarizes verification process
- Generates detailed explanations
- Formats evidence citations
- Creates comprehensive reports
"""

import os
import re
import json
import logging
from typing import Dict, List, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from TRUST_agents.agents.explainer_tools import (
    summarize_verification_tool,
    generate_explanation_tool,
    cite_evidence_tool,
    create_report_tool,
)

load_dotenv()
logger = logging.getLogger("Explainer.Agent")


async def run_explainer_agent(
    claim: str,
    verdict_data: Dict[str, Any],
    evidence: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate explanation for verification result.
    
    Args:
        claim: The original claim
        verdict_data: Dictionary with verdict, confidence, and reasoning
        evidence: List of evidence passages used in verification
    
    Returns:
        Dictionary with complete fact-check report
    """
    # Model
    model_name = os.getenv("MODEL", "gpt-4.1-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        openai_api_key=api_key,
    )

    logger.info("[AGENT] Explainer Agent initialized")

    # Convert to JSON strings for tools
    verdict_json = json.dumps(verdict_data, ensure_ascii=False)
    evidence_json = json.dumps(evidence[:5], ensure_ascii=False)
    
    verdict = verdict_data.get("verdict", "uncertain")
    confidence = verdict_data.get("confidence", 0.5)

    # Agent prompt
    agent_prompt = f"""
You are an Explainer agent. Your task is to create a clear, comprehensive explanation of the fact-check result.

Claim: {claim}
Verdict: {verdict} (confidence: {confidence:.1%})

Verdict Data: {verdict_json[:500]}...
Evidence Data: {evidence_json[:500]}...

You have access to these tools:
- summarize_verification_tool: Create a concise summary
- generate_explanation_tool: Generate detailed explanation with citations
- cite_evidence_tool: Format evidence citations
- create_report_tool: Compile comprehensive report

Your PROCESS:
1. Use summarize_verification_tool to create a summary
2. Use cite_evidence_tool to format citations
3. Use generate_explanation_tool to create detailed explanation
4. Use create_report_tool to compile everything

After creating the report, return JSON with the complete fact-check report.
""".strip()

    tools = [
        summarize_verification_tool,
        generate_explanation_tool,
        cite_evidence_tool,
        create_report_tool,
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

    # Extract report from final AI message
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
            # Try to extract JSON report
            if "{" in content:
                # Look for the most complete JSON object
                json_match = re.search(r'\{[^{}]*"claim"[^{}]*"verdict"[^{}]*\}', content, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    logger.info("[AGENT] Successfully extracted report")
                    return parsed
        except Exception as e:
            logger.warning("[AGENT] Could not parse report JSON: %s", e)

    logger.warning("[AGENT] No report found, creating basic report")
    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "label": verdict,
        "summary": f"The claim is {verdict} with {confidence:.1%} confidence.",
        "explanation": verdict_data.get("reasoning", "Based on available evidence."),
        "evidence_count": len(evidence)
    }


def run_explainer_agent_sync(
    claim: str,
    verdict_data: Dict[str, Any],
    evidence: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Synchronous wrapper for run_explainer_agent."""
    import asyncio
    return asyncio.run(run_explainer_agent(claim, verdict_data, evidence))