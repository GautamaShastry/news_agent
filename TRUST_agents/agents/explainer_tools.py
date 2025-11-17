# -*- coding: utf-8 -*-

"""
Explainer Agent Tools - Tools for generating explanations of verification results.

Tools used by the Explainer ReAct Agent:
- Summarize Tool: Summarize the verification process
- Generate Explanation Tool: Create natural language explanation
- Cite Evidence Tool: Format evidence citations
- Create Report Tool: Generate comprehensive report

FIXED: All tools now handle both JSON strings and Python objects (dicts/lists)
"""

import os
import json
import logging
from typing import Dict, List, Union, Any

from langchain_core.tools import tool
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("TRUST_agents.agents.explainer_tools")
logger.propagate = True


# ============================================================
# HELPER FUNCTION: Safe JSON parsing
# ============================================================
def safe_json_parse(data: Union[str, dict, list, Any], default: Any = None) -> Any:
    """
    Safely parse JSON data that might be a string, dict, list, or other type.
    
    This is needed because LangGraph sometimes passes JSON strings and sometimes
    passes Python objects directly.
    
    Args:
        data: Input data of any type
        default: Default value if parsing fails
        
    Returns:
        Parsed data or default value
    """
    if data is None:
        return default
    
    # Already a dict or list - return as is
    if isinstance(data, (dict, list)):
        return data
    
    # Try to parse as JSON string
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON, return the string or default
            return default if default is not None else data
    
    # Other types - convert to string representation or return default
    return default if default is not None else str(data)


@tool()
async def summarize_verification_tool(claim: str, verdict: str, confidence: float, evidence_summary: str) -> str:
    """
    Summarize the verification process and key findings.
    
    Args:
        claim: The original claim
        verdict: The verification verdict (true/false/uncertain)
        confidence: Confidence score (0.0-1.0)
        evidence_summary: Summary of evidence used
        
    Returns:
        JSON string with verification summary
    """
    logger.info(f"[DEBUG] summarize_verification_tool called")
    
    try:
        # Handle various input types for confidence
        if not isinstance(confidence, (int, float)):
            try:
                confidence = float(confidence)
            except:
                confidence = 0.5
        
        model = os.getenv("MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""Summarize this fact-checking verification in 2-3 clear sentences.

Claim: {claim}
Verdict: {verdict}
Confidence: {confidence:.1%}
Evidence: {evidence_summary}

Create a concise summary that:
1. States the claim
2. States the verdict with confidence
3. Mentions key supporting/contradicting evidence

Return ONLY valid JSON:
{{
    "summary": "2-3 sentence summary",
    "key_finding": "one sentence key finding",
    "verdict_statement": "The claim is [verdict]"
}}"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a fact-checking expert. Write clear, concise summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        content = content.replace('```json\n', '').replace('```\n', '').replace('```', '').strip()
        
        parsed = json.loads(content)
        logger.info(f"summarize_verification_tool completed")
        
        return json.dumps(parsed)
        
    except Exception as e:
        logger.error(f"Error summarizing verification: {e}")
        return json.dumps({
            "summary": f"The claim '{claim}' was assessed as {verdict} with {confidence:.1%} confidence.",
            "key_finding": f"Verdict: {verdict}",
            "verdict_statement": f"The claim is {verdict}.",
            "error": str(e)
        })


@tool()
async def generate_explanation_tool(
    claim: str, 
    verdict_data: Union[str, dict], 
    evidence_list: Union[str, list]
) -> str:
    """
    Generate detailed natural language explanation of the verification.
    
    Args:
        claim: The original claim
        verdict_data: JSON string OR dict with verdict, confidence, and reasoning
        evidence_list: JSON string OR list with evidence passages
        
    Returns:
        JSON string with detailed explanation
    """
    logger.info(f"[DEBUG] generate_explanation_tool called")
    
    try:
        # Parse inputs with safe handling
        verdict_info = safe_json_parse(
            verdict_data, 
            {"verdict": "uncertain", "confidence": 0.5, "reasoning": "Unknown"}
        )
        if not isinstance(verdict_info, dict):
            verdict_info = {"verdict": "uncertain", "confidence": 0.5, "reasoning": str(verdict_info)}
        
        evidence = safe_json_parse(evidence_list, [])
        if not isinstance(evidence, list):
            evidence = [evidence] if evidence else []
        
        model = os.getenv("MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Format evidence
        evidence_text = "\n\n".join([
            f"Evidence {i+1}:\n{ev.get('text', str(ev))[:200] if isinstance(ev, dict) else str(ev)[:200]}..."
            for i, ev in enumerate(evidence[:3])
        ])
        
        verdict = verdict_info.get("verdict", "uncertain")
        confidence = verdict_info.get("confidence", 0.5)
        reasoning = verdict_info.get("reasoning", "Based on available evidence")
        
        prompt = f"""Generate a clear, detailed explanation for this fact-check result.

Claim: {claim}
Verdict: {verdict} (confidence: {confidence:.1%})
Reasoning: {reasoning}

Evidence Used:
{evidence_text}

Write a 3-4 paragraph explanation that:
1. Restates the claim being verified
2. Explains the verdict and why
3. References specific evidence (cite as [Evidence 1], [Evidence 2], etc.)
4. Discusses confidence level and any limitations

Make it accessible to general readers while being accurate.

Return ONLY valid JSON:
{{
    "explanation": "Full multi-paragraph explanation with [Evidence N] citations",
    "conclusion": "One sentence conclusion",
    "limitations": "Any caveats or limitations"
}}"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a fact-checking expert. Write clear, evidence-based explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        content = content.replace('```json\n', '').replace('```\n', '').replace('```', '').strip()
        
        parsed = json.loads(content)
        logger.info(f"generate_explanation_tool completed")
        
        return json.dumps(parsed)
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        logger.error(f"verdict_data type: {type(verdict_data)}, evidence_list type: {type(evidence_list)}")
        verdict_str = verdict_info.get('verdict', 'uncertain') if isinstance(verdict_info, dict) else 'uncertain'
        return json.dumps({
            "explanation": f"The claim was verified as {verdict_str} based on available evidence.",
            "conclusion": f"Verdict: {verdict_str}",
            "limitations": "Unable to generate detailed explanation.",
            "error": str(e)
        })


@tool()
async def cite_evidence_tool(
    evidence_list: Union[str, list], 
    explanation_text: str = ""
) -> str:
    """
    Format evidence citations for the explanation.
    
    Args:
        evidence_list: JSON string OR list with evidence passages
        explanation_text: The explanation text with citation markers (optional)
        
    Returns:
        JSON string with formatted citations
    """
    logger.info(f"[DEBUG] cite_evidence_tool called")
    
    try:
        # Parse evidence with safe handling
        evidence = safe_json_parse(evidence_list, [])
        if not isinstance(evidence, list):
            evidence = [evidence] if evidence else []
        
        # Format citations
        citations = []
        for i, ev in enumerate(evidence, 1):
            if isinstance(ev, dict):
                citation = {
                    "id": i,
                    "text": ev.get("text", str(ev))[:300],
                    "source": ev.get("source", ev.get("title", ev.get("metadata", {}).get("id", "Unknown source"))),
                    "relevance_score": ev.get("hybrid_score", ev.get("score", 0.5))
                }
            else:
                citation = {
                    "id": i,
                    "text": str(ev)[:300],
                    "source": "Unknown source",
                    "relevance_score": 0.5
                }
            citations.append(citation)
        
        result = {
            "citations": citations,
            "citation_count": len(citations),
            "format": "Use [Evidence N] to reference citations in text"
        }
        
        logger.info(f"cite_evidence_tool completed: {len(citations)} citations")
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error citing evidence: {e}")
        logger.error(f"evidence_list type: {type(evidence_list)}")
        return json.dumps({
            "citations": [],
            "citation_count": 0,
            "error": str(e)
        })


@tool()
async def create_report_tool(
    claim: str, 
    verdict_data: Union[str, dict, list], 
    explanation_data: Union[str, dict, list], 
    citations_data: Union[str, dict, list]
) -> str:
    """
    Create comprehensive fact-check report.
    
    Args:
        claim: The original claim
        verdict_data: JSON string OR dict/list with verdict and confidence
        explanation_data: JSON string OR dict/list with explanation text
        citations_data: JSON string OR dict/list with formatted citations
        
    Returns:
        JSON string with complete report
    """
    logger.info(f"[DEBUG] create_report_tool called")
    
    try:
        # Debug logging
        logger.debug(f"verdict_data type: {type(verdict_data)}")
        logger.debug(f"explanation_data type: {type(explanation_data)}")
        logger.debug(f"citations_data type: {type(citations_data)}")
        
        # Parse all inputs with safe handling
        verdict = safe_json_parse(verdict_data, {"verdict": "uncertain", "confidence": 0.5})
        
        # Handle list inputs - take first element if it's a list
        if isinstance(verdict, list):
            logger.warning(f"verdict_data is a list with {len(verdict)} elements, taking first")
            verdict = verdict[0] if verdict else {"verdict": "uncertain", "confidence": 0.5}
        
        if not isinstance(verdict, dict):
            logger.warning(f"verdict is not a dict: {type(verdict)}, creating default")
            verdict = {"verdict": "uncertain", "confidence": 0.5}
        
        explanation = safe_json_parse(explanation_data, {"explanation": "No explanation available", "conclusion": ""})
        if isinstance(explanation, list):
            logger.warning(f"explanation_data is a list, taking first")
            explanation = explanation[0] if explanation else {"explanation": "No explanation available", "conclusion": ""}
        
        if not isinstance(explanation, dict):
            logger.warning(f"explanation is not a dict: {type(explanation)}, creating default")
            explanation = {"explanation": "No explanation available", "conclusion": ""}
        
        citations = safe_json_parse(citations_data, {"citations": [], "citation_count": 0})
        if isinstance(citations, list):
            logger.warning(f"citations_data is a list, wrapping")
            citations = {"citations": citations, "citation_count": len(citations)}
        
        if not isinstance(citations, dict):
            logger.warning(f"citations is not a dict: {type(citations)}, creating default")
            citations = {"citations": [], "citation_count": 0}
        
        # Create comprehensive report
        report = {
            "claim": claim,
            "verdict": verdict.get("verdict", "uncertain"),
            "confidence": verdict.get("confidence", 0.5),
            "label": verdict.get("verdict", "uncertain"),  # For evaluation compatibility
            "summary": explanation.get("conclusion", f"The claim is {verdict.get('verdict', 'uncertain')}"),
            "explanation": explanation.get("explanation", "No detailed explanation available"),
            "limitations": explanation.get("limitations", "None specified"),
            "evidence_citations": citations.get("citations", []),
            "evidence_count": citations.get("citation_count", 0),
            "methodology": {
                "claim_extraction": "NER + Dependency Parsing + LLM",
                "evidence_retrieval": "Hybrid BM25 + Dense Semantic Search",
                "verification": "Multi-evidence aggregation with LLM",
                "explanation": "LLM-generated with evidence citations"
            },
            "metadata": {
                "model": os.getenv("MODEL", "gpt-4.1-mini"),
                "agent_system": "TRUST Agents (LangGraph)"
            }
        }
        
        logger.info(f"create_report_tool completed")
        
        return json.dumps(report, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error creating report: {e}")
        logger.error(f"verdict_data: {verdict_data}")
        logger.error(f"explanation_data: {explanation_data}")
        logger.error(f"citations_data: {citations_data}")
        return json.dumps({
            "claim": claim,
            "verdict": "error",
            "confidence": 0.0,
            "label": "error",
            "summary": f"Error creating report: {str(e)}",
            "explanation": f"Error: {str(e)}",
            "error": str(e)
        })