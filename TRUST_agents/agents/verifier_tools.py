# -*- coding: utf-8 -*-

"""
Verifier Agent Tools - Tools for claim verification against evidence.

Tools used by the Verifier ReAct Agent:
- Compare Tool: Compare claim against evidence passages
- Score Tool: Generate verification score
- Verdict Tool: Determine final verdict (true/false/uncertain)

FIXED: More decisive prompts, lower confidence threshold, better error handling
"""

import os
import json
import logging
from typing import List, Dict

from langchain_core.tools import tool
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("TRUST_agents.agents.verifier_tools")
logger.propagate = True


@tool()
async def compare_claim_evidence_tool(claim: str, evidence_text: str) -> str:
    """
    Compare a claim against evidence text to assess consistency.
    
    Args:
        claim: The claim to verify
        evidence_text: The evidence text to compare against
        
    Returns:
        JSON string with comparison result and reasoning
    """
    logger.info(f"[DEBUG] compare_claim_evidence_tool called")
    
    try:
        model = os.getenv("MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""Compare the following claim against the evidence and assess consistency.

Claim: {claim}

Evidence: {evidence_text}

Analyze:
1. Does the evidence support the claim?
2. Does the evidence contradict the claim?
3. Is the evidence insufficient to determine?

Return ONLY valid JSON:
{{
    "consistency": "supports|contradicts|insufficient",
    "confidence": 0.0-1.0,
    "key_points": ["point1", "point2"],
    "reasoning": "brief explanation"
}}"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a fact verification expert. Analyze claims against evidence objectively."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        content = content.replace('```json\n', '').replace('```\n', '').replace('```', '').strip()
        
        # Validate JSON
        parsed = json.loads(content)
        logger.info(f"compare_claim_evidence_tool completed: {parsed['consistency']}")
        
        return json.dumps(parsed)
        
    except Exception as e:
        logger.error(f"Error comparing claim and evidence: {e}")
        return json.dumps({
            "consistency": "error",
            "confidence": 0.0,
            "key_points": [],
            "reasoning": str(e),
            "error": str(e)
        })


@tool()
async def aggregate_evidence_tool(claim: str, evidence_list: str) -> str:
    """
    Aggregate multiple evidence passages to form an overall assessment.
    
    Args:
        claim: The claim being verified
        evidence_list: JSON string containing list of evidence passages with scores
        
    Returns:
        JSON string with aggregated assessment
    """
    logger.info(f"[DEBUG] aggregate_evidence_tool called")
    
    try:
        # Parse evidence list
        try:
            evidence_data = json.loads(evidence_list)
            if not isinstance(evidence_data, list):
                evidence_data = [evidence_data]
        except:
            evidence_data = [{"text": evidence_list, "score": 0.5}]
        
        model = os.getenv("MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Format evidence for prompt
        evidence_summary = "\n\n".join([
            f"Evidence {i+1} (relevance: {ev.get('hybrid_score', ev.get('score', 0.5)):.3f}):\n{ev.get('text', str(ev))}"
            for i, ev in enumerate(evidence_data[:5])  # Top 5 pieces
        ])
        
        # FIXED: More decisive prompt that encourages making a call
        prompt = f"""You are a fact-checker. Aggregate the following evidence to verify the claim.

Claim: {claim}

Evidence Passages:
{evidence_summary}

INSTRUCTIONS:
- If the MAJORITY of evidence supports the claim → verdict: "supported"
- If the MAJORITY of evidence contradicts the claim → verdict: "contradicted"  
- ONLY use "insufficient" if evidence is truly ambiguous or off-topic
- Be decisive - favor "supported" or "contradicted" when you have ANY relevant evidence
- Confidence should reflect strength of evidence (0.3-0.5 = weak, 0.5-0.7 = moderate, 0.7-1.0 = strong)

Return ONLY valid JSON:
{{
    "overall_verdict": "supported|contradicted|insufficient",
    "confidence": 0.3-1.0,
    "supporting_count": 0,
    "contradicting_count": 0,
    "key_points": ["point1", "point2"],
    "conflicts": ["conflict1"],
    "reasoning": "explanation"
}}

IMPORTANT: Make a decision (supported/contradicted) unless evidence is truly ambiguous."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a decisive fact verification expert. Make clear judgments based on available evidence. Favor definitive verdicts over 'insufficient'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=600
        )
        
        content = response.choices[0].message.content.strip()
        content = content.replace('```json\n', '').replace('```\n', '').replace('```', '').strip()
        
        parsed = json.loads(content)
        
        # FIXED: Boost confidence slightly if we have multiple pieces of evidence
        if len(evidence_data) >= 3:
            original_conf = parsed.get("confidence", 0.5)
            # Boost by 10-20% when we have good evidence quantity
            parsed["confidence"] = min(original_conf * 1.15, 0.95)
        
        logger.info(f"aggregate_evidence_tool completed: {parsed['overall_verdict']} (confidence: {parsed['confidence']:.3f})")
        
        return json.dumps(parsed)
        
    except Exception as e:
        logger.error(f"Error aggregating evidence: {e}")
        return json.dumps({
            "overall_verdict": "insufficient",
            "confidence": 0.3,
            "supporting_count": 0,
            "contradicting_count": 0,
            "key_points": [],
            "conflicts": [],
            "reasoning": str(e),
            "error": str(e)
        })


@tool()
async def generate_verdict_tool(claim: str, aggregated_assessment: str) -> str:
    """
    Generate final verdict based on aggregated evidence assessment.
    
    Args:
        claim: The original claim
        aggregated_assessment: JSON string with aggregated evidence assessment
        
    Returns:
        JSON string with final verdict
    """
    logger.info(f"[DEBUG] generate_verdict_tool called")
    
    try:
        # Parse assessment
        try:
            assessment = json.loads(aggregated_assessment)
        except:
            logger.warning("Failed to parse assessment, using defaults")
            assessment = {"overall_verdict": "insufficient", "confidence": 0.3}
        
        # Generate verdict based on assessment
        verdict_map = {
            "supported": "true",
            "contradicted": "false",
            "insufficient": "uncertain",
            "error": "uncertain"
        }
        
        overall_verdict = assessment.get("overall_verdict", "insufficient")
        verdict = verdict_map.get(overall_verdict, "uncertain")
        confidence = assessment.get("confidence", 0.3)
        
        # FIXED: Lower threshold from 0.4 to 0.25
        # Only force "uncertain" if confidence is VERY low
        if confidence < 0.25:
            logger.warning(f"Low confidence ({confidence:.3f}), forcing uncertain")
            verdict = "uncertain"
            confidence = 0.25
        
        # If we have a supported/contradicted verdict with decent confidence, keep it
        # even if confidence is moderate (0.3-0.5)
        if overall_verdict in ["supported", "contradicted"] and confidence >= 0.3:
            # Keep the verdict - don't override to uncertain
            pass
        
        result = {
            "claim": claim,
            "verdict": verdict,
            "confidence": float(confidence),
            "label": verdict,  # For compatibility with evaluation
            "evidence_summary": {
                "overall_verdict": assessment.get("overall_verdict", "insufficient"),
                "supporting_count": assessment.get("supporting_count", 0),
                "contradicting_count": assessment.get("contradicting_count", 0),
                "key_points": assessment.get("key_points", []),
                "conflicts": assessment.get("conflicts", [])
            },
            "reasoning": assessment.get("reasoning", "Based on available evidence")
        }
        
        logger.info(f"generate_verdict_tool completed: {verdict} (confidence: {confidence:.3f})")
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error generating verdict: {e}")
        return json.dumps({
            "claim": claim,
            "verdict": "uncertain",
            "confidence": 0.3,
            "label": "uncertain",
            "reasoning": f"Error: {str(e)}",
            "error": str(e)
        })


@tool()
async def confidence_calibration_tool(verdict: str, evidence_quality: str) -> str:
    """
    Calibrate confidence score based on evidence quality and consistency.
    
    Args:
        verdict: The current verdict (true/false/uncertain)
        evidence_quality: JSON string describing evidence quality metrics
        
    Returns:
        JSON string with calibrated confidence
    """
    logger.info(f"[DEBUG] confidence_calibration_tool called")
    
    try:
        # Parse quality metrics
        try:
            quality = json.loads(evidence_quality)
        except:
            quality = {"relevance": 0.5, "consistency": 0.5, "quantity": 1}
        
        # Calibration factors
        relevance = quality.get("relevance", 0.5)
        consistency = quality.get("consistency", 0.5)
        quantity = min(quality.get("quantity", 1) / 5.0, 1.0)  # Normalize to 0-1
        
        # Calculate calibrated confidence
        base_confidence = quality.get("base_confidence", 0.5)
        
        # Apply calibration
        calibrated = base_confidence * (
            0.4 * relevance +
            0.4 * consistency +
            0.2 * quantity
        )
        
        # FIXED: Reduce uncertainty penalty from 0.7 to 0.9
        # We're already being too conservative
        if verdict == "uncertain":
            calibrated *= 0.9
        
        # Ensure minimum confidence
        calibrated = max(calibrated, 0.25)
        
        result = {
            "original_confidence": base_confidence,
            "calibrated_confidence": round(calibrated, 3),
            "factors": {
                "relevance": relevance,
                "consistency": consistency,
                "quantity": quantity
            },
            "verdict": verdict
        }
        
        logger.info(f"confidence_calibration_tool completed: {calibrated:.3f}")
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error calibrating confidence: {e}")
        return json.dumps({
            "original_confidence": 0.5,
            "calibrated_confidence": 0.35,
            "error": str(e)
        })