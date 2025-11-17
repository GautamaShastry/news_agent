# -*- coding: utf-8 -*-

"""
TRUST Agents Orchestrator - Coordinates all agents in the fact-checking pipeline.

Pipeline:
1. Claim Extractor: Extract factual claims from text
2. Evidence Retriever: Find relevant evidence for each claim
3. Verifier: Verify claims against evidence
4. Explainer: Generate comprehensive explanations

This orchestrator runs the complete end-to-end pipeline.

FIXED: Normalizes verifier output to ensure consistent verdict format and confidence range
"""

import os
import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

# Import all agents
from TRUST_agents.agents.claim_extractor import run_claim_extractor_agent_sync
from TRUST_agents.agents.evidence_retrieval import run_evidence_retrieval_agent_sync
from TRUST_agents.agents.verifier import run_verifier_agent_sync
from TRUST_agents.agents.explainer import run_explainer_agent_sync

load_dotenv()
logger = logging.getLogger("TRUST_agents.orchestrator")


@dataclass
class TRUSTResult:
    """Result from complete TRUST Agents pipeline."""
    original_text: str
    claims: List[str]
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]


class TRUSTOrchestrator:
    """
    Orchestrator for TRUST Agents multi-agent system.
    
    Coordinates:
    - Claim Extractor
    - Evidence Retriever
    - Verifier
    - Explainer
    """
    
    def __init__(self, index_dir: str = "retrieval_index", top_k_evidence: int = 5):
        """
        Initialize TRUST orchestrator.
        
        Args:
            index_dir: Directory for retrieval index
            top_k_evidence: Number of evidence passages to retrieve per claim
        """
        self.index_dir = index_dir
        self.top_k_evidence = top_k_evidence
        logger.info("[ORCHESTRATOR] TRUST Agents initialized")
    
    def _normalize_verdict(self, verdict_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize verifier output to ensure consistent format.
        
        Fixes:
        - Maps "supported" -> "true", "contradicted" -> "false"
        - Extracts verdict from text like "The claim is uncertain" -> "uncertain"
        - Ensures confidence is 0.0-1.0 (not 0-100)
        - Handles errors and edge cases
        
        Args:
            verdict_data: Raw output from verifier
            
        Returns:
            Normalized verdict data
        """
        # Make a copy to avoid modifying original
        normalized = verdict_data.copy()
        
        # Extract verdict
        raw_verdict = verdict_data.get("verdict", "uncertain")
        
        # Clean up verdict text
        if isinstance(raw_verdict, str):
            verdict_lower = raw_verdict.lower().strip()
            
            # Map verifier outputs to standard labels
            verdict_map = {
                "supported": "true",
                "true": "true",
                "contradicted": "false",
                "false": "false",
                "insufficient": "uncertain",
                "uncertain": "uncertain",
                "error": "uncertain"
            }
            
            # Try direct mapping first
            if verdict_lower in verdict_map:
                normalized["verdict"] = verdict_map[verdict_lower]
                normalized["label"] = verdict_map[verdict_lower]
            else:
                # Extract from text like "The claim is false" or "The verdict is uncertain"
                if "false" in verdict_lower or "contradicted" in verdict_lower:
                    normalized["verdict"] = "false"
                    normalized["label"] = "false"
                elif "true" in verdict_lower or "supported" in verdict_lower:
                    normalized["verdict"] = "true"
                    normalized["label"] = "true"
                else:
                    # Default to uncertain for anything else
                    normalized["verdict"] = "uncertain"
                    normalized["label"] = "uncertain"
                    logger.warning(f"[ORCHESTRATOR] Unusual verdict format: {raw_verdict}, defaulting to uncertain")
        else:
            # Non-string verdict
            logger.warning(f"[ORCHESTRATOR] Non-string verdict: {type(raw_verdict)}, defaulting to uncertain")
            normalized["verdict"] = "uncertain"
            normalized["label"] = "uncertain"
        
        # Normalize confidence to 0.0-1.0
        raw_confidence = verdict_data.get("confidence", 0.3)
        
        try:
            if isinstance(raw_confidence, str):
                # Try to parse string
                confidence = float(raw_confidence)
            else:
                confidence = float(raw_confidence)
            
            # If confidence is > 1, assume it's in 0-100 format
            if confidence > 1.0:
                confidence = confidence / 100.0
                logger.debug(f"[ORCHESTRATOR] Converted confidence from percentage: {raw_confidence} -> {confidence}")
            
            # Clamp to valid range
            confidence = max(0.0, min(1.0, confidence))
            
        except (ValueError, TypeError) as e:
            logger.warning(f"[ORCHESTRATOR] Could not parse confidence: {raw_confidence}, defaulting to 0.3")
            confidence = 0.3
        
        normalized["confidence"] = float(confidence)
        
        # Ensure reasoning exists
        if "reasoning" not in normalized or not normalized["reasoning"]:
            normalized["reasoning"] = normalized.get("evidence_summary", {}).get("reasoning", "No reasoning provided")
        
        return normalized
    
    def process_text(self, text: str, skip_evidence: bool = False) -> TRUSTResult:
        """
        Process text through complete TRUST pipeline.
        
        Args:
            text: Input text to fact-check
            skip_evidence: If True, skip evidence retrieval (for testing)
        
        Returns:
            TRUSTResult with complete analysis
        """
        logger.info("[ORCHESTRATOR] Starting TRUST pipeline")
        logger.info("[ORCHESTRATOR] Input text length: %d characters", len(text))
        
        # Step 1: Extract Claims
        logger.info("[ORCHESTRATOR] STEP 1: Extracting claims...")
        claims = run_claim_extractor_agent_sync(text)
        logger.info("[ORCHESTRATOR] Extracted %d claims", len(claims))
        
        if not claims:
            logger.warning("[ORCHESTRATOR] No claims extracted, stopping pipeline")
            return TRUSTResult(
                original_text=text,
                claims=[],
                results=[],
                summary={"status": "no_claims", "message": "No claims found in text"}
            )
        
        # Process each claim
        results = []
        for i, claim in enumerate(claims, 1):
            logger.info("[ORCHESTRATOR] Processing claim %d/%d: %s", i, len(claims), claim[:80])
            
            try:
                result = self._process_single_claim(claim, skip_evidence)
                results.append(result)
                logger.info("[ORCHESTRATOR] Claim %d complete: verdict=%s", i, result.get("verdict"))
            except Exception as e:
                logger.error("[ORCHESTRATOR] Error processing claim %d: %s", i, e, exc_info=True)
                results.append({
                    "claim": claim,
                    "verdict": "uncertain",
                    "confidence": 0.0,
                    "label": "uncertain",
                    "reasoning": f"Error: {str(e)}",
                    "error": str(e)
                })
        
        # Create summary
        summary = self._create_summary(results)
        
        logger.info("[ORCHESTRATOR] Pipeline complete. Processed %d claims", len(results))
        
        return TRUSTResult(
            original_text=text,
            claims=claims,
            results=results,
            summary=summary
        )
    
    def _process_single_claim(self, claim: str, skip_evidence: bool = False) -> Dict[str, Any]:
        """
        Process a single claim through the pipeline.
        
        Args:
            claim: The claim to verify
            skip_evidence: If True, skip evidence retrieval
        
        Returns:
            Dictionary with verification result
        """
        # Step 2: Retrieve Evidence
        if skip_evidence:
            logger.info("[ORCHESTRATOR] Skipping evidence retrieval")
            evidence = []
        else:
            logger.info("[ORCHESTRATOR] STEP 2: Retrieving evidence...")
            try:
                evidence = run_evidence_retrieval_agent_sync(claim, top_k=self.top_k_evidence)
                logger.info("[ORCHESTRATOR] Retrieved %d evidence passages", len(evidence))
            except Exception as e:
                logger.error(f"[ORCHESTRATOR] Evidence retrieval failed: {e}")
                evidence = []
        
        # Step 3: Verify Claim
        logger.info("[ORCHESTRATOR] STEP 3: Verifying claim...")
        try:
            if evidence:
                verdict_data = run_verifier_agent_sync(claim, evidence)
            else:
                # No evidence available
                verdict_data = {
                    "claim": claim,
                    "verdict": "uncertain",
                    "confidence": 0.1,
                    "label": "uncertain",
                    "reasoning": "No evidence available for verification"
                }
            
            # CRITICAL FIX: Normalize the verdict output
            verdict_data = self._normalize_verdict(verdict_data)
            
            logger.info("[ORCHESTRATOR] Verification complete: %s (%.1f%%)", 
                       verdict_data.get("verdict"), verdict_data.get("confidence", 0) * 100)
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Verification failed: {e}", exc_info=True)
            verdict_data = {
                "claim": claim,
                "verdict": "uncertain",
                "confidence": 0.1,
                "label": "uncertain",
                "reasoning": f"Verification error: {str(e)}",
                "error": str(e)
            }
        
        # Step 4: Generate Explanation
        logger.info("[ORCHESTRATOR] STEP 4: Generating explanation...")
        try:
            report = run_explainer_agent_sync(claim, verdict_data, evidence)
            
            # Ensure report has normalized verdict
            if "verdict" in report:
                report = self._normalize_verdict(report)
            else:
                # Merge verdict data into report if not present
                report.update({
                    "verdict": verdict_data["verdict"],
                    "confidence": verdict_data["confidence"],
                    "label": verdict_data["label"]
                })
            
            logger.info("[ORCHESTRATOR] Explanation complete")
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Explanation failed: {e}", exc_info=True)
            # Use verdict data as fallback
            report = verdict_data.copy()
            report["summary"] = f"Verdict: {verdict_data['verdict']}"
            report["explanation"] = f"Error generating explanation: {str(e)}"
            report["error"] = str(e)
        
        return report
    
    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics for all results."""
        if not results:
            return {"total_claims": 0}
        
        # Count verdicts
        verdict_counts = {"true": 0, "false": 0, "uncertain": 0, "error": 0}
        confidences = []
        
        for result in results:
            verdict = result.get("verdict", "uncertain")
            
            # Normalize verdict for counting
            if verdict in ["true", "supported"]:
                verdict_counts["true"] += 1
            elif verdict in ["false", "contradicted"]:
                verdict_counts["false"] += 1
            elif verdict in ["uncertain", "insufficient"]:
                verdict_counts["uncertain"] += 1
            else:
                verdict_counts["error"] += 1
            
            conf = result.get("confidence", 0.0)
            try:
                conf = float(conf)
                if conf > 1.0:  # Handle percentage format
                    conf = conf / 100.0
                if 0.0 <= conf <= 1.0:
                    confidences.append(conf)
            except (ValueError, TypeError):
                pass
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total_claims": len(results),
            "verdicts": verdict_counts,
            "average_confidence": round(avg_confidence, 3),
            "high_confidence_claims": sum(1 for c in confidences if c > 0.7),
            "low_confidence_claims": sum(1 for c in confidences if c < 0.4)
        }


def run_trust_pipeline_sync(text: str, top_k_evidence: int = 5, skip_evidence: bool = False) -> Dict[str, Any]:
    """
    Run complete TRUST pipeline on text.
    
    Args:
        text: Input text to fact-check
        top_k_evidence: Number of evidence passages per claim
        skip_evidence: If True, skip evidence retrieval (for testing)
    
    Returns:
        Dictionary with complete results
    """
    orchestrator = TRUSTOrchestrator(top_k_evidence=top_k_evidence)
    result = orchestrator.process_text(text, skip_evidence=skip_evidence)
    return asdict(result)


# Convenience function
def fact_check(text: str, top_k_evidence: int = 5) -> Dict[str, Any]:
    """
    Fact-check text using TRUST Agents.
    
    Simple interface for fact-checking.
    
    Args:
        text: Text to fact-check
        top_k_evidence: Number of evidence passages per claim
    
    Returns:
        Dictionary with fact-check results
    """
    return run_trust_pipeline_sync(text, top_k_evidence=top_k_evidence)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TRUST Agents fact-checking pipeline")
    parser.add_argument("--text", required=True, help="Text to fact-check")
    parser.add_argument("--top-k", type=int, default=5, help="Evidence passages per claim")
    parser.add_argument("--skip-evidence", action="store_true", help="Skip evidence retrieval")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    # Run pipeline
    result = run_trust_pipeline_sync(
        args.text,
        top_k_evidence=args.top_k,
        skip_evidence=args.skip_evidence
    )
    
    # Print results
    print("\n" + "="*70)
    print("TRUST AGENTS - FACT-CHECK RESULTS")
    print("="*70)
    print(f"\nOriginal Text: {args.text[:200]}...")
    print(f"\nClaims Found: {len(result['claims'])}")
    
    for i, claim_result in enumerate(result['results'], 1):
        print(f"\n--- Claim {i} ---")
        print(f"Claim: {claim_result['claim']}")
        print(f"Verdict: {claim_result['verdict']} (confidence: {claim_result.get('confidence', 0):.1%})")
        print(f"Summary: {claim_result.get('summary', 'N/A')}")
    
    print(f"\n--- Summary ---")
    summary = result['summary']
    print(f"Total Claims: {summary['total_claims']}")
    print(f"Verdicts: {summary.get('verdicts', {})}")
    print(f"Average Confidence: {summary.get('average_confidence', 0):.1%}")
    
    # Save if requested
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {args.output}")