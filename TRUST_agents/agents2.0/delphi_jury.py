# -*- coding: utf-8 -*-
"""
Delphi-Style Multi-Agent Jury System

Multiple verifier personas with dynamic trust scoring and weighted voting.
Based on: Delphi method + Cleanlab BOLAA trust scoring
"""

import os
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger("TRUST_agents.agents.delphi_jury")


@dataclass
class VerifierPersona:
    """A verifier agent with specific personality/focus"""
    name: str
    system_prompt: str
    focus_areas: List[str]
    weight: float = 1.0


class DelphiJury:
    """
    Multi-agent jury system with trust-weighted voting.
    
    Personas:
    - StrictLegalist: Only accepts high-credibility sources
    - OpenWebPragmatist: Considers broader sources
    - CausalSkeptic: Focuses on temporal/numeric consistency
    - ConspiracyDetector: Specialized for conspiratorial patterns
    """
    
    def __init__(self, model: str = None):
        self.model = model or os.getenv("MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.personas = self._create_personas()
    
    def _create_personas(self) -> List[VerifierPersona]:
        """Create verifier personas with different focuses"""
        
        return [
            VerifierPersona(
                name="StrictLegalist",
                system_prompt="""You are a STRICT fact-checker who only trusts high-credibility sources.
                
RULES:
- Only accept evidence from: .gov, .edu, major news outlets (Reuters, AP, BBC)
- Require multiple independent sources for claims
- Default to "uncertain" if evidence quality is questionable
- Be extremely conservative with "true" verdicts""",
                focus_areas=["source_credibility", "evidence_quality"]
            ),
            
            VerifierPersona(
                name="OpenWebPragmatist",
                system_prompt="""You are a PRAGMATIC fact-checker who considers broader sources.
                
RULES:
- Accept evidence from various sources if content is verifiable
- Focus on content quality over source prestige
- Use common sense and contextual reasoning
- Balance between being too strict and too lenient""",
                focus_areas=["content_quality", "contextual_reasoning"]
            ),
            
            VerifierPersona(
                name="CausalSkeptic",
                system_prompt="""You are a CAUSAL fact-checker focused on temporal and numeric consistency.
                
RULES:
- Verify temporal ordering (cause before effect)
- Check numeric claims against data
- Look for logical fallacies (correlation ≠ causation)
- Question causal claims without mechanism""",
                focus_areas=["temporal_consistency", "numeric_verification", "causal_logic"]
            ),
            
            VerifierPersona(
                name="ConspiracyDetector",
                system_prompt="""You are specialized in detecting CONSPIRATORIAL patterns.
                
RULES:
- Identify conspiracy theory markers (secret cabals, hidden agendas)
- Check for cherry-picked evidence
- Look for logical leaps and unfalsifiable claims
- Flag claims that rely on "they don't want you to know"
- Be skeptical of extraordinary claims without extraordinary evidence""",
                focus_areas=["conspiracy_patterns", "logical_fallacies"]
            )
        ]
    
    def verify_with_jury(
        self,
        claim: str,
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify claim using multi-agent jury with trust-weighted voting.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence passages
            
        Returns:
            Aggregated verdict with trust scores
        """
        logger.info(f"[DELPHI_JURY] Convening jury for claim: {claim[:100]}...")
        
        # Get verdicts from each persona
        persona_verdicts = []
        for persona in self.personas:
            verdict = self._get_persona_verdict(persona, claim, evidence)
            persona_verdicts.append(verdict)
        
        # Compute trust scores for each verdict
        trust_scores = self._compute_trust_scores(persona_verdicts, evidence)
        
        # Aggregate with trust weighting
        final_verdict = self._aggregate_with_trust(persona_verdicts, trust_scores)
        
        logger.info(f"[DELPHI_JURY] Final verdict: {final_verdict['verdict']} ({final_verdict['confidence']:.2f})")
        
        return final_verdict
    
    def _get_persona_verdict(
        self,
        persona: VerifierPersona,
        claim: str,
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get verdict from a specific persona"""
        
        logger.info(f"[DELPHI_JURY] Consulting {persona.name}...")
        
        # Format evidence
        evidence_text = "\n\n".join([
            f"Evidence {i+1} (score: {e.get('hybrid_score', 0.5):.3f}, source: {e.get('source', 'unknown')}):\n{e.get('text', '')[:300]}..."
            for i, e in enumerate(evidence[:5])
        ])
        
        prompt = f"""Verify the following claim against the evidence.

Claim: {claim}

Evidence:
{evidence_text}

Analyze from your perspective ({persona.name}) and return ONLY valid JSON:
{{
    "verdict": "true|false|uncertain",
    "confidence": 0.0-1.0,
    "reasoning": "your analysis",
    "key_concerns": ["concern1", "concern2"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": persona.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            content = content.replace('```json\n', '').replace('```\n', '').replace('```', '').strip()
            
            parsed = json.loads(content)
            parsed['persona'] = persona.name
            
            logger.info(f"[DELPHI_JURY] {persona.name}: {parsed['verdict']} ({parsed['confidence']:.2f})")
            
            return parsed
            
        except Exception as e:
            logger.error(f"[DELPHI_JURY] Error with {persona.name}: {e}")
            return {
                "persona": persona.name,
                "verdict": "uncertain",
                "confidence": 0.3,
                "reasoning": f"Error: {str(e)}",
                "error": True
            }
    
    def _compute_trust_scores(
        self,
        verdicts: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Compute trust scores for each verdict.
        
        Based on:
        - Evidence quality (retrieval scores)
        - Internal consistency (agreement with evidence)
        - Confidence calibration
        """
        trust_scores = []
        
        for verdict in verdicts:
            # Base trust from evidence quality
            if evidence:
                avg_evidence_score = np.mean([e.get('hybrid_score', 0.5) for e in evidence])
            else:
                avg_evidence_score = 0.3
            
            # Confidence calibration
            confidence = verdict.get('confidence', 0.5)
            
            # Penalty for errors
            if verdict.get('error', False):
                trust = 0.2
            else:
                # Combine factors
                trust = (
                    0.4 * avg_evidence_score +
                    0.4 * confidence +
                    0.2 * (1.0 if not verdict.get('error') else 0.0)
                )
            
            # Boost for decisive verdicts with good evidence
            if verdict.get('verdict') in ['true', 'false'] and avg_evidence_score > 0.6:
                trust *= 1.1
            
            trust = min(max(trust, 0.0), 1.0)
            trust_scores.append(trust)
            
            logger.debug(f"[DELPHI_JURY] {verdict.get('persona')}: trust={trust:.3f}")
        
        return trust_scores
    
    def _aggregate_with_trust(
        self,
        verdicts: List[Dict[str, Any]],
        trust_scores: List[float]
    ) -> Dict[str, Any]:
        """Aggregate verdicts using trust-weighted voting"""
        
        # Weighted votes
        weighted_votes = {"true": 0.0, "false": 0.0, "uncertain": 0.0}
        total_weight = 0.0
        
        for verdict, trust in zip(verdicts, trust_scores):
            v = verdict.get('verdict', 'uncertain')
            conf = verdict.get('confidence', 0.5)
            
            # Weight = trust * confidence
            weight = trust * conf
            
            weighted_votes[v] += weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for k in weighted_votes:
                weighted_votes[k] /= total_weight
        
        # Get winner
        final_verdict = max(weighted_votes, key=weighted_votes.get)
        final_confidence = weighted_votes[final_verdict]
        
        # Generate reasoning
        reasoning_parts = [f"Jury Decision (Trust-Weighted Voting):"]
        for verdict, trust in zip(verdicts, trust_scores):
            reasoning_parts.append(
                f"- {verdict['persona']}: {verdict['verdict']} "
                f"(conf={verdict['confidence']:.2f}, trust={trust:.2f})"
            )
        reasoning_parts.append(f"\nWeighted votes: {weighted_votes}")
        reasoning_parts.append(f"Final: {final_verdict} ({final_confidence:.2f})")
        
        return {
            "verdict": final_verdict,
            "confidence": final_confidence,
            "label": final_verdict,
            "reasoning": "\n".join(reasoning_parts),
            "jury_verdicts": verdicts,
            "trust_scores": trust_scores,
            "weighted_votes": weighted_votes
        }


def verify_with_delphi_jury_sync(claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Synchronous wrapper for Delphi jury verification"""
    jury = DelphiJury()
    return jury.verify_with_jury(claim, evidence)


if __name__ == "__main__":
    # Test Delphi jury
    test_claim = "The COVID-19 vaccine contains microchips for tracking"
    test_evidence = [
        {
            "text": "No evidence supports the claim that COVID-19 vaccines contain microchips. This is a debunked conspiracy theory.",
            "source": "reuters.com",
            "hybrid_score": 0.85
        },
        {
            "text": "Fact-check: COVID vaccines do not contain microchips or tracking devices.",
            "source": "factcheck.org",
            "hybrid_score": 0.82
        }
    ]
    
    result = verify_with_delphi_jury_sync(test_claim, test_evidence)
    
    print("\n" + "="*70)
    print("DELPHI JURY VERDICT")
    print("="*70)
    print(f"\nClaim: {test_claim}")
    print(f"\nFinal Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"\nReasoning:\n{result['reasoning']}")
