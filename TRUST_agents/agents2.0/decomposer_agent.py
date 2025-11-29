# -*- coding: utf-8 -*-
"""
Decomposer Agent - LoCal-inspired claim decomposition.

Decomposes complex claims into:
- Atomic sub-claims
- Logical structure (AND/OR/IMPLIES)
- Causal relationships

Based on: LoCal (ACM 2024)
"""

import os
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger("TRUST_agents.agents.decomposer")


@dataclass
class DecomposedClaim:
    """Result of claim decomposition"""
    original_claim: str
    atomic_claims: List[str]
    logic_structure: str
    causal_edges: List[Dict[str, str]]
    complexity_score: float


class DecomposerAgent:
    """
    Decomposes complex claims into atomic sub-claims with logical structure.
    
    Example:
        Input: "After policy X passed in 2020, unemployment decreased by 5%"
        Output:
            - Atomic claims: ["Policy X passed in 2020", "Unemployment decreased by 5%"]
            - Logic: "C1 AND C2"
            - Causal: [{"cause": "Policy X passed", "effect": "unemployment decreased"}]
    """
    
    def __init__(self, model: str = None):
        self.model = model or os.getenv("MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def decompose(self, claim: str) -> DecomposedClaim:
        """
        Decompose a claim into atomic sub-claims with logical structure.
        
        Args:
            claim: The complex claim to decompose
            
        Returns:
            DecomposedClaim with atomic claims, logic, and causal edges
        """
        logger.info(f"[DECOMPOSER] Decomposing claim: {claim[:100]}...")
        
        prompt = f"""Decompose the following claim into atomic sub-claims with logical structure.

Claim: {claim}

INSTRUCTIONS:
1. Break down into ATOMIC claims (each verifiable independently)
2. Identify LOGICAL structure (AND, OR, IMPLIES, NOT)
3. Detect CAUSAL relationships (cause → effect)
4. Assign complexity score (0.0-1.0)

Return ONLY valid JSON:
{{
    "atomic_claims": [
        "atomic claim 1",
        "atomic claim 2"
    ],
    "logic_structure": "C1 AND C2",
    "causal_edges": [
        {{"cause": "event X", "effect": "outcome Y", "temporal": true}}
    ],
    "complexity_score": 0.7,
    "reasoning": "brief explanation"
}}

EXAMPLES:

Claim: "Biden won the 2020 election and became president in 2021"
{{
    "atomic_claims": ["Biden won the 2020 election", "Biden became president in 2021"],
    "logic_structure": "C1 AND C2",
    "causal_edges": [{{"cause": "Biden won election", "effect": "became president", "temporal": true}}],
    "complexity_score": 0.3
}}

Claim: "If unemployment rises, then either inflation increases or GDP decreases"
{{
    "atomic_claims": ["unemployment rises", "inflation increases", "GDP decreases"],
    "logic_structure": "C1 IMPLIES (C2 OR C3)",
    "causal_edges": [
        {{"cause": "unemployment rises", "effect": "inflation increases", "temporal": false}},
        {{"cause": "unemployment rises", "effect": "GDP decreases", "temporal": false}}
    ],
    "complexity_score": 0.8
}}

Now decompose the given claim:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at logical claim decomposition."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            content = content.replace('```json\n', '').replace('```\n', '').replace('```', '').strip()
            
            parsed = json.loads(content)
            
            result = DecomposedClaim(
                original_claim=claim,
                atomic_claims=parsed.get("atomic_claims", [claim]),
                logic_structure=parsed.get("logic_structure", "C1"),
                causal_edges=parsed.get("causal_edges", []),
                complexity_score=parsed.get("complexity_score", 0.5)
            )
            
            logger.info(f"[DECOMPOSER] Decomposed into {len(result.atomic_claims)} atomic claims")
            logger.info(f"[DECOMPOSER] Logic structure: {result.logic_structure}")
            logger.info(f"[DECOMPOSER] Causal edges: {len(result.causal_edges)}")
            
            return result
            
        except Exception as e:
            logger.error(f"[DECOMPOSER] Error: {e}")
            # Fallback: treat as single atomic claim
            return DecomposedClaim(
                original_claim=claim,
                atomic_claims=[claim],
                logic_structure="C1",
                causal_edges=[],
                complexity_score=0.5
            )


def decompose_claim_sync(claim: str) -> DecomposedClaim:
    """Synchronous wrapper for claim decomposition"""
    agent = DecomposerAgent()
    return agent.decompose(claim)


if __name__ == "__main__":
    # Test decomposer
    test_claims = [
        "Biden won the 2020 election and became president in 2021",
        "After the policy passed in 2020, unemployment decreased by 5%",
        "If inflation rises above 5%, then either the Fed will raise rates or the economy will slow down"
    ]
    
    for claim in test_claims:
        print(f"\n{'='*70}")
        print(f"Claim: {claim}")
        print('='*70)
        
        result = decompose_claim_sync(claim)
        
        print(f"\nAtomic Claims ({len(result.atomic_claims)}):")
        for i, ac in enumerate(result.atomic_claims, 1):
            print(f"  C{i}: {ac}")
        
        print(f"\nLogic Structure: {result.logic_structure}")
        
        print(f"\nCausal Edges ({len(result.causal_edges)}):")
        for edge in result.causal_edges:
            print(f"  {edge['cause']} → {edge['effect']}")
        
        print(f"\nComplexity Score: {result.complexity_score:.2f}")
