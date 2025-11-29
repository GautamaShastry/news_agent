# -*- coding: utf-8 -*-
"""
Logic Aggregator Agent - Reconstructs truth from atomic claim verdicts.

Uses logical structure to combine atomic claim verdicts into final verdict.
Based on: LoCal (ACM 2024)
"""

import logging
from typing import List, Dict, Any
import re

logger = logging.getLogger("TRUST_agents.agents.logic_aggregator")


class LogicAggregator:
    """
    Aggregates atomic claim verdicts using logical structure.
    
    Supports: AND, OR, NOT, IMPLIES
    """
    
    def __init__(self):
        self.truth_map = {
            "true": True,
            "supported": True,
            "false": False,
            "contradicted": False,
            "refuted": False,
            "uncertain": None
        }
    
    def aggregate(
        self,
        atomic_verdicts: List[Dict[str, Any]],
        logic_structure: str
    ) -> Dict[str, Any]:
        """
        Aggregate atomic verdicts using logical structure.
        
        Args:
            atomic_verdicts: List of verdicts for atomic claims
            logic_structure: Logical formula (e.g., "C1 AND C2", "C1 IMPLIES (C2 OR C3)")
            
        Returns:
            Final verdict with reasoning
        """
        logger.info(f"[LOGIC_AGG] Aggregating {len(atomic_verdicts)} verdicts")
        logger.info(f"[LOGIC_AGG] Logic structure: {logic_structure}")
        
        # Map verdicts to truth values
        truth_values = {}
        for i, verdict in enumerate(atomic_verdicts, 1):
            label = verdict.get('verdict', 'uncertain').lower()
            truth_values[f'C{i}'] = self.truth_map.get(label, None)
        
        logger.debug(f"[LOGIC_AGG] Truth values: {truth_values}")
        
        # Evaluate logical formula
        try:
            result = self._evaluate_logic(logic_structure, truth_values)
            
            # Convert back to verdict
            if result is True:
                final_verdict = "true"
                confidence = self._compute_confidence(atomic_verdicts, "true")
            elif result is False:
                final_verdict = "false"
                confidence = self._compute_confidence(atomic_verdicts, "false")
            else:
                final_verdict = "uncertain"
                confidence = 0.3
            
            reasoning = self._generate_reasoning(
                logic_structure,
                atomic_verdicts,
                final_verdict
            )
            
            logger.info(f"[LOGIC_AGG] Final verdict: {final_verdict} ({confidence:.2f})")
            
            return {
                "verdict": final_verdict,
                "confidence": confidence,
                "label": final_verdict,
                "reasoning": reasoning,
                "logic_structure": logic_structure,
                "atomic_verdicts": atomic_verdicts
            }
            
        except Exception as e:
            logger.error(f"[LOGIC_AGG] Error evaluating logic: {e}")
            # Fallback: use majority vote
            return self._majority_vote_fallback(atomic_verdicts)
    
    def _evaluate_logic(self, formula: str, truth_values: Dict[str, bool]) -> bool:
        """Evaluate logical formula with truth values"""
        
        # Replace claim variables with truth values
        expr = formula
        for var, value in truth_values.items():
            if value is None:
                # Uncertain - can't evaluate
                return None
            expr = expr.replace(var, str(value))
        
        # Replace logical operators
        expr = expr.replace('AND', 'and')
        expr = expr.replace('OR', 'or')
        expr = expr.replace('NOT', 'not')
        expr = expr.replace('IMPLIES', '<=')  # A IMPLIES B = (not A) or B
        
        # Handle IMPLIES properly
        if '<=' in expr:
            # Convert "A <= B" to "(not A) or B"
            parts = expr.split('<=')
            if len(parts) == 2:
                expr = f"(not ({parts[0].strip()})) or ({parts[1].strip()})"
        
        # Evaluate
        try:
            result = eval(expr)
            return result
        except:
            return None
    
    def _compute_confidence(
        self,
        atomic_verdicts: List[Dict[str, Any]],
        final_verdict: str
    ) -> float:
        """Compute confidence for final verdict"""
        
        # Average confidence of atomic verdicts that agree with final verdict
        agreeing_confidences = []
        
        for verdict in atomic_verdicts:
            v = verdict.get('verdict', 'uncertain').lower()
            c = verdict.get('confidence', 0.5)
            
            if (final_verdict == "true" and v in ["true", "supported"]) or \
               (final_verdict == "false" and v in ["false", "contradicted"]):
                agreeing_confidences.append(c)
        
        if agreeing_confidences:
            return sum(agreeing_confidences) / len(agreeing_confidences)
        else:
            return 0.3
    
    def _generate_reasoning(
        self,
        logic_structure: str,
        atomic_verdicts: List[Dict[str, Any]],
        final_verdict: str
    ) -> str:
        """Generate human-readable reasoning"""
        
        reasoning_parts = []
        
        # Explain atomic verdicts
        for i, verdict in enumerate(atomic_verdicts, 1):
            claim = verdict.get('claim', f'Claim {i}')
            v = verdict.get('verdict', 'uncertain')
            c = verdict.get('confidence', 0.5)
            reasoning_parts.append(f"C{i} ({claim[:50]}...): {v} ({c:.2f})")
        
        # Explain logical combination
        reasoning_parts.append(f"\nLogical structure: {logic_structure}")
        reasoning_parts.append(f"Final verdict: {final_verdict}")
        
        return "\n".join(reasoning_parts)
    
    def _majority_vote_fallback(self, atomic_verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback to majority voting if logic evaluation fails"""
        
        logger.warning("[LOGIC_AGG] Using majority vote fallback")
        
        votes = {"true": 0, "false": 0, "uncertain": 0}
        
        for verdict in atomic_verdicts:
            v = verdict.get('verdict', 'uncertain').lower()
            if v in ["true", "supported"]:
                votes["true"] += 1
            elif v in ["false", "contradicted"]:
                votes["false"] += 1
            else:
                votes["uncertain"] += 1
        
        final_verdict = max(votes, key=votes.get)
        confidence = votes[final_verdict] / len(atomic_verdicts)
        
        return {
            "verdict": final_verdict,
            "confidence": confidence,
            "label": final_verdict,
            "reasoning": f"Majority vote: {votes}",
            "fallback": True
        }


if __name__ == "__main__":
    # Test logic aggregator
    aggregator = LogicAggregator()
    
    # Test case 1: Simple AND
    print("\n" + "="*70)
    print("Test 1: C1 AND C2")
    print("="*70)
    
    verdicts = [
        {"claim": "Biden won 2020 election", "verdict": "true", "confidence": 0.9},
        {"claim": "Biden became president in 2021", "verdict": "true", "confidence": 0.95}
    ]
    
    result = aggregator.aggregate(verdicts, "C1 AND C2")
    print(f"Result: {result['verdict']} ({result['confidence']:.2f})")
    print(f"Reasoning:\n{result['reasoning']}")
    
    # Test case 2: OR with one false
    print("\n" + "="*70)
    print("Test 2: C1 OR C2")
    print("="*70)
    
    verdicts = [
        {"claim": "Claim A", "verdict": "false", "confidence": 0.8},
        {"claim": "Claim B", "verdict": "true", "confidence": 0.7}
    ]
    
    result = aggregator.aggregate(verdicts, "C1 OR C2")
    print(f"Result: {result['verdict']} ({result['confidence']:.2f})")
    
    # Test case 3: IMPLIES
    print("\n" + "="*70)
    print("Test 3: C1 IMPLIES C2")
    print("="*70)
    
    verdicts = [
        {"claim": "Condition", "verdict": "true", "confidence": 0.9},
        {"claim": "Consequence", "verdict": "false", "confidence": 0.8}
    ]
    
    result = aggregator.aggregate(verdicts, "C1 IMPLIES C2")
    print(f"Result: {result['verdict']} ({result['confidence']:.2f})")
