# -*- coding: utf-8 -*-
"""
TRUST Agents Research Orchestrator - LoCal + Delphi + MEGA-RAG

This is the research-grade pipeline combining:
- LoCal-style claim decomposition and logic aggregation
- Delphi-style multi-agent jury with trust scoring
- MEGA-RAG evidence refinement (coming soon)

Expected improvements: +15-25% accuracy over baseline
"""

import os
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

# Import research agents (agents2.0)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "agents2.0"))

from decomposer_agent import DecomposerAgent, DecomposedClaim
from logic_aggregator import LogicAggregator
from delphi_jury import DelphiJury

# Import existing agents
from TRUST_agents.agents.evidence_retrieval import run_evidence_retrieval_agent_sync
from TRUST_agents.agents.explainer import run_explainer_agent_sync

load_dotenv()
logger = logging.getLogger("TRUST_agents.orchestrator_research")


@dataclass
class ResearchTRUSTResult:
    """Result from research TRUST pipeline"""
    original_text: str
    decomposed_claim: DecomposedClaim
    atomic_verdicts: List[Dict[str, Any]]
    logic_aggregation: Dict[str, Any]
    final_verdict: Dict[str, Any]
    metadata: Dict[str, Any]


class ResearchTRUSTOrchestrator:
    """
    Research-grade TRUST orchestrator with:
    - Claim decomposition (LoCal)
    - Multi-agent jury (Delphi)
    - Logic aggregation
    """
    
    def __init__(
        self,
        index_dir: str = "retrieval_index",
        top_k_evidence: int = 10,
        use_delphi_jury: bool = True
    ):
        """
        Initialize research orchestrator.
        
        Args:
            index_dir: Directory for retrieval index
            top_k_evidence: Number of evidence passages per claim
            use_delphi_jury: Use multi-agent jury (vs single verifier)
        """
        self.index_dir = index_dir
        self.top_k_evidence = top_k_evidence
        self.use_delphi_jury = use_delphi_jury
        
        # Initialize agents
        self.decomposer = DecomposerAgent()
        self.logic_aggregator = LogicAggregator()
        if use_delphi_jury:
            self.delphi_jury = DelphiJury()
        
        logger.info("[RESEARCH_ORCHESTRATOR] Initialized with Delphi=%s", use_delphi_jury)
    
    def process_text(self, text: str, skip_evidence: bool = False) -> ResearchTRUSTResult:
        """
        Process text through research TRUST pipeline.
        
        Args:
            text: Input text to fact-check
            skip_evidence: If True, skip evidence retrieval
            
        Returns:
            ResearchTRUSTResult with complete analysis
        """
        logger.info("[RESEARCH_ORCHESTRATOR] Starting research pipeline")
        logger.info("[RESEARCH_ORCHESTRATOR] Input length: %d chars", len(text))
        
        # Step 1: Decompose into atomic claims
        logger.info("[RESEARCH_ORCHESTRATOR] STEP 1: Decomposing claim...")
        decomposed = self.decomposer.decompose(text)
        logger.info("[RESEARCH_ORCHESTRATOR] Decomposed into %d atomic claims", 
                   len(decomposed.atomic_claims))
        logger.info("[RESEARCH_ORCHESTRATOR] Logic structure: %s", 
                   decomposed.logic_structure)
        
        # Step 2: Verify each atomic claim
        atomic_verdicts = []
        for i, atomic_claim in enumerate(decomposed.atomic_claims, 1):
            logger.info("[RESEARCH_ORCHESTRATOR] Processing atomic claim %d/%d: %s",
                       i, len(decomposed.atomic_claims), atomic_claim[:80])
            
            try:
                verdict = self._verify_atomic_claim(atomic_claim, skip_evidence)
                atomic_verdicts.append(verdict)
                logger.info("[RESEARCH_ORCHESTRATOR] Atomic claim %d: %s (%.2f)",
                           i, verdict['verdict'], verdict['confidence'])
            except Exception as e:
                logger.error("[RESEARCH_ORCHESTRATOR] Error on atomic claim %d: %s", i, e)
                atomic_verdicts.append({
                    "claim": atomic_claim,
                    "verdict": "uncertain",
                    "confidence": 0.3,
                    "error": str(e)
                })
        
        # Step 3: Aggregate using logic structure
        logger.info("[RESEARCH_ORCHESTRATOR] STEP 3: Logic aggregation...")
        logic_result = self.logic_aggregator.aggregate(
            atomic_verdicts,
            decomposed.logic_structure
        )
        logger.info("[RESEARCH_ORCHESTRATOR] Logic aggregation: %s (%.2f)",
                   logic_result['verdict'], logic_result['confidence'])
        
        # Step 4: Generate final explanation
        logger.info("[RESEARCH_ORCHESTRATOR] STEP 4: Generating explanation...")
        try:
            # Use first atomic claim's evidence for explanation
            first_evidence = atomic_verdicts[0].get('evidence', []) if atomic_verdicts else []
            explanation = run_explainer_agent_sync(
                text,
                logic_result,
                first_evidence
            )
        except Exception as e:
            logger.error("[RESEARCH_ORCHESTRATOR] Explanation error: %s", e)
            explanation = logic_result.copy()
            explanation['summary'] = f"Verdict: {logic_result['verdict']}"
        
        # Metadata
        metadata = {
            "num_atomic_claims": len(decomposed.atomic_claims),
            "complexity_score": decomposed.complexity_score,
            "num_causal_edges": len(decomposed.causal_edges),
            "used_delphi_jury": self.use_delphi_jury,
            "avg_atomic_confidence": sum(v['confidence'] for v in atomic_verdicts) / len(atomic_verdicts) if atomic_verdicts else 0.0
        }
        
        if self.use_delphi_jury and atomic_verdicts:
            # Add jury-specific metadata
            jury_agreements = []
            for v in atomic_verdicts:
                if 'jury_verdicts' in v:
                    verdicts = [jv['verdict'] for jv in v['jury_verdicts']]
                    agreement = verdicts.count(v['verdict']) / len(verdicts)
                    jury_agreements.append(agreement)
            
            if jury_agreements:
                metadata['avg_jury_agreement'] = sum(jury_agreements) / len(jury_agreements)
        
        logger.info("[RESEARCH_ORCHESTRATOR] Pipeline complete")
        
        return ResearchTRUSTResult(
            original_text=text,
            decomposed_claim=decomposed,
            atomic_verdicts=atomic_verdicts,
            logic_aggregation=logic_result,
            final_verdict=explanation,
            metadata=metadata
        )
    
    def _verify_atomic_claim(
        self,
        claim: str,
        skip_evidence: bool = False
    ) -> Dict[str, Any]:
        """Verify a single atomic claim"""
        
        # Retrieve evidence
        if skip_evidence:
            evidence = []
        else:
            try:
                evidence = run_evidence_retrieval_agent_sync(claim, top_k=self.top_k_evidence)
            except Exception as e:
                logger.error(f"[RESEARCH_ORCHESTRATOR] Evidence retrieval failed: {e}")
                evidence = []
        
        # Verify with Delphi jury or single verifier
        if self.use_delphi_jury and evidence:
            verdict = self.delphi_jury.verify_with_jury(claim, evidence)
        else:
            # Fallback to simple verification
            from TRUST_agents.agents.verifier import run_verifier_agent_sync
            verdict = run_verifier_agent_sync(claim, evidence)
        
        # Add claim and evidence to result
        verdict['claim'] = claim
        verdict['evidence'] = evidence
        verdict['num_evidence'] = len(evidence)
        
        return verdict


def run_research_pipeline_sync(
    text: str,
    top_k_evidence: int = 10,
    skip_evidence: bool = False,
    use_delphi_jury: bool = True
) -> Dict[str, Any]:
    """
    Run research TRUST pipeline.
    
    Args:
        text: Input text
        top_k_evidence: Evidence passages per claim
        skip_evidence: Skip retrieval
        use_delphi_jury: Use multi-agent jury
        
    Returns:
        Dictionary with results
    """
    orchestrator = ResearchTRUSTOrchestrator(
        top_k_evidence=top_k_evidence,
        use_delphi_jury=use_delphi_jury
    )
    result = orchestrator.process_text(text, skip_evidence=skip_evidence)
    return asdict(result)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run research TRUST pipeline")
    parser.add_argument("--text", required=True, help="Text to fact-check")
    parser.add_argument("--top-k", type=int, default=10, help="Evidence per claim")
    parser.add_argument("--skip-evidence", action="store_true")
    parser.add_argument("--no-delphi", action="store_true", help="Disable Delphi jury")
    parser.add_argument("--output", help="Save to JSON")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    # Run pipeline
    result = run_research_pipeline_sync(
        text=args.text,
        top_k_evidence=args.top_k,
        skip_evidence=args.skip_evidence,
        use_delphi_jury=not args.no_delphi
    )
    
    # Print results
    print("\n" + "="*70)
    print("RESEARCH TRUST PIPELINE - RESULTS")
    print("="*70)
    print(f"\nOriginal Text: {args.text[:200]}...")
    print(f"\nAtomic Claims: {len(result['atomic_verdicts'])}")
    for i, v in enumerate(result['atomic_verdicts'], 1):
        print(f"  {i}. {v['claim'][:60]}... → {v['verdict']} ({v['confidence']:.2f})")
    
    print(f"\nLogic Structure: {result['decomposed_claim']['logic_structure']}")
    print(f"Final Verdict: {result['logic_aggregation']['verdict']} ({result['logic_aggregation']['confidence']:.2f})")
    
    print(f"\nMetadata:")
    for k, v in result['metadata'].items():
        print(f"  {k}: {v}")
    
    # Save if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n✓ Saved to {args.output}")
