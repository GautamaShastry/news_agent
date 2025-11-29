# -*- coding: utf-8 -*-
"""
TRUST Agents 2.0 - Research-Grade Components

This module contains cutting-edge research components:
- Decomposer Agent (LoCal-inspired)
- Logic Aggregator
- Delphi Jury (Multi-agent system)

Based on 2024-2025 research papers.
"""

from .decomposer_agent import DecomposerAgent, decompose_claim_sync
from .logic_aggregator import LogicAggregator
from .delphi_jury import DelphiJury, verify_with_delphi_jury_sync

__all__ = [
    'DecomposerAgent',
    'decompose_claim_sync',
    'LogicAggregator',
    'DelphiJury',
    'verify_with_delphi_jury_sync',
]
