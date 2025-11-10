"""Pydantic models for claim extraction endpoints"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ClaimTextRequest(BaseModel):
    """Request model for claim extraction"""
    claim_text: str = Field(..., description="Text containing claims to extract", min_length=1)
    

class ClaimItem(BaseModel):
    """Individual claim item"""
    claim_text: str = Field(..., description="The extracted claim statement")


class ClaimExtractionResponse(BaseModel):
    """Response model for claim extraction"""
    claims: List[ClaimItem] = Field(..., description="List of extracted claims")
    raw_output: Optional[str] = Field(None, description="Raw output from the crew")

