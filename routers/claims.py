"""Router for claim extraction endpoints"""

from fastapi import APIRouter, HTTPException
import os
import json
import re

from claim_extractor.crew import ClaimExtractor
from schemas.claims import ClaimTextRequest, ClaimItem, ClaimExtractionResponse

# Create router (no app instance here - app is created in app.py)
router = APIRouter(prefix="/api/v1", tags=["Claim Extractor"])


@router.post("/extract", response_model=ClaimExtractionResponse)
async def extract_claims(request: ClaimTextRequest):
    """
    Extract claims from the provided text.
    
    This endpoint uses crewAI to analyze the text and extract distinct factual claims
    that can be fact-checked and verified.
    """
    try:
        # Check if OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY not set. Please set it in your .env file."
            )
        
        # Prepare inputs for crew
        inputs = {
            'claim_text': request.claim_text
        }
        
        # Run the crew
        crew_instance = ClaimExtractor()
        result = crew_instance.crew().kickoff(inputs=inputs)
        
        # Parse the result
        # The result might be a string or CrewOutput object
        raw_output = str(result)
        
        # Try to extract claims from the result
        # This is a simplified version - you may need to adjust based on actual output format
        claims = []
        try:
            # Try to parse as JSON if the output is JSON
            if raw_output.strip().startswith('['):
                parsed = json.loads(raw_output)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and 'claim_text' in item:
                            claims.append(ClaimItem(claim_text=item['claim_text']))
                        elif isinstance(item, str):
                            claims.append(ClaimItem(claim_text=item))
        except json.JSONDecodeError:
            # If not JSON, try to extract claims from text
            # Look for JSON-like patterns in the output
            json_pattern = r'\[.*?\]'
            matches = re.findall(json_pattern, raw_output, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and 'claim_text' in item:
                                claims.append(ClaimItem(claim_text=item['claim_text']))
                            elif isinstance(item, str):
                                claims.append(ClaimItem(claim_text=item))
                except json.JSONDecodeError:
                    continue
        
        # If no claims were extracted, return the raw output as a single claim
        if not claims:
            # Fallback: return the original text as a claim
            claims.append(ClaimItem(claim_text=request.claim_text))
        
        return ClaimExtractionResponse(
            claims=claims,
            raw_output=raw_output
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting claims: {str(e)}"
        )
