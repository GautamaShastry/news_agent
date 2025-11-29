import json
import re
import logging
from typing import Any, Union, Dict, List

logger = logging.getLogger("TRUST.Utils")

def clean_and_parse_json(text: str) -> Union[Dict, List, None]:
    """
    Robustly extracts and parses JSON from LLM output, handling markdown fences,
    conversational filler, and common syntax errors.
    
    Args:
        text: Raw text from LLM that may contain JSON
        
    Returns:
        Parsed JSON object (dict or list) or None if parsing fails
    """
    if not text:
        return None
    
    # 1. Remove Markdown Code Blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # 2. Try Direct Parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 3. Heuristic Extraction: Find outermost { } or [ ]
    try:
        # Look for object pattern
        match_obj = re.search(r'(\{.*\})', text, re.DOTALL)
        if match_obj:
            return json.loads(match_obj.group(1))
        
        # Look for list pattern
        match_list = re.search(r'(\[.*\])', text, re.DOTALL)
        if match_list:
            return json.loads(match_list.group(1))
    except json.JSONDecodeError:
        pass
    
    # 4. Last Resort: Python Literal Eval (handle single quotes)
    try:
        import ast
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        logger.error(f"Failed to parse JSON. Raw text snippet: {text[:100]}...")
        return None