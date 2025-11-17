"""
Claim Extractor Tools - Tools for NLP-based claim extraction.

Tools used by the Claim Extractor ReAct Agent:
- NER Tool: Named Entity Recognition
- Dependency Parsing Tool: Sentence structure analysis
- LLM Tool: Zero-shot reasoning

Each tool is self-contained, logs its actions, and returns status summaries.
"""

import os
import re
import json
import logging

import spacy
from dotenv import load_dotenv
from langchain_core.tools import tool

# OpenAI client only imported when needed
from openai import OpenAI

load_dotenv()
logger = logging.getLogger("TRUST_agents.agents.claim_extractor_tools")
logger.propagate = True  # let messages bubble to root handler configured by entry script

@tool()
async def ner_claim_extraction_tool(text: str) -> str:
    """
    Extract claims from text using Named Entity Recognition (NER).
    Identifies sentences containing named entities (people, organizations, locations, etc.).
    
    Args:
        text: Input text to extract claims from
        
    Returns:
        JSON string with claims list and method
    """
    logger.info(f"[DEBUG] ner_claim_extraction_tool called")
    
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        entity_types = ["PERSON", "ORG", "GPE", "LOC", "EVENT", "DATE", "MONEY"]
        claims = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_doc = nlp(sent_text)
            
            entities = [ent.text for ent in sent_doc.ents if ent.label_ in entity_types]
            
            if entities and _looks_like_claim(sent_doc):
                claims.append(sent_text)
        
        logger.info(f"ner_claim_extraction_tool completed: {len(claims)} claims found")
        result = {"claims": claims, "method": "ner", "ner_done": True}
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error during NER extraction: {e}")
        result = {"claims": [], "error": str(e)}
        return json.dumps(result)


@tool()
async def dependency_claim_extraction_tool(text: str) -> str:
    """
    Extract claims from text using dependency parsing.
    Identifies claim patterns: subject-verb-object structures with factual content.
    
    Args:
        text: Input text to extract claims from
        
    Returns:
        JSON string with claims list and method
    """
    logger.info(f"[DEBUG] dependency_claim_extraction_tool called")
    
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        claim_verbs = {
            # Statement verbs
            "say", "claim", "state", "report", "announce", "declare", "assert", "allege",
            "argue", "maintain", "contend", "insist", "affirm", "attest", "testify",
            # Disclosure verbs
            "reveal", "disclose", "admit", "acknowledge", "confess", "confirm", "deny",
            # Evidence verbs
            "show", "prove", "demonstrate", "indicate", "suggest", "imply", "reveal",
            # Observation verbs
            "note", "observe", "find", "discover", "detect", "notice",
            # Emphasis verbs
            "emphasize", "stress", "highlight", "underscore", "point out",
            # Prediction verbs
            "predict", "forecast", "warn", "caution", "anticipate", "expect",
            # Explanation verbs
            "explain", "describe", "characterize", "define", "specify",
            # Belief verbs
            "believe", "think", "consider", "regard", "view",
            # Estimation verbs
            "estimate", "calculate", "determine", "assess", "evaluate",
            # Being verbs (copula)
            "is", "are", "was", "were", "be", "been", "being",
            # Possession verbs
            "has", "have", "had", "contain", "include", "involve"
        }
        claims = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_doc = nlp(sent_text)
            
            has_claim_verb = any(token.lemma_.lower() in claim_verbs for token in sent_doc)
            has_subject = any(token.dep_ == "nsubj" for token in sent_doc)
            has_object = any(token.dep_ in ["dobj", "pobj", "attr"] for token in sent_doc)
            
            if has_claim_verb and has_subject and (has_object or len(sent_doc) > 5):
                if not sent_text.endswith("?") and _looks_like_claim(sent_doc):
                    claims.append(sent_text)
        
        logger.info(f"dependency_claim_extraction_tool completed: {len(claims)} claims found")
        result = {"claims": claims, "method": "dependency", "dependency_done": True}
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error during dependency parsing: {e}")
        result = {"claims": [], "error": str(e)}
        return json.dumps(result)


@tool()
async def llm_claim_extraction_tool(text: str) -> str:
    """
    Extract claims from text using LLM zero-shot reasoning.
    Uses GPT to identify and extract factual claims.
    
    Args:
        text: Input text to extract claims from
        
    Returns:
        JSON string with claims list and method
    """
    logger.info(f"[DEBUG] llm_claim_extraction_tool called")
    
    try:
        model = os.getenv("MODEL", "gpt-4.1-mini")
        llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""Analyze the following text and extract all distinct factual claims.

Text: {text}

Extract clear, verifiable claims that can be fact-checked.
Return ONLY a valid JSON array: [{{"claim_text": "..."}}, {{"claim_text": "..."}}]
No markdown, no additional text."""

        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Extract factual claims and return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_completion_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        content = re.sub(r'```json\n?', '', content)
        content = re.sub(r'```\n?', '', content)
        content = content.strip()
        
        parsed = json.loads(content)
        if isinstance(parsed, list):
            claims = []
            for item in parsed:
                if isinstance(item, dict) and 'claim_text' in item:
                    claims.append(item['claim_text'].strip())
                elif isinstance(item, str):
                    claims.append(item.strip())
            
            logger.info(f"llm_claim_extraction_tool completed: {len(claims)} claims found")
            result = {"claims": claims, "method": "llm", "llm_done": True}
            return json.dumps(result)
        
        logger.warning("Invalid LLM response format")
        result = {"claims": [], "error": "Invalid LLM response"}
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error during LLM extraction: {e}")
        result = {"claims": [], "error": str(e)}
        return json.dumps(result)


def _looks_like_claim(doc) -> bool:
    """Check if a sentence looks like a factual claim."""
    text = doc.text.strip()
    if text.endswith("?") or text.endswith("!"):
        return False
    if len(doc) < 3:
        return False
    return any(token.pos_ == "VERB" for token in doc)