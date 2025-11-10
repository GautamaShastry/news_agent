"""
Claim Extractor using NLP techniques: NER, dependency parsing, and zero-shot reasoning.

This module implements claim extraction using:
1. Named Entity Recognition (NER) - identifies entities that might be part of claims
2. Dependency Parsing - analyzes sentence structure to identify claim patterns
3. Zero-shot reasoning (LLM) - refines and validates extracted claims
"""

import json
import re
import os
from typing import List
from dataclasses import dataclass

import spacy
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Claim:
    """Represents an extracted claim"""
    text: str
    entities: List[str]
    method: str


class ClaimExtractor:
    """
    Claim extractor using NLP techniques: NER, dependency parsing, and zero-shot reasoning.
    
    Uses hybrid approach combining all three techniques.
    """
    
    def __init__(self, use_ner: bool = True, use_dependency: bool = True, use_llm: bool = True):
        """Initialize the claim extractor."""
        self.use_ner = use_ner
        self.use_dependency = use_dependency
        self.use_llm = use_llm
        
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize OpenAI client
        self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract(self, text: str) -> List[str]:
        """
        Extract claims from text using hybrid approach (NER + dependency parsing + LLM).
        
        Args:
            text: Input text to extract claims from
        
        Returns:
            List of extracted claim texts
        """
        return self._extract_hybrid(text)
    
    def _extract_hybrid(self, text: str) -> List[str]:
        """Extract claims using all available methods and merge results."""
        all_claims = []
        
        if self.use_ner:
            all_claims.extend(self.extract_with_ner(text))
        if self.use_dependency:
            all_claims.extend(self.extract_with_dependency(text))
        if self.use_llm:
            all_claims.extend(self.extract_with_llm(text))
        
        return self._deduplicate_claims(all_claims)
    
    def extract_with_ner(self, text: str) -> List[Claim]:
        """Extract claims using Named Entity Recognition."""
        claims = []
        doc = self.nlp(text)
        
        # Relevant entity types for claims
        entity_types = ["PERSON", "ORG", "GPE", "LOC", "EVENT", "DATE", "MONEY"]
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_doc = self.nlp(sent_text)
            
            # Find named entities
            entities = [ent.text for ent in sent_doc.ents if ent.label_ in entity_types]
            
            # Extract if has entities and looks like a claim
            if entities and self._looks_like_claim(sent_doc):
                claims.append(Claim(
                    text=sent_text,
                    entities=entities,
                    method="ner"
                ))
        
        return claims
    
    def extract_with_dependency(self, text: str) -> List[Claim]:
        """Extract claims using dependency parsing."""
        claims = []
        doc = self.nlp(text)
        
        # Verbs that indicate factual statements
        claim_verbs = {"say", "claim", "state", "report", "announce", "declare", 
                      "is", "are", "was", "were", "has", "have", "had"}
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_doc = self.nlp(sent_text)
            
            # Check for claim patterns: subject + verb + object
            has_claim_verb = any(token.lemma_.lower() in claim_verbs for token in sent_doc)
            has_subject = any(token.dep_ == "nsubj" for token in sent_doc)
            has_object = any(token.dep_ in ["dobj", "pobj", "attr"] for token in sent_doc)
            
            # Extract if matches pattern and looks like a claim
            if has_claim_verb and has_subject and (has_object or len(sent_doc) > 5):
                if not sent_text.endswith("?") and self._looks_like_claim(sent_doc):
                    claims.append(Claim(
                        text=sent_text,
                        entities=[ent.text for ent in sent_doc.ents],
                        method="dependency"
                    ))
        
        return claims
    
    def extract_with_llm(self, text: str) -> List[Claim]:
        """Extract claims using LLM zero-shot reasoning."""
        prompt = f"""Analyze the following text and extract all distinct factual claims.

Text: {text}

Extract clear, verifiable claims that can be fact-checked.
Return ONLY a valid JSON array: [{{"claim_text": "..."}}, {{"claim_text": "..."}}]
No markdown, no additional text."""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract factual claims and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            claim_texts = self._parse_llm_response(content)
            
            return [Claim(
                text=claim_text,
                entities=[],
                method="llm"
            ) for claim_text in claim_texts]
            
        except Exception as e:
            print(f"Error in LLM extraction: {e}")
            return []
    
    def _parse_llm_response(self, content: str) -> List[str]:
        """Parse claims from LLM JSON response."""
        # Remove markdown code blocks if present
        content = re.sub(r'```json\n?', '', content)
        content = re.sub(r'```\n?', '', content)
        content = content.strip()
        
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                return []
            
            claims = []
            for item in parsed:
                if isinstance(item, dict) and 'claim_text' in item:
                    claims.append(item['claim_text'].strip())
                elif isinstance(item, str):
                    claims.append(item.strip())
            
            return [c for c in claims if c]
            
        except json.JSONDecodeError:
            return []
    
    def _looks_like_claim(self, doc) -> bool:
        """Check if a sentence looks like a factual claim."""
        text = doc.text.strip()
        
        # Filter out questions and exclamations
        if text.endswith("?") or text.endswith("!"):
            return False
        
        # Too short
        if len(doc) < 3:
            return False
        
        # Should have at least one verb
        return any(token.pos_ == "VERB" for token in doc)
    
    def _deduplicate_claims(self, claims: List[Claim]) -> List[str]:
        """Remove duplicate claims."""
        seen = set()
        unique = []
        
        for claim in claims:
            claim_norm = claim.text.lower().strip()
            if claim_norm not in seen:
                unique.append(claim.text)
                seen.add(claim_norm)
        
        return unique
