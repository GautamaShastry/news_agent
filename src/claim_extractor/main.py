#!/usr/bin/env python
import warnings
from dotenv import load_dotenv
from claim_extractor.crew import ClaimExtractor

load_dotenv()

def run():
    """Simple interactive claim extractor - just type your claim text"""
    print("\nEnter claim text (or 'quit' to exit):")
    
    while True:
        claim_text = input("\n> ").strip()
        
        if claim_text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not claim_text:
            continue
        
        # Prepare inputs for crew
        inputs = {
            'claim_text': claim_text
        }
        
        try:
            # Run the crew
            result = ClaimExtractor().crew().kickoff(inputs=inputs)
            print(f"\nResult:\n{result}\n")
            
        except Exception as e:
            print(f"Error: {e}")
