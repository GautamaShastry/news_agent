"""
Baseline claim extractor - runs claim extraction on dataset and saves predictions.
Uses ReAct Agent with NLP techniques:
- Named Entity Recognition (NER)
- Dependency Parsing  
- Zero-shot reasoning (LLM)

Follows the same pattern as baseline_llm.py
"""

import argparse
import json
import os
import logging
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv

load_dotenv()

# Configure logging ONCE at entry point (force=True overrides any prior configs)
os.environ["PYTHONUNBUFFERED"] = "1"  # flush stdout/stderr immediately

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True  # override any prior configs from imports
)

# Suppress noisy third-party logs
for noisy in ["httpcore", "httpx", "openai", "langchain", "langgraph", "asyncio"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Import agent AFTER logging is configured
from TRUST_agents.agents.claim_extractor import run_claim_extractor_agent_sync

@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(5))
def extract_claims(text: str) -> list:
    """
    Extract claims from text using ReAct Agent with NLP techniques.
    
    The agent uses:
    - Named Entity Recognition (NER)
    - Dependency Parsing
    - Zero-shot reasoning (LLM)
    
    Args:
        text: Input text to extract claims from
    
    Returns:
        List of extracted claim texts
    """
    try:
        claims = run_claim_extractor_agent_sync(text)
        return claims if claims else []
    except Exception as e:
        print(f"Error extracting claims: {e}")
        return []

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run claim extraction baseline using ReAct Agent with NLP techniques (NER + dependency parsing + zero-shot reasoning)"
    )
    ap.add_argument("--dataset", choices=["liar", "fnn"], required=True)
    ap.add_argument("--proc_dir", default="data/processed")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    
    # Check for required dependencies
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set. Required for LLM-based extraction. Make sure it's in your .env file")
    
    split_file = f"{a.dataset}_{a.split}.jsonl"
    inp = Path(a.proc_dir) / split_file
    # Match baseline_llm.py output pattern: {dataset}_predictions.jsonl (no split in filename)
    out = Path(a.out or f"outputs/claim_extractor/{a.dataset}_predictions.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    
    n = 0
    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as w:
        for line in f:
            if a.limit and n >= a.limit:
                break
            ex = json.loads(line)
            try:
                claims = extract_claims(ex["text"])
                rec = {
                    "id": ex["id"],
                    "text": ex["text"],
                    "gold": ex.get("label", "unknown"),
                    "extracted_claims": claims,
                    "num_claims": len(claims)
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 10 == 0:
                    print(f"Processed {n} examples...")
            except Exception as e:
                print(f"Error on example {n}: {e}")
                continue
    print(f"Saved {n} predictions to {out}")

