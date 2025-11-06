import argparse, json, os
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SCHEMA = {
  "name":"binary_schema","strict":True,
  "schema":{
    "type":"object",
    "properties":{
      "label":{"type":"string","enum":["true","false"]},
      "confidence":{"type":"number","minimum":0,"maximum":1},
      "rationale":{"type":"string"}
    },
    "required":["label","confidence","rationale"],
    "additionalProperties":False
  }
}
PROMPT = (
  "You are a fact-consistency rater.\n"
  "Given a short political/news statement, label it 'true' or 'false' based on plausibility.\n"
  "Respond ONLY as JSON with keys: label, confidence, rationale (<=40 words).\n"
)

@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(5))
def call_llm(client, model, text):
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":PROMPT},
            {"role":"user","content":text}
        ],
        response_format={"type":"json_schema","json_schema":SCHEMA},
        temperature=0.2,
        max_tokens=150
    )
    return json.loads(r.choices[0].message.content)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["liar","fnn"], required=True)
    ap.add_argument("--proc_dir", default="data/processed")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model", default=os.getenv("TRUST_MODEL","gpt-4o-mini"))
    ap.add_argument("--out", default=None)
    a=ap.parse_args()

    key=os.getenv("OPENAI_API_KEY")
    if not key: 
        raise SystemExit("OPENAI_API_KEY not set. Make sure it's in your .env file")
    client=OpenAI(api_key=key)

    split = f"{a.dataset}_val.jsonl"
    inp = Path(a.proc_dir) / split
    out = Path(a.out or f"outputs/llm_baseline/{a.dataset}_predictions.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n=0
    with open(inp,"r",encoding="utf-8") as f, open(out,"w",encoding="utf-8") as w:
        for line in f:
            if a.limit and n>=a.limit: break
            ex=json.loads(line)
            try:
                ans=call_llm(client,a.model,ex["text"])
                rec={"id":ex["id"],"text":ex["text"],"gold":ex["label"],"pred":ans["label"],"confidence":ans.get("confidence")}
                w.write(json.dumps(rec,ensure_ascii=False)+"\n")
                n+=1
                if n % 10 == 0:
                    print(f"Processed {n} examples...")
            except Exception as e:
                print(f"Error on example {n}: {e}")
                continue
    print(f"Saved {n} predictions to {out}")