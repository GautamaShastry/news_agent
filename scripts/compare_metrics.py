import argparse, json
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["liar","fnn"], required=True)
ap.add_argument("--out", required=True)
args=ap.parse_args()

out = {"dataset": args.dataset, "baselines": {}}

def maybe_read(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

# HF
b = maybe_read(f"outputs/hf/bert_{args.dataset}_metrics.json")
if b: out["baselines"]["bert"] = b
r = maybe_read(f"outputs/hf/roberta_{args.dataset}_metrics.json")
if r: out["baselines"]["roberta"] = r

# GPT
g = maybe_read(f"outputs/llm_baseline/metrics_{args.dataset}.json")
if g: out["baselines"]["gpt4.1-nano"] = g

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
