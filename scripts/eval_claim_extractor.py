import argparse
import json
from sklearn.metrics import accuracy_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--split", default="validation")
a = ap.parse_args()

original_texts = []
extracted_claims_list = []

with open(a.preds, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            o = json.loads(line)
            original_texts.append(o["text"])
            extracted_claims_list.append(o.get("extracted_claims", []))

def normalize(text: str) -> str:
    return text.lower().strip()

def has_matching_claim(extracted_claims: list, original_text: str) -> bool:
    if not extracted_claims:
        return False
    original_norm = normalize(original_text)
    for claim in extracted_claims:
        claim_norm = normalize(claim)
        if claim_norm == original_norm:
            return True
        if original_norm in claim_norm or claim_norm in original_norm:
            return True
    return False

preds = []
gold = []

for extracted_claims, original_text in zip(extracted_claims_list, original_texts):
    matched = has_matching_claim(extracted_claims, original_text)
    preds.append(1 if matched else 0)
    gold.append(1)

acc = float(accuracy_score(gold, preds))
f1 = float(f1_score(gold, preds))
print(json.dumps({"split": a.split, "accuracy": acc, "f1": f1}, indent=2))

