import argparse, json
from sklearn.metrics import accuracy_score, f1_score

ap=argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--split", default="validation")
a=ap.parse_args()

preds=[]; gold=[]
with open(a.preds,"r",encoding="utf-8") as f:
    for line in f:
        o=json.loads(line)
        preds.append(1 if o["pred"]=="true" else 0)
        gold.append(1 if o["gold"]=="true" else 0)

acc=float(accuracy_score(gold,preds)); f1=float(f1_score(gold,preds))
print(json.dumps({"split":a.split,"accuracy":acc,"f1":f1}, indent=2))
