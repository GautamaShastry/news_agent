import argparse, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--model_out", required=True)
ap.add_argument("--dataset", choices=["liar","fnn"], required=True)
ap.add_argument("--proc_dir", default="data/processed")
a = ap.parse_args()

ds = load_dataset("json", data_files={"test": f"{a.proc_dir}/{a.dataset}_test.jsonl"})
tok = AutoTokenizer.from_pretrained(a.model_out)
model = AutoModelForSequenceClassification.from_pretrained(a.model_out)
model.eval()  # Set to evaluation mode

def enc(ex): 
    return tok(ex["text"], truncation=True, padding="max_length", max_length=256)

te = ds["test"].map(enc, batched=True).rename_column("label","labels")

# Convert labels to integers (if they're strings like "true"/"false")
if isinstance(te[0]["labels"], str):
    label_map = {"false": 0, "true": 1}  # Adjust based on your label format
    te = te.map(lambda x: {"labels": label_map[x["labels"]]})

# Set format for PyTorch
te.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Create DataLoader for batching
dataloader = DataLoader(te, batch_size=64)

logits = []
gold = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for batch in dataloader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits.append(outputs.logits.detach().cpu().numpy())
    gold.append(labels.numpy())

logits = np.concatenate(logits)
gold = np.concatenate(gold)
preds = logits.argmax(axis=1)

print(json.dumps({
    "accuracy": float(accuracy_score(gold, preds)), 
    "f1": float(f1_score(gold, preds, average='binary'))
}, indent=2))