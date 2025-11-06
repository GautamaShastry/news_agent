import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def load_ds(proc_dir="data/processed", dataset="liar"):
    files = {"train": f"{proc_dir}/{dataset}_train.jsonl", "validation": f"{proc_dir}/{dataset}_val.jsonl"}
    return load_dataset("json", data_files=files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["liar","fnn"], required=True)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()

    ds = load_ds(dataset=a.dataset)
    tok = AutoTokenizer.from_pretrained(a.model)
    def enc(ex): return tok(ex["text"], truncation=True, padding="max_length", max_length=256)
    ds = ds.map(enc, batched=True).rename_column("label","labels").class_encode_column("labels")

    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2, id2label={0:"false",1:"true"})
    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids,preds), "f1": f1_score(p.label_ids,preds)}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=a.out, num_train_epochs=a.epochs,
            per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="f1", report_to=[]
        ),
        train_dataset=ds["train"], eval_dataset=ds["validation"], tokenizer=tok, compute_metrics=metrics
    )
    trainer.train()
    trainer.save_model(a.out)

if __name__ == "__main__":
    main()
