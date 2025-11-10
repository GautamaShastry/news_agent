.PHONY: setup data.liar hf.bert hf.roberta eval.bert eval.roberta llm.run llm.eval compare claim.run claim.eval

setup:
	python -m pip install -e .
	python -m spacy download en_core_web_sm || echo "Note: spaCy model download failed. Run manually: python -m spacy download en_core_web_sm"

data.liar:
	python scripts/download_liar.py && python scripts/prepare_liar.py

data.fnn:
	python scripts/download_fnn.py && python scripts/prepare_fnn.py

hf.bert:
	python scripts/train_hf.py --dataset liar --model bert-base-uncased --out outputs/hf/bert_liar

eval.bert:
	python scripts/eval_hf.py --model_out outputs/hf/bert_liar --dataset liar > outputs/hf/bert_liar_metrics.json

hf.roberta:
	python scripts/train_hf.py --dataset liar --model roberta-base --out outputs/hf/roberta_liar

eval.roberta:
	python scripts/eval_hf.py --model_out outputs/hf/roberta_liar --dataset liar > outputs/hf/roberta_liar_metrics.json

llm.run:
	python scripts/baseline_llm.py --dataset liar --limit 200

llm.eval:
	python scripts/eval_llm.py --preds outputs/llm_baseline/liar_predictions.jsonl --split validation > outputs/llm_baseline/metrics_liar.json

compare:
	python scripts/compare_metrics.py --dataset liar --out results/baselines_liar.json

# Claim extraction baseline (follows same pattern as llm baseline)
claim.run:
	python scripts/baseline_claim_extractor.py --dataset liar --limit 200

claim.eval:
	python scripts/eval_claim_extractor.py --preds outputs/claim_extractor/liar_predictions.jsonl --split validation > outputs/claim_extractor/metrics_liar.json
