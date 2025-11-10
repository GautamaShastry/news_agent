.PHONY: setup data.liar hf.bert hf.roberta eval.bert eval.roberta llm.run llm.eval compare api.dev api.run

setup:
	python -m pip install -e .

api.dev:
	uv run fastapi dev app.py

api.run:
	uv run fastapi run app.py

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
	OPENAI_API_KEY=$$OPENAI_API_KEY python scripts/baseline_llm.py --dataset liar --limit 200

llm.eval:
	python scripts/eval_llm.py --preds outputs/llm_baseline/liar_predictions.jsonl --split validation > outputs/llm_baseline/metrics_liar.json

compare:
	python scripts/compare_metrics.py --dataset liar --out results/baselines_liar.json
