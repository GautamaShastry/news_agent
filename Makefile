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

# ============================================================
# TRUST AGENTS: FULL MULTI-AGENT SYSTEM
# ============================================================

trust.setup:
	@echo "Setting up TRUST Agents evidence corpus..."
	python scripts/prepare_evidence_corpus.py --dataset data/processed/liar_train.jsonl --max-docs 1000
	@echo "Indexing evidence corpus (this may take a few minutes)..."
	python scripts/index_evidence.py --corpus data/evidence_corpus
	@echo "✓ TRUST Agents setup complete!"

trust.test:
	@echo "Running TRUST Agents in test mode (no evidence retrieval)..."
	python scripts/run_trust_agents.py --dataset liar --split val --limit 10 --skip-evidence

trust.run:
	@echo "Running TRUST Agents on validation set..."
	python scripts/run_trust_agents.py --dataset liar --split val --limit 200 --top-k 5

trust.run.full:
	@echo "Running TRUST Agents on FULL validation set (this will take a while)..."
	python scripts/run_trust_agents.py --dataset liar --split val --top-k 5

trust.eval:
	python scripts/eval_trust_agents.py --preds outputs/trust_agents/liar_predictions.jsonl --dataset liar --split val --output outputs/trust_agents/metrics_liar.json

trust.pipeline:
	@echo "Running complete TRUST agents pipeline..."
	$(MAKE) trust.run
	@echo ""
	$(MAKE) trust.eval

compare.all:
	@echo "Comparing all methods..."
	python scripts/compare_all_metrics.py --dataset liar --out results/all_methods_liar.json

clean.outputs:
	rm -rf outputs/

clean.index:
	rm -rf retrieval_index/

clean.all:
	rm -rf outputs/ retrieval_index/ data/evidence_corpus/

help:
	@echo "TRUST Agents - Makefile Commands"
	@echo "=================================="
	@echo ""
	@echo "TRUST AGENTS:"
	@echo "  make trust.setup    - Setup evidence corpus and index"
	@echo "  make trust.test     - Quick test (10 examples)"
	@echo "  make trust.run      - Run TRUST (200 examples)"
	@echo "  make trust.eval     - Evaluate TRUST predictions"
	@echo "  make trust.pipeline - Complete pipeline"
	@echo "  make compare.all    - Compare all methods"
	@echo "  make help           - Show this help"