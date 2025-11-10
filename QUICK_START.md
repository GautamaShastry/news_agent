# Quick Start Guide

## Prerequisites
1. Install dependencies: `make setup` or `pip install -e .`
2. Create `.env` file with: `OPENAI_API_KEY=your_key_here`

## Essential Commands

### Setup
```bash
make setup                    # Install all dependencies
```

### Dataset Preparation

**LIAR Dataset:**
```bash
make data.liar                # Download + prepare LIAR dataset
```

**FNN Dataset:**
```bash
# First: Edit scripts/download_fnn.py with Google Drive URLs
make data.fnn                 # Download + prepare FNN dataset
```

### Training & Evaluation

**Train Models:**
```bash
make hf.bert                  # Train BERT on LIAR
make hf.roberta               # Train RoBERTa on LIAR
```

**Evaluate Models:**
```bash
make eval.bert                # Evaluate BERT
make eval.roberta             # Evaluate RoBERTa
```

**LLM Baseline:**
```bash
make llm.run                  # Run GPT-4o-mini baseline
make llm.eval                 # Evaluate LLM baseline
```

**Compare All:**
```bash
make compare                  # Compare all baseline metrics
```

## Complete Workflow (LIAR Dataset)

```bash
# 1. Setup
make setup

# 2. Prepare data
make data.liar

# 3. Train models
make hf.bert
make hf.roberta

# 4. Evaluate models (one-step: model → metrics)
make eval.bert    # Runs inference and computes metrics directly
make eval.roberta # Runs inference and computes metrics directly

# 5. LLM baseline (two-step: run → eval)
make llm.run      # Step 1: Generate predictions
make llm.eval     # Step 2: Evaluate predictions

# 6. Claim extractor baseline (two-step: run → eval, same as LLM)
make claim.run    # Step 1: Generate predictions
make claim.eval   # Step 2: Evaluate predictions

# 7. Compare results
make compare
```

**Note on Evaluation Approaches:**
- **BERT/RoBERTa**: One-step evaluation (model → metrics) - faster, no intermediate file
- **LLM/Claim Extractor**: Two-step evaluation (run → eval) - allows re-evaluation without re-running

## Custom Commands

**Train on FNN:**
```bash
python scripts/train_hf.py --dataset fnn --model bert-base-uncased --out outputs/hf/bert_fnn
```

**Custom training:**
```bash
python scripts/train_hf.py --dataset liar --model bert-base-uncased --out outputs/hf/bert_liar --epochs 3 --bs 32
```

## Claim Extractor Baseline

**Two-Step Evaluation (same pattern as LLM baseline):**

```bash
# Step 1: Run claim extraction and save predictions
make claim.run                  # Generates: outputs/claim_extractor/liar_predictions.jsonl

# Step 2: Evaluate predictions and compute metrics
make claim.eval                 # Generates: outputs/claim_extractor/metrics_liar.json
```

**Note:** This follows the exact same pattern as LLM baseline:
- **Step 1 (`claim.run`)**: Uses `val` split (same as `llm.run`), saves to `{dataset}_predictions.jsonl`
- **Step 2 (`claim.eval`)**: Reads predictions file, outputs JSON to stdout (redirected to metrics file)
- **Metrics**: Match rate (like accuracy), has_claims_rate, avg_claims
- **Simple evaluation**: No semantic similarity, just like `eval_llm.py` - direct and simple

**Compare with Other Baselines:**
```bash
make compare                    # Compare all baselines (BERT, RoBERTa, GPT, Claim Extractor)
```

**Custom Usage:**
```bash
# Run on different dataset/split
python scripts/baseline_claim_extractor.py --dataset fnn --split val

# Evaluate predictions
python scripts/eval_claim_extractor.py --preds outputs/claim_extractor/fnn_predictions.jsonl --split validation > outputs/claim_extractor/metrics_fnn.json
```

## Output Locations

- **Processed Data**: `data/processed/*.jsonl`
- **Trained Models**: `outputs/hf/*/`
- **Metrics**: `outputs/hf/*_metrics.json`, `outputs/llm_baseline/metrics_*.json`, `outputs/claim_extractor/metrics_*.json`
- **Results**: `results/baselines_*.json`
- **Predictions**: `outputs/llm_baseline/*_predictions.jsonl`, `outputs/claim_extractor/*_predictions.jsonl`

For detailed instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

