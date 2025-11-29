# How to Run TRUST Agents

Complete guide for running the fact-checking pipelines.

## Prerequisites

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
python -m spacy download en_core_web_sm
```

### 2. API Configuration
Create `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key
MODEL=gpt-4.1-mini
```

**Model Options:**
| Model | Cost | Notes |
|-------|------|-------|
| gpt-4.1-nano | Very low | High uncertainty rate |
| gpt-4.1-mini | Low | Used in our experiments |
| gpt-4o-mini | Medium | Recommended for better results |
| gpt-4o | High | Best accuracy |

### 3. Data Preparation
```bash
make data.liar
# or manually:
python scripts/download_liar.py
python scripts/prepare_liar.py
```

## Running the Pipelines

### Original TRUST Pipeline
```bash
# Quick test (10 examples)
python scripts/run_trust_agents.py --dataset liar --split val --limit 10 --skip-evidence

# Full run (200 examples)
python scripts/run_trust_agents.py --dataset liar --split val --limit 200 --skip-evidence
```

### Research Pipeline (LoCal + Delphi)
```bash
# Quick test
python scripts/run_trust_research.py --dataset liar --split val --limit 10 --skip-evidence

# Full run
python scripts/run_trust_research.py --dataset liar --split val --limit 200 --skip-evidence

# Without Delphi jury (faster)
python scripts/run_trust_research.py --dataset liar --split val --limit 200 --skip-evidence --no-delphi
```

### With Evidence Retrieval
```bash
# Setup evidence corpus (one-time)
make trust.setup

# Run with evidence
python scripts/run_trust_research.py --dataset liar --split val --limit 200 --top-k 10
```

## Evaluation

### Basic Evaluation
```bash
python scripts/evaluate_predictions.py --preds outputs/trust_research/liar_predictions.jsonl
```

### Handling Uncertain Predictions
The pipeline outputs three classes (true/false/uncertain) but gold labels are binary.

```bash
# Map uncertain → false (default)
python scripts/evaluate_predictions.py --preds outputs/trust_research/liar_predictions.jsonl --uncertain-as false

# Map uncertain → true
python scripts/evaluate_predictions.py --preds outputs/trust_research/liar_predictions.jsonl --uncertain-as true

# Drop uncertain predictions
python scripts/evaluate_predictions.py --preds outputs/trust_research/liar_predictions.jsonl --uncertain-as drop
```

### Understanding the Output
```
Total predictions: 200
Evaluated: 200
Dropped (uncertain): 0

--- Metrics (uncertain → false) ---
Accuracy: 49.50%
F1 (macro): 0.3632

--- Distribution ---
Gold: {'false': 100, 'true': 100}
Pred: {'uncertain': 165, 'false': 26, 'true': 9}
```

Key metrics:
- **Accuracy**: Overall correctness
- **F1 (macro)**: Balanced F1 across classes
- **Pred distribution**: Shows how many uncertain predictions

## Makefile Commands

```bash
# Data
make data.liar              # Download and prepare LIAR dataset

# Original Pipeline
make trust.test             # Quick test (10 examples)
make trust.run              # Run on 200 examples
make trust.eval             # Evaluate predictions

# Research Pipeline
make research.test          # Test components
make research.run           # Run on 200 examples
make research.eval          # Evaluate predictions

# Baselines
make hf.bert                # Train BERT
make hf.roberta             # Train RoBERTa
make eval.bert              # Evaluate BERT
make eval.roberta           # Evaluate RoBERTa
```

## Command-Line Options

### run_trust_research.py
| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset (liar/fnn) | Required |
| `--split` | Data split (train/val/test) | val |
| `--limit` | Max examples | None |
| `--top-k` | Evidence passages per claim | 10 |
| `--skip-evidence` | Skip evidence retrieval | False |
| `--no-delphi` | Disable multi-agent jury | False |
| `--out` | Output file path | Auto |

### evaluate_predictions.py
| Option | Description | Default |
|--------|-------------|---------|
| `--preds` | Predictions file | Required |
| `--uncertain-as` | How to handle uncertain (true/false/drop) | false |
| `--output` | Save metrics to file | None |

## Output Files

```
outputs/
├── trust_agents/
│   └── liar_predictions.jsonl    # Original pipeline predictions
├── trust_research/
│   ├── liar_predictions.jsonl    # Research pipeline predictions
│   └── metrics_liar.json         # Evaluation metrics
└── hf/
    ├── bert_liar/                # Fine-tuned BERT
    └── roberta_liar/             # Fine-tuned RoBERTa
```

## Prediction Format

```json
{
  "id": "example_123",
  "text": "Biden won 2020 and became president",
  "gold": "true",
  "pred": "uncertain",
  "confidence": 0.45,
  "num_atomic_claims": 2,
  "logic_structure": "C1 AND C2",
  "atomic_verdicts": [
    {"claim": "Biden won 2020", "verdict": "true", "confidence": 0.8},
    {"claim": "Biden became president", "verdict": "uncertain", "confidence": 0.4}
  ]
}
```

## Troubleshooting

### High Uncertainty Rate
If most predictions are "uncertain":
1. Use a better model (gpt-4o-mini or gpt-4o)
2. Enable evidence retrieval (remove --skip-evidence)
3. Check prompt formatting in agent files

### Low Accuracy
1. Check prediction distribution with evaluate_predictions.py
2. Try different --uncertain-as options
3. Compare with baseline models

### API Errors
1. Verify OPENAI_API_KEY in .env
2. Check rate limits
3. Reduce --limit for testing

### Memory Issues
1. Reduce --limit
2. Use --no-delphi for single-agent mode
3. Reduce --top-k for evidence

## Experimental Results Summary

### Our Setup
- Model: GPT-4.1-mini
- Dataset: LIAR validation (200 examples)
- Evidence: Skipped (--skip-evidence)

### Results
| Method | Accuracy |
|--------|----------|
| BERT (fine-tuned) | 65.2% |
| RoBERTa (fine-tuned) | 64.1% |
| GPT-4.1-nano baseline | 58.0% |
| Original Pipeline | ~19% |
| Research Pipeline | ~50% |

### Key Finding
82.5% of research pipeline predictions were "uncertain", indicating the model's conservative behavior with complex multi-step reasoning tasks.

## Next Steps

To improve results:
1. **Better model**: Use gpt-4o-mini or gpt-4o
2. **Evidence**: Enable evidence retrieval
3. **Prompt tuning**: Reduce uncertainty rate
4. **Hybrid approach**: Combine fine-tuned + LLM
