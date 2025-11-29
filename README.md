# TRUST Agents - Multi-Agent Fact-Checking System

A research-grade multi-agent system for automated fact-checking using Large Language Models (LLMs).

## Project Overview

This project explores multi-agent architectures for fact-checking, implementing both a baseline pipeline and an advanced research pipeline based on recent academic papers. The goal is to investigate whether decomposing claims, using multiple verification agents, and applying logical reasoning can improve fact-checking accuracy.

## What We Built

### 1. Baseline Pipeline (`agents/`)
A standard fact-checking pipeline with:
- **Claim Extraction**: ReAct-based extraction using NER, dependency parsing, and LLM
- **Evidence Retrieval**: Hybrid search combining BM25 and dense embeddings (FAISS)
- **Verification**: Single-agent LLM verification
- **Explanation**: Human-readable explanation generation

### 2. Research Pipeline (`agents2.0/`)
An advanced pipeline implementing cutting-edge techniques:
- **LoCal-style Decomposition**: Breaking complex claims into atomic sub-claims with logical structure (AND/OR/IMPLIES)
- **Delphi Jury**: Multi-agent verification with multiple personas and trust-weighted voting
- **Logic Aggregation**: Combining atomic verdicts using logical formulas

## Research Methodology

### LoCal-Inspired Claim Decomposition
Based on the LoCal paper (ACM 2024), we decompose complex claims into:
- **Atomic claims**: Individual verifiable facts
- **Logical structure**: How claims relate (e.g., "C1 AND C2")
- **Causal edges**: Cause-effect relationships

Example:
```
Input: "Biden won the 2020 election and became president in 2021"
Output:
  - C1: "Biden won the 2020 election"
  - C2: "Biden became president in 2021"
  - Logic: "C1 AND C2"
```

### Delphi Multi-Agent Jury
Inspired by the Delphi method for consensus-building:
- Multiple AI personas independently verify claims
- Each persona has a different perspective (Strict, Balanced)
- Verdicts are aggregated using confidence-weighted voting

### Logic Aggregation
Reconstructs the truth value of complex claims:
- Evaluates logical formulas (AND, OR, IMPLIES)
- Handles uncertain atomic verdicts appropriately
- Falls back to majority voting when logic evaluation fails

## Experimental Results

### Baseline Models (Fine-tuned)
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| BERT | 65.2% | 0.726 |
| RoBERTa | 64.1% | 0.726 |
| GPT-4.1-nano (LLM baseline) | 58.0% | 0.528 |

### Agent Pipelines (GPT-4.1-mini)
| Pipeline | Accuracy | Notes |
|----------|----------|-------|
| Original TRUST Pipeline | ~19% | High uncertain rate |
| Research Pipeline (LoCal + Delphi) | ~50% | 82.5% uncertain predictions |

## Analysis: Why Agent Pipelines Underperformed

The agent-based pipelines showed lower accuracy compared to fine-tuned baselines. This is attributable to several factors:

### 1. Model Constraints
- **Token Limitations**: GPT-4.1-mini has strict token limits, requiring aggressive prompt truncation
- **Conservative Behavior**: Smaller models tend to output "uncertain" when confidence is low
- **Cost-Accuracy Tradeoff**: Using more capable models (GPT-4o, GPT-4-turbo) would likely improve results but at higher cost

### 2. Three-Class vs Binary Classification
- **Gold labels**: Binary (true/false)
- **Predictions**: Three-class (true/false/uncertain)
- **Impact**: "Uncertain" predictions (82.5% of outputs) are counted as incorrect, severely impacting accuracy

### 3. Pipeline Complexity
- Multi-step pipelines (decompose → retrieve → verify → aggregate) compound errors
- Each LLM call introduces potential for mistakes
- More complex architectures require more capable models to function effectively

### 4. Evidence Retrieval Challenges
- Running without evidence (`--skip-evidence`) forces the model to rely solely on parametric knowledge
- Evidence quality and relevance significantly impact verification accuracy

### 5. Prompt Engineering Sensitivity
- LLM outputs are highly sensitive to prompt wording
- Balancing between decisive predictions and appropriate uncertainty is challenging
- Smaller models struggle to follow complex multi-step instructions

## Key Insights

1. **Architecture vs Model Capability**: Sophisticated architectures (LoCal, Delphi) require capable models to realize their potential
2. **Uncertainty Handling**: The three-class output space (true/false/uncertain) needs careful handling during evaluation
3. **Fine-tuning Advantage**: Task-specific fine-tuned models (BERT, RoBERTa) outperform general-purpose LLMs on structured classification
4. **Cost-Performance Tradeoff**: Better models (GPT-4o) would likely improve agent pipeline performance but at 10-20x the cost

## Project Structure

```
news_agent/
├── TRUST_agents/
│   ├── agents/                    # Original pipeline agents
│   │   ├── claim_extractor.py
│   │   ├── evidence_retrieval.py
│   │   ├── verifier.py
│   │   └── explainer.py
│   ├── agents2.0/                 # Research pipeline agents
│   │   ├── decomposer_agent.py    # LoCal-style decomposition
│   │   ├── delphi_jury.py         # Multi-agent jury
│   │   └── logic_aggregator.py    # Logical aggregation
│   ├── orchestrator.py            # Original pipeline
│   └── orchestrator_research.py   # Research pipeline
├── scripts/
│   ├── run_trust_agents.py        # Run original pipeline
│   ├── run_trust_research.py      # Run research pipeline
│   ├── evaluate_predictions.py    # Evaluation with uncertain handling
│   └── ...
├── data/
│   └── processed/                 # LIAR dataset
└── outputs/
    ├── trust_agents/              # Original pipeline outputs
    └── trust_research/            # Research pipeline outputs
```

## Quick Start

### Installation
```bash
pip install -e .
python -m spacy download en_core_web_sm
```

### Configuration
Create `.env` file:
```bash
OPENAI_API_KEY=your_key_here
MODEL=gpt-4.1-mini
```

### Run Pipelines
```bash
# Prepare data
make data.liar

# Run original pipeline
python scripts/run_trust_agents.py --dataset liar --split val --limit 200 --skip-evidence

# Run research pipeline
python scripts/run_trust_research.py --dataset liar --split val --limit 200 --skip-evidence

# Evaluate
python scripts/evaluate_predictions.py --preds outputs/trust_research/liar_predictions.jsonl
```

## Evaluation

The evaluation script handles the three-class to binary mapping:
```bash
# Default: uncertain → false
python scripts/evaluate_predictions.py --preds outputs/trust_research/liar_predictions.jsonl

# Alternative: uncertain → true
python scripts/evaluate_predictions.py --preds outputs/trust_research/liar_predictions.jsonl --uncertain-as true

# Drop uncertain predictions
python scripts/evaluate_predictions.py --preds outputs/trust_research/liar_predictions.jsonl --uncertain-as drop
```

## Future Work

1. **Use More Capable Models**: GPT-4o or GPT-4-turbo for better reasoning
2. **Evidence Integration**: Full evidence retrieval pipeline
3. **Prompt Optimization**: Systematic prompt engineering to reduce uncertainty
4. **Hybrid Approaches**: Combine fine-tuned classifiers with LLM reasoning
5. **Confidence Calibration**: Better handling of model uncertainty

## References

- **LoCal**: Logical and Causal Fact-Checking (ACM 2024)
- **Delphi Method**: Multi-expert consensus building
- **LIAR Dataset**: Benchmark for fake news detection
- **TRUST-VL/TRUST-Instruct**: News assistance with LLMs (ACL 2024)

## License

MIT License
