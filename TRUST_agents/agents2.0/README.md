# TRUST Agents 2.0 - Research Pipeline

This module implements a research-grade fact-checking pipeline based on recent academic papers, exploring whether advanced multi-agent architectures can improve fact-checking accuracy.

## Research Motivation

Traditional fact-checking pipelines use a single verification step. We hypothesized that:
1. **Decomposing claims** into atomic parts could improve verification of complex statements
2. **Multiple verification agents** with different perspectives could reduce bias
3. **Logical aggregation** could properly handle compound claims

## Architecture

```
Input Claim
    │
    ▼
┌─────────────────────────────────────┐
│       DECOMPOSER AGENT              │
│   (LoCal-inspired decomposition)    │
│                                     │
│   Extracts:                         │
│   - Atomic claims (C1, C2, ...)     │
│   - Logic structure (C1 AND C2)     │
│   - Causal relationships            │
└─────────────────────────────────────┘
    │
    ▼ (for each atomic claim)
┌─────────────────────────────────────┐
│         DELPHI JURY                 │
│   (Multi-agent verification)        │
│                                     │
│   Personas:                         │
│   - Strict: Conservative checker    │
│   - Balanced: Pragmatic checker     │
│                                     │
│   Output: Confidence-weighted vote  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│       LOGIC AGGREGATOR              │
│   (Combine atomic verdicts)         │
│                                     │
│   Evaluates: C1 AND C2 = ?          │
│   Handles: AND, OR, IMPLIES, NOT    │
└─────────────────────────────────────┘
    │
    ▼
Final Verdict (true/false/uncertain)
```

## Components

### 1. Decomposer Agent (`decomposer_agent.py`)

**Based on**: LoCal - Logical and Causal Fact-Checking (ACM 2024)

Breaks complex claims into verifiable atomic parts:

```python
Input: "Biden won 2020 and became president"

Output:
  atomic_claims: ["Biden won the 2020 election", "Biden became president"]
  logic_structure: "C1 AND C2"
  complexity_score: 0.3
```

### 2. Delphi Jury (`delphi_jury.py`)

**Based on**: Delphi method for consensus-building

Multiple AI personas independently verify claims:

| Persona | Behavior |
|---------|----------|
| Strict | Conservative, requires strong evidence |
| Balanced | Pragmatic, uses common sense |

Verdicts are aggregated using confidence-weighted voting.

### 3. Logic Aggregator (`logic_aggregator.py`)

**Based on**: LoCal logical reasoning

Combines atomic verdicts using logical formulas:
- `C1 AND C2`: Both must be true
- `C1 OR C2`: At least one must be true
- `C1 IMPLIES C2`: If C1 then C2

## Experimental Setup

### Model Used
- **GPT-4.1-mini**: Cost-effective model with reasonable capabilities
- Token limits required prompt optimization

### Dataset
- **LIAR**: 6-class fake news dataset mapped to binary (true/false)
- Validation split: 200 examples

### Configuration
```bash
MODEL=gpt-4.1-mini
--skip-evidence  # No external evidence retrieval
--limit 200      # 200 validation examples
```

## Results

### Prediction Distribution
```
Gold: {'false': 100, 'true': 100}
Pred: {'uncertain': 165, 'false': 26, 'true': 9}
```

### Accuracy (with different uncertain handling)
| Uncertain Mapping | Accuracy | F1 (macro) |
|-------------------|----------|------------|
| → false | 49.5% | 0.363 |
| → true | 52.0% | 0.444 |
| dropped | 54.3% | 0.493 |

### Comparison with Baselines
| Method | Accuracy |
|--------|----------|
| BERT (fine-tuned) | 65.2% |
| RoBERTa (fine-tuned) | 64.1% |
| GPT-4.1-nano (LLM baseline) | 58.0% |
| Research Pipeline | ~50% |

## Analysis: Why Results Were Lower Than Expected

### 1. High Uncertainty Rate (82.5%)
The model predicted "uncertain" for 165/200 examples. This is because:
- **Model conservatism**: GPT-4.1-mini defaults to uncertainty when not confident
- **No evidence**: Running without evidence retrieval limits verification capability
- **Complex prompts**: Multi-step reasoning is challenging for smaller models

### 2. Model Capability Constraints
- **Token limits**: Required aggressive prompt truncation
- **Reasoning depth**: Smaller models struggle with multi-hop reasoning
- **Instruction following**: Complex agent prompts may not be followed precisely

### 3. Pipeline Complexity
Each step introduces potential errors:
1. Decomposition may miss nuances
2. Jury personas may both be uncertain
3. Logic aggregation propagates uncertainty

### 4. Evaluation Mismatch
- Pipeline outputs 3 classes (true/false/uncertain)
- Gold labels are binary (true/false)
- "Uncertain" predictions are penalized regardless of correctness

## What We Learned

### Positive Findings
1. **Architecture is sound**: The decompose → verify → aggregate approach is logically coherent
2. **Modular design**: Easy to swap components and experiment
3. **Interpretable**: Can trace decisions through atomic claims and jury votes

### Challenges Identified
1. **Model dependency**: Architecture effectiveness depends heavily on model capability
2. **Uncertainty handling**: Need better strategies for uncertain predictions
3. **Cost-accuracy tradeoff**: Better models (GPT-4o) would likely improve results significantly

## Recommendations for Future Work

### Short-term
1. Use GPT-4o-mini or GPT-4o for better reasoning
2. Implement evidence retrieval for grounded verification
3. Tune prompts to reduce uncertainty rate

### Long-term
1. Fine-tune a model specifically for claim decomposition
2. Train a classifier to handle uncertain predictions
3. Hybrid approach: Use fine-tuned model for easy cases, LLM for complex ones

## Usage

### Test Components
```bash
python TRUST_agents/agents2.0/decomposer_agent.py
python TRUST_agents/agents2.0/delphi_jury.py
python TRUST_agents/agents2.0/logic_aggregator.py
```

### Run Pipeline
```bash
python scripts/run_trust_research.py \
    --dataset liar \
    --split val \
    --limit 200 \
    --skip-evidence
```

### Evaluate
```bash
python scripts/evaluate_predictions.py \
    --preds outputs/trust_research/liar_predictions.jsonl
```

## Files

```
agents2.0/
├── __init__.py              # Module exports
├── decomposer_agent.py      # Claim decomposition
├── delphi_jury.py           # Multi-agent jury
├── logic_aggregator.py      # Logic aggregation
└── README.md                # This file
```

## References

1. **LoCal**: Logical and Causal Fact-Checking (ACM 2024)
2. **Delphi Method**: Consensus-building technique
3. **LIAR Dataset**: Fake news detection benchmark
4. **TRUST-VL**: Vision-language fact-checking (ACL 2024)

## Conclusion

While the research pipeline did not outperform fine-tuned baselines in our experiments, the architecture demonstrates a principled approach to fact-checking that could be effective with:
- More capable models (GPT-4o, Claude)
- Evidence retrieval integration
- Prompt optimization
- Hybrid fine-tuned + LLM approaches

The high uncertainty rate (82.5%) indicates that the model recognizes its limitations, which could be valuable in a production system where uncertain cases are escalated for human review.
