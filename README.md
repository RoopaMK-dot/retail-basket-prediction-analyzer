# Retail Basket Prediction Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ujs8npxmwb/badge.svg)](https://ask.thili.ai/b/ujs8npxmwb)

This project provides a comparative analysis tool for retail analysts. It trains both an n-gram model and a small language model on synthetic shopping basket data to predict the next item a customer might purchase. The system then evaluates and compares the performance of these two models, offering insights into their respective strengths for retail basket prediction.

## What you are building

**Goal.** You will build a system for retail analysts to compare the next-item prediction accuracy of an n-gram model and a small language model, both trained on simulated shopping basket data.

**Done when.** You will have trained both an n-gram model and a small language model on the same simulated retail basket data. You will then evaluate their next-item prediction performance using relevant metrics and present a comparison of their results.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Generate a synthetic dataset of shopping baskets with planted itemsets.**  
      Hint: Use the data generation technique from 'Teach a Language Model What Sells Together'.
- [ ] **2. Implement the n-gram model to predict next items in a basket.**  
      Hint: Apply the n-gram model technique from 'Build a Next-Word Predictor from Scratch'.
- [ ] **3. Pretrain a small language model (like in the drill) on the basket sequences.**  
      Hint: Use the language model training technique from 'Teach a Language Model What Sells Together'.
- [ ] **4. Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).**  
      Hint: Implement evaluation metrics as discussed in 'Build a Next-Word Predictor from Scratch' or 'Teach a Language Model What Sells Together'.
- [ ] **5. Present a comparison of the models' performance.**  
      Hint: Summarize and visualize the evaluation results.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Explore different n-gram sizes or language model architectures to see how they impact performance.
- Implement additional evaluation metrics beyond accuracy and rank, such as precision@k or recall@k.
- Analyze specific types of itemsets where one model performs significantly better or worse than the other.
