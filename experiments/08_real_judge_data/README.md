# Experiment 08 — Real Judge Data

## Goal

Validate the scaling law on real LLM-as-a-judge data.

## Method

1. Obtain public judge datasets:
   - Chatbot Arena human preference data
   - MT-Bench multi-judge evaluations
   - Other public LLM evaluation datasets
2. Treat the full judge panel as the gold reference.
3. Subsample K judges from the panel, compute aggregated score.
4. Measure error vs K.
5. Fit the scaling law and estimate empirical alpha.

## Expected output

- Empirical Error vs K curves on real data.
- Comparison of empirical alpha to theoretical predictions.
- Analysis of how real-world judge correlations affect scaling.
