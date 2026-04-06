# Experiment 05 — Aggregation Rules Comparison

## Goal

Compare different aggregation methods across the scenarios from experiments
01–04.

## Methods to compare

1. Majority vote
2. Simple mean
3. Weighted mean (inverse-variance weighting)
4. Median
5. Dawid-Skene EM
6. Spectral method

## Evaluation

For each method, measure:
- Aggregated error vs K
- Convergence rate (alpha)
- Computational cost (wall-clock time)
- Robustness to correlated and adversarial judges

## Expected output

- Comparison table of alpha values across methods and scenarios.
- Pareto plot: accuracy vs. computation.
- Recommendation for which aggregation to use in which regime.
