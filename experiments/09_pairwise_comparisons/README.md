# Experiment 07 — Pairwise Comparison Protocol

## Goal

Extend the framework to pairwise (Bradley-Terry style) judgments.

## Hypothesis

When judges compare pairs of items instead of rating them, the convergence
rate depends on both K (judges) and m (comparisons per item).

## Method

1. Generate a true quality vector for n items.
2. Each judge compares m random pairs using a Bradley-Terry noise model.
3. Aggregate via Bradley-Terry MLE or spectral ranking.
4. Measure Kendall tau distance to the true ranking.
5. Sweep K and m independently.

## Expected output

- Ranking error vs K for fixed m.
- Ranking error vs m for fixed K.
- Joint scaling law: Error(K, m).
