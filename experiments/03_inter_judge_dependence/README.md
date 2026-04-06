# Experiment 03 — Inter-Judge Dependence

## Goal

Quantify how correlated judges slow down convergence.

## Hypothesis

Positive inter-judge correlation rho reduces the effective number of
independent judges, increasing the scaling exponent (slowing convergence).
At rho = 1, adding judges provides no benefit.

## Method

1. Use a multivariate normal copula to generate correlated binary judgments.
2. Fix pairwise correlation rho (same for all judge pairs).
3. Sweep rho in {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}.
4. For each rho, sweep K and compute aggregated error.
5. Fit alpha(rho) and plot alpha vs rho.

## Expected output

- Error vs K curves for each rho.
- Plot of alpha vs rho.
- Comparison: effective number of independent judges = K / (1 + (K-1)*rho)
  vs. empirical convergence rate.
