# Experiment 02 — Effect of Judge Quality Distribution

## Goal

Study how heterogeneous judge quality affects the scaling exponent alpha.

## Hypothesis

When judge accuracies are drawn from a distribution rather than fixed,
the effective scaling exponent alpha depends on both the mean and variance
of the quality distribution.

## Method

1. Draw each judge's accuracy p_k from Beta(a, b) with varying (a, b).
2. Configurations:
   - High quality, low variance: Beta(9, 1) — mean 0.9
   - Moderate quality, moderate variance: Beta(3, 2) — mean 0.6
   - Mixed quality, high variance: Beta(1, 1) — uniform on [0, 1]
   - Bimodal: 50% from Beta(9, 1) + 50% from Beta(2, 8)
3. For each config, sweep K and compute aggregated error.
4. Fit alpha and compare across distributions.

## Expected output

- Error vs K curves for each quality distribution.
- Alpha values as a function of mean and variance of quality.
- Insight into when heterogeneity helps vs. hurts.
