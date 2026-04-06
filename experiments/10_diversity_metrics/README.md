# Experiment 09 — Diversity Metrics

## Goal

Measure and relate judge diversity to convergence rate.

## Hypothesis

Judge diversity (as measured by disagreement rate, entropy, or effective
independent count) is a strong predictor of convergence speed.

## Method

1. Define diversity metrics:
   - Disagreement rate: fraction of item pairs where judges disagree.
   - Panel entropy: entropy of the empirical judge-score distribution.
   - Effective independent judges: K_eff = K / (1 + (K-1)*mean_rho).
2. Across experiments 01–04, compute each diversity metric for each
   configuration.
3. Correlate diversity with the fitted alpha.
4. Test: is diversity a sufficient statistic for alpha?

## Expected output

- Scatter plots of alpha vs. each diversity metric.
- Correlation coefficients.
- Regression model predicting alpha from diversity.
