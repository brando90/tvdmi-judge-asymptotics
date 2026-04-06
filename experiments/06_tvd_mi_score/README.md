# Experiment 06 — TVD-MI Score Computation

## Goal

Implement and validate the TVD-MI score estimator for finite judge panels.

## Hypothesis

The plug-in TVD-MI estimator has bias that decreases with K and n. A debiased
estimator should have lower MSE for small panels.

## Method

1. Implement plug-in estimators for TVD and MI from judge score matrices.
2. Implement debiased estimators (Miller correction for MI, bootstrap for TVD).
3. Generate data from known distributions where TVD and MI are analytically
   computable.
4. Sweep K and n, measure bias and variance of each estimator.
5. Compare MSE: plug-in vs. debiased.

## Expected output

- Bias and variance curves as functions of K and n.
- MSE comparison plots.
- Recommended estimator for practical panel sizes.
