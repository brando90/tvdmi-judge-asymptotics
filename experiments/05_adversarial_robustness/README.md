# Experiment 05 -- Adversarial Robustness and Breakdown

## Theorem
Ensemble correct iff delta < 2*gamma/(1 + 2*gamma). Robust scalar aggregation: error ~ O(K^{-1/2} + delta).

## Design

**Part A -- Binary voting breakdown:**
- Consensus setup with gamma = 0.1, fraction delta adversarial (always output 1 - h*).
- K = 256, sweep delta in {0, 0.05, 0.1, 0.2, 0.3, 0.45}.
- Plot gap vs delta, mark breakdown threshold delta* = 2*gamma/(1+2*gamma).

**Part B -- Scalar score contamination:**
- Honest: S_j ~ N(tau(c), sigma^2). Adversarial: S_j ~ N(1-tau(c), sigma^2).
- Aggregators: mean, trimmed mean (10%), median.
- RMSE vs K at delta=0.1; RMSE vs delta at K=256.
- 500 reps per config.

## Files
- `exp5_adversarial.py`
- `exp5_binary_breakdown.{pdf,png}`
- `exp5_scalar_aggregators.{pdf,png}`
