# Experiment 02 — Consensus (Majority-Vote) Scaling Law

**Date:** 2026-04-05 20:12

## Setup

- Grid size d = 20
- Correlation = 0.5
- True TVD = 0.182024
- tau_c(h*) = 0.182024
- K sweep: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
- N_REPS = 500
- Gammas: [0.01, 0.05, 0.1, 0.2, 0.4]

## Theory

Majority-vote gap bound: tau(c) - Delta_c(b_K) <= 2*exp(-2*K*gamma^2)

## Results: Fitted Decay Rates

| gamma_true | gamma_fitted | R^2 |
|------------|-------------|------|
| 0.0100 | 0.035612 | 0.988467 |
| 0.0500 | 0.077351 | 0.992184 |
| 0.1000 | 0.141066 | 0.996981 |
| 0.2000 | 0.248274 | 0.994770 |
| 0.4000 | 0.422343 | 0.961555 |

## Plots

- Semi-log plot: `results/exp2_semilog.pdf` / `.png`
- Decay rates CSV: `results/exp2_decay_rates.csv`
