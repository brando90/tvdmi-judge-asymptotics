# Experiment 06 - End-to-End Scaling Law with LLM Judges

**NOTE: SYNTHETIC PROXY - No real LLM APIs were used.**

**Date:** 2026-04-05 20:07

## Setup

- n_items = 100 (gold scores ~ Uniform(1, 10))
- max_judges = 50
- Inter-judge correlation rho = 0.30
- Empirical pairwise Spearman correlation = 0.8750
- Judge biases ~ N(0, 0.5)
- Judge noise levels ~ Uniform(0.5, 2.0)
- K values: [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
- N_SUBSETS = 200 per K
- Aggregators: ['mean', 'median', 'max']

## Synthetic Judge Model

Score for judge j on item i:

    S_{ji} = gold_i + b_j + sigma_j * (sqrt(rho) * Z_common_i + sqrt(1-rho) * Z_{ji})

where b_j ~ N(0, 0.5), sigma_j ~ U(0.5, 2.0), Z ~ N(0, 1), clipped to [1, 10].

## Results: Scaling Law Fits

| Aggregator | Power alpha | Power R^2 | Exp rate | Exp R^2 | Better fit |
|------------|-------------|-----------|----------|---------|------------|
| mean | 0.7792 | 0.9989 | 0.4508 | 0.9836 | power_law |
| median | 0.6784 | 0.9966 | 0.3748 | 0.9725 | power_law |
| max | 5.0000 | 0.4427 | 0.0000 | -0.0000 | power_law |

## Correlation with Gold at Selected K

| K | mean corr | median corr | max corr |
|---|---------|---------|---------|
| 1 | 0.9115 | 0.9081 | 0.9153 |
| 2 | 0.9377 | 0.9377 | 0.9265 |
| 3 | 0.9466 | 0.9435 | 0.9269 |
| 5 | 0.9571 | 0.9556 | 0.9309 |
| 8 | 0.9636 | 0.9627 | 0.9316 |
| 12 | 0.9657 | 0.9670 | 0.9289 |
| 16 | 0.9676 | 0.9706 | 0.9277 |
| 20 | 0.9683 | 0.9709 | 0.9275 |
| 25 | 0.9696 | 0.9717 | 0.9251 |
| 30 | 0.9703 | 0.9727 | 0.9246 |
| 40 | 0.9709 | 0.9735 | 0.9207 |
| 50 | 0.9715 | 0.9750 | 0.9195 |

## Plots

- Correlation vs K: `../results/exp6_correlation_vs_K.pdf` / `.png`
- Error scaling (log-log): `../results/exp6_error_scaling.pdf` / `.png`
- Scaling fits CSV: `../results/exp6_scaling_fit.csv`
