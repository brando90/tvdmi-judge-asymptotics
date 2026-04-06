# Experiment 01 — Basic Scaling Law

## Goal

Verify the basic scaling law `Error(K) ~ K^{-alpha}` with independent,
identically distributed judges.

## Hypothesis

When K i.i.d. judges each have accuracy p > 0.5, the aggregated error from
majority vote or mean decreases as a power law in K.

## Method

1. Fix n items with known ground-truth labels.
2. For each judge k in {1, ..., K}, generate binary judgments with accuracy p.
3. Aggregate via majority vote and via mean.
4. Compute error = fraction of items where the aggregate disagrees with
   ground truth.
5. Sweep K in {1, 2, 3, 5, 10, 20, 50, 100, 200, 500}.
6. Sweep p in {0.55, 0.6, 0.7, 0.8, 0.9}.
7. Sweep n in {100, 500, 1000, 5000}.
8. Repeat each configuration 100 times for confidence intervals.

## Expected output

- Log-log plot of Error vs K for each (p, n) combination.
- Fitted alpha for each setting.
- Table of alpha values vs p.
- Comparison: theoretical alpha (from Condorcet/CLT) vs empirical.

## Run

```bash
python main.py
```
