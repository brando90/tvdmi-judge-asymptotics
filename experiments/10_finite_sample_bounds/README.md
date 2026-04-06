# Experiment 10 — Finite-Sample Bounds

## Goal

Empirically validate finite-sample concentration inequalities from the
consistency theorem.

## Hypothesis

The theoretical bounds from the consistency theorem are achievable and
reasonably tight in moderate-K regimes.

## Method

1. For given K and n, compute the theoretical error bound from the
   consistency theorem (Hoeffding, Bennett, or custom bound).
2. Run simulations and measure actual error.
3. Compare predicted bound vs. observed error.
4. Sweep K in {5, 10, 20, 50, 100, 200} and n in {100, 500, 1000}.

## Expected output

- Plot: theoretical bound vs. empirical error vs. K.
- Identification of regimes where bounds are tight vs. loose.
- Guidance on minimum K needed for a target error level.
