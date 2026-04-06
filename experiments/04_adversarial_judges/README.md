# Experiment 04 — Adversarial Judges

## Goal

Find the robustness threshold for adversarial contamination.

## Hypothesis

There exists a critical adversarial fraction delta* below which aggregation
still converges, and above which it fails. For majority vote with i.i.d.
judges of accuracy p, the threshold should be near delta* = 1 - 1/(2p).

## Method

1. Fix K judges with accuracy p.
2. Replace a fraction delta of judges with adversaries.
3. Adversary types: (a) always-wrong, (b) random, (c) strategic (anti-correlated
   with the correct answer).
4. Sweep delta in {0, 0.05, 0.10, 0.15, ..., 0.49}.
5. Aggregation methods: naive mean, trimmed mean, median.
6. Compute error for each (delta, adversary type, aggregation method).

## Expected output

- Error vs delta curves for each adversary type and aggregation.
- Identification of critical delta*.
- Comparison of naive vs. robust aggregation under contamination.
