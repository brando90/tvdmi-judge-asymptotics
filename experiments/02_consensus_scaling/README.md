# Experiment 02 -- Consensus (Majority-Vote) Scaling Law

## Theorem
tau(c) - Delta_c(b_K) <= 2 exp(-2K*gamma^2), where gamma is the per-judge competence margin above chance.

## Design
- Ground truth: same discretized bivariate Gaussian as Exp 1.
- Judges: each votes b_j(u,v) = h*(u,v) with prob 1/2 + gamma, else flips.
- Aggregation: majority vote b_K(u,v) = mode of {b_1,...,b_K}.
- Sweep: K = 1..1024, gamma in {0.01, 0.05, 0.1, 0.2, 0.4}.
- 500 Monte Carlo reps per config.

## Expected output
- Semi-log plot (linear K, log gap), one curve per gamma, with 2 exp(-2K*gamma^2) overlay.
- CSV of decay rates (gamma_true, gamma_fitted, R^2).

## Files
- `exp2_consensus_scaling.py`
- `exp2_semilog.{pdf,png}`
- `exp2_decay_rates.csv`
