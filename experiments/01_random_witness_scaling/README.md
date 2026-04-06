# Experiment 01 -- Random-Witness Scaling Law

## Theorem
Pr(tau(c) - tau_hat(c) > eps) <= exp(-K * p_eps(c)), where p_eps proportional to eps^beta gives Error ~ O_p(K^{-1/beta}).

## Design
- Ground truth: discretized bivariate Gaussian on d x d grid, compute tau(c) = TV(P_UV, P_U tensor P_V).
- Judges: random binary critics h_j with quality q_j ~ Beta(alpha_shape, 1); h_j = h* with prob q_j, else random. alpha_shape controls beta.
- Estimator: best-of-K, tau_hat_K = max_j Delta_c(h_j) (population-level, no finite-sample noise).
- Sweep: K = 1..1024 (powers of 2), beta via alpha_shape in {0.5, 1, 2, 4}.
- 500 Monte Carlo reps per config.

## Expected output
- Log-log plot of mean gap vs K, one curve per beta, with K^{-1/beta} overlay.
- CSV of fitted exponents (beta, alpha_predicted, alpha_fitted, R^2).

## Files
- `exp1_random_witness_scaling.py`
- `exp1_scaling_loglog.{pdf,png}`
- `exp1_fitted_exponents.csv`
