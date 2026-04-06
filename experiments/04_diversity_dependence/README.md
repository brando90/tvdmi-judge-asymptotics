# Experiment 04 -- Diversity and Dependence (rho Effect)

## Theorem
Var(S_bar_K | c) = sigma^2 * ((1 - rho)/K + rho). When rho > 0, irreducible error floor at sigma^2 * rho.

## Design
- Judge scores: S_j = sqrt(rho)*Z_common + sqrt(1-rho)*Z_j, Z ~ N(0, sigma^2), clipped to [0,1].
- Aggregate: S_bar_K = mean of {S_1,...,S_K}.
- Sweep: K = 1..1024, rho in {0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8}.
- 1000 reps per config. sigma^2 = 0.1, tau(c) = 0.5.

## Expected output
- Log-log RMSE vs K, one curve per rho. Slope = -0.5 for rho=0, flattening to sqrt(sigma^2*rho) for rho>0.
- Fitted exponent alpha_hat vs rho.

## Files
- `exp4_diversity.py`
- `exp4_rmse_vs_K.{pdf,png}`
- `exp4_exponent_vs_rho.{pdf,png}`
