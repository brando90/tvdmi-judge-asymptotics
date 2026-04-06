# Experiment 03 -- Two-Axis Decomposition (K x n)

## Theorem
Pr(tau(c) - max_{j<=K} Delta_hat_c(h_j) > eps + eta) <= exp(-K*p_eps) + 2K*exp(-2n*eta^2).

## Design
- Same random-witness setup as Exp 1 with fixed beta.
- Finite-sample estimation: Delta_hat_c(h_j) from n samples instead of population-level.
- Sweep: K in {4, 16, 64, 256, 1024}, n in {50, 100, 500, 1000, 5000}.
- 200 Monte Carlo reps per config.

## Expected output
- Heatmap of mean gap over the (K, n) grid.
- Marginal line plots: gap vs K (fixed n=1000), gap vs n (fixed K=64).
- Contour plot of K-n tradeoff frontier for gap < 0.05.

## Files
- `exp3_two_axis.py`
- `exp3_heatmap.{pdf,png}`
- `exp3_marginals.{pdf,png}`
- `exp3_tradeoff_contour.{pdf,png}`
