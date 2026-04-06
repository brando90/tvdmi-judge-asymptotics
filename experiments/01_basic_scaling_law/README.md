# Experiment 01 — Random-Witness Scaling Law

**Tests:** Theorem 1 (random-witness consistency)

## Theory prediction

Each judge is a random witness h_j drawn from a distribution μ over critic
functions. The best-of-K estimator is τ̂(c) = max_{j≤K} Δ_c(h_j). The
approximation gap satisfies:

    Pr(τ(c) − τ̂(c) > ε) ≤ exp(−K · p_ε(c))

where p_ε(c) = μ{h : Δ_c(h) ≥ τ(c) − ε}. When p_ε ∝ ε^β, the error
scales as O_p(K^{−1/β}).

## Simulation design

1. **Ground truth**: Fix a joint distribution P_UV (bivariate Gaussian
   with known correlation, discretized to a d×d grid, d=20). Compute the
   true τ(c) = TV(P_UV, P_U ⊗ P_V) exactly on the grid.

2. **Judge distribution**: Each judge h_j is a random binary function on
   the d×d grid. To control β, use a parametric family:
   - Draw a random "quality" parameter q_j ~ Beta(α_shape, 1) where
     α_shape controls the tail (higher α_shape → higher β → thinner tail
     near the optimum → slower convergence).
   - Given q_j, construct h_j as a noisy version of the optimal critic h*,
     where h_j(u,v) = h*(u,v) with probability q_j, else random.
   - This gives p_ε(c) ∝ ε^β with β determined by α_shape.

3. **Protocol**: For each K in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}:
   - Draw K judges from the distribution.
   - Compute Δ_c(h_j) for each judge (population-level, no finite-sample
     noise — that is Experiment 03).
   - Record τ̂_K = max_j Δ_c(h_j).
   - Repeat 500 times. Record mean and std of the gap τ(c) − τ̂_K.

4. **β sweep**: β ∈ {0.5, 1, 2, 4}.

## Plots

- Log-log plot of mean gap vs K, one curve per β value.
- Overlay the theoretical prediction K^{−1/β} as a dashed line.
- Fit slope via OLS on the log-log data; report α̂ and compare to 1/β.

## Output files

- `main.py`
- `results/exp1_scaling_loglog.pdf` / `.png`
- `results/exp1_fitted_exponents.csv` (columns: beta, alpha_predicted, alpha_fitted, R²)

## Run

```bash
python main.py
```
