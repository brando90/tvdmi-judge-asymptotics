# Experiment 03 — Two-Axis Decomposition (K × n)

**Tests:** Theorem 3 (two-axis error decomposition)

## Theory prediction

    Pr(τ(c) − max_{j≤K} Δ̂_c(h_j) > ε + η) ≤ exp(−K·p_ε) + 2K·exp(−2nη²)

This cleanly separates judge approximation error (controlled by K) from
finite-sample estimation error (controlled by n).

## Simulation design

1. Use the same setup as Experiment 01 (random witnesses with a fixed β).

2. **Finite-sample estimation**: For each judge h_j, instead of computing
   Δ_c(h_j) exactly, estimate it from n samples:
   - Draw n pairs (u_i, v_i) ~ P_UV (positive samples).
   - Draw n pairs (u_i', v_i') ~ P_U ⊗ P_V (negative samples).
   - Δ̂_c(h_j) = (1/n) Σ h_j(u_i,v_i) − (1/n) Σ h_j(u_i',v_i').

3. **Protocol**: Sweep over a grid of (K, n) values.
   K = [4, 16, 64, 256, 1024], n = [50, 100, 500, 1000, 5000].
   For each (K, n), record mean gap over 200 repetitions.

## Plots

- Heatmap of mean gap over the (K, n) grid.
- Two marginal line plots: gap vs K (for fixed n=1000) and gap vs n (for
  fixed K=64).
- Contour plot of the (K, n) combinations that achieve a target error
  (e.g., gap < 0.05), showing the K-n tradeoff frontier.

## Output files

- `main.py`
- `results/exp3_heatmap.pdf` / `.png`
- `results/exp3_marginals.pdf` / `.png`
- `results/exp3_tradeoff_contour.pdf` / `.png`

## Run

```bash
python main.py
```
