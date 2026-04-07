# Experiment 03 - Two-Axis Decomposition Results

**Generated:** 2026-04-05 22:24:50

## Parameters
- d = 20, correlation = 0.5
- alpha_shape = 1.0, beta = 1.0
- K_grid = [4, 16, 64, 256, 1024]
- n_grid = [50, 100, 500, 1000, 5000]
- N_REPS = 200
- tau_c(h*) = 0.182024

## Gap Matrix (K x n)

| K \ n | 50 | 100 | 500 | 1000 | 5000 |
|-------|-------|-------|-------|-------|-------|
|     4 | 0.0559 | 0.0626 | 0.0670 | 0.0710 | 0.0741 |
|    16 | 0.0034 | 0.0109 | 0.0178 | 0.0164 | 0.0199 |
|    64 | 0.0000 | 0.0000 | 0.0010 | 0.0014 | 0.0038 |
|   256 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
|  1024 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Power-Law Fits

### Gap vs K (fixed n = 1000)
- alpha = 1.0178
- R^2 = 0.9994

### Gap vs n (fixed K = 64)
- alpha = 5.0000
- R^2 = -0.0000

## Plots
- Heatmap: `results/exp3_heatmap.pdf`
- Marginals: `results/exp3_marginals.pdf`
- Contour: `results/exp3_tradeoff_contour.pdf`
