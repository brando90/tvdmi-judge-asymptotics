# Experiment 04 — Diversity and Dependence (ρ Effect)

**Tests:** Proposition 5.1 (variance of equicorrelated ensemble mean)

## Theory prediction

For correlated scalar scores with pairwise correlation ρ:

    Var(S̄_K | c) = σ² · ((1 − ρ)/K + ρ)

When ρ > 0, there is an irreducible error floor of σ²ρ.

## Simulation design

1. **Judge score model**: For a fixed candidate c with true quality τ(c):
   - Generate K correlated scores S_1,...,S_K from a multivariate normal
     with mean τ(c), variance σ² = 0.1, and equicorrelation ρ.
   - Use the standard equicorrelation construction:
     S_j = √ρ · Z_common + √(1−ρ) · Z_j + τ(c), where Z's are i.i.d.
     N(0, σ²).
   - Clip scores to [0,1].
   - Aggregate: S̄_K = mean of {S_1,...,S_K}.

2. **Protocol**: For each K in the sweep and each ρ in
   {0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8}:
   - Generate 1000 repetitions.
   - Record RMSE of S̄_K relative to τ(c).

## Plots

- Log-log plot of RMSE vs K, one curve per ρ.
  - For ρ = 0, verify slope = −0.5.
  - For ρ > 0, show the curve flattening to the floor √(σ²ρ).
  - Overlay theoretical prediction as dashed lines.
- Separate plot: fitted exponent α̂ vs ρ, showing how effective scaling
  degrades with dependence.

## Output files

- `main.py`
- `results/exp4_rmse_vs_K.pdf` / `.png`
- `results/exp4_exponent_vs_rho.pdf` / `.png`

## Run

```bash
python main.py
```
