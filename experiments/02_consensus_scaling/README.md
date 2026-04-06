# Experiment 02 — Consensus (Majority-Vote) Scaling Law

**Tests:** Theorem 2 (consensus convergence)

## Theory prediction

Each judge votes b_j(z) ∈ {0,1} with Pr(b_j = h*(z) | z) ≥ ½ + γ.
Majority vote converges exponentially:

    τ(c) − Δ_c(b_K) ≤ 2 exp(−2Kγ₀²)

## Simulation design

1. **Ground truth**: Same discretized joint distribution as Experiment 01
   (bivariate Gaussian on d×d grid, d=20). Compute h*(u,v) =
   1[p_uv(u,v) > p_u(u)·p_v(v)] (the Bayes-optimal critic).

2. **Judge model**: For each (u,v) pair, judge j outputs b_j(u,v) =
   h*(u,v) with probability ½ + γ, else flips, independently across judges.

3. **Aggregation**: Majority vote b_K(u,v) = mode of {b_1,...,b_K}.

4. **Protocol**: For each K in the sweep and each γ in {0.01, 0.05, 0.1, 0.2, 0.4}:
   - Generate K votes for every (u,v) cell.
   - Compute Δ_c(b_K) using the population distribution.
   - Record gap τ(c) − Δ_c(b_K).
   - Repeat 500 times.

## Plots

- Semi-log plot (linear K, log gap) — one curve per γ.
- Overlay theoretical bound 2·exp(−2Kγ²) as dashed.
- Fit exponential decay rate; report γ̂ vs true γ.

## Output files

- `main.py`
- `results/exp2_semilog.pdf` / `.png`
- `results/exp2_decay_rates.csv` (columns: gamma_true, gamma_fitted, R²)

## Run

```bash
python main.py
```
