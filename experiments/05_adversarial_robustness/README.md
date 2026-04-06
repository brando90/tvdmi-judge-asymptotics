# Experiment 05 — Adversarial Robustness and Breakdown

**Tests:** Theorem 6.1 (adversarial breakdown threshold)

## Theory prediction

With fraction δ adversarial, the ensemble is correct only if
δ < 2γ/(1 + 2γ). Beyond this, consistency breaks.

For robust scalar aggregation: error ~ O(K^{−1/2} + δ).

## Simulation design

### Part A — Binary voting breakdown

1. Use the consensus setup from Experiment 02 with a fixed γ = 0.1.
2. Fraction 1 − δ of judges are honest (correct with prob ½ + γ).
3. Fraction δ are adversarial (always output 1 − h*(z)).
4. For each (K, δ), compute majority-vote gap. Use K = 256 (large enough
   that finite-K noise is small) and sweep δ in
   {0, 0.05, 0.1, 0.2, 0.3, 0.45}.
5. **Plot**: Gap vs δ for fixed K. Mark the theoretical breakdown threshold
   δ* = 2γ/(1 + 2γ). Show that below δ*, error → 0 with K, and above δ*,
   error is bounded away from 0.

### Part B — Scalar score contamination

1. Honest judges: S_j ~ N(τ(c), σ²), clipped to [0,1].
2. Adversarial judges: S_j ~ N(1 − τ(c), σ²), clipped to [0,1].
3. Aggregators:
   - Mean (not robust).
   - Trimmed mean (trim top and bottom 10%).
   - Median.
4. For each (K, δ), record RMSE of each aggregator over 500 reps.
5. **Plots**:
   - RMSE vs K for each aggregator, at δ = 0.1, showing K^{−1/2} + δ
     behavior.
   - RMSE vs δ for fixed K = 256, showing breakdown curves for each
     aggregator.

## Output files

- `main.py`
- `results/exp5_binary_breakdown.pdf` / `.png`
- `results/exp5_scalar_aggregators.pdf` / `.png`
- `results/exp5_breakdown_curves.pdf` / `.png`

## Run

```bash
python main.py
```
