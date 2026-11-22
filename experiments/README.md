# Experiments

This directory contains all experiments for the TVD-MI judge asymptotics
project. Each experiment is in a numbered folder. Run each experiment from
its own folder.

## Shared utilities

`utils.py` — shared across all experiments. Provides:
- `make_joint_distribution(d, correlation)` — discretized bivariate Gaussian
- `compute_true_tvd(P_UV, P_U, P_V)` — exact TVD on a grid
- `optimal_critic(P_UV, P_U, P_V)` — Bayes-optimal binary critic h*
- `compute_delta(h, P_UV, P_U, P_V)` — Δ_c(h) for a given critic
- `fit_power_law(K_vals, errors, allow_negative_alpha=False)` — fit error = aK^{-α} + b
- `fit_loglog_power_law(K_vals, errors, k_min=8)` — pure log-log OLS without free offset artifact
- `compute_local_elasticity(K_vals, rmse_vals, k_lo=16, k_hi=256)` — unclipped local scaling elasticity
- `fit_exponential_decay(K_vals, errors)` — fit error = a·exp(-bK) + c
- Consistent matplotlib styling and color palette

## Notation

- **K**: number of judges (sweep: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
- **n**: samples per judge for estimation (sweep: 50, 100, 500, 1000, 5000)
- **ρ**: pairwise inter-judge correlation (sweep: 0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8)
- **δ**: fraction of adversarial judges (sweep: 0, 0.05, 0.1, 0.2, 0.3, 0.45)
- **γ**: per-judge competence margin above chance (sweep: 0.01, 0.05, 0.1, 0.2, 0.4)
- **β**: near-optimal witness mass geometry (sweep: 0.5, 1, 2, 4)
- **τ(c)**: ground-truth TVD-MI value

Use 500 Monte Carlo repetitions per configuration unless noted otherwise.

## Experiment plan

| # | Name | Tests | X-axis | Y-axis | Expected shape |
|---|------|-------|--------|--------|----------------|
| 00 | Lean formalization | — | — | — | Machine-checked proofs |
| 01 | Random-witness scaling | Theorem 1 | K (log) | Gap (log) | Slope = −1/β |
| 02 | Consensus scaling | Theorem 2 | K (linear) | Gap (log) | Straight line, slope = −2γ² |
| 03 | Two-axis decomposition | Theorem 3 | K × n grid | Gap | Heatmap, two error sources |
| 04 | Diversity & dependence | Variance formula | K (log) | RMSE (log) | Slope −0.5 then floor at √(σ²ρ) |
| 05 | Adversarial robustness | Breakdown threshold | δ | Gap | Phase transition at δ* = 2γ/(1+2γ) |
| 06 | LLM judges (optional) | All of the above | K | Corr with gold | Power-law or exponential |
| 07 | Aggregation rules | Comparison | K | Error | Method-dependent |
| 08 | TVD-MI score estimator | Bias/variance | K, n | MSE | Decreasing |
| 09 | Pairwise comparisons | Bradley-Terry | K, m | Ranking error | Decreasing |
| 10 | Diversity metrics | Correlation | Diversity | α | Predictive |
| 11 | Finite-sample bounds | Concentration | K | Bound vs error | Tight/loose |
| 12 | `12_repo_audit_bugfixes_and_next_theory_roadmap_start_21Nov2026` | Repo audit & fixes | K, B, rho, delta | Gap / RMSE / bias | Corrected `1/beta`, split-sample `K*`, oriented peer TVD-MI |
| 13 | `13_real_llm_validation_hierarchical_rb_spectral_theory_and_lean_start_21Nov2026` | Real-LLM & Lean `NewTheory` | M clones, delta, budget | Spearman / Acc / ICC(A,1) | Clone hijacking vs `RB-Spec` `1/F` recovery on SummEval & MT-Bench |
| 14 | `14_affine_calibration_rewardbench2_and_confounder_em_start_21Nov2026` | Calibration, RB2, Confounders, Davis–Kahan | m anchors, c confounder, n samples | ICC(A,1) / ECE / Acc / sin^2(theta) | Bias-calibrated ICC agreement, `RB2` natural clones/cost inversion, `CR-RB-Spec`, `O(1/n)` sin^2 decay |

Experiments 01–05 are the core synthetic experiments that directly validate
the theorems. Experiment 06 validates on real data. Experiments 07–11 are
secondary extensions.

## Adding a new experiment

1. Create the next numbered folder: `NN_short_name/`.
2. Write a `README.md` inside it describing the goal, hypothesis, method,
   and expected output.
3. Implement the experiment as a standalone Python script.
4. Store results in a `results/` subfolder (gitignored if large).
