# Signal-Deconfounded Shrinkage (`SRB-Spec`), Real 33-Judge `JudgeBench`, Partial Proxies, Multi-Factor & Continuous Measure-Theoretic Lean 4 (`11-21-2026`)

Date: 11-21-2026

**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/15_shrinkage_partial_confounders_and_multifactor_lean_start_21Nov2026/SHRINKAGE_PARTIAL_CONFOUNDERS_JUDGEBENCH_AND_MULTIFACTOR_LEAN_11-21-2026.md>

**TLDR:** Complete mathematical proofs and Lean 4.29.0 theorem mapping (`MultiFactorContinuous.lean`, `12` new theorems, `66` total across `TvdmiJudge`, zero `sorry`), paired with empirical results across `SRB-Spec` (`RewardBench 2`, `SummEval`, `MT-Bench`), real `33`-judge `JudgeBench` (`350` pairs), imperfect proxy sweeps (`r in [0, 1]`), and `rank-r` (`r=1,2,3`) Davis–Kahan subspace convergence (`0` bound violations across `10,500` draws).

**Typeset PDF:** [`SHRINKAGE_PARTIAL_CONFOUNDERS_JUDGEBENCH_AND_MULTIFACTOR_LEAN_11-21-2026.pdf`](SHRINKAGE_PARTIAL_CONFOUNDERS_JUDGEBENCH_AND_MULTIFACTOR_LEAN_11-21-2026.pdf) · **Results ledger:** [`RESULTS_11-21-2026.md`](RESULTS_11-21-2026.md)

---

## 1. What These Results Show & Explicit Scope Limits

1. **Why Marginal `RB-Spec` Paid a Clean-Regime Penalty & How `SRB-Spec` Fixes It:** Under `marginal_cov_confounds_signal_and_cloning`, distinct accurate judge families have positive off-family covariance `a_j a_k > 0`, so vanilla `RB-Spec` penalizes strong distinct families (`0.7872` vs `0.8334` Naive SVD on clean `RewardBench 2`). Residualizing out Stage-1 consensus `E_res = C_std - z_0 a_hat^T` isolates within-family cloning while tempered precision weights (`w(alpha)`) close the clean gap (`0.8311` Auto, `0.8363` at `alpha=1.00`) without sacrificing clone (`0.8266`–`0.8323`) or inverted polarity (`0.8357`–`0.8374`) protection.
2. **Real Multi-Paradigm Pairwise Evaluation (`JudgeBench`, ICLR 2025):** Across `33` heterogeneous judges (`0.1314` `PandaLM-7B` to `0.7486` `o1-preview`), train-only length-residualized `CR-SRB-Spec` hits `0.7143` clean, `0.7086` under `32` weak clones (`p=9.50e-04` vs SVD), and `0.7114` under `46%` inverted clones (`p=0.00380` vs Mean).
3. **Imperfect Proxy Covariates Retain Strict `(1 - r^2) c^2 v_W` Floor:** When observed covariates correlate only at `r^2 < 1` with universal confounder `W`, residualization cannot eliminate the remaining `(1 - r^2) c^2 v_W` variance floor without human anchors (`500`-rep Monte Carlo matches the Lean formula within `<= 0.022` MSE across all `r in [0, 1]`, and to `0.002` at `r = 1`).
4. **Multi-Factor & Continuous Measure-Theoretic Limits:** Rank-`r` spectral projection error scales as `2 r eps / (lambda_r - lambda_{r+1})` (`0` violations across `10,500` draws for `r=1,2,3`), while `measure_tvd_event_witness_attainment` and `partition_tvd_refinement_monotone` formalize general measurable-space TVD attainment and prove finite discretizations never overestimate continuous TVD.

---

## 2. Formalized Theorems (`TvdmiJudge/MultiFactorContinuous.lean`)

1. **Signal Deconfounding & Shrinkage Dominance (`marginal_cov_confounds_signal_and_cloning`, `convex_shrinkage_mse_formula`, `interior_shrinkage_strictly_dominates_both`, `tempered_precision_denom_pos_and_bounded`):**
   $$\lambda^\star = \frac{v_U - c_{UB}}{v_U + v_B - 2c_{UB}} \in (0, 1) \implies R(\lambda^\star) < \min(v_U, v_B).$$
2. **Partial Proxy Covariate Floor (`partial_proxy_residual_confounder_variance`, `partial_proxy_strict_floor_when_imperfect`, `partial_proxy_monotone_in_r_sq`):**
   $$\mathrm{MSE}\!\left(\bar{S}^{\mathrm{CR-RB}}(r)\right) = (1 - r^2) c^2 v_W + \mathrm{Var}_{\mathrm{bal}} \le (1 - r^2) c^2 v_W + \frac{B}{F}.$$
3. **Multi-Factor `Rank-r` Subspace Perturbation (`multifactor_rayleigh_trace_sin_sq_bound`, `multifactor_subspace_rate_vanishes`, `multifactor_reduces_to_rank_one`):**
   $$\sum_{k=1}^r \sin^2\theta_k \le \frac{2 r \varepsilon}{\lambda_r - \lambda_{r+1}} = O\!\left(\frac{r}{\sqrt{n}}\right) \to 0.$$
4. **Continuous Measure-Theoretic TVD & Partition Monotonicity (`measure_tvd_event_witness_attainment`, `partition_tvd_refinement_monotone`):**
   $$\left|\int_S 1\,\mathrm{d}P - \int_S 1\,\mathrm{d}Q\right| = |P_{\mathbb{R}}(S) - Q_{\mathbb{R}}(S)|, \qquad \frac{1}{2}\sum_{b=1}^B \left|\sum_{j=1}^M (p_{b,j} - q_{b,j})\right| \le \frac{1}{2}\sum_{b=1}^B \sum_{j=1}^M |p_{b,j} - q_{b,j}|.$$
