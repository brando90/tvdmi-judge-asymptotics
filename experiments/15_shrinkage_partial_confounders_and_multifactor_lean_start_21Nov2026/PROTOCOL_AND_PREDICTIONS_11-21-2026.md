# Pre-Registered Protocol & Predictions: Experiment 15 (`SRB-Spec`, `JudgeBench`, Partial Proxies & Multi-Factor/Continuous Lean 4, `11-21-2026`)

Date: 11-21-2026

**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/15_shrinkage_partial_confounders_and_multifactor_lean_start_21Nov2026/PROTOCOL_AND_PREDICTIONS_11-21-2026.md>

**TLDR:** Pre-registers all hypotheses, falsification criteria, non-leakage rules, and numerical predictions before running Experiment 15 across `SRB-Spec` residualized shrinkage, real `33`-judge `JudgeBench` (`350` pairs), imperfect proxy covariate sweeps `r in [0, 1]`, multi-factor `rank-r` Davis–Kahan perturbation, and `12` new Lean 4 theorems (`MultiFactorContinuous.lean`, bringing `TvdmiJudge` to `66` total theorems).

---

## 1. Hypotheses & Quantitative Predictions (Frozen Before Execution)

1. **Module 1 (`SRB-Spec` Signal-Deconfounded Residual Balancing & Shrinkage across `RewardBench 2`, `SummEval`, and `MT-Bench`):**
   - *Diagnosis:* Marginal correlation `R0 = Corr(C_std)` mixes shared quality signal (`a_j * a_k`) with redundant cloning (`tau_fam`), causing `RB-Spec` to downweight strong distinct-model families (`Skywork`, `allenai`, `GRM`) relative to legacy singletons on clean `RewardBench 2` (`0.7872` vs Naive SVD `0.8334`).
   - *Method (`SRB-Spec(alpha)`):* Compute flood-immune Stage 1 `z0` via `RB-Spec`, residualize `E_res = C_std - z0 * a_hat^T`, estimate clone mass `m_res` on `E_res` (with exact collinear guard `|R0| > 0.985`), and apply tempered Cauchy–Schwarz precision weights `(a_hat / ((1 - alpha) + alpha * max(1 - a_hat^2, 0.15))) / (m_res * sd)`, plus unsupervised train-split `SRB-Spec (Auto)`.
   - *Prediction:* `SRB-Spec` will close the clean `RewardBench 2 (K=23)` accuracy gap from `0.7872` to `>= 0.830` (matching Naive SVD) while remaining invariant under `M=32` `OpenAssistant` clones (`>= 0.81` vs `0.4324` Naive SVD) and `54.9%` inverted `PKU-beaver-v1.0-reward` clones (`>= 0.81` vs `0.1455` Naive SVD), and preserving/improving `SummEval` (`>= 0.63`) and `MT-Bench` (`>= 0.79`).
   - *Falsification criterion:* If `SRB-Spec (Auto)` fails to beat Vanilla `RB-Spec` by `>= +0.03` on clean `RewardBench 2` or degrades by `> 0.03` under `PKU`/`Weak` floods, report failure openly.

2. **Module 2 (`JudgeBench` ICLR 2025, `33` Official Real Judges x `350` Hard Objective Pairs):**
   - *Dataset & Non-Leakage:* Official `ScalerLab/JudgeBench` (`response_model=gpt-4o-2024-05-13`) JSONLs cached in `/tmp/judgebench_public/` (`350` pairs spanning `mmlu-pro`, `livebench-reasoning`, `livebench-math`, `livecodebench`), decoded with official two-pass position-swap scoring. 5-fold pair-disjoint CV with train-only weight fitting.
   - *Prediction:* On `Clean K=33`, `SRB-Spec` achieves `>= 0.70` accuracy (matching/beating Uniform `~0.68`–`0.71`), while under `Weak-Clone Flood K=65` and `Adversarial Flood K=61` Naive SVD and Uniform Vote collapse (`< 0.50` under polarity flood) whereas `SRB-Spec` holds `>= 0.70`.

3. **Module 3 (Imperfect Proxy Covariate Sweep `r in {0.0, 0.25, 0.50, 0.70, 0.85, 0.95, 1.0}`):**
   - *Theory (`partial_proxy_residual_confounder_variance`):* Under universal cross-family confounder `c * W` (`c = 2.0`), residualizing against a proxy covariate `X(r) = r * W + sqrt(1 - r^2) * xi` leaves residual floor `(1 - r^2) * c^2 * v_W`.
   - *Prediction:* Both real benchmarks (`JudgeBench`, `RewardBench 2`, `SummEval`) and 500-rep Monte Carlo (`K=24, F=6`) will exhibit strictly monotone improvement from `r=0` (`r^2=0`, maximal degradation) to `r=1` (`r^2=1`, full recovery), matching the closed-form `(1 - r^2) * c^2 * v_W + Var_bal` MSE curve in Monte Carlo.

4. **Module 4 (Multi-Factor `Rank-r` Spectral Subspace Perturbation on `RewardBench 2`):**
   - *Theory (`multifactor_rayleigh_trace_sin_sq_bound`):* Top-`r` empirical subspace trace leakage across `r in {1, 2, 3}` satisfies `0` violations of `(2 * r * eps) / (lambda_r - lambda_{r+1})` across all `10,500` draws (`500` `replace=True` draws x `7` `n` values x `3` ranks `r`), decaying at `O(1/n)` (`slope approx -1.0`).

5. **Lean 4 (`MultiFactorContinuous.lean`):** `12` new theorems (`66` total across `TvdmiJudge`), zero `sorry`, `lake build` clean, `#print axioms` strictly `[propext, Classical.choice, Quot.sound]`.
