# Semi-Supervised Calibration, Cross-Family Confounder Recovery (`CR-RB-Spec`), `RewardBench 2` Validation, and Davis–Kahan Spectral Bounds (`11-21-2026`)

Date: 11-21-2026

**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/14_affine_calibration_rewardbench2_and_confounder_em_start_21Nov2026/AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026.md>

**TLDR:** Complete proofs and Lean 4.29.0 theorem mapping (`SpectralCalibration.lean`, `11` new theorems, `54` total in `TvdmiJudge`, zero `sorry`), paired with empirical results across `SummEval` (`m`-anchor affine calibration), `RewardBench 2` (`1,753` prompts, `27` RMs), universal cross-family confounders (`CR-RB-Spec` & `DS-EM`), and Davis–Kahan `sin^2(theta)` convergence (`0` bound violations).

**Typeset PDF:** [AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026.pdf](AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026.pdf) · **Results ledger:** [RESULTS_11-21-2026.md](RESULTS_11-21-2026.md)

---

## 1. What These Results Show and Do Not Show (Explicit Limits)

1. **Likert Severity vs. Bradley–Terry Logit Calibration:** Scalar Likert judges (`SummEval`) require `m = 5` to `20` human anchors to remove constant severity offsets (`bias: -1.1039 -> +0.0074`, `ICC(A,1): 0.3341 -> 0.6441`, `MAE: 1.1295 -> 0.3702`), whereas Bradley–Terry RLHF reward models (`RewardBench 2`) already output near-calibrated binary cross-entropy margins (`m = 0` `ECE = 0.0219`), so small-`m` (`m <= 10`) Platt refits overfit slightly before stabilizing at `m = 50`.
2. **Clean vs. Flooded `RewardBench 2` Tradeoff:** On clean `RewardBench 2` (`K = 23`), where strong correlated families already hold a `17/23` majority, unsupervised `RB-Spec` trades slight clean-regime accuracy (`0.7872` vs. `0.8334` Naive SVD) to survive `M = 32` weak-clone floods (`0.7781` vs. `0.4324`, `p = 1.65e-124`) and `54.9%` naturally inverted `PKU-beaver-v1.0-reward` floods (`0.7963` vs. `0.1455`, `p = 2.11e-229`).
3. **Cross-Family Confounder Vulnerability vs. `CR-RB-Spec`:** Vanilla `RB-Spec` still breaks when **every** family loads onto the same confounder (`c^2 v_W` floor in `cross_family_confounder_variance_floor`). Residualizing observable confounders (`CR-RB-Spec`) requires candidate length / prompt-overlap covariates (or disambiguating anchors) to isolate quality cleanly (`0.5936` `SummEval`, `0.7709` `MT-Bench`, `0.8915` MC).

---

## 2. Formalized Theorems (`TvdmiJudge/SpectralCalibration.lean`)

1. **Bias–Variance & Agreement ICC Gap (`mse_bias_variance_decomp`, `strict_mse_reduction_from_debiasing`, `icc_agreement_le_consistency`, `icc_agreement_lt_consistency_of_bias`, `icc_agreement_eq_consistency_of_calibrated`, `anchor_calibration_excess_mse_le`, `anchor_calibration_excess_tendsto_zero`):**
   $$\mathrm{ICC}(A,1) = \frac{2\,\mathrm{Cov}(\widehat{Y}, Y)}{\mathrm{Var}(\widehat{Y}) + \mathrm{Var}(Y) + (\mu_{\widehat{Y}} - \mu_Y)^2} < \frac{2\,\mathrm{Cov}(\widehat{Y}, Y)}{\mathrm{Var}(\widehat{Y}) + \mathrm{Var}(Y)} = \mathrm{ICC}(C,1) \quad (\mu_{\widehat{Y}} \neq \mu_Y),$$
   closing identically when affine calibration sets $\mu_{\widehat{Y}} = \mu_Y$ with excess anchor MSE $O(1/m) \to 0$.
2. **Cross-Family Confounder Floor & Residualization (`cross_family_confounder_variance_floor`, `confounder_residualization_recovery`):**
   $$\mathrm{Var}(\bar{S}^{\mathrm{bal}}) = c^2 v_W + \frac{1}{F^2}\sum_{f=1}^F \left(\tau_f^2 + \frac{\sigma_f^2}{K_f}\right) \ge c^2 v_W > 0, \qquad \mathrm{Var}(\bar{S}^{\mathrm{CR\text{-}RB}}) \xrightarrow{\hat{c}=c} \frac{1}{F^2}\sum_{f=1}^F \left(\tau_f^2 + \frac{\sigma_f^2}{K_f}\right) \le \frac{B}{F}.$$
3. **Davis–Kahan Spectral Perturbation (`davis_kahan_rayleigh_sin_sq_bound`, `davis_kahan_sample_rate_vanishes`):**
   For eigengap $\gamma_{\mathrm{gap}} = \lambda_1 - \lambda_2 > 0$ and quadratic form perturbation $|\widehat{Q}(u) - Q(u)| \le \varepsilon$,
   $$\sin^2\theta(\hat{u}, u_1) \le \frac{2\varepsilon}{\lambda_1 - \lambda_2}, \qquad \varepsilon_n = \frac{C}{\sqrt{n}} \implies \frac{2\varepsilon_n}{\gamma_{\mathrm{gap}}} \to 0.$$
