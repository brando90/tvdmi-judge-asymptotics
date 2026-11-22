# Pre-Registered Protocol & Predictions — Experiment 14 (`11-21-2026`)

- **Start timestamp:** `11-21-2026 19:28:30 PST`
- **Base commit (`main`):** `b0f1029`
- **Datasets (cached strictly in `/tmp/`, zero repo commits):**
  1. `SummEval` (`/tmp/corr-ce-public-ordinal/summeval`): `100` articles, `16` systems = `1,600` summaries, `3` expert raters, `38` wild judges.
  2. `LMSYS MT-Bench` (`/tmp/mt_bench_public`): `80` questions, `2,575` decisive human pairs, `K=17` confounded pool.
  3. `RewardBench 2` (`/tmp/rewardbench2_public`): `1,865` test rows (`1,763` non-`Ties` prompts x `4` completions = `7,052` candidates across `Factuality`, `Math`, `Precise IF`, `Focus`, `Safety`), evaluated across `23+` real open-source reward models from `allenai/reward-bench-2-results` (`Skywork` x8, `allenai` Tulu/Llama x5, `GRM` x4, `OpenAssistant` x3, diverse strong singletons x3, plus `PKU-Alignment/beaver-*-cost` natural inverted cost judges & `beaver-*-reward`).

## Pre-Registered Hypotheses & Falsification Criteria

1. **H1 (`SummEval` Semi-Supervised `m`-Anchor Affine Calibration):**
   - Double-centered `RB-Spec TVD-MI` (`m=0`) retains `-1.1114` mean bias (`ICC(A,1) ~ 0.352`, `ICC(C,1) ~ 0.688`).
   - *Prediction:* Fitting monotone OLS affine calibration `(a_hat * z_RB + b_hat)` clipped to `[1, 5]` on only `m = 5` to `20` labeled anchor articles eliminates mean bias toward `0` (`|bias| < 0.05` at `m=20`), cuts MAE from `1.12` below `0.62`, and raises `ICC(A,1)` from `0.352` up to ~`ICC(C,1)` (`>= 0.65`) without altering rank order (`item Spearman >= 0.685`).
   - *Falsification:* If affine calibration leaves `|bias| > 0.15` or degrades rank Spearman by `> 0.02`.

2. **H2 (`RewardBench 2` Natural Clone Hijacking, Natural Cost Polarity Inversion & `m`-Anchor Platt Calibration):**
   - Using real multi-family reward models (`8` Skywork, `5` Tulu/RB2, `4` GRM, `3` OpenAssistant legacy + clones, plus natural `PKU-Alignment/beaver-*-cost` models that score harm/cost positively so chosen < rejected):
   - *Prediction:* Under `OpenAssistant` clone floods (`M=0..32`) and natural `PKU-cost` polarity floods (`>50%` raw headcount), `Uniform Z-Mean`, `Median`, and `Naive SVD` drop steeply or invert, whereas `RB-Spec TVD-MI` maintains Best-of-4 accuracy `> 0.78`. On the `7,052` candidate logits, `m=10..50` anchor Platt calibration cuts held-out `15`-bin `ECE` and `Brier Score` substantially below temperature-1 uncalibrated margins.
   - *Falsification:* If `RB-Spec` accuracy drops below `Uniform Mean` on clean or flooded regimes, or if Platt calibration increases `ECE`.

3. **H3 (Cross-Family Shared Confounder Breakdown & Recovery via `CR-RB-Spec` / Bi-Factor EM):**
   - When a universal confounding signal (`+ c * z_length`, `c in {0.0, 0.5, 1.0, 1.5, 2.0}`) affects *every* judge family across `SummEval`, `MT-Bench`, and `RewardBench 2`:
   - *Prediction:* Vanilla `RB-Spec` breaks down monotonically as `c` exceeds the true task eigenvalue, matching our Lean theorem `cross_family_confounder_variance_floor`. Pre-residualizing judge scores within each training fold against observable confounder covariates (response word length + prompt lexical overlap) via **`CR-RB-Spec`** (Confounder-Residualized `RB-Spec`) or `5`-anchor 2-factor disambiguation / `RB-Spec + Dawid-Skene EM` restores high held-out Spearman / pairwise accuracy (`>= 0.65` `SummEval`, `>= 0.77` `MT-Bench`).
   - *Falsification:* If `CR-RB-Spec` still collapses (`Spearman < 0.30` at `c=2.0`) under cross-family verbosity confounding.

4. **H4 (Finite-Sample Davis–Kahan Eigenvector Perturbation Convergence):**
   - Formalized in Lean 4 (`TvdmiJudge/SpectralCalibration.lean`, `davis_kahan_rayleigh_sin_sq_bound`, `davis_kahan_sample_rate_vanishes`).
   - *Prediction:* Both `500`-rep Monte Carlo and subsampled `RewardBench 2` prompt sweeps (`n in {10, 20, 40, 80, 160, 320, 640}`) obey empirical `sin^2(theta)` `O(1/n)` (log-log slope near `-1.0`) and stay strictly below the machine-verified `2 * eps / (lambda_1 - lambda_2)` bound.
