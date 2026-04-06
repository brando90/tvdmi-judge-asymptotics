# TVD-MI Judge Ensemble Scaling Laws: Experiment Plan & Code Prompt

## Context for Claude

You are implementing empirical scaling-law experiments for a research project on
reference-free evaluation with judge ensembles. The core object is TVD-MI (Total
Variation Distance-Mutual Information). The theory predicts specific scaling
behaviors as the number of judges K grows. Your job is to simulate these
scenarios, measure error as a function of K, fit scaling laws, and produce
publication-quality plots.

All code should be in Python. Use numpy, scipy, matplotlib, and seaborn.
Save every plot as both .pdf and .png. Use a consistent style: white background,
labeled axes with LaTeX, legend, and a title indicating which experiment it is.

---

## Notation

- **K**: number of judges (main scaling variable, sweep over K = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
- **n**: number of samples per judge used to estimate each judge's score (sweep: 50, 100, 500, 1000, 5000)
- **rho**: pairwise correlation between judges (sweep: 0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8)
- **delta**: fraction of adversarial judges (sweep: 0, 0.05, 0.1, 0.2, 0.3, 0.45)
- **gamma**: per-judge competence margin above chance (sweep: 0.01, 0.05, 0.1, 0.2, 0.4)
- **beta**: controls near-optimal witness mass geometry (sweep: 0.5, 1, 2, 4)
- **tau(c)**: ground-truth TVD-MI value (set to known values like 0.3, 0.5, 0.7)

Use 500 Monte Carlo repetitions per configuration unless noted otherwise.

---

## Experiment 1: Random-Witness Scaling Law

### Theory prediction
Each judge is a random witness h_j drawn from a distribution mu over critic
functions. The best-of-K estimator is tau_hat(c) = max_{j<=K} Delta_c(h_j). The
approximation gap satisfies:

    Pr(tau(c) - tau_hat(c) > eps) <= exp(-K * p_eps(c))

where p_eps(c) = mu{h : Delta_c(h) >= tau(c) - eps}. When p_eps proportional to eps^beta, the error
scales as O_p(K^{-1/beta}).

### Simulation design

1. **Ground truth**: Fix a joint distribution P_UV (e.g., bivariate Gaussian
   with known correlation, discretized to a d x d grid). Compute tau(c) = TV(P_UV,
   P_U tensor P_V) exactly on the grid.

2. **Judge distribution**: Each judge h_j is a random binary function on the
   d x d grid. To control beta, use a parametric family:
   - Draw a random "quality" parameter q_j ~ Beta(alpha_shape, 1) where
     alpha_shape controls the tail (higher alpha_shape -> higher beta -> thinner tail
     near the optimum -> slower convergence).
   - Given q_j, construct h_j as a noisy version of the optimal critic h*,
     where h_j(u,v) = h*(u,v) with probability q_j, else random.
   - This gives p_eps(c) proportional to eps^beta with beta determined by alpha_shape.

3. **Protocol**: For each K in the sweep:
   - Draw K judges from the distribution.
   - Compute Delta_c(h_j) for each judge (population-level, no finite-sample noise
     yet -- that is Experiment 3).
   - Record tau_hat_K = max_j Delta_c(h_j).
   - Repeat 500 times. Record mean and std of the gap tau(c) - tau_hat_K.

4. **Plots**:
   - Log-log plot of mean gap vs K, one curve per beta value.
   - Overlay the theoretical prediction K^{-1/beta} as a dashed line.
   - Fit power law to the log-log data; report alpha_hat and compare to 1/beta.

### Output files
- `exp1_random_witness_scaling.py`
- `exp1_scaling_loglog.pdf`
- `exp1_fitted_exponents.csv` (columns: beta, alpha_predicted, alpha_fitted, R^2)

---

## Experiment 2: Consensus (Majority-Vote) Scaling Law

### Theory prediction
Each judge votes b_j(z) in {0,1} with Pr(b_j = h*(z) | z) >= 1/2 + gamma.
Majority vote converges exponentially:

    tau(c) - Delta_c(b_K) <= 2 exp(-2K*gamma_0^2)

### Simulation design

1. **Ground truth**: Same discretized joint distribution as Experiment 1.
   Compute h*(u,v) = 1[p_uv(u,v) > p_u(u)*p_v(v)] (the Bayes-optimal critic).

2. **Judge model**: For each (u,v) pair, judge j outputs b_j(u,v) = h*(u,v)
   with probability 1/2 + gamma, else flips, independently across judges.

3. **Aggregation**: Majority vote b_K(u,v) = mode of {b_1,...,b_K}.

4. **Protocol**: For each K and each gamma:
   - Generate K votes for every (u,v) cell.
   - Compute Delta_c(b_K) using the population distribution.
   - Record gap tau(c) - Delta_c(b_K).

5. **Plots**:
   - Semi-log plot (linear K, log gap) -- one curve per gamma.
   - Overlay theoretical bound 2 exp(-2K*gamma^2) as dashed.
   - Fit exponential decay rate; report gamma_hat vs true gamma.

### Output files
- `exp2_consensus_scaling.py`
- `exp2_semilog.pdf`
- `exp2_decay_rates.csv` (columns: gamma_true, gamma_fitted, R^2)

---

## Experiment 3: Two-Axis Decomposition (K x n)

### Theory prediction

    Pr(tau(c) - max_{j<=K} Delta_hat_c(h_j) > eps + eta) <= exp(-K*p_eps) + 2K*exp(-2n*eta^2)

### Simulation design

1. Use the same setup as Experiment 1 (random witnesses with a fixed beta).

2. **Finite-sample estimation**: For each judge h_j, instead of computing
   Delta_c(h_j) exactly, estimate it from n samples:
   - Draw n pairs (u_i, v_i) ~ P_UV (positive samples).
   - Draw n pairs (u_i', v_i') ~ P_U tensor P_V (negative samples).
   - Delta_hat_c(h_j) = (1/n) sum h_j(u_i,v_i) - (1/n) sum h_j(u_i',v_i').

3. **Protocol**: Sweep over a grid of (K, n) values.
   K = [4, 16, 64, 256, 1024], n = [50, 100, 500, 1000, 5000].
   Use 200 repetitions.

4. **Plots**:
   - Heatmap of mean gap over the (K, n) grid.
   - Two marginal line plots: gap vs K (for fixed n = 1000) and gap vs n (for
     fixed K = 64).
   - Contour plot of the (K, n) combinations that achieve a target error (e.g.,
     gap < 0.05), showing the K-n tradeoff frontier.

### Output files
- `exp3_two_axis.py`
- `exp3_heatmap.pdf`
- `exp3_marginals.pdf`
- `exp3_tradeoff_contour.pdf`

---

## Experiment 4: Diversity and Dependence (rho Effect)

### Theory prediction
For correlated scalar scores with pairwise correlation rho:

    Var(S_bar_K | c) = sigma^2 * ((1 - rho)/K + rho)

When rho > 0, there is an irreducible error floor of sigma^2 * rho.

### Simulation design

1. **Judge score model**: For a fixed candidate c with true quality tau(c):
   - Generate K correlated scores S_1,...,S_K from a multivariate normal with
     mean tau(c), variance sigma^2 = 0.1, and equicorrelation rho.
   - Use the standard equicorrelation construction:
     S_j = sqrt(rho) * Z_common + sqrt(1-rho) * Z_j, where Z ~ N(0, sigma^2).
   - Clip scores to [0,1].
   - Aggregate: S_bar_K = mean of {S_1,...,S_K}.

2. **Protocol**: For each K and rho:
   - Generate 1000 repetitions.
   - Record RMSE of S_bar_K relative to tau(c).

3. **Plots**:
   - Log-log plot of RMSE vs K, one curve per rho.
   - For rho = 0, verify slope = -0.5.
   - For rho > 0, show the curve flattening to the floor sqrt(sigma^2 * rho).
   - Overlay theoretical prediction as dashed lines.
   - Separate plot: fitted exponent alpha_hat vs rho, showing how effective scaling
     degrades with dependence.

### Output files
- `exp4_diversity.py`
- `exp4_rmse_vs_K.pdf`
- `exp4_exponent_vs_rho.pdf`

---

## Experiment 5: Adversarial Robustness and Breakdown

### Theory prediction
With fraction delta adversarial, the ensemble is correct only if
delta < 2*gamma/(1 + 2*gamma). Beyond this, consistency breaks.

For robust scalar aggregation: error ~ O(K^{-1/2} + delta).

### Simulation design

**Part A -- Binary voting breakdown:**

1. Use the consensus setup from Experiment 2 with a fixed gamma = 0.1.
2. Fraction (1 - delta) of judges are honest (correct with prob 1/2 + gamma).
3. Fraction delta are adversarial (always output 1 - h*(z)).
4. For each (K, delta), compute majority-vote gap. Use K = 256 (large enough
   that finite-K noise is small) and sweep delta.
5. **Plot**: Gap vs delta for fixed K. Mark the theoretical breakdown threshold
   delta* = 2*gamma/(1 + 2*gamma). Show that below delta*, error -> 0 with K, and above delta*,
   error is bounded away from 0.

**Part B -- Scalar score contamination:**

1. Honest judges: S_j ~ N(tau(c), sigma^2), clipped to [0,1].
2. Adversarial judges: S_j ~ N(1 - tau(c), sigma^2), clipped to [0,1].
3. Aggregators:
   - Mean (not robust).
   - Trimmed mean (trim top and bottom 10%).
   - Median.
4. For each (K, delta), record RMSE of each aggregator over 500 reps.
5. **Plots**:
   - RMSE vs K for each aggregator, at delta = 0.1, showing K^{-1/2} + delta behavior.
   - RMSE vs delta for fixed K = 256, showing breakdown curves for each aggregator.

### Output files
- `exp5_adversarial.py`
- `exp5_binary_breakdown.pdf`
- `exp5_scalar_aggregators.pdf`

---

## Experiment 6: End-to-End Scaling Law with LLM Judges (Optional/Real-Data)

### Purpose
Validate the synthetic findings with real LLM-as-judge evaluations.

### Design

1. **Dataset**: Use a public evaluation benchmark where gold scores exist
   (e.g., MT-Bench, Chatbot Arena, or AlpacaEval with human annotations).

2. **Judges**: Use a pool of diverse LLMs as judges. If API access is limited,
   use different prompting strategies or temperature settings on the same model
   to create pseudo-diverse judges. E.g.:
   - Same model, 10 different system prompts (strict, lenient, focused on
     different quality dimensions).
   - Same model at temperatures {0.0, 0.3, 0.5, 0.7, 1.0}.
   - Different models if available (e.g., Claude, GPT-4, Gemini, Llama, Mistral).

3. **Protocol**:
   - Collect scores from all available judges on all items.
   - For each K = 1, 2, ..., max_judges:
     - Subsample K judges (average over all (max_judges choose K) subsets, or
       sample 200 random subsets).
     - Aggregate scores (mean, median, max).
     - Compute correlation with gold (Spearman rho, Kendall tau, or MSE).
   - Plot correlation-with-gold vs K.

4. **Fit scaling law**: Fit both power-law (a*K^{-alpha} + b) and exponential
   (a*exp(-b*K) + c) models. Report which fits better (AIC/BIC).

5. **Estimate rho**: Compute average pairwise correlation among judges. Compare
   the observed scaling curve to the theoretical prediction from Experiment 4
   using the empirically estimated rho.

### Output files
- `exp6_llm_judges.py`
- `exp6_correlation_vs_K.pdf`
- `exp6_scaling_fit.csv`

---

## Shared Utilities (create first)

See `utils.py` in the experiments/ directory.

---

## How to Use This Document

Feed this entire document to Claude as a prompt. Then say:

> "Implement Experiment N. Create all the Python files, run the simulation,
> and produce the plots."

Or to run everything:

> "Implement all 5 synthetic experiments (Experiments 1-5). Start with utils.py,
> then implement and run each experiment in order. Save all plots and CSVs."

For Experiment 6 (real LLM judges), you will need to provide API keys or a
dataset of pre-collected judge scores.

---

## Summary Table

| Experiment | Tests | X-axis | Y-axis | Expected shape |
|---|---|---|---|---|
| 1. Random Witness | Theorem 1 | K (log) | Gap (log) | Slope = -1/beta |
| 2. Consensus | Theorem 2 | K (linear) | Gap (log) | Straight line, slope = -2*gamma^2 |
| 3. Two-Axis | Theorem 3 | K x n grid | Gap | Heatmap, two error sources |
| 4. Diversity | Variance formula | K (log) | RMSE (log) | Slope -0.5 then floor at sqrt(sigma^2 * rho) |
| 5. Adversarial | Breakdown threshold | delta | Gap | Phase transition at delta* = 2*gamma/(1+2*gamma) |
| 6. LLM (optional) | All of the above | K | Corr with gold | Power-law or exponential |
