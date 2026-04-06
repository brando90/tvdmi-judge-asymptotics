# Experiments

This directory contains all experiments for the TVD-MI judge asymptotics
project. Each experiment is in a numbered folder. Run each experiment from
its own folder.

## Experiment plan

The experiments below are organized to systematically validate the theoretical
claims of the project. They progress from foundational formalization to
increasingly complex empirical scenarios.

### 00 — Lean 4 formalization (`00_lean_formalization/`)

**Goal:** Formalize the natural-language proof of the main consistency theorem
in Lean 4.

- Input: natural-language proof (to be provided).
- Output: machine-checked Lean 4 proof of the core theorems.
- This is not a simulation; it is a formal verification task.

### 01 — Basic scaling law (`01_basic_scaling_law/`)

**Goal:** Verify the basic scaling law `Error(K) ~ K^{-alpha}` with
independent, identically distributed judges.

- Simulate K judges, each with fixed accuracy p > 0.5, scoring n items.
- Aggregate via majority vote or mean.
- Plot error vs K on log-log axes and fit alpha.
- Sweep p in {0.55, 0.6, 0.7, 0.8, 0.9} and n in {100, 500, 1000, 5000}.

### 02 — Effect of judge quality distribution (`02_judge_quality_distribution/`)

**Goal:** Study how heterogeneous judge quality affects the scaling exponent.

- Draw each judge's accuracy from a distribution (e.g., Beta(a,b)).
- Vary the distribution to control mean quality and variance.
- Measure how alpha changes with quality spread.

### 03 — Inter-judge dependence (`03_inter_judge_dependence/`)

**Goal:** Quantify how correlated judges slow down convergence.

- Introduce pairwise correlation rho between judge scores.
- Use a multivariate normal copula to generate correlated binary judgments.
- Sweep rho in {0, 0.1, 0.2, ..., 0.9} and measure Error(K) for each.
- Plot alpha vs rho.

### 04 — Adversarial judges (`04_adversarial_judges/`)

**Goal:** Find the robustness threshold for adversarial contamination.

- Fix K judges, replace a fraction delta with adversaries (always wrong or
  strategic).
- Sweep delta in {0, 0.05, 0.1, ..., 0.49}.
- Identify the critical delta beyond which aggregation breaks.
- Compare naive mean vs. robust aggregation (trimmed mean, median).

### 05 — Aggregation rules comparison (`05_aggregation_rules/`)

**Goal:** Compare different aggregation methods.

- Methods: majority vote, mean, weighted mean (inverse-variance), median,
  Dawid-Skene EM, spectral methods.
- Evaluate each under the settings of experiments 01–04.
- Report error, convergence rate, and computational cost.

### 06 — TVD-MI score computation (`06_tvd_mi_score/`)

**Goal:** Implement and validate the TVD-MI score itself.

- Define TVD and MI estimators for finite judge panels.
- Measure bias and variance of the estimator as K and n grow.
- Compare plug-in estimator vs. debiased estimator.

### 07 — Pairwise comparison protocol (`07_pairwise_comparisons/`)

**Goal:** Extend the framework to pairwise (Bradley-Terry style) judgments.

- Each judge compares pairs of items instead of rating them.
- Vary m (comparisons per item) and K.
- Aggregate via Bradley-Terry MLE or spectral ranking.
- Measure convergence to the true ranking.

### 08 — Real judge data (`08_real_judge_data/`)

**Goal:** Validate the scaling law on real LLM-as-a-judge data.

- Use public judge datasets (e.g., Chatbot Arena, MT-Bench).
- Subsample K judges from the full panel.
- Measure error vs K using the full panel as the gold reference.
- Fit the scaling law and compare empirical alpha to theory.

### 09 — Diversity metrics (`09_diversity_metrics/`)

**Goal:** Measure and relate judge diversity to convergence rate.

- Define diversity metrics: disagreement rate, entropy of judge panel,
  effective number of independent judges.
- Correlate diversity with convergence speed across experiments 01–04.
- Test whether diversity is a sufficient statistic for predicting alpha.

### 10 — Finite-sample bounds (`10_finite_sample_bounds/`)

**Goal:** Empirically validate finite-sample concentration inequalities.

- For given K and n, compute theoretical error bounds from the consistency
  theorem.
- Compare predicted bounds to observed error in simulations.
- Identify regimes where the bounds are tight vs. loose.

## Adding a new experiment

1. Create the next numbered folder: `NN_short_name/`.
2. Write a `README.md` inside it describing the goal, hypothesis, method,
   and expected output.
3. Implement the experiment as a standalone Python script.
4. Store results in a `results/` subfolder (gitignored if large).
