# Results Report: Experiment 12 — Repository Audit, Bugfixes & Theory Verification (11-21-2026)

Date: 11-21-2026

**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/12_repo_audit_bugfixes_and_next_theory_roadmap_start_21Nov2026/RESULTS_audit_and_fixes_11-21-2026.md>

**TLDR:** All 5 verification checks (`run_audit_and_verifications.py`, single-process Python 3.11 simulation, 500 MC reps, 10,000 seed-bootstrap CIs), Lean 4 build (`lake build`, 3,287 jobs, standard axioms only), and NeurIPS paper compilation (`latex_paper/neurips/main.pdf`, 16 pages) succeeded. Exact fixes resolve all four historical simulation/math bugs and validate prompt-copying failure + split-sample cross-fitted TVD-MI.

## 1. Environment & Verification Provenance

- **Simulation runner:** Single-process Python 3.11 (`run_audit_and_verifications.py`, seed `42`, `N_REPS=500`, `B_BOOT=10000` percentile bootstrap over independent MC reps; no LLM harness).
- **Lean 4:** `experiments/00_lean_formalization/lean/` compiled with Lean `v4.29.0` (`Build completed successfully (3287 jobs)`). Verified `#print axioms` on `variance_equicorrelated_mean`, `adversarial_threshold_div`, `ranking_preserved`, `miss_prob_le_exp`, `consensus_bound_vanishes`, and `two_axis_sum_vanishes` — all depend solely on `[propext, Classical.choice, Quot.sound]` (zero `sorryAx`).
- **LaTeX:** `latex_paper/neurips/main.pdf` compiled cleanly (`16` pages, `288,221` bytes); `THEORY_AND_NEXT_EXPERIMENTS_11-21-2026.pdf` compiled cleanly (`12` pages) and visually inspected across all rendered PNG pages.

## 2. Check A: Unconditional Prompt-Response TVD-MI vs. Oriented Conditional Peer TVD-MI (`n=400`, `K=16`)

| Candidate | Gold Acc. | Prompt TVD-MI [95% CI] | Unsigned Peer TVD-MI [95% CI] | Oriented Peer Delta [95% CI] |
|---|---|---|---|---|
| Oracle | 1.00 | 0.499 [0.499, 0.499] | 0.249 [0.249, 0.250] | +0.249 [+0.249, +0.250] |
| Noisy Solver | 0.80 | 0.299 [0.297, 0.301] | 0.149 [0.148, 0.150] | +0.149 [+0.148, +0.150] |
| Prompt-Copier (`Y=X`) | 0.00 | **0.948 [0.948, 0.948]** | 0.020 [0.019, 0.020] | -0.000 [-0.001, +0.001] |
| Systematic Inverter | 0.00 | 0.499 [0.499, 0.499] | **0.249 [0.249, 0.250]** | -0.249 [-0.250, -0.249] |

- Spearman rank correlation with gold accuracy: Unconditional Prompt TVD-MI = `-0.500`; Unsigned Peer TVD-MI = `+0.389`; Oriented Peer Witness = **`+0.949`**.

## 3. Check B: `experiments/01_basic_scaling_law/` Bug Fix (`Beta(1, beta)` + `K >= 8` Log-Log Slope)

| `beta` | Predicted `1/beta` | Buggy `Beta(beta,1)` | Fixed Cont. (all `K`) | Fixed Cont. (`K >= 8` log-log) | Fixed Grid (`K >= 8` fit) |
|---|---|---|---|---|---|
| 0.5 | 2.0000 | 0.5205 | 1.1475 | **1.9596** (`R^2=0.9972`) | 1.8750 |
| 1.0 | 1.0000 | 0.6975 | 0.7024 | **0.9871** (`R^2=0.9995`) | 1.0464 |
| 2.0 | 0.5000 | 0.8645 | 0.4194 | **0.4906** (`R^2=0.9981`) | 0.5054 |
| 4.0 | 0.2500 | 0.8532 | 0.2270 | **0.2464** (`R^2=0.9988`) | 0.2588 |

## 4. Check C: Naive Max Overfitting vs. Split-Sample Cross-Fitted Best-of-`K`

### Check C1: Null Distribution (`tau(c) = 0`, `n = 50`)

| `K` | Naive Max Mean [95% CI] | Split-Sample Mean [95% CI] | `p`-val (Split bias = 0) | Wilcoxon `p` (`|Naive|` vs `|Split|`) |
|---|---|---|---|---|
| 1 | +0.0008 [-0.0078, +0.0095] | +0.0042 [-0.0084, +0.0170] | 0.657 | 1.75e-21 |
| 4 | +0.1026 [+0.0965, +0.1087] | +0.0010 [-0.0120, +0.0139] | 0.911 | 0.244 |
| 16 | +0.1756 [+0.1706, +0.1808] | +0.0002 [-0.0122, +0.0127] | 0.698 | 7.96e-41 |
| 64 | +0.2314 [+0.2276, +0.2354] | +0.0070 [-0.0048, +0.0188] | 0.172 | 4.59e-75 |
| 256 | +0.2744 [+0.2711, +0.2778] | +0.0068 [-0.0050, +0.0189] | 0.176 | 2.45e-81 |
| 1024 | **+0.3155 [+0.3122, +0.3189]** | **+0.0063 [-0.0058, +0.0184]** | 0.448 | 7.60e-83 |

### Check C2: Compute-Optimal Budget Tradeoff (`B = K * n = 4096`, `beta = 1`, `tau(c) = 0.182024`)

- Split-sample RMSE achieves its minimum at interior allocation `K* = 8`, `n = 512`: `0.0456 [0.0428, 0.0483]` (bias `-0.0179 [-0.0216, -0.0142]`), matching theoretical scaling `K*(B) ~ (B / log B)^{1/3}`. Naive RMSE at `K = 256, n = 16` degrades to `0.1490 [0.1415, 0.1567]` (bias flipped positive `+0.0093`).

## 5. Check D: `experiments/05_adversarial_robustness/` Part B Non-Degenerate Contamination (`K = 1024`, `tau_c = 0.80`, `mu_adv = 0.20`, `sigma = 0.20`, `delta = 0.15`)

| Aggregator | RMSE [95% CI] | Mean Bias (est - clipped honest) [95% CI] | Wilcoxon `p` vs. Mean |
|---|---|---|---|
| Simple Mean | 0.0863 [0.0858, 0.0868] | -0.0861 [-0.0866, -0.0856] | 1.0 |
| 10%-Trimmed Mean | 0.0513 [0.0507, 0.0519] | -0.0509 [-0.0514, -0.0503] | 1.26e-83 |
| 20%-Trimmed Mean | 0.0375 [0.0369, 0.0382] | -0.0369 [-0.0375, -0.0362] | 1.26e-83 |
| Median | 0.0302 [0.0294, 0.0309] | -0.0289 [-0.0297, -0.0282] | 1.26e-83 |

## 6. Check E: `experiments/04_diversity_dependence/` Local Scaling Elasticity (`K in [16, 256]`, `sigma^2 = 0.1`)

| `rho` | Theory Floor | Buggy Free-`b` `alpha` | Fixed Local `alpha_eff` (`16..256`) | Theory Chord (`16..256`) | Exact Deriv. at `K=64` |
|---|---|---|---|---|---|
| 0.00 | 0.0000 | 0.4681 | **0.4990** | 0.5000 | 0.5000 |
| 0.01 | 0.0316 | 0.5310 | **0.2647** | 0.2967 | 0.3037 |
| 0.05 | 0.0707 | 0.6379 | **0.1418** | 0.1282 | 0.1145 |
| 0.10 | 0.1000 | 0.6592 | **0.0915** | 0.0743 | 0.0616 |
| 0.30 | 0.1732 | 0.7258 | **0.0487** | 0.0229 | 0.0176 |
| 0.50 | 0.2236 | 1.0325 | **0.0260** | 0.0102 | 0.0077 |
| 0.80 | 0.2828 | 0.9251 | **0.0160** | 0.0026 | 0.0019 |

**TLDR-end:** [tvdmi: repo-audit-and-theory-roadmap] Experiment 12 verified all four simulation/math bugfixes, Lean 4 compilation (`3,287` jobs, standard axioms), and NeurIPS paper compilation (`16` pages), while confirming oriented conditional peer TVD-MI (`Spearman 0.949`) and unbiased split-sample cross-fitted scaling (`K*=8` at `B=4096`).
