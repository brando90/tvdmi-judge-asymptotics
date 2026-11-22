# Redundancy-Balanced Spectral Peer TVD-MI: Detailed Mathematical Derivations, Lean 4 Formalization, and Real-LLM Falsification & Recovery (11-21-2026)

Date: 11-21-2026

**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/13_real_llm_validation_hierarchical_rb_spectral_theory_and_lean_start_21Nov2026/RB_SPEC_TVDMI_THEORY_LEAN_AND_REAL_LLM_11-21-2026.md>

**TLDR:** Complete mathematical proofs, Lean 4.29.0 theorem mapping (`experiments/00_lean_formalization/lean/TvdmiJudge/NewTheory.lean`, `15` theorems, zero `sorry`), and real-LLM empirical verification on `SummEval` (`1,600` items) and `LMSYS MT-Bench` (`2,575` pairs) showing why naive prompt-response TVD-MI and uncalibrated uniform ensembles fail under prompt-copying and clone hijacking, and proving how `RB-Spec TVD-MI` recovers family-level $1/F$ convergence and $\delta_{\mathrm{fam}} < 1/2$ adversarial tolerance.

**Typeset PDF:** [RB_SPEC_TVDMI_THEORY_LEAN_AND_REAL_LLM_11-21-2026.pdf](RB_SPEC_TVDMI_THEORY_LEAN_AND_REAL_LLM_11-21-2026.pdf) · **Results ledger:** [RESULTS_11-21-2026.md](RESULTS_11-21-2026.md)

---

## 1. What This Theory Shows and Does Not Show (Explicit Metric Limits)

1. **What TVD-MI measures vs. semantic correctness:** Total Variation Distance–Mutual Information $\tau(c) = \mathrm{TV}(P^{(c)}_{X,Y}, P^{(c)}_X \otimes P^{(c)}_Y)$ measures joint statistical dependence between prompt $X$ and output $Y$, **not** correctness. Any injection $Y = f(X)$ attains $1 - \sum_x P_X(x)^2$ (verified on `SummEval`: verbatim prompt copy `0.4851` vs top real system `0.3277`).
2. **What double-centering removes and leaves:** Prompt centering + global standardization cancels *arbitrary additive prompt difficulty* $a(x)$ and *additive leniency/verbosity* $b(y)$ identically (`delta = 4.00e-15`), isolating multiplicative candidate–task interactions $\gamma\,\psi(x,y)$. Without a human anchor set, linear/spectral scores do **not** calibrate absolute Likert severity (on `SummEval`, `RB-Spec` achieves high rank correlation `+0.689` and consistency `ICC(C,1) = 0.688` while retaining `-1.111` mean bias and `ICC(A,1) = 0.351`).
3. **What hierarchical redundancy balancing recovers:** When $F$ distinct judge families produce unbalanced clone counts $K_1, \dots, K_F$, `RB-Spec TVD-MI` balances collinear block mass via $m_j = \sum_{j'} |\widehat{R}_{jj'}|^6$ and $d_j = m_j^{-1/2}$, guaranteeing $O(1/F)$ family variance and $\delta_{\mathrm{fam}} < 1/2$ adversarial breakdown regardless of how many clones $K_f$ are injected into a family. It still cannot rescue a panel where **every** family shares the same confounder $W$ or more than half of independent families are adversarial.

---

## 2. Core Theorems & Lean 4 Verification Mapping

### 2.1 Finite-PMF Variational Duality & Injection Blindness (`NewTheory.lean` §A)

Using $|d| = 2\max(d,0) - d$ on finite equal-mass vectors $\sum_x p(x) = \sum_x q(x)$, for any critic $h : \mathcal{A} \to [0,1]$,

$$\sum_x \big(p(x) - q(x)\big)h(x) \le \sum_x \max\big(p(x) - q(x), 0\big) = \frac{1}{2}\sum_x \big|p(x) - q(x)\big|,$$

with equality at $h^\star(x) = \mathbf{1}\{p(x) \ge q(x)\}$ (`finite_tvd_variational_le`, `finite_tvd_variational_eq`). Setting $Y = f(X)$ for any injection $f : \mathcal{U} \to \mathcal{V}$ gives diagonal discrepancy $p(x)(1-p(x))$ and off-diagonal sum $\sum_{y \neq f(x)} p(x)q(y) = p(x)(1-p(x))$, proving (`injection_tvd_blindness`):

$$\mathrm{TV}\!\big(P_{X,f(X)}, P_X \otimes P_{f(X)}\big) = 1 - \sum_{x \in \mathcal{U}} P_X(x)^2.$$

### 2.2 Double-Centering Invariance & Oriented Peer Witness (`NewTheory.lean` §B)

If $h(x,y) = a(x) + b(y) + \gamma\,\psi(x,y)$, summing over $P_{XY}$ and $P_X \otimes P_Y$ cancels $a(x)$ and $b(y)$ identically because $\sum_y P_{XY}(x,y) = P_X(x)$ and $\sum_x P_{XY}(x,y) = P_Y(y)$ (`tvdmi_double_centering_invariance`):

$$\Delta(a(x) + b(y) + \gamma\,\psi(x,y)) = \gamma\,\Delta(\psi).$$

On latent binary gold targets $Z^\star$ with peer predictor $W_j$ ($q_j > 1/2$), the oriented contrast gives $\Delta^{\mathrm{orient}}_{c,j} = 2(2q_j - 1)(2p_c - 1)\mathrm{Var}(Z^\star)$ (`oriented_peer_strict_mono`, `oriented_peer_rejects_inverter`).

### 2.3 Clone-Hijacking Falsification & `RB-Spec` $1/F$ Recovery (`NewTheory.lean` §C)

Under hierarchical random effects $S_j = \mu + u_f + e_j$ ($\mathrm{Var}(u_f)=\tau_f^2$, $\mathrm{Var}(e_j)=\sigma_f^2$), uniform averaging across $(K_1, K_2)$ obeys (`clone_hijacking_tendsto`, `clone_hijacking_strict_increase_counterexample`):

$$\mathrm{Var}\!\left(\bar{S}^{\mathrm{unif}}_{K_1, K_2}\right) = \frac{K_1^2 \tau_1^2 + K_1 \sigma_1^2 + K_2^2 \tau_2^2 + K_2 \sigma_2^2}{(K_1 + K_2)^2} \xrightarrow[K_1 \to \infty]{} \tau_1^2.$$

By contrast, balancing weights $1/(F K_f)$ per judge satisfy (`var_family_balanced_le_inv_F`, `inv_F_rate_vanishes`):

$$\mathrm{Var}\!\left(\bar{S}^{\mathrm{bal}}_{F, \{K_f\}}\right) = \frac{1}{F^2}\sum_{f=1}^F \left(\tau_f^2 + \frac{\sigma_f^2}{K_f}\right) \le \frac{\max_f(\tau_f^2 + \sigma_f^2)}{F} \xrightarrow[F\to\infty]{} 0 \qquad \forall\, K_1,\dots,K_F \ge 1.$$

Our unsupervised estimator **`RB-Spec TVD-MI`** computes within-prompt centered standardized scores $\widetilde{Z}$, empirical correlation $\widehat{R}$, redundancy mass $m_j = \sum_{j'} |\widehat{R}_{jj'}|^6$, balanced projection $Z^{\mathrm{bal}}_{i,j} = m_j^{-1/2}\widetilde{Z}_{i,j}$, leading singular vector $v_1(Z^{\mathrm{bal}})$, and balanced sign orientation $\mathrm{sign}(\sum_j m_j^{-1} v_{1,j})$.

### 2.4 Effective-Family Breakdown, Split-Sample Bound & Cauchy–Schwarz Weights (`NewTheory.lean` §D)

1. **Effective-family threshold (`effective_family_adversarial_threshold`):** $(1-\delta_{\mathrm{fam}})(\tfrac{1}{2}+\gamma) + \delta_{\mathrm{fam}}(\tfrac{1}{2}-\gamma) > \tfrac{1}{2} \iff \delta_{\mathrm{fam}} < \tfrac{1}{2}$ independently of raw clone counts.
2. **Corrected Hoeffding & Split-Sample Bound (`hoeffding_two_sample_exponent`):** $\sum_{m=1}^{2n}(b_m - a_m)^2 = 2/n \implies 2\exp(-n\eta^2)$, giving Theorem 3' with $K^\star(B) \asymp (B/\log B)^{\beta/(\beta+2)}$.
3. **Cauchy–Schwarz / Nitzan–Paroush weights (`optimal_judge_weights_cauchy_schwarz`, `optimal_judge_weights_equality`):** Under $\sum w_j \mu_j = 1$ and $v_j > 0$, $\sum w_j^2 v_j \ge (\sum \mu_j^2/v_j)^{-1}$ with equality at $w_j^\star \propto \mu_j / v_j$.

---

## 3. Summary of Verified Numbers

Complete step-by-step proofs, full tables for `SummEval` (`1,600` items), `MT-Bench` (`2,575` pairs), and `500`-rep Monte Carlo checks are in [RB_SPEC_TVDMI_THEORY_LEAN_AND_REAL_LLM_11-21-2026.pdf](RB_SPEC_TVDMI_THEORY_LEAN_AND_REAL_LLM_11-21-2026.pdf) and [RESULTS_11-21-2026.md](RESULTS_11-21-2026.md).
