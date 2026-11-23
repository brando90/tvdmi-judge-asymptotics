# Mega QA Report: Full-Repo 3-Stage Sequential `agy` Audit (`11-23-2026`)

Date: 11-23-2026

**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/16_mega_qa_repo_audit_and_reconciliation_start_23Nov2026/MEGA_QA_REPORT_11-23-2026.md>

**TLDR:** Executed a 3-stage sequential `agy` Mega Quality Assurance (`QA`) audit over `tvdmi-judge-asymptotics` from `QA_BASE = cb2a2c0` through `a3583e7`. Because Anthropic models in `agy` (`claude-opus-5-5-high` and `claude-sonnet-5-5-high`) hit `RESOURCE_EXHAUSTED` (HTTP `429`, resetting in `~59h 18m`–`59h 43m`), Stage 1 and Stage 3 ran cleanly in fresh contexts on `agy gemini-pro-orchestrator-neon` (recording the missing cross-company diversity), while Stage 2 ran on `agy gemini-4-neon-xhigh` (`xhigh`). Stages 1 and 2 caught and fixed the Huber median Taylor factor-of-2 error (`-\sqrt{\pi/2}\,\sigma\delta`), missing table factor `2`, stale appendix Proposition proof, Exp 14 HTML link labels and raw math underscores, Exp 15 cluster-vs-pooled CI note, `JudgeBench` acknowledgments parity, all LaTeX overfull/float warnings across `main.pdf` and standalone reports, and dead Python imports/temp files; Stage 3 returned `VERDICT: PASS` with zero remaining critical or major issues, and all deterministic orchestrator checks (`lake build` `3,290` jobs, `66/66` theorem axiom check strictly `[propext, Classical.choice, Quot.sound]`, `20/20` `py_compile`, `main.pdf` Page 9 anonymity check, Rule 91 HTML scans, secret scan) passed.

---

## 1. Stage Chain, Requested vs. Actual Models, and Verdicts

| Stage | Role | Requested CLI / Model / Effort | Actual Executed CLI / Model / Effort & Fallback Reason | Verdict Block (Verbatim) |
|---|---|---|---|---|
| **Stage 1** | Clean-Eyes Pass 1 | `agy` / `claude-opus-5-5-high` (`high`) | `claude-opus-5-5-high` (`high`) & retry `claude-sonnet-5-5-high` (`high`) both returned HTTP `429` (`59h 43m` reset); `gemini-pro-orchestrator-neon --effort xhigh` failed CLI flag validation (`--effort` unsupported); ran **`gemini-pro-orchestrator-neon`** (no `--effort`). Missing cross-company diversity recorded. | `VERDICT: FIXED` \| `CRITICAL_ISSUES: 0` \| `MAJOR_ISSUES: 4` \| `FIXES_APPLIED: 9` \| `STRUCTURAL: IMPROVED` |
| **Stage 2** | Builder Intent & Reconciliation | `agy` / `gemini-4-neon-xhigh` (`xhigh`) | **`gemini-4-neon-xhigh`** (`xhigh`) — ran directly as planned. | `VERDICT: FIXED` \| `CRITICAL_ISSUES: 0` \| `MAJOR_ISSUES: 0` \| `FIXES_APPLIED: 3` \| `STRUCTURAL: PASS` |
| **Stage 3** | Fresh-Context Clean-Eyes Final Pass | `agy` / `claude-opus-5-5-high` (`high`) | `claude-opus-5-5-high` (`high`) returned HTTP `429` (`59h 18m` reset); skipped smaller Anthropic retry per `qa-correctness.md` provider-exhaustion rule; ran fresh **`gemini-pro-orchestrator-neon`** (no `--effort`). Missing cross-company diversity recorded. | `VERDICT: PASS` \| `CRITICAL_ISSUES: 0` \| `MAJOR_ISSUES: 0` \| `FIXES_APPLIED: 0` \| `STRUCTURAL: PASS` |

> **Verbatim Stage 1 Summary:** Lean 4 (66 theorems, standard axioms) and all experimental pipelines/anonymity/page-limit checks passed, while 4 major issues (Huber median Taylor factor-of-2 error, stale appendix proof for Proposition 7.4, Exp 15 cluster-CI vs. pooled-estimate documentation, and LaTeX overfull/float warnings) plus 1 HTML link-label issue and 4 Python dead-code cleanups were fixed and verified cleanly.
>
> **Verbatim Stage 2 Summary:** Reconciled and retained all Stage 1 fixes, verified all 66 Lean 4 theorems (standard axioms, zero sorry), non-leakage and numerical consistency across Experiments 12–15 and the NeurIPS/arXiv papers, eliminated 3 remaining raw math underscores in the Experiment 14 HTML companion, added JudgeBench to acknowledgments with clean PDF rebuilds (0 warnings, anonymous Page 9 limit intact), and removed 36 stale temp/bytecode files.
>
> **Verbatim Stage 3 Summary:** Independent clean-eyes verification across git diff, Lean 4 (66/66 theorems, zero sorry, standard axioms only), Python leakage/numerical/HTML checks, and NeurIPS/arXiv PDF compliance passed with zero remaining defects.

---

## 2. Detailed Reconciliations Applied Across Stages 1–2

1. **Huber Median Taylor Expansion Factor-of-2 Fix (`Exp 12`):** Corrected $-\sigma\,\Phi^{-1}(\frac{1}{2(1-\delta)}) = -\sqrt{\pi/2}\,\sigma\delta + O(\delta^2)$ (previously mistyped with an extraneous $/2$) and added the missing factor `2` in `2(2q_j-1)(2p_c-1)\Var(Z^\star)$ in Table 1 across `experiments/12_repo_audit_bugfixes_and_next_theory_roadmap_start_21Nov2026/{THEORY_AND_NEXT_EXPERIMENTS_11-21-2026.tex, THEORY_AND_NEXT_EXPERIMENTS_11-21-2026.md, make_equation_images.py}`, re-rendered `doc_equation_images_11-21-2026/eq05_huber_median_bias.png` (visually verified), and recompiled the 12-page PDF (`0` warnings).
2. **Appendix Adversarial Robustness Proof (`latex_paper/neurips/98_appendix_proofs.tex`):** Replaced the stale proof of `Proposition~\ref{prop:robust-aggregation}` with the exact derivation matching Section 7 (`07_adversarial_robustness.tex`) for sample mean shift $\delta(\mu_{\mathrm{adv}}-\tau_c)$ and Gaussian one-sided median shift $-\sqrt{\pi/2}\,\sigma\delta + O(\delta^2)$.
3. **Exp 15 Cluster-CI vs. Pooled-Estimate Documentation (`experiments/15_.../RESULTS_11-21-2026.md`):** Documented explicitly that `SummEval` (`1,600` items) and `MT-Bench` (`2,575` pairs) tables pair pooled point estimates with 95% cluster-bootstrap CIs of per-cluster means (`0` frozen JSON/CSV numbers altered).
4. **Exp 14 HTML Companion Links & Math Formatting (`experiments/14_.../AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026_gdoc_companion.html`):** Fixed misleading `(Google Drive)` labels on GitHub links, removed the duplicate PDF link, and replaced `3` raw underscore occurrences outside `<code>` (`v_W`, `lambda_1`, `lambda_2`) with HTML `<sup>`/`<sub>` tags.
5. **NeurIPS / arXiv Acknowledgements & LaTeX Warning Elimination (`latex_paper/neurips/` + Exp 12/13 `.tex`):** Added `\textsc{JudgeBench}` to `97_acknowledgments.tex`; fixed the `8.18pt` `Overfull \hbox` in `04_consensus_convergence.tex`, `[h]`->`[ht]` float warnings and `non-ties`->`non-tie` typo in `98_appendix_experiments.tex` / `99_supplementary.tex`, and overfull/float warnings in Exp 12 & Exp 13 standalone `.tex` files; rebuilt clean PDFs (`main.pdf` 21p, `main_arxiv.pdf` 23p, Exp 12 12p, Exp 13 5p, Exp 14 3p, Exp 15 3p).
6. **Structural Dead-Code & Temp Artifact Cleanup:** Removed unused imports/locals in `experiments/12..15/*.py` while preserving identical `rng` call streams and `experiments/utils.py` public APIs, and purged `36` stale `_build/` and `__pycache__/` artifacts. Assessed repo RAMP level **L2** (`CLAUDE.md`, `AGENTS.md`, `README.md`, `theory/assumptions.md`, `experiments/README.md`) and `experiments/utils.py` AST health (`148` LOC, max CC `5`, `0` erosion regressions).

---

## 3. Orchestrator Post-Mega-QA Deterministic Verification (Executed Directly)

- **Lean 4.29.0:** `lake build` succeeded cleanly (`3,290` jobs, `0` errors); AST/regex check confirmed **`66/66` theorems** (`28` `Basic.lean`, `15` `NewTheory.lean`, `11` `SpectralCalibration.lean`, `12` `MultiFactorContinuous.lean`), **`0` `sorry`/`admit`/`native_decide`/custom axioms**, and `#print axioms` across all `66/66` theorems strictly within `[propext, Classical.choice, Quot.sound]`.
- **Python & `utils.py`:** `20/20` Python scripts under `experiments/` compiled cleanly; functional test confirmed `compute_true_tvd == compute_delta` (`0.177823 == 0.177823`, gap `< 1e-12`).
- **PDFs, Visual Inspection & Anonymity:** Verified `main.pdf` (`21`p, zero author/affiliation/owner leaks across all pages, Sections 1–12 + Limitations ending on **Page 9**, `References` starting at **Page 10**, `0` `[TODO]`s in checklist), `main_arxiv.pdf` (`23`p), Exp 12 (`12`p), Exp 13 (`5`p), Exp 14 (`3`p), Exp 15 (`3`p). Visually inspected rendered PNGs for `main` pp. `1, 9, 10` and `eq05_huber_median_bias.png`.
- **HTML Companions & Security:** Stripped plain text of both Exp 14 & Exp 15 HTML companions and Exp 16 companion verified `FORBIDDEN_MATH_HITS == []`; `git diff --check` clean; secret scan across all tracked/edited paths returned `0` hits.
- **Open Cross-Company Note (Transparent Record):** Because Anthropic quota on `agy` is currently rate-limited (`~59h`), Stages 1 and 3 executed on Google's `gemini-pro-orchestrator-neon` inside `agy` per Brando's explicit `"mega qa with agy."` request and `qa-correctness.md` fallback policy. An optional opposite-company Claude/Codex pass can still be run later once Anthropic/Codex quota is targeted without blocking the verified fixes above.
