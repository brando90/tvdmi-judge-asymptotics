# Live Results: Experiment 15 — `SRB-Spec`, `JudgeBench`, Partial Proxies & Multi-Factor/Continuous Lean 4 (`11-21-2026`)

Date: 11-21-2026

**Status:** COMPLETE (`11-21-2026`, `agy gemini-4-neon-xhigh` Light QA & Post-QA Gate PASSED; see [RESULTS_11-21-2026.md](RESULTS_11-21-2026.md) & [QA_REPORT_11-21-2026.md](QA_REPORT_11-21-2026.md))
**Last updated:** 11-21-2026 21:17 PST
**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/15_shrinkage_partial_confounders_and_multifactor_lean_start_21Nov2026/results.md>

**TLDR:** Completed Experiment 15 end-to-end: Lean 4 `MultiFactorContinuous.lean` (`12` new theorems -> `66` total in `TvdmiJudge`, `3,290` jobs, `[propext, Classical.choice, Quot.sound]`), `run_exp15_suite.py` (`seed=42`, `25.4s`, `0` warnings/`nan`s), Rule-91 `220 DPI` equation PNGs + `0`-overfull 3-page PDF, anonymous Page-9 `main.pdf` (`21`p) & named `main_arxiv.pdf` (`23`p), and `agy gemini-4-neon-xhigh` Light QA. Google Drive upload stays honestly marked pending OAuth refresh (`~/keys/gspread_authorized_user.json` expired `Nov 21 20:04 PST` with `invalid_grant`).

| Phase | Status | Notes |
|---|---|---|
| 1. Pre-registration & Lean 4 `MultiFactorContinuous.lean` | DONE | `12` new theorems -> `66` total (`0` `sorry`, standard axioms only) |
| 2. Cache `JudgeBench` (`33` judges) & run `run_exp15_suite.py` | DONE | `SRB-Spec` `0.8311`–`0.8374` `RB2`, `CR-SRB` `0.7143`/`0.7114` `JudgeBench`, proxy MSE curve, `0/10,500` `rank-r` violations |
| 3. Rule 91 PDF/PNGs & NeurIPS venue/arXiv PDFs | DONE | `0` overfulls in 3-page PDF, anonymous `main.pdf` (`21`p, Page 9 cutoff) & `main_arxiv.pdf` (`23`p), `0` checklist `TODO`s |
| 4. Light QA (`agy gemini-4-neon-xhigh --effort xhigh`) & Closeout | DONE | Light QA reconciled 6 items; post-QA `lake build`, `py_compile`, PDF, diff, & secret checks PASSED |

**TLDR-end:** [tvdmi: exp15-srb-judgebench-proxies-lean] Completed Experiment 15, Light QA, and post-QA deterministic checks (`66` Lean theorems, `SRB-Spec`, `33`-judge `JudgeBench`, partial-proxy law, `rank-r` perturbation, NeurIPS/arXiv PDFs).
