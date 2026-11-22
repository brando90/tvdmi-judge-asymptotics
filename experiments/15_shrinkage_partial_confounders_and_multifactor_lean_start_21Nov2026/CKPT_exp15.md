Created: 11-21-2026 20:51 PST
Last updated: 11-21-2026 21:17 PST

1. **Goal** -- Execute Experiment 15 end-to-end: (a) Lean 4 `MultiFactorContinuous.lean` (`12` new theorems -> `66` total), (b) residualized shrinkage `SRB-Spec` closing clean vs flooded `RewardBench 2` gap + `SummEval` + `MT-Bench`, (c) real `33`-judge `JudgeBench` (`350` pairs), (d) imperfect proxy `r in [0,1]` confounder sweep across real benchmarks + MC, (e) multi-factor `rank-r` spectral perturbation (`r=1,2,3`), (f) Rule 91 PDF/PNGs + NeurIPS venue (`<=9`p body) & arXiv rebuild + `agy gemini-4-neon-xhigh` Light QA + publish to `main`, `agent-sync`, and email.
2. **What ran** -- `lake build` (`3,290` jobs, `66` theorems, `[propext, Classical.choice, Quot.sound]`); cached `33` official `JudgeBench` judges (`350` pairs) in `/tmp/judgebench_public/`; executed `run_exp15_suite.py` (`seed=42`, `25.4s`, `0` warnings/`nan`s); ran `make_exp15_equation_images.py` (`220 DPI` `eq01..eq06.png`); built 3-page Exp 15 PDF (`0` overfulls), anonymous `main.pdf` (`21`p, Page 9 cutoff), & `main_arxiv.pdf` (`23`p); ran `agy --model gemini-4-neon-xhigh --effort xhigh` Light QA + post-QA verification suite.
3. **Results** -- Clean `RewardBench 2`: `SRB-Spec` `0.8311` (Auto) / `0.8363` (`alpha=1.00`) vs `0.7872` Vanilla `RB-Spec` (`p=6.96e-13`) and `0.8334` SVD, while keeping clone (`0.8266`/`0.8323` vs `0.4353` SVD) and `PKU` (`0.8357`/`0.8374` vs `0.1426` SVD) dominance. `JudgeBench`: `CR-SRB-Spec` `0.7143` clean / `0.7114` adv-flood (`p=0.00380`). Proxy `r` sweep reproduces `(1-r^2)c^2 v_W + Var_bal` across `500` MC reps & 3 benchmarks. `Rank-r` (`r=1,2,3`) Davis–Kahan bounds hold with `0/10,500` violations.
4. **Stuck / blocked** -- Google Drive OAuth `~/keys/gspread_authorized_user.json` unchanged (`Nov 14 20:04`, expired `Nov 21 20:04 PST` with `invalid_grant`); SNAP unreachable from laptop network check. Mitigated transparently by archiving a Rule-91 HTML companion in git with verified live GitHub PNG URLs and attaching the typeset 3-page PDF directly via email.
5. **Next steps** -- Commit & push implementation to `origin/main`, verify raw GitHub PNGs & save Rule-91 HTML companion, log to `~/agent-sync`, send Rule-14 email to `brando.science@gmail.com`.
6. **Brando to-do** -- Send outreach email to Rylan Schaeffer (co-authorship invite) and refresh Google Drive OAuth (`gspread_authorized_user.json`) when ready to sync Drive links.

## Log

- 11-21-2026 20:51 PST -- Pre-registered Experiment 15 (`SRB-Spec`, `JudgeBench`, partial proxy confounders, multi-factor/continuous Lean 4).
- 11-21-2026 21:05 PST -- Completed Lean 4 `MultiFactorContinuous.lean` (`66` total theorems), `run_exp15_suite.py` (`25.4s`), and `eq01..eq06.png`.
- 11-21-2026 21:17 PST -- Completed Page-9 NeurIPS/arXiv rebuild, `agy gemini-4-neon-xhigh` Light QA (`6` fixes reconciled), and post-QA deterministic checks.
