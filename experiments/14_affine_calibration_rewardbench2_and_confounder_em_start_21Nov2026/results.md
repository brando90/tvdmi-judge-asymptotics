# Live Results: Experiment 14 — Affine Calibration, `RewardBench 2`, Cross-Family Confounders & Davis–Kahan Lean 4 (`11-21-2026`)

Date: 11-21-2026

**Status:** COMPLETE on GitHub `main` (`e540915`); Google Drive/Doc companion upload pending OAuth refresh of `~/keys/gspread_authorized_user.json` (`invalid_grant`: 7-day token expired `11-21-2026 20:04 PST`) and `google-drive` MCP sign-in
**Last updated:** 11-21-2026 20:22 PST
**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/14_affine_calibration_rewardbench2_and_confounder_em_start_21Nov2026/RESULTS_11-21-2026.md>
**Primary Implementation Commit (`main`):** <https://github.com/brando90/tvdmi-judge-asymptotics/commit/e540915>
**Derivation PDF / Rule-91 HTML Companion:** [`AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026.pdf`](AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026.pdf) · [`AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026_gdoc_companion.html`](AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026_gdoc_companion.html)
**NeurIPS Venue / arXiv PDFs (`main`):** [`main.pdf`](../../latex_paper/neurips/main.pdf) · [`main_arxiv.pdf`](../../latex_paper/neurips/main_arxiv.pdf)

**TLDR:** Executed all 4 prioritized roadmap extensions and machine-checked `11` new Lean 4.29.0 theorems in `TvdmiJudge/SpectralCalibration.lean` (`54` theorems total across `TvdmiJudge`, `3,289` build jobs, strictly standard axioms `[propext, Classical.choice, Quot.sound]`, zero `sorry`). Tier 2 Light QA (`agy --model gemini-4-neon-xhigh --effort xhigh`) passed after fixing finite-population `replace=False -> replace=True` Davis–Kahan sampling (`-1.03` true `O(1/n)` slope), rounding alignments (`0.145`, `6.59e-13`, `1,753`), and NeurIPS checklist float flushing (`\clearpage`).

| Phase | Status | Notes |
|---|---|---|
| 1. `README.md` Citation & Rylan TODO + Exp 14 pre-reg | DONE | BibTeX (`Brando Miranda`, `Sanmi Koyejo`) & `Rylan Schaeffer` TODO; row `14` added |
| 2. Lean 4 `SpectralCalibration.lean` | DONE | `11` new theorems (`54` total); `[propext, Classical.choice, Quot.sound]` |
| 3. `run_exp14_suite.py` (`SummEval`, `RB2`, Confounder, DK) | DONE | Python 3.11 (`seed=42`, `11.02s`), `1,753` deduplicated `RB2` prompts |
| 4. Rule 91 PDF + PNGs + NeurIPS Checklist & Rule 89 PDFs | DONE | Zero `[TODO]`, `main.pdf` body inside pp. 1–9, `main_arxiv.pdf` named authors |
| 5. Light QA (`agy gemini-4-neon-xhigh --effort xhigh`) | PASS | Reconciled 3 major QA findings and re-verified all builds (`QA_REPORT_11-21-2026.md`) |

**TLDR-end:** [tvdmi: exp14-calibration-rb2-confounders-lean] Experiment 14 completed and passed Tier 2 Light QA (`54` Lean 4 theorems, `SummEval` `5–20` anchor calibration, `RewardBench 2` `1,753` prompts, `CR-RB-Spec` confounder recovery, `O(1/n)` Davis–Kahan convergence).
