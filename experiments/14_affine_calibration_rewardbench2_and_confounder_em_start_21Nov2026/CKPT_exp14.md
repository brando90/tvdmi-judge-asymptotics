Created: 11-21-2026 19:35 PST
Last updated: 11-21-2026 20:22 PST

1. **Goal** -- Execute all 4 ranked roadmap items (`m`-anchor affine & Platt calibration on `SummEval` + `RewardBench 2`, cross-family shared confounder testing & `CR-RB-Spec` / EM recovery across real LLMs + 500-rep MC, NeurIPS checklist completion & anonymous/arXiv authorship toggle + citation, Lean 4 `SpectralCalibration.lean` proofs for `ICC(A,1)/ICC(C,1)`, confounder floor, and Davis-Kahan perturbation), run Light QA via `agy gemini-4-neon-xhigh --effort xhigh`, and publish to `main`, Drive, Google Docs, `~/agent-sync`, and email.
2. **What ran** -- `lake build` (`3,289` jobs, `54` theorems across `Basic.lean`, `NewTheory.lean`, `SpectralCalibration.lean`), `#print axioms` on all 11 new theorems, `run_exp14_suite.py` (`seed=42`, `11.02s`), `make_exp14_equation_images.py` (`220 DPI` `eq01..eq06.png`), `pdflatex`/`bibtex` on Exp 14 Rule 91 PDF + `main.pdf` (anonymous 9-page body) + `main_arxiv.pdf` (named authors), Tier 2 Light QA via `agy --model gemini-4-neon-xhigh --effort xhigh` (`PASS`), pushed commit `e540915` to `origin/main`, verified HTTP 200 on all 7 raw GitHub PNG URLs, and preserved `AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026_gdoc_companion.html`.
3. **Results** -- Full tables in [`RESULTS_11-21-2026.md`](RESULTS_11-21-2026.md) and QA report in [`QA_REPORT_11-21-2026.md`](QA_REPORT_11-21-2026.md). Pushed to `main` (`e540915`).
4. **Stuck / blocked** -- Google Drive/Doc companion upload blocked on expired/revoked OAuth refresh token in `~/keys/gspread_authorized_user.json` (`google.auth.exceptions.RefreshError: invalid_grant`, 7-day token expired `11-21-2026 20:04 PST`) and `google-drive` MCP `Unauthorized: rejected by transport`. Reproducible HTML companion (`AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026_gdoc_companion.html`) preserved in git.
5. **Next steps** -- Commit HTML companion + updated ledgers to `origin/main`, sync `~/agent-sync/coding/log.md`, send Trigger Rule 14 + Rule 91 email (attaching Exp 14 PDF directly via `gmail_app_password.txt`).
6. **Brando to-do** -- Ask Rylan Schaeffer if he wants to be a co-author (`README.md` TODO and `main.tex` comment added); re-authorize `~/keys/gspread_authorized_user.json` / `google-drive` MCP to upload the saved HTML companion + PDFs to Google Drive.

## Log

- 11-21-2026 19:35 PST -- Pre-registered Experiment 14.
- 11-21-2026 20:14 PST -- Completed Lean 4 `SpectralCalibration.lean` (`54` total theorems), `run_exp14_suite.py` (`1,753` deduplicated `RB2` prompts across `27` models), NeurIPS checklist (`0` `TODO`s, 9-page body, venue/arXiv toggle), and `agy` Light QA (`PASS`).
- 11-21-2026 20:22 PST -- Pushed `e540915` to `origin/main` (all 7 raw PNG URLs HTTP 200); preserved Rule-91 HTML companion in git after `gspread_authorized_user.json` returned `invalid_grant` and `google-drive` MCP returned `Unauthorized`.
