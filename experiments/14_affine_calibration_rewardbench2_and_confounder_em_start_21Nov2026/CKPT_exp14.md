Created: 11-21-2026 19:35 PST
Last updated: 11-21-2026 20:14 PST

1. **Goal** -- Execute all 4 ranked roadmap items (`m`-anchor affine & Platt calibration on `SummEval` + `RewardBench 2`, cross-family shared confounder testing & `CR-RB-Spec` / EM recovery across real LLMs + 500-rep MC, NeurIPS checklist completion & anonymous/arXiv authorship toggle + citation, Lean 4 `SpectralCalibration.lean` proofs for `ICC(A,1)/ICC(C,1)`, confounder floor, and Davis-Kahan perturbation), run Light QA via `agy gemini-4-neon-xhigh --effort xhigh`, and publish to `main`, Drive, Google Docs, `~/agent-sync`, and email.
2. **What ran** -- `lake build` (`3,289` jobs, `54` theorems across `Basic.lean`, `NewTheory.lean`, `SpectralCalibration.lean`), `#print axioms` on all 11 new theorems, `run_exp14_suite.py` (`seed=42`, `11.02s`), `make_exp14_equation_images.py` (`220 DPI` `eq01..eq06.png`), `pdflatex`/`bibtex` on Exp 14 Rule 91 PDF + `main.pdf` (anonymous 9-page body) + `main_arxiv.pdf` (named authors), and Tier 2 Light QA via `agy --model gemini-4-neon-xhigh --effort xhigh`.
3. **Results** -- Full tables in [`RESULTS_11-21-2026.md`](RESULTS_11-21-2026.md) and QA report in [`QA_REPORT_11-21-2026.md`](QA_REPORT_11-21-2026.md).
4. **Stuck / blocked** -- nothing.
5. **Next steps** -- Run post-QA verification command, push to `origin/main`, upload Drive PDF + Rule-91 Google Doc, backfill verified links, sync `~/agent-sync`, send email.
6. **Brando to-do** -- Ask Rylan Schaeffer if he wants to be a co-author (`README.md` TODO and `main.tex` comment added).

## Log

- 11-21-2026 19:35 PST -- Pre-registered Experiment 14.
- 11-21-2026 20:14 PST -- Completed Lean 4 `SpectralCalibration.lean` (`54` total theorems), `run_exp14_suite.py` (`1,753` deduplicated `RB2` prompts across `27` models), NeurIPS checklist (`0` `TODO`s, 9-page body, venue/arXiv toggle), and `agy` Light QA (`PASS`).
