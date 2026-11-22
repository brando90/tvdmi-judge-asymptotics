# Results Ledger: Experiment 13 — Real-LLM Validation, `RB-Spec TVD-MI` & Lean 4 Verification (`11-21-2026`)

Date: 11-21-2026

**Status:** COMPLETE (`11-21-2026 19:25 PST`; published to GitHub `main`, Google Drive, and Google Docs)
**Last updated:** 11-21-2026 19:25 PST
**Doc link:** <https://docs.google.com/document/d/1ObShZ0GCxgpH_Di4DS46tvJc05Jggl2RuRo5JmI18Go/edit?usp=drivesdk>
**Drive Derivation PDF:** <https://drive.google.com/file/d/1-XyUnWIX-B6MTyC7QVlIwXgBYmRYVIcc/view?usp=drivesdk>
**Drive NeurIPS Venue / arXiv PDFs:** <https://drive.google.com/file/d/1x8TjkHtVkEbC3XTmY1PWZM9Vk_JpOlB4/view?usp=drivesdk> · <https://drive.google.com/file/d/1tygAthFmh6kBakFjnCEVjXJvg9crlD7Z/view?usp=drivesdk>
**Drive Folder:** <https://drive.google.com/drive/folders/1lsRCjER2ZBguuQ59eYqgDpK3fS5WHZcr>
**Primary Implementation Commit (`main`):** <https://github.com/brando90/tvdmi-judge-asymptotics/commit/6820a39>

**TLDR:** Completed pre-registered real-LLM validation (`SummEval` + `MT-Bench`), 15 new Lean 4.29.0 proofs (`NewTheory.lean`, 43 total in `TvdmiJudge`, `3,288` jobs, standard axioms only), historical script reconciliations (`01`–`11`), NeurIPS Rule 89 venue (`main.pdf`) + arXiv (`main_arxiv.pdf`) builds, Rule 91 PDF + 8 220 DPI equation PNGs, and Tier 2 Light QA via `agy gemini-4-neon-xhigh --effort xhigh` (`QA_REPORT_11-21-2026.md`). Full numerical tables and p-values live in [`RESULTS_11-21-2026.md`](RESULTS_11-21-2026.md).

| Phase | Status | Verified Artifact / Output |
|---|---|---|
| 1. Protocol, predictions & dataset check | DONE | [`PROTOCOL_AND_PREDICTIONS_11-21-2026.md`](PROTOCOL_AND_PREDICTIONS_11-21-2026.md) |
| 2. Lean 4 `NewTheory.lean` (`15` new, `43` total) | DONE | `lake build` (`3,288` jobs), `#print axioms` `[propext, Classical.choice, Quot.sound]` |
| 3. Real-LLM (`SummEval` & `MT-Bench`) + MC suite | DONE | [`RESULTS_11-21-2026.md`](RESULTS_11-21-2026.md), `results/summary_metrics.json` |
| 4. Historical scripts (`01`–`11`) + NeurIPS Rule 89 build | DONE | `main.pdf` (`18` pp.), `main_arxiv.pdf` (`19` pp.), `py_compile` clean |
| 5. Rule 91 derivation PDF & equation PNGs | DONE | [`RB_SPEC_TVDMI_THEORY_LEAN_AND_REAL_LLM_11-21-2026.pdf`](RB_SPEC_TVDMI_THEORY_LEAN_AND_REAL_LLM_11-21-2026.pdf), `doc_equation_images_11-21-2026/eq01..eq08.png` |
| 6. Tier 2 Light QA & post-QA deterministic checks | DONE | [`QA_REPORT_11-21-2026.md`](QA_REPORT_11-21-2026.md); Drive/Doc upload next |

**TLDR-end:** [tvdmi: real-llm-validation-rb-spec-theory-and-lean] Experiment 13 empirical evaluation, 15 new Lean 4 proofs (`43` total), NeurIPS venue + arXiv PDFs, and Tier 2 Light QA verification are complete; proceeding to GitHub push and Drive/Doc publication.
