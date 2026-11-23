# Live Results: Experiment 17 — Cross-Company Mega QA & Collaborator Brief (`11-23-2026`)

Date: 11-23-2026

**Status:** VERIFIED (`11-23-2026 12:07 PST`, started `11-23-2026 10:52:43 PST`)
**Last updated:** 11-23-2026 12:08 PST
**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/17_cross_company_mega_qa_and_collaborator_brief_start_23Nov2026/results.md>

**TLDR:** Completed all 3 cross-company Mega QA stages (`Stage 1: Codex FIXED -> Stage 2: Claude Code FIXED -> Stage 3: Codex FIXED`) and post-Stage-3 deterministic verification (`lake build` `3,290` jobs, `66/66` axioms, `21/21` Python files, `21/23/1`-page warning-free PDFs, zero temp files/secrets/Rule-91 violations), plus `collaborators.py` push (`8b8246b`); now landing `main` + `agent-sync` and dispatching email to `Sanmi Koyejo` and `Henry Bosch`.

| Stage / Gate | CLI / Model & Effort (`Requested -> Actual`) | Status | Verdict / Notes |
|---|---|---|---|
| Setup: Summary Slide & Draft Email | `TVDMI_SUMMARY_SLIDE_11-23-2026.pdf` + draft brief | DONE | 1-page slide (`0` overfulls) + visually inspected; email polished (`SUBJECT:`, `brando90` links, defined terms, verbatim signature) |
| Stage 1 (Cross-Company Pass 1) | `codex exec` `gpt-6.2-astra` (`xhigh`) -> `gpt-6.2-astra` (`xhigh`) | DONE | `FIXED` (`0` crit, `6` maj, `8` fixes, `IMPROVED`): fixed CV leakage (`MT-Bench` `0.7938`), recovered `4` raw `JudgeBench` votes (`o1-preview` `75.4%`, `CR-SRB` `71.1%` vs Vote `63.1%`, `p=0.00253`), pooled cluster CIs (`0.647`, `0.803`), strengthened `MultiFactorContinuous.lean` & Hoeffding exponent, replayed Exps `12`–`15` + `11` smokes + `66/66` axioms |
| Stage 2 (Opposite-Company Pass 2) | `claude` `opus[1m]` (`xhigh`) via `~/.claude-su` -> `claude-opus-5-5` (`xhigh`) (personal `~/.claude` `fable` hit 11pm limit) | DONE | `FIXED` (`0` crit, `0` maj, `1` fix, `PASS`): independently rebuilt 33-judge `JudgeBench` vote matrix (`4` cells fixed -> `0.7543`), fixed `0.5114 -> 0.5086` typo in Exp 15 results line 40, verified `66/66` axioms, Exps `12`–`15` byte-identical replay, PDFs; flagged `mtbench_features.py` & `LMSYS MT-Bench` |
| Stage 3 (Fresh Clean-Eyes Pass 3) | `codex exec` `gpt-6.2-astra` (`xhigh`) -> `gpt-6.2-astra` (`xhigh`) (`191,576` tokens) | DONE | `FIXED` (`0` crit, `2` maj, `7` fixes, `IMPROVED`): separated known-family `1/F` bound from empirical shrinkage weights in Exp 13 & brief, fixed Exp 12 tie term `(1/2) P(Q=1/2)` & expectation over random votes, clarified Exp `06` as synthetic simulation (`CLAUDE.md`, `experiments/README.md`), refined `98_appendix_experiments.tex` |
| Post-Stage-3 Deterministic Gate & Contact DB | Orchestrator CLI checks + `~/ultimate-utils` (`8b8246b`) | DONE | `3,290`-job `lake build`, `66/66` stripped `#print axioms`, `21/21` `py_compile`, `utils.py` (`TV=0.1828`), `main.pdf` (`21`p)/`main_arxiv.pdf` (`23`p)/`TVDMI_SUMMARY_SLIDE_11-23-2026.pdf` (`1`p) `0` warnings, `0` temp files, `diff --check` & secrets clean, `Rule-91` clean, `Henry Bosch` pushed (`8b8246b`), report + HTML companion written |
| Git Push (`main` + `agent-sync`) & Email | Git CLI + SMTP (`brandojazz@gmail.com`) | IN PROGRESS | Staging explicit files (`mtbench_features.py` + Exp 17), pushing `main`, logging `agent-sync`, verifying HTTP `200`, and emailing Sanmi + Henry B (`BCC: brando.science@gmail.com`) |

**TLDR-end:** [tvdmi: xcomp-mega-qa-email] Initialized Experiment 17 (`11-23-2026 10:52 PST`); building summary slide and email draft before Stage 1 Codex Mega QA.
