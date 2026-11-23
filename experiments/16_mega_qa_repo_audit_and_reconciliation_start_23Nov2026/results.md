# Live Results: Experiment 16 — Full-Repo 3-Stage `agy` Mega QA (`11-23-2026`)

Date: 11-23-2026

**Status:** COMPLETE (`11-23-2026 09:59 PST`, all 3 Mega QA stages + orchestrator deterministic verification PASSED; see [MEGA_QA_REPORT_11-23-2026.md](MEGA_QA_REPORT_11-23-2026.md))
**Last updated:** 11-23-2026 09:59 PST
**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/16_mega_qa_repo_audit_and_reconciliation_start_23Nov2026/results.md>

**TLDR:** Completed 3-stage sequential `agy` Mega QA (`Stage 1: gemini-pro-orchestrator-neon` fallback -> `Stage 2: gemini-4-neon-xhigh` -> `Stage 3: fresh gemini-pro-orchestrator-neon` fallback; Anthropic `429` `59h18m`–`59h43m` reset recorded with missing cross-company diversity noted) over `QA_BASE = cb2a2c0` and passed all post-QA deterministic gates (`lake build` `3,290` jobs, `66/66` theorem axiom checks, `20/20` `py_compile`, `main.pdf` 21p anonymous Page-9 cutoff, HTML Rule 91, secret scan). Google Drive upload remains pending user OAuth refresh (`~/keys/gspread_authorized_user.json` `Nov 14 20:04`).

| Stage / Gate | Model & Effort (`Requested -> Actual`) | Status | Verdict / Notes |
|---|---|---|---|
| Stage 1 (Clean-Eyes Pass 1) | `claude-opus-5-5-high` (`high`) -> `claude-sonnet-5-5-high` (`high`) -> `gemini-pro-orchestrator-neon` (no `--effort`) | DONE | `FIXED` (`0` crit, `4` maj, `9` fixes, `IMPROVED`); Anthropic `429` (`59h42m`); missing cross-company diversity noted |
| Stage 2 (Builder Flagship Intent Pass) | `gemini-4-neon-xhigh` (`xhigh`) -> `gemini-4-neon-xhigh` (`xhigh`) | DONE | `FIXED` (`0` crit, `0` maj, `3` fixes, `PASS`); retained all Stage 1 fixes; fixed Exp 14 HTML underscores, `JudgeBench` ack, cleaned 36 temp files |
| Stage 3 (Clean-Eyes Final Pass 2) | `claude-opus-5-5-high` (`high`) -> fresh `gemini-pro-orchestrator-neon` (no `--effort`) | DONE | `PASS` (`0` crit, `0` maj, `0` fixes, `PASS`); Anthropic `429` (`59h18m`); missing cross-company diversity noted |
| Deterministic Post-Mega-QA Verification Gate | Orchestrator CLI checks | DONE | `lake build` `3,290` jobs + `66/66` standard axioms; `20/20` Python compile + `utils.py` TVD check; `main.pdf` `21`p Page-9 anon + `0` TODO; Rule 91 HTML & secret scan clean |

**TLDR-end:** [tvdmi: mega-qa] Completed 3-stage `agy` Mega QA and full deterministic verification (`66/66` Lean theorems, `20/20` Python scripts, NeurIPS/arXiv PDFs, and HTML/secret scans clean).
