# Experiment 17: Cross-Company Mega QA & Collaborator Brief Report (`11-23-2026`)

- **Date:** `11-23-2026`
- **Campaign Anchor (`QA_BASE`):** `cb2a2c0` | **Branch Base (`Exp 17`):** `7ea00af` (`origin/main`)
- **Started:** `11-23-2026 10:52:43 PST` | **Verified:** `11-23-2026 12:07 PST`
- **External Contact Update:** `brando90/ultimate-utils` commit `8b8246b` (`Henry Bosch` -> `hbosch@stanford.edu`, `"Henry B"`, `"Stanford"`)

## 1. Executive Verdict

True cross-company 3-stage sequential Mega Quality Assurance (`OpenAI Codex -> Anthropic Claude Code -> OpenAI Codex` fresh context) plus an independent deterministic orchestrator verification gate passed with **zero open critical or major findings** across `/Users/brandomiranda/tvdmi-judge-asymptotics`, the 1-page summary slide (`TVDMI_SUMMARY_SLIDE_11-23-2026.pdf`), and the collaborator brief (`collaborator_email_sanmi_henry.txt`).

## 2. Three-Stage Cross-Company Audit Ledger

| Stage | Provider / CLI | Model & Effort (`Requested -> Actual`) | Auth / Profile Note | Tokens / Exit | Verdict | Key Substantive Fixes & Findings |
|---|---|---|---|---|---|---|
| Stage 1 | OpenAI Codex (`codex exec`) | `gpt-6.2-astra` (`xhigh`) -> `gpt-6.2-astra` (`xhigh`) | Default `~/.codex` (API env scrubbed) | `540,748` / `exit 0` | `VERDICT: FIXED` (`0` crit, `6` maj, `8` fixes, `IMPROVED`) | Eliminated `MT-Bench` TF-IDF & Platt fold leakage (`0.7950 -> 0.7938`, created `mtbench_features.py`); recovered `4` reverse-order `JudgeBench` votes (`o1-preview = 264/350 = 75.4%`, `CR-SRB` `71.1%` vs Vote `63.1%`, `p = 0.00253`, vs SVD `66.0%`, `p = 0.0143`); fixed pooled cluster bootstrap CIs (`SummEval` `0.647 [0.612, 0.681]`, `MT-Bench` `0.803 [0.777, 0.829]`); strengthened `MultiFactorContinuous.lean`; fixed two-sample exponent (`main_theorem.tex`) & attributions (`references.bib`, `08_gold_consistency.tex`); updated checklist Open Access answer to `\answerNo{}`; replayed Exps `12`–`15` + `11` legacy smokes. |
| Stage 2 | Anthropic Claude Code (`claude`) | `opus[1m]` (`xhigh`) -> `claude-opus-5-5` (`xhigh`) | Stanford `~/.claude-su` via `CLAUDE_CODE_OAUTH_TOKEN` (personal `~/.claude` `fable` hit 11pm limit) | `exit 0` | `VERDICT: FIXED` (`0` crit, `0` maj, `1` fix, `PASS`) | Independently rebuilt 33-judge `JudgeBench` vote matrix (`4` cells fixed -> `0.7543`); fixed `0.5114 -> 0.5086` table typo in `experiments/15_.../RESULTS_11-21-2026.md` line 40; verified `66/66` Lean axioms, byte-identical Exps `12`–`15` replay, `21`/`23`/`1`-page PDFs; flagged untracked `mtbench_features.py` & `"the chat benchmark"` -> `LMSYS MT-Bench`. |
| Stage 3 (Fresh Context) | OpenAI Codex (`codex exec`) | `gpt-6.2-astra` (`xhigh`) -> `gpt-6.2-astra` (`xhigh`) | Default `~/.codex` (fresh context, API env scrubbed) | `191,576` / `exit 0` | `VERDICT: FIXED` (`0` crit, `2` maj, `7` fixes, `IMPROVED`) | Separated known-family equal-weighted `O(1/F)` variance guarantee from empirical covariance-shrinkage weights in Exp 13 & collaborator brief; restored missing tie term `(1/2) P(Q = 1/2)`, conditionally i.i.d. hypothesis given `(Z, W)`, equicorrelated RMSE, and random-vote expectation in Exp 12 summary; clarified Exp `06` as synthetic judge simulation vs Exps `13`–`15` real benchmark evaluations (`CLAUDE.md`, `experiments/README.md`); updated `98_appendix_experiments.tex` note on five-fold splits vs clone-noise draws. |

## 3. Post-Stage-3 Orchestrator Polishing & Deterministic Verification Gate

- **Collaborator Brief Polish (`collaborator_email_sanmi_henry.txt`):** Restored live `brando90` GitHub links alongside the attached PDFs (verified before sending once pushed), defined `TVD-MI`, `covariance shrinkage`, `SVD`, and `ICC` on first use in plain language, kept 5 verified bullet groups with explicit baselines, 95% intervals, and caveats, expanded `Large Model Systems Organization (LMSYS) Multi-Turn Benchmark (MT-Bench)`, and restored the verbatim Stanford signature (`Brando (drafted with Antigravity)`).
- **Lean 4 Formalization (`experiments/00_lean_formalization/lean/`):** `lake build` (`3,290` jobs) succeeded; comment-stripped source audit found `0` `sorry`, `admit`, `native_decide`, or custom axioms; `#print axioms` across all `66 / 66` theorems (`28 + 15 + 11 + 12`) reported only `[propext, Classical.choice, Quot.sound]`.
- **Python Suite & Utilities:** `21 / 21` Python files compiled cleanly; `experiments/utils.py` smoke test (`make_joint_distribution` + `compute_true_tvd`) returned `TV = 0.1828`; all temporary `__pycache__` directories were removed.
- **LaTeX & PDF Verification:**
  - `latex_paper/neurips/main.pdf`: `21` pages (`References` starts on Page 10, `0` warnings/overfulls, `Anonymous Author(s)` verified, `0` author/affiliation leaks, `0` `[TODO]`s).
  - `latex_paper/neurips/main_arxiv.pdf`: `23` pages (`0` warnings/overfulls, `557,758` bytes).
  - `experiments/17_cross_company_mega_qa_and_collaborator_brief_start_23Nov2026/TVDMI_SUMMARY_SLIDE_11-23-2026.pdf`: `1` page (`0` warnings/overfulls, `94,440` bytes).
- **Hygiene, Diff & Security Scans:** `0` temporary `.aux/.log/.nav/.out/.snm/.toc/.vrb/.bbl/.blg/.pyc/.fls/.fdb_latexmk` files or `__pycache__` dirs; `git -c core.whitespace=cr-at-eol diff --check` passed on both `cb2a2c0` and `7ea00af`; `0` secret regex hits; `0` Rule-91 forbidden characters across HTML companions and the email brief.

## 4. External Artifacts & Dispatch Outcomes

- **`collaborators.py` (`~/ultimate-utils`):** Added `hbosch@stanford.edu`, `"Henry B"`, and `"Stanford"` for `Henry Bosch` (`8b8246b`) and added `"Ethan H"` alias to `Ethan Hersch` (`ehersch@stanford.edu`, `3171bb8`) via detached worktrees on `brando90/ultimate-utils`; `resolve("Henry B")` and `resolve("Ethan H")` verified.
- **`tvdmi-judge-asymptotics` & `~/agent-sync`:** Pushed `12c0771` to `origin/main` (HTTP `200` confirmed on all 4 linked GitHub URLs); surgically removed Stage 1 Codex's premature unstaged line from `~/agent-sync/coding/log.md` preserving concurrent entries, and logged Exp 17 via detached worktree (`8bb650b`).
- **Dispatched Collaborator Emails (plus internal Rule 14 receipts to `brando.science@gmail.com`):**
  - `Sanmi Koyejo <sanmi@stanford.edu>, Henry Bosch <hbosch@stanford.edu>` (`BCC: brando.science@gmail.com`), attachments `TVDMI_SUMMARY_SLIDE_11-23-2026.pdf` + `main_arxiv.pdf`: `Message-ID=<179546458245.60794.7492552956154424024@gmail.com>` (Rule 14 receipt `<179546458246.60794.14250693798573697056@gmail.com>`).
  - Follow-up `Ethan Hersch <ehersch@stanford.edu>` (`BCC: brando.science@gmail.com`, standalone without duplicate send to Sanmi/Henry), attachments `TVDMI_SUMMARY_SLIDE_11-23-2026.pdf` + `main_arxiv.pdf`: `Message-ID=<179546477648.61503.14514811252898834304@gmail.com>` (Rule 14 receipt `<179546477648.61503.4577546329141733005@gmail.com>`).
