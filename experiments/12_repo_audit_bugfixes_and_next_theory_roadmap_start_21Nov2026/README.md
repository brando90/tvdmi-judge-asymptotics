# Experiment 12 — Repository Audit, Bugfixes, and Next Theory & Experiment Roadmap (Started 11-21-2026)

Date: 11-21-2026

**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/12_repo_audit_bugfixes_and_next_theory_roadmap_start_21Nov2026/README.md>

**TLDR:** Full audit of `brando90/tvdmi-judge-asymptotics` (`experiments/00_`–`11_`, `latex_paper/neurips/`, `theory/assumptions.md`) identifying four simulation/math bugs, verifying Lean 4 (`lake build`) and LaTeX paper compilation, validating exact fixes plus two new theoretical results (prompt-copy bijection blindness vs. oriented conditional peer TVD-MI, and split-sample cross-fitted best-of-$K$ compute-optimal scaling), and delivering a Trigger Rule 91 LaTeX derivation PDF + Google Drive / Google Doc companion.

## Google Drive & Companion Links

- Typeset Derivation PDF (`THEORY_AND_NEXT_EXPERIMENTS_11-21-2026.pdf`): *pending upload update*
- Google Doc Companion (typeset equation PNGs + audit summary): *pending upload update*
- GitHub Markdown derivation: [THEORY_AND_NEXT_EXPERIMENTS_11-21-2026.md](THEORY_AND_NEXT_EXPERIMENTS_11-21-2026.md)
- Live results ledger: [results.md](results.md)

## Goal

1. Audit every experiment (`00`–`11`), the NeurIPS draft (`latex_paper/neurips/`), and the Lean 4 formalization (`experiments/00_lean_formalization/lean/`) for mathematical errors, simulation bugs, missing verification, and unaddressed theoretical/empirical gaps.
2. Reproduce and verify exact fixes for the four defects found without overwriting historical April 2026 records (`01_`, `03_`, `04_`, `05_`, `11_`).
3. Derive, verify numerically, and typeset (Trigger Rule 91) the next core theorems and ranked experimental roadmap (including zero-cost real-LLM reuse from sister repos, synthetic/LLM calibration, and Lean 4 formalization upgrades).

## Hypothesis

- Replacing `Beta(beta, 1)` with `Beta(1, beta)` and `b = 0` power-law fitting in Experiment 01 restores the monotone `1/beta` exponent ordering (`2.0, 1.0, 0.5, 0.25`).
- Naive empirical best-of-$K$ overfits under the null (`tau(c) = 0`) at rate `O(sqrt(log(K)/n))`, whereas split-sample cross-fitted best-of-$K$ achieves zero mean bias under the null and yields an interior compute-optimal frontier `K*(B) ~ (B / log B)^{beta/(beta+2)}`.
- Unconditional prompt-response TVD-MI ranks prompt-copying (`Y = X`) above the oracle solver and cannot separate oracle from systematic inversion, whereas oriented conditional peer TVD-MI restores exact rank ordering.
- Non-degenerate adversarial scalar contamination (`tau_c = 0.80`, `mu_adv = 0.20`, `delta = 0.15`) exposes both the `O(delta)` simple-mean error floor and the exact `O(sigma * delta)` Huber quantile shift of the sample median.
- Measuring local log-log elasticity (`K in [16, 256]`) instead of `fit_power_law` with free offset `b` recovers the theoretical monotone degradation of effective exponent with correlation `rho`.

## Files

- `run_audit_and_verifications.py` — Standalone deterministic audit & verification script (seed `42`, `500` MC reps, `10,000` seed-bootstrap draws for 95% CIs).
- `make_equation_images.py` — Compiles standalone LaTeX equation PNGs (`doc_equation_images_11-21-2026/`) for the Google Doc companion.
- `THEORY_AND_NEXT_EXPERIMENTS_11-21-2026.tex`, `.pdf`, `.md` — Detailed Trigger Rule 91 derivations, metric limits, new theorems, and ranked experiment roadmap.
- `results/` — Verification CSVs, `summary_metrics.json`, and 4-panel figure (`exp12_audit_and_fixes_4panel.{pdf,png}`).
- `results.md` — Live execution ledger with measured tables and TLDR-end.
- `CKPT_repo_audit_and_theory_roadmap.md` — Resumable task checkpoint.

## How to Run

```bash
cd experiments/12_repo_audit_bugfixes_and_next_theory_roadmap_start_21Nov2026
python3 run_audit_and_verifications.py
```
