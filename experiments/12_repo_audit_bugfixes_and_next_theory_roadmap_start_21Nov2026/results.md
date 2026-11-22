# Live Results: Experiment 12 — Repo Audit, Bugfixes & Next Theory Roadmap (11-21-2026)

**Status:** COMPLETE (verified 11-21-2026 18:32 PST)
**Last updated:** 11-21-2026 18:32 PST
**Doc link:** <https://github.com/brando90/tvdmi-judge-asymptotics/blob/main/experiments/12_repo_audit_bugfixes_and_next_theory_roadmap_start_21Nov2026/results.md>

## Phases

| Phase | Status | Notes |
|---|---|---|
| Paper PDF build (`latex_paper/neurips/main.pdf`) | DONE | Compiled cleanly with `latexmk` (`16` pages, `288,221` bytes, zero undefined refs) |
| Lean 4 build (`experiments/00_lean_formalization/lean`) | DONE | `lake build` passed (`3,287` jobs); `#print axioms` verified standard axioms only |
| Python verification suite (`run_audit_and_verifications.py`) | DONE | Checks A–E complete (`94.54s`, `500` MC reps, `10,000` bootstrap draws) |
| Typeset Rule 91 PDF + Markdown + equation PNGs | DONE | [THEORY_AND_NEXT_EXPERIMENTS_11-21-2026.pdf](THEORY_AND_NEXT_EXPERIMENTS_11-21-2026.pdf) (`12` pages), [THEORY_AND_NEXT_EXPERIMENTS_11-21-2026.md](THEORY_AND_NEXT_EXPERIMENTS_11-21-2026.md), [RESULTS_audit_and_fixes_11-21-2026.md](RESULTS_audit_and_fixes_11-21-2026.md) |
| GitHub push + Google Drive / Google Doc sync + email | DONE | Pushed to `main` (`58679cb`+); Drive PDF <https://drive.google.com/file/d/1m3m9cs_tkTv4vUcLjUvq129Grsm0yOCp/view?usp=drivesdk>, Doc <https://docs.google.com/document/d/14JuGIAOMoWrBrI4pLcxj20oExt-okDbwU18rzdpZuLA/edit?usp=drivesdk> (`anyone` `commenter`, `8` PNGs, `0` raw math) |

## Key Measured Results (500 MC Reps, 95% Seed-Bootstrap CIs)

- **Check A (Bijection & Inversion Blindness):** Prompt-Copier gets unconditional prompt TVD-MI `0.948 [0.948, 0.948]` vs. Oracle `0.499 [0.499, 0.499]` (Spearman `-0.500`), whereas oriented task-semantic peer TVD-MI yields Oracle `+0.249`, Noisy `+0.149`, Copier `-0.000`, Inverter `-0.249` (Spearman `+0.949`).
- **Check B (`01_basic_scaling_law` Fix):** `Beta(1, beta)` + `K >= 8` log-log slopes recover `1.9596, 0.9871, 0.4906, 0.2464` vs. predicted `2.0, 1.0, 0.5, 0.25` (`R^2 >= 0.9972`).
- **Check C (Null Overfitting & Split-Sample Frontier):** At `tau(c)=0, n=50, K=1024`, Naive Max inflates to `+0.3155 [+0.3122, +0.3189]` while Split-Sample stays `+0.0063 [-0.0058, +0.0184]` (`p = 0.448`; Wilcoxon `p = 7.60e-83`); under `B = 4096, beta = 1`, split-sample attains minimum RMSE at `K* = 8, n = 512` (`0.0456 [0.0428, 0.0483]`).
- **Check D (`05_adversarial_robustness` Fix):** Non-degenerate `tau_c=0.80, mu_adv=0.20, delta=0.15, K=1024` yields Mean RMSE `0.0863` (bias `-0.0861`), 10%-Trim `0.0513`, 20%-Trim `0.0375`, Median `0.0302` (bias `-0.0289`, Wilcoxon `p = 1.26e-83` vs. Mean).
- **Check E (`04_diversity_dependence` Fix):** Local log-log elasticity (`K in [16, 256]`) degrades monotonically from `0.4990` (`rho=0`) to `0.0160` (`rho=0.80`), eliminating the free-offset `alpha -> 1` artifact.

**TLDR-end:** [tvdmi: repo-audit-and-theory-roadmap] Experiment 12 verified Lean 4 (`3,287` jobs), the NeurIPS paper PDF (`16` pages), and all five simulation/math fixes across Checks A–E, and delivered the typeset derivation PDF and Google Doc companion on Google Drive and GitHub `main`.
