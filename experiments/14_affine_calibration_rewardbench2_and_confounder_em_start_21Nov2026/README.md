# Experiment 14: Semi-Supervised Calibration, `RewardBench 2`, Cross-Family Confounders & Davis–Kahan Lean 4

- **Date:** `11-21-2026`
- **Status:** Executed locally (`seed=42`, `Lean v4.29.0`); pending final Drive link backfill after Light QA & `main` push.
- **Doc link:** Pending Drive upload (local ledgers: [`RESULTS_11-21-2026.md`](RESULTS_11-21-2026.md), [`AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026.pdf`](AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026.pdf)).
- **TLDR:** Proved `11` new Lean 4 theorems in [`SpectralCalibration.lean`](../00_lean_formalization/lean/TvdmiJudge/SpectralCalibration.lean) (`54` theorems total across `TvdmiJudge`, `3,289` jobs, strictly standard axioms) and validated all 4 prioritized extensions on `SummEval` (`1,600` summaries), `LMSYS MT-Bench` (`2,575` pairs), and `RewardBench 2` (`1,753` deduplicated non-`Ties` prompts across `27` open-source reward models).

## Reproducibility Commands

```bash
# 1. Verify Lean 4 formalization (`54` theorems total, zero `sorry`)
cd /Users/brandomiranda/tvdmi-judge-asymptotics/experiments/00_lean_formalization/lean
lake build

# 2. Run Experiment 14 full real-LLM & Monte Carlo suite (~13s)
cd /Users/brandomiranda/tvdmi-judge-asymptotics/experiments/14_affine_calibration_rewardbench2_and_confounder_em_start_21Nov2026
python3 run_exp14_suite.py

# 3. Render 220 DPI equation cards & compile Rule 91 derivation PDF
python3 make_exp14_equation_images.py
pdflatex -interaction=nonstopmode AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026.tex
pdflatex -interaction=nonstopmode AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026.tex
```

## Artifact Map

- [`PROTOCOL_AND_PREDICTIONS_11-21-2026.md`](PROTOCOL_AND_PREDICTIONS_11-21-2026.md) — Pre-registered hypotheses & falsification rules.
- [`run_exp14_suite.py`](run_exp14_suite.py) — Deterministic evaluation script across `SummEval`, `MT-Bench`, `RewardBench 2`, and 500-rep Monte Carlo.
- [`RESULTS_11-21-2026.md`](RESULTS_11-21-2026.md) — Full metric tables, confidence intervals, and Wilcoxon tests.
- [`AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026.pdf`](AFFINE_CALIBRATION_RB2_CONFOUNDERS_AND_DAVIS_KAHAN_11-21-2026.pdf) / `.tex` / `.md` — Typeset Rule 91 derivations.
- `doc_equation_images_11-21-2026/eq01..eq06.png` — Standalone 220 DPI equation cards for Google Doc companion.
- `results/{summary_metrics.json,trace_one_item_rb2.json,*.csv,exp14_calibration_rb2_and_confounders_6panel.{png,pdf}}` — Machine-readable outputs and 6-panel figure.
- [`CKPT_exp14.md`](CKPT_exp14.md) — Live session checkpoint.
