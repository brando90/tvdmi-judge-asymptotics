# Experiment 13: Real-LLM Validation, RB-Spec TVD-MI Theory, and Lean 4 Proofs

- **Date:** `11-21-2026`
- **Status:** COMPLETE (`11-21-2026 19:25 PST`; `seed=42`, `Lean v4.29.0`, Light QA `gemini-4-neon-xhigh --effort xhigh` reconciled, Drive/Doc & `main` published).
- **Doc link:** <https://docs.google.com/document/d/1ObShZ0GCxgpH_Di4DS46tvJc05Jggl2RuRo5JmI18Go/edit?usp=drivesdk>
- **Primary Drive Derivation PDF:** <https://drive.google.com/file/d/1-XyUnWIX-B6MTyC7QVlIwXgBYmRYVIcc/view?usp=drivesdk>
- **Drive NeurIPS Venue PDF / arXiv PDF:** <https://drive.google.com/file/d/1x8TjkHtVkEbC3XTmY1PWZM9Vk_JpOlB4/view?usp=drivesdk> · <https://drive.google.com/file/d/1tygAthFmh6kBakFjnCEVjXJvg9crlD7Z/view?usp=drivesdk>
- **Google Drive Experiment 13 Folder:** <https://drive.google.com/drive/folders/1lsRCjER2ZBguuQ59eYqgDpK3fS5WHZcr>
- **TLDR:** Unconditional prompt-response TVD-MI fails on real `SummEval` data (`0.4851` verbatim prompt copy beats every real model `<= 0.3277`), and naive `1/K` judge ensembles suffer clone hijacking (`0.9853 -> 0.8265` across `64` BART-CNN prompt paraphrases) and flip sign (`-0.5001` naive SVD, `-0.2008` uniform mean) under cloned adversaries. Our proposed redundancy-balanced spectral conditional peer estimator (`RB-Spec TVD-MI`) achieves `+0.6890` item Spearman (`+0.9265` system) in wild `SummEval` (`K=38`), `+0.7033` (`+0.9735` system) under `66.7%` raw adversarial flooding, and `0.7950` held-out pairwise accuracy (`+0.9429` model win-rate Spearman) on `LMSYS MT-Bench`, backed by 15 new Lean 4 theorems (`TvdmiJudge/NewTheory.lean`, zero `sorry`, standard axioms `[propext, Classical.choice, Quot.sound]`).

---

## Pre-Registration & Protocol

See [PROTOCOL_AND_PREDICTIONS_11-21-2026.md](PROTOCOL_AND_PREDICTIONS_11-21-2026.md) for hypotheses, falsification criteria, and dataset SHA-256 verification before running code.

## Reproducibility Commands

Public raw benchmark caches stay outside git in `/tmp/corr-ce-public-ordinal/summeval` (`100` articles x `16` systems = `1,600` summaries, `G-Eval` GPT-4 20-sample completions, `BARTScore` 70 `en` + 70 `de` prompt variants, `BART-para`, `BART-CNN`, `PRISM`) and `/tmp/mt_bench_public/` (`human.parquet` `3,355` rows, `gpt4.parquet` `2,400` rows, `80` questions, `2,575` decisive human pairs).

```bash
# 1. Verify Lean 4 proofs (zero sorry, standard axioms)
cd /Users/brandomiranda/tvdmi-judge-asymptotics/experiments/00_lean_formalization/lean
lake build

# 2. Run the full 6-module real-LLM + Monte Carlo verification suite (~72s)
cd /Users/brandomiranda/tvdmi-judge-asymptotics/experiments/13_real_llm_validation_hierarchical_rb_spectral_theory_and_lean_start_21Nov2026
python3 run_real_llm_and_theory_suite.py

# 3. Render standalone 220 DPI equation cards and compile the Rule 91 PDF
python3 make_exp13_equation_images.py
pdflatex -interaction=nonstopmode RB_SPEC_TVDMI_THEORY_LEAN_AND_REAL_LLM_11-21-2026.tex
pdflatex -interaction=nonstopmode RB_SPEC_TVDMI_THEORY_LEAN_AND_REAL_LLM_11-21-2026.tex
```

## Artifact Map

- `run_real_llm_and_theory_suite.py` — End-to-end deterministic validator (`SummEval`, `MT-Bench`, 500-rep Monte Carlo, Figure 1 & CSV/JSON outputs).
- `results/exp13_real_llm_and_theory_6panel.png` / `.pdf` — 6-panel main figure (`SummEval` copy blindness, `M=0..64` clone sweep, null elimination, `K=38` & `Adv-Flood`, `MT-Bench`, Lean MC).
- `results/summary_metrics.json`, `results/summeval_heldout_regimes.csv`, `results/summeval_clone_sweep.csv`, `results/mtbench_pairwise.csv`, `results/trace_one_item.json` — Raw structured outputs.
- `RESULTS_11-21-2026.md` & `results.md` — Full numerical tables, paired Wilcoxon tests, ICC/MAE calibration boundaries, and scope limits.
- `RB_SPEC_TVDMI_THEORY_LEAN_AND_REAL_LLM_11-21-2026.{pdf,tex,md}` — Publication-grade derivation PDF, LaTeX source, and structured companion markdown.
- `doc_equation_images_11-21-2026/eq01_*.png` .. `eq08_*.png` — Standalone 220 DPI equation figures.
- `make_exp13_equation_images.py` — Deterministic script rendering `eq01`–`eq08`.
- `../00_lean_formalization/lean/TvdmiJudge/NewTheory.lean` — 15 machine-checked Lean 4 theorems supporting Sections 2–8 of the writeup.
- `CKPT_real_llm_and_lean_theory.md` — Session checkpoint log.
