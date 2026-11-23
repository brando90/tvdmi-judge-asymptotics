# Experiment 00 — Lean 4 Formalization

## Goal

Formalize the main consistency theorems for TVD-MI judge ensembles in Lean 4
so the core results are machine-checked.

## Status as of 11-23-2026

The project contains 66 checked theorems. The mathematical overview is in [main_theorem.tex](main_theorem.tex). The four [Lean modules](lean/TvdmiJudge) contain the exact hypotheses and conclusions.

| Module | Theorems | Coverage |
|---|---:|---|
| Basic | 28 | Algebraic and limiting statements for coverage, consensus, dependence, and ranking |
| NewTheory | 15 | Finite total variation, peer contrasts, clone families, and weight inequalities |
| SpectralCalibration | 11 | Bias, calibration, shared variance, and conditional spectral perturbation |
| MultiFactorContinuous | 12 | Shrinkage, proxy projection, conditional rank bounds, and measurable event witnesses |

Run `lake build` from [lean/](lean/). The rank bounds start from Rayleigh inequalities. The continuous event result takes the positive and negative set conditions of a Hahn decomposition as hypotheses.
