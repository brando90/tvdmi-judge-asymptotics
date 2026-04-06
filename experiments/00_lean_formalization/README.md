# Experiment 00 — Lean 4 Formalization

## Goal

Formalize the main consistency theorems for TVD-MI judge ensembles in Lean 4
so the core results are machine-checked.

## Status

Natural-language proof available in `main_theorem.tex`. Lean 4 formalization
in progress.

## Files

- `main_theorem.tex` — Full natural-language statements and proofs of all
  main theorems (coverage regime, consensus regime, two-axis decomposition,
  dependence/impossibility, adversarial robustness, gold-metric consistency).
- Lean 4 source files will be added as formalization proceeds.

## Theorems to formalize

1. **Random-witness consistency** (Theorem 2.1): Best-of-K ensemble converges
   to optimal TVD witness at rate controlled by near-optimal witness mass.
2. **Consensus convergence** (Theorem 3.1): Majority vote converges
   exponentially fast to optimal critic under positive competence.
3. **Two-axis decomposition** (Theorem 4.1): Joint judge-count and sample-size
   error bound.
4. **Variance floor** (Proposition 5.1): Equicorrelated judges have irreducible
   variance when dependence persists.
5. **Adversarial breakdown** (Theorem 6.1): Contamination threshold for
   majority vote.
6. **Gold-metric rank consistency** (Theorem 7.1): Monotone validity link
   transfers consistency to gold ranking.

## Plan

1. Set up Lean 4 project with Mathlib dependency.
2. Define core types: judge, witness function, critic distribution.
3. Formalize each theorem statement.
4. Write proofs, relying on Mathlib for probability and measure theory.
5. Ensure everything compiles and type-checks.
