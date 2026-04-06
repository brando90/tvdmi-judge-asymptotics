# Experiment 00 — Lean 4 Formalization

## Goal

Formalize the main consistency theorem for judge ensembles in Lean 4 so the
core result is machine-checked.

## Status

Waiting for the natural-language proof to be finalized (will be copied from
GPT-5 Pro once ready).

## Plan

1. Receive the natural-language proof of the consistency theorem.
2. Translate definitions into Lean 4: judge model, aggregation rule, error
   metric, convergence statement.
3. Formalize assumptions (judge quality, independence/dependence structure,
   adversarial fraction).
4. Write the Lean 4 proof, lemma by lemma.
5. Ensure the proof compiles and type-checks.

## Inputs

- Natural-language proof (to be provided).
- Notation from `latex_paper/neurips/main.tex`.

## Outputs

- Lean 4 source files with machine-checked proofs.
- These will eventually live in a top-level `lean/` directory.
