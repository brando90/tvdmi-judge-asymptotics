# tvdmi-judges

This repository studies the mathematics of reference-free evaluation with judge
ensembles. The main object is Total Variation Distance–Mutual Information
(TVD-MI) and related reference-free scores built from many imperfect but
diverse judges.

The central question is:

> If we increase the number of judges, can an aggregated reference-free score
> converge to a gold or oracle evaluation signal? If yes, at what rate, and
> under what assumptions?

This repo has three goals:

1. **Scaling laws.** Measure how evaluation error changes as the number of
   judges grows.

2. **Consistency theory.** Prove when the aggregated score converges to the
   gold target.

3. **Lean 4 formalization.** Write the definitions and theorems in Lean 4 so
   the core proofs can be machine-checked.

## Core variables

- **K**: number of judges in the ensemble.  
  This is the main scaling parameter.

- **n**: number of items, tasks, or examples.  
  This is the sample-size parameter used to estimate population quantities and
  validate the scaling laws.

- **m**: number of pairwise comparisons per item, when the protocol is
  pairwise.

- **rho**: dependence between judges.  
  Higher dependence means fewer effectively independent judges.

- **delta**: fraction of adversarial judges.

A simple way to read the project is:

- Increase **K** to study asymptotic consistency.
- Increase **n** to estimate the error curve reliably.
- Track **rho** and **delta** to understand when scaling helps and when it
  breaks.

## Main hypotheses

- As **K** grows, aggregate error decreases, ideally following a scaling law
  of the form `Error(K) ~ K^{-alpha}`.

- Under suitable assumptions on judge quality, diversity, and dependence, the
  aggregated TVD-MI-style score converges to a gold or oracle target as
  `K -> infinity`.

- If a nontrivial fraction **delta** of judges is adversarial, there may be
  robustness thresholds or impossibility results.

## What is in this repo?

- `paper/` — LaTeX sources for the theory and empirical papers
- `theory/` — definitions, assumptions, theorem statements, and proof sketches
  in natural language
- `lean/` — Lean 4 formalization of the setup and core results
- `experiments/` — simulations and judge-scaling experiments
- `notes/` — working notes, open problems, and design decisions

## Planned outputs

- a theory paper on asymptotic consistency for judge ensembles
- an empirical paper on scaling laws for reference-free evaluation
- Lean 4 definitions and machine-checked proofs for the core theorems

## Status

- [ ] Formalize the evaluation setup and target metric
- [ ] Define the judge ensemble model and aggregation rule
- [ ] State the consistency theorem with explicit assumptions
- [ ] Derive or estimate the scaling-law exponent `alpha`
- [ ] Analyze dependence, diversity, and adversarial judges
- [ ] Formalize the core results in Lean 4

This repository is about one core idea: reference-free evaluation should not
only be measured empirically; it should also be stated mathematically and,
where possible, proved formally.
