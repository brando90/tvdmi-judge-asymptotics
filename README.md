# tvdmi-judge-asymptotics

This repository studies the mathematics of reference-free evaluation with judge
ensembles, with a focus on Total Variation Distance–Mutual Information
(TVD-MI) and related judge-aggregation methods.

The core question is simple:

> As we increase the number of judges, can a reference-free evaluation signal
> converge to a gold or oracle target? If so, at what rate, under what
> assumptions, and with what failure modes?

This repo has three main goals:

1. **Scaling laws**
   Measure how evaluation error changes as the number of judges grows.

2. **Consistency theory**
   Prove when and why aggregated judge-based scores converge to a gold target.

3. **Lean 4 formalization**
   Formalize the setup, definitions, and core theorems in Lean 4 so the main
   results can be machine-checked.

## Core variables

- **K**: number of judges in the ensemble
  This is the main scaling parameter.

- **n**: number of items, tasks, or examples
  This is the sample-size parameter used to estimate population quantities and
  validate empirical scaling laws.

- **m**: number of pairwise comparisons per item
  Relevant when the evaluation protocol is pairwise.

- **rho**: dependence between judges
  Higher dependence means fewer effectively independent judges.

- **delta**: fraction of adversarial judges
  This controls robustness and possible impossibility regimes.

A good way to read this project is:

- increase **K** to study asymptotic consistency,
- increase **n** to estimate the error curve reliably,
- track **rho** and **delta** to understand when scaling helps and when it
  breaks.

## Main hypotheses

- As **K** increases, aggregate error decreases, ideally following a scaling
  law of the form

  `Error(K) ~ K^{-alpha}`

- Under suitable assumptions on judge quality, diversity, and dependence, the
  aggregated score converges to a gold or oracle target as `K -> infinity`.

- When a nontrivial fraction **delta** of judges is adversarial, there may be
  sharp robustness thresholds or impossibility results.

## What this repository contains

- `paper/`
  LaTeX sources for the theory and empirical papers

- `theory/`
  Definitions, assumptions, theorem statements, and proof sketches in natural
  language

- `lean/`
  Lean 4 formalization of the setup and the main results

- `experiments/`
  Simulations and judge-scaling experiments

- `notes/`
  Working notes, open problems, and design decisions

## Planned outputs

- a theory paper on asymptotic consistency for judge ensembles
- an empirical paper on scaling laws for reference-free evaluation
- Lean 4 definitions and machine-checked proofs for the core theorems

## Current focus

- formalize the evaluation setup and target metric
- define the judge ensemble model and aggregation rule
- state the consistency theorem with explicit assumptions
- characterize or estimate the scaling-law exponent `alpha`
- analyze dependence, diversity, and adversarial judges
- formalize the core results in Lean 4

## One-sentence summary

This repository is about making reference-free evaluation mathematically
precise, empirically testable, and formally verifiable.
