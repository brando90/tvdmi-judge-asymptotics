# Reference-Free Evaluation with Judge Ensembles: Scaling Laws, Consistency Proofs and Lean 4 (TVD-MI)

> **TLDR (11-21-2026)**
> - **What it is:** a research repo on reference-free evaluation with judge ensembles (scoring with TVD-MI, Total Variation Distance–Mutual Information): does the aggregated score converge to a gold target as the number of judges K grows, at what rate, and when does it break (dependent, cloned, or adversarial judges, or shared confounders)?
> - **Status:** Lean 4 formalization compiled across `Basic.lean`, `NewTheory.lean`, and `SpectralCalibration.lean`; historical scripts `01`–`11` reconciled in Experiment 12–13; real-LLM validation & falsification completed on `SummEval` (`1,600` items), `LMSYS MT-Bench` (`2,575` pairs), and `RewardBench 2` (`1,753` deduplicated non-tie prompts, `27` open-source reward/cost models: `23` main + `4` PKU Beaver models) in Experiments 13–14; dual NeurIPS (`main.pdf`) and arXiv (`main_arxiv.pdf`) builds maintained.
> - **Where things are:** `experiments/` (`00` Lean 4, `01`–`14` empirical/theoretical experiments, roadmap in `experiments/README.md`), `theory/assumptions.md`, `latex_paper/neurips/main.tex` & `main_arxiv.tex`, `CLAUDE.md`, and `AGENTS.md`.

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

- `latex_paper/neurips/`
  NeurIPS paper (LaTeX, using neurips_2025 template)

- `experiments/`
  Numbered experiment folders (00–10+), each self-contained with a README
  describing goal, hypothesis, method, and expected output.
  See `experiments/README.md` for the full experiment plan.

- `lean/`
  Lean 4 formalization of the setup and the main results (future)

- `notes/`
  Working notes, open problems, and design decisions (future)

- `CLAUDE.md`
  Agent instructions and project conventions

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

## Citation

If you find this repository, mathematical derivations, Lean 4 formalization, or `RB-Spec TVD-MI` experiments useful, please cite:

```bibtex
@misc{miranda2026tvdmijudgeasymptotics,
  title  = {Asymptotic Consistency of Reference-Free Evaluation with TVD-MI Judge Ensembles},
  author = {Miranda, Brando and Koyejo, Sanmi},
  year   = {2026},
  url    = {https://github.com/brando90/tvdmi-judge-asymptotics},
  note   = {Preprint and Lean 4 formalization}
}
```

## TODO

- [ ] **TODO (Authorship):** Ask Rylan Schaeffer if he wants to be an author on this paper/project, and update the paper author list and BibTeX citation above accordingly.
