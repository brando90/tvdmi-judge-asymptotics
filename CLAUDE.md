# Claude Code: read `~/agents-config/INDEX_RULES.md` for all agent documentation and routing.
# If `~/agents-config/` is not present, clone it: `git clone https://github.com/brando90/agents-config.git ~/agents-config`

# CLAUDE.md — Agent Instructions for tvdmi-judge-asymptotics

## Project overview

This repository studies asymptotic consistency and scaling laws for
reference-free evaluation using judge ensembles. The core object is TVD-MI
(Total Variation Distance–Mutual Information). We aim to prove (and formalize
in Lean 4) that aggregating K diverse judges yields a score converging to a
gold target as K grows.

## Repository structure

```
tvdmi-judge-asymptotics/
├── CLAUDE.md              # this file — agent instructions
├── README.md              # project overview
├── LICENSE                # Apache 2.0
├── .gitignore
├── latex_paper/           # NeurIPS paper (LaTeX)
│   └── neurips/           # NeurIPS template files + main.tex
├── experiments/           # numbered experiment folders
│   ├── 00_lean_formalization/   # Lean 4 proof formalization
│   ├── 01_basic_scaling_law/    # best-of-K, gap ~ K^{-1/beta}
│   ├── 02_consensus_scaling/    # majority-vote, exp convergence
│   ├── 03_two_axis_decomposition/ # K x n tradeoff
│   ├── 04_diversity_dependence/ # correlated judges (rho)
│   ├── 05_adversarial_robustness/ # breakdown threshold
│   ├── 06_llm_judges/          # real LLM-judge validation
│   ├── 07–11                    # secondary extensions
│   ├── utils.py               # shared utilities
│   ├── EXPERIMENT_PLAN.md     # full implementation spec
│   └── README.md              # experiment index
├── lean/                  # Lean 4 project (future)
└── notes/                 # working notes (future)
```

## Conventions

### Experiments

- All experiments live under `experiments/` at the repo root.
- Each experiment gets a numbered folder: `00_name/`, `01_name/`, etc.
- Each experiment folder contains:
  - `README.md` or `prompt.md` — describes the experiment goal, hypothesis,
    method, and expected output.
  - Source code files (Python unless otherwise noted).
  - A `results/` subfolder (gitignored if large) for output artifacts.
- Experiment numbering is sequential. Never skip numbers.
- Experiment 00 is reserved for Lean 4 proof formalization work.

### LaTeX paper

- The paper lives in `latex_paper/neurips/`.
- Use the NeurIPS 2025 template (`neurips_2025.sty`).
- The main file is `main.tex`.
- Keep one `.bib` file: `references.bib`.

### Code style

- Python 3.10+.
- Use `numpy`, `scipy`, `matplotlib` for simulations.
- Type hints encouraged but not required in experiment scripts.
- Each experiment should be runnable standalone: `python main.py` or
  `python run.py` from the experiment folder.

### Git

- Feature branches off `main`.
- Commit messages: imperative mood, concise, one-line summary.
- Do not commit large binary files or datasets. Use `.gitignore`.

### Key notation (keep consistent across code and paper)

- `K` — number of judges
- `n` — number of samples per judge
- `rho` — inter-judge dependence (pairwise correlation)
- `delta` — adversarial fraction
- `gamma` — per-judge competence margin above chance
- `beta` — controls near-optimal witness mass geometry
- `alpha` — scaling exponent in `Error(K) ~ K^{-alpha}`
- `tau(c)` — ground-truth TVD-MI value

## What agents should do

1. Before writing code, read the relevant experiment's `README.md` to
   understand the goal.
2. Before modifying the paper, read `latex_paper/neurips/main.tex`.
3. Keep experiments self-contained and reproducible.
4. When creating a new experiment, assign the next available number and
   write the `README.md` first.
5. When unsure about math notation, refer to `latex_paper/neurips/main.tex`
   and this file for the canonical variable names.
