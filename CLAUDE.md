# Claude Code: read `~/agents-config/INDEX_RULES.md` for all agent documentation and routing.

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
│   ├── 01_basic_scaling_law/    # basic K-scaling simulation
│   ├── ...
│   └── README.md          # experiment plan overview
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
- `n` — number of items/examples
- `m` — pairwise comparisons per item
- `rho` — inter-judge dependence
- `delta` — adversarial fraction
- `alpha` — scaling exponent in `Error(K) ~ K^{-alpha}`

## What agents should do

1. Before writing code, read the relevant experiment's `README.md` to
   understand the goal.
2. Before modifying the paper, read `latex_paper/neurips/main.tex`.
3. Keep experiments self-contained and reproducible.
4. When creating a new experiment, assign the next available number and
   write the `README.md` first.
5. When unsure about math notation, refer to `latex_paper/neurips/main.tex`
   and this file for the canonical variable names.
