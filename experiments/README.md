# Experiments

Each experiment lives in a numbered folder. Run from its own directory. See each folder's `README.md` for details. The full implementation spec is in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).

| # | Folder | Theorem | One-line summary |
|---|--------|---------|-----------------|
| 00 | `00_lean_formalization/` | -- | Formalize consistency theorems in Lean 4 (machine-checked proofs) |
| 01 | `01_random_witness_scaling/` | Thm 1 | Best-of-K witnesses: gap ~ K^{-1/beta}, controlled by tail geometry |
| 02 | `02_consensus_scaling/` | Thm 2 | Majority-vote: exponential convergence 2 exp(-2K*gamma^2) |
| 03 | `03_two_axis_decomposition/` | Thm 3 | K x n tradeoff: approximation error + estimation error |
| 04 | `04_diversity_dependence/` | Var formula | Correlated judges (rho): irreducible error floor at sigma^2*rho |
| 05 | `05_adversarial_robustness/` | Breakdown | Phase transition at delta* = 2*gamma/(1+2*gamma) |
| 06 | `06_llm_judges/` | All | Validate on real LLM-as-judge data (optional) |

Shared utilities: [`utils.py`](utils.py)

## Adding a new experiment

1. Create the next numbered folder: `NN_short_name/`.
2. Write a `README.md` inside it describing goal, method, and expected output.
3. Implement as a standalone Python script (`python main.py`).
4. Store results in a `results/` subfolder (gitignored if large).
