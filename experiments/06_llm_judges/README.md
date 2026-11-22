# Experiment 06 — End-to-End Scaling Law with LLM Judges (Optional)

**As of 11-21-2026:** `experiments/06_llm_judges/main.py` remains a historical synthetic Gaussian simulation. Real LLM-as-judge validation (on `SummEval` `1,600` summaries x `224` reference-free judge outputs and `LMSYS MT-Bench` `3,355` human + `2,400` GPT-4 pairwise judgments) is implemented in [`experiments/13_real_llm_validation_hierarchical_rb_spectral_theory_and_lean_start_21Nov2026/`](../13_real_llm_validation_hierarchical_rb_spectral_theory_and_lean_start_21Nov2026/).

**Tests:** All theorems on real data

## Purpose

Validate the synthetic findings with real LLM-as-judge evaluations.

## Simulation design

1. **Dataset**: Use a public evaluation benchmark where gold scores exist
   (e.g., MT-Bench, Chatbot Arena, or AlpacaEval with human annotations).

2. **Judges**: Use a pool of diverse LLMs as judges. If API access is limited,
   use different prompting strategies or temperature settings on the same model
   to create pseudo-diverse judges. E.g.:
   - Same model, 10 different system prompts (strict, lenient, focused on
     different quality dimensions).
   - Same model at temperatures {0.0, 0.3, 0.5, 0.7, 1.0}.
   - Different models if available (e.g., Claude, GPT-4, Gemini, Llama,
     Mistral).

3. **Protocol**:
   - Collect scores from all available judges on all items.
   - For each K = 1, 2, ..., max_judges:
     - Subsample K judges (average over all (max_judges choose K) subsets,
       or sample 200 random subsets).
     - Aggregate scores (mean, median, max).
     - Compute correlation with gold (Spearman ρ, Kendall τ, or MSE).
   - Plot correlation-with-gold vs K.

4. **Fit scaling law**: Fit both power-law (aK^{−α} + b) and exponential
   (a·exp(−bK) + c) models. Report which fits better (AIC/BIC).

5. **Estimate ρ**: Compute average pairwise correlation among judges. Compare
   the observed scaling curve to the theoretical prediction from Experiment 04
   using the empirically estimated ρ.

## Output files

- `main.py`
- `results/exp6_correlation_vs_K.pdf` / `.png`
- `results/exp6_scaling_fit.csv`

## Run

```bash
python main.py
```
