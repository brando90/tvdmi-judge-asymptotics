# Experiment 06 -- End-to-End Scaling Law with LLM Judges (Optional)

## Purpose
Validate synthetic findings with real LLM-as-judge evaluations.

## Design
- Dataset: public benchmark with gold scores (MT-Bench, Chatbot Arena, AlpacaEval).
- Judges: pool of diverse LLMs or same model with varied prompts/temperatures.
- Protocol: subsample K judges, aggregate (mean, median, max), measure correlation with gold.
- Fit power-law and exponential models, compare via AIC/BIC.
- Estimate empirical rho among judges; compare to Exp 4 predictions.

## Files
- `exp6_llm_judges.py`
- `exp6_correlation_vs_K.{pdf,png}`
- `exp6_scaling_fit.csv`
