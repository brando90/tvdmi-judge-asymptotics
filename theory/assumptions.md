# assumptions

## Baseline setup

Let:

- `K` be the number of judges
- `n` be the number of items or tasks
- `m` be the number of pairwise comparisons per item when the protocol is pairwise
- `rho` measure dependence among judges
- `delta` be the adversarial fraction of judges

## Initial assumptions to refine

1. Judges are sampled from a distribution over evaluators.
2. Judge outputs are bounded.
3. Judge dependence is controlled rather than arbitrary.
4. The aggregation rule is fixed and measurable.
5. There exists a target gold or oracle score to approximate.
6. Adversarial judges, if present, are bounded by a contamination model.

## Open choice points

- exact definition of the gold target
- exact aggregation rule
- whether convergence is pointwise, uniform, or in expectation
- whether the main error notion is absolute error, ranking error, or calibration error
