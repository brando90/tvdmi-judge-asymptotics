"""Experiment 10 — Diversity Metrics.

Measure and relate judge diversity to convergence rate.
Test whether diversity metrics are sufficient statistics for predicting
the convergence speed (alpha).

Theory: correlated judges (rho > 0) converge slower than independent ones.
K_eff = K / (1 + (K-1)*rho). Different diversity metrics should predict
the effective convergence rate alpha.

See README.md for full specification.
"""

import os
import sys
import csv
import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import shared utilities from parent experiments/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    fit_power_law,
    COLORS,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RHO_VALUES = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
GAMMA_VALUES = [0.05, 0.1, 0.2]
K_MAX = 64
K_SWEEP = [1, 2, 4, 8, 16, 32, 64]
N_REPS = 200          # Monte Carlo reps per K value
N_ITEMS = 200         # number of (u,v) items for diversity metric computation
D_GRID = 20           # grid size for the discrete distribution

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# Helper: generate correlated binary judge votes
# ===========================================================================
def generate_correlated_votes(K, n_items, gamma, rho, rng=None):
    """Generate correlated binary votes for K judges on n_items items.

    Each judge votes correctly (1) with marginal probability 1/2 + gamma.
    Equicorrelation rho is induced via a latent Gaussian model:
        S_j = sqrt(rho) * Z_common + sqrt(1-rho) * Z_j
    Judge j votes correctly if S_j > threshold, where threshold is set
    so that P(vote correct) = 1/2 + gamma.

    Returns:
        votes: (n_items, K) array of binary votes (1 = correct, 0 = incorrect)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Threshold so marginal P(S_j > threshold) = 1/2 + gamma
    threshold = sp_stats.norm.ppf(0.5 - gamma)

    # Z_common: (n_items,)
    Z_common = rng.standard_normal(n_items)

    # Z_j: (n_items, K)
    Z_indiv = rng.standard_normal((n_items, K))

    sqrt_rho = np.sqrt(max(rho, 0.0))
    sqrt_1mrho = np.sqrt(max(1.0 - rho, 0.0))

    # S_j = sqrt(rho) * Z_common[:, None] + sqrt(1-rho) * Z_indiv
    S = sqrt_rho * Z_common[:, None] + sqrt_1mrho * Z_indiv

    # Vote correctly if S > threshold
    votes = (S > threshold).astype(int)
    return votes


# ===========================================================================
# Diversity metrics
# ===========================================================================
def compute_disagreement_rate(votes):
    """Fraction of (item, judge-pair) where judges disagree.

    votes: (n_items, K) binary array.
    """
    n_items, K = votes.shape
    if K < 2:
        return 0.0
    # For each item, count disagreements among all pairs
    # Disagreement for a pair on an item = |v_i - v_j|
    # Mean votes per item
    total_disagree = 0.0
    total_pairs = 0
    for i in range(n_items):
        row = votes[i]
        n_ones = row.sum()
        n_zeros = K - n_ones
        # Number of disagreeing pairs = n_ones * n_zeros
        total_disagree += n_ones * n_zeros
        total_pairs += K * (K - 1) / 2
    return total_disagree / total_pairs if total_pairs > 0 else 0.0


def compute_panel_entropy(votes):
    """Panel entropy: average over items of H(vote distribution).

    For each item, compute H = -sum p_k log(p_k) where p_k is the fraction
    of judges giving label k. Then average over items.
    """
    n_items, K = votes.shape
    entropies = np.zeros(n_items)
    for i in range(n_items):
        row = votes[i]
        p1 = row.mean()
        p0 = 1.0 - p1
        h = 0.0
        if p1 > 0:
            h -= p1 * np.log2(p1)
        if p0 > 0:
            h -= p0 * np.log2(p0)
        entropies[i] = h
    return entropies.mean()


def compute_mean_pairwise_correlation(votes):
    """Mean pairwise Pearson correlation among judges.

    votes: (n_items, K) binary array.
    Returns the average over all judge pairs of Pearson r.
    """
    n_items, K = votes.shape
    if K < 2:
        return 0.0
    # Compute correlation matrix
    # Center each judge's votes
    centered = votes - votes.mean(axis=0, keepdims=True)
    stds = votes.std(axis=0, ddof=0)
    # Avoid division by zero
    stds[stds == 0] = 1.0
    normed = centered / stds
    corr_matrix = (normed.T @ normed) / n_items
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((K, K), dtype=bool), k=1)
    pairwise_corrs = corr_matrix[mask]
    return pairwise_corrs.mean()


def compute_k_eff(K, mean_rho):
    """Effective independent judges: K_eff = K / (1 + (K-1)*mean_rho)."""
    denom = 1.0 + (K - 1) * max(mean_rho, 0.0)
    return K / denom


# ===========================================================================
# Consensus model: estimate error at each K
# ===========================================================================
def consensus_error_curve(K_sweep, gamma, rho, n_reps, n_items, rng):
    """Run consensus (majority vote) and compute error vs K.

    For each K in K_sweep:
      - Draw correlated votes for K judges on n_items items.
      - Majority vote estimate of the correct label for each item.
      - Error = fraction of items where majority vote is wrong.
      - Repeat n_reps times and average the error.

    Returns: list of mean errors, one per K value.
    """
    errors = []
    for K in K_sweep:
        rep_errors = np.zeros(n_reps)
        for rep in range(n_reps):
            votes = generate_correlated_votes(K, n_items, gamma, rho, rng=rng)
            # Majority vote: for each item, 1 if majority says 1, else 0
            # "Correct" label is 1 (since votes=1 means correct)
            majority_correct = (votes.mean(axis=1) > 0.5).astype(float)
            # Handle ties: votes.mean == 0.5 => break randomly
            ties = (votes.mean(axis=1) == 0.5)
            if ties.any():
                majority_correct[ties] = rng.integers(0, 2, size=ties.sum()).astype(float)
            rep_errors[rep] = 1.0 - majority_correct.mean()
        errors.append(rep_errors.mean())
    return errors


# ===========================================================================
# Main simulation
# ===========================================================================
print("=" * 70)
print("Experiment 10 — Diversity Metrics")
print("=" * 70)

all_results = []
rng = np.random.default_rng(42)

total_configs = len(RHO_VALUES) * len(GAMMA_VALUES)
config_idx = 0

for gamma in GAMMA_VALUES:
    for rho in RHO_VALUES:
        config_idx += 1
        print(f"\n[{config_idx}/{total_configs}] gamma={gamma}, rho={rho}")

        # ------------------------------------------------------------------
        # 1. Compute error curve across K sweep
        # ------------------------------------------------------------------
        error_curve = consensus_error_curve(
            K_SWEEP, gamma, rho, N_REPS, N_ITEMS, rng
        )
        print(f"  Error curve: {[f'{e:.4f}' for e in error_curve]}")

        # ------------------------------------------------------------------
        # 2. Fit power law to get alpha
        # ------------------------------------------------------------------
        alpha_hat, a_fit, b_fit, R2 = fit_power_law(K_SWEEP, error_curve)
        print(f"  Fitted alpha={alpha_hat:.4f}, R^2={R2:.4f}")

        # ------------------------------------------------------------------
        # 3. Compute diversity metrics at K=K_MAX
        # ------------------------------------------------------------------
        votes_full = generate_correlated_votes(
            K_MAX, N_ITEMS, gamma, rho, rng=rng
        )

        disagreement = compute_disagreement_rate(votes_full)
        entropy = compute_panel_entropy(votes_full)
        mean_corr = compute_mean_pairwise_correlation(votes_full)
        k_eff = compute_k_eff(K_MAX, mean_corr)

        print(f"  Disagreement={disagreement:.4f}, Entropy={entropy:.4f}, "
              f"MeanCorr={mean_corr:.4f}, K_eff={k_eff:.2f}")

        all_results.append({
            "rho": rho,
            "gamma": gamma,
            "disagreement": disagreement,
            "entropy": entropy,
            "k_eff": k_eff,
            "mean_corr": mean_corr,
            "alpha_fitted": alpha_hat,
            "alpha_R2": R2,
        })

# ===========================================================================
# Save main results CSV
# ===========================================================================
csv_path = os.path.join(RESULTS_DIR, "exp10_results.csv")
fieldnames = ["rho", "gamma", "disagreement", "entropy", "k_eff",
              "mean_corr", "alpha_fitted", "alpha_R2"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)
print(f"\nSaved results to {csv_path}")


# ===========================================================================
# Analysis: correlations and regressions
# ===========================================================================
metrics = {
    "disagreement": [r["disagreement"] for r in all_results],
    "entropy": [r["entropy"] for r in all_results],
    "k_eff": [r["k_eff"] for r in all_results],
    "mean_corr": [r["mean_corr"] for r in all_results],
}
alphas = np.array([r["alpha_fitted"] for r in all_results])

# Filter out NaN alphas for correlation/regression
valid = ~np.isnan(alphas)
alphas_valid = alphas[valid]

corr_summary = []
for name, vals in metrics.items():
    vals_arr = np.array(vals)[valid]

    # Pearson r
    if len(vals_arr) > 2 and np.std(vals_arr) > 1e-12 and np.std(alphas_valid) > 1e-12:
        pearson_r, _ = sp_stats.pearsonr(vals_arr, alphas_valid)
        spearman_rho, _ = sp_stats.spearmanr(vals_arr, alphas_valid)
    else:
        pearson_r = np.nan
        spearman_rho = np.nan

    # Linear regression: alpha = slope * metric + intercept
    if len(vals_arr) > 2 and np.std(vals_arr) > 1e-12:
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
            vals_arr, alphas_valid
        )
        R2 = r_value ** 2
    else:
        slope, intercept, R2 = np.nan, np.nan, np.nan

    corr_summary.append({
        "metric": name,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "R2": R2,
        "regression_slope": slope,
        "regression_intercept": intercept,
    })
    print(f"  {name}: Pearson r={pearson_r:.4f}, Spearman rho={spearman_rho:.4f}, "
          f"R^2={R2:.4f}, slope={slope:.4f}, intercept={intercept:.4f}")

# Save correlation summary CSV
corr_csv_path = os.path.join(RESULTS_DIR, "exp10_correlation_summary.csv")
corr_fields = ["metric", "pearson_r", "spearman_rho", "R2",
               "regression_slope", "regression_intercept"]
with open(corr_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=corr_fields)
    writer.writeheader()
    writer.writerows(corr_summary)
print(f"Saved correlation summary to {corr_csv_path}")


# ===========================================================================
# Plotting helpers
# ===========================================================================
METRIC_LABELS = {
    "disagreement": r"Disagreement Rate",
    "entropy": r"Panel Entropy (bits)",
    "k_eff": r"$K_{\mathrm{eff}}$",
    "mean_corr": r"Mean Pairwise Correlation $\bar{\rho}$",
}

METRIC_FILENAMES = {
    "disagreement": "disagreement",
    "entropy": "entropy",
    "k_eff": "keff",
    "mean_corr": "mean_corr",
}

GAMMA_MARKERS = {0.05: 'o', 0.1: 's', 0.2: '^'}
GAMMA_COLORS = {0.05: COLORS[0], 0.1: COLORS[1], 0.2: COLORS[2]}


def _scatter_with_regression(ax, metric_name, metric_vals, alpha_vals, results):
    """Scatter plot of alpha vs a diversity metric, colored by gamma."""
    vals_arr = np.array(metric_vals)
    alpha_arr = np.array(alpha_vals)
    valid_mask = ~np.isnan(alpha_arr)

    # Plot points colored by gamma
    for gamma in GAMMA_VALUES:
        mask = np.array([r["gamma"] == gamma for r in results]) & valid_mask
        if mask.any():
            ax.scatter(
                vals_arr[mask], alpha_arr[mask],
                marker=GAMMA_MARKERS[gamma],
                color=GAMMA_COLORS[gamma],
                s=60, edgecolors='k', linewidths=0.5,
                label=rf"$\gamma={gamma}$",
                zorder=3,
            )

    # Regression line
    v = vals_arr[valid_mask]
    a = alpha_arr[valid_mask]
    if len(v) > 2 and np.std(v) > 1e-12:
        slope, intercept, r_value, _, _ = sp_stats.linregress(v, a)
        x_line = np.linspace(v.min(), v.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, '--', color='gray', linewidth=1.5, alpha=0.7,
                label=rf"$R^2={r_value**2:.3f}$")

    ax.set_xlabel(METRIC_LABELS[metric_name])
    ax.set_ylabel(r"Convergence exponent $\alpha$")
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)


# ===========================================================================
# Plot 1-4: Individual scatter plots
# ===========================================================================
for metric_name in metrics:
    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_with_regression(
        ax, metric_name, metrics[metric_name], alphas, all_results
    )
    ax.set_title(rf"Convergence Exponent $\alpha$ vs {METRIC_LABELS[metric_name]}")

    fname = METRIC_FILENAMES[metric_name]
    for ext in ("pdf", "png"):
        path = os.path.join(RESULTS_DIR, f"exp10_alpha_vs_{fname}.{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


# ===========================================================================
# Plot 5: Correlation summary bar chart
# ===========================================================================
fig5, ax5 = plt.subplots(figsize=(7, 5))
metric_names_display = [METRIC_LABELS[cs["metric"]] for cs in corr_summary]
R2_vals = [cs["R2"] for cs in corr_summary]
bar_colors = [COLORS[i % len(COLORS)] for i in range(len(corr_summary))]

bars = ax5.bar(range(len(corr_summary)), R2_vals, color=bar_colors,
               edgecolor='k', linewidth=0.5)
ax5.set_xticks(range(len(corr_summary)))
ax5.set_xticklabels(metric_names_display, rotation=20, ha='right', fontsize=10)
ax5.set_ylabel(r"$R^2$")
ax5.set_title(r"Predictive Power of Diversity Metrics for $\alpha$")
ax5.set_ylim(0, 1.05)
ax5.grid(True, axis='y', ls=':', alpha=0.4)

# Add R^2 values on top of bars
for bar, r2 in zip(bars, R2_vals):
    if not np.isnan(r2):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{r2:.3f}", ha='center', va='bottom', fontsize=10)

for ext in ("pdf", "png"):
    path = os.path.join(RESULTS_DIR, f"exp10_correlation_summary.{ext}")
    fig5.savefig(path)
    print(f"Saved {path}")
plt.close(fig5)


# ===========================================================================
# Plot 6: 2x2 dashboard
# ===========================================================================
fig6, axes = plt.subplots(2, 2, figsize=(14, 10))
metric_order = ["disagreement", "entropy", "k_eff", "mean_corr"]

for idx, metric_name in enumerate(metric_order):
    row, col = divmod(idx, 2)
    ax = axes[row][col]
    _scatter_with_regression(
        ax, metric_name, metrics[metric_name], alphas, all_results
    )
    ax.set_title(rf"$\alpha$ vs {METRIC_LABELS[metric_name]}", fontsize=12)

fig6.suptitle("Diversity Metrics vs Convergence Exponent (Experiment 10)",
              fontsize=14, y=1.01)
fig6.tight_layout()

for ext in ("pdf", "png"):
    path = os.path.join(RESULTS_DIR, f"exp10_diversity_dashboard.{ext}")
    fig6.savefig(path)
    print(f"Saved {path}")
plt.close(fig6)


# ===========================================================================
# Summary printout
# ===========================================================================
print("\n" + "=" * 70)
print("Experiment 10 — Summary")
print("=" * 70)
print(f"\nConfigurations tested: {len(all_results)}")
print(f"  rho values: {RHO_VALUES}")
print(f"  gamma values: {GAMMA_VALUES}")
print(f"  K sweep: {K_SWEEP}")
print(f"  Monte Carlo reps per K: {N_REPS}")
print(f"  Items per evaluation: {N_ITEMS}")

print("\nCorrelation of diversity metrics with alpha:")
print(f"  {'Metric':<25s} {'Pearson r':>10s} {'Spearman':>10s} {'R^2':>8s}")
print("  " + "-" * 55)
for cs in corr_summary:
    print(f"  {cs['metric']:<25s} {cs['pearson_r']:>10.4f} "
          f"{cs['spearman_rho']:>10.4f} {cs['R2']:>8.4f}")

# Identify best predictor
best = max(corr_summary, key=lambda x: x["R2"] if not np.isnan(x["R2"]) else -1)
print(f"\nBest predictor of alpha: {best['metric']} (R^2 = {best['R2']:.4f})")

print(f"\nOutput files in {RESULTS_DIR}/")
print("  exp10_results.csv")
print("  exp10_correlation_summary.csv")
print("  exp10_alpha_vs_disagreement.{pdf,png}")
print("  exp10_alpha_vs_entropy.{pdf,png}")
print("  exp10_alpha_vs_keff.{pdf,png}")
print("  exp10_alpha_vs_mean_corr.{pdf,png}")
print("  exp10_correlation_summary.{pdf,png}")
print("  exp10_diversity_dashboard.{pdf,png}")

print("\nExperiment 10 complete.")
