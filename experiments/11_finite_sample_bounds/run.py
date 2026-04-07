"""Experiment 11 — Finite-Sample Bounds.

Empirically validate finite-sample concentration inequalities from the
consistency theorem. Compare theoretical error bounds vs actual observed
error across K and n.

The main consistency theorem gives:
    Pr(tau(c) - max_{j<=K} Delta_hat_c(h_j) > eps + eta)
        <= exp(-K * p_eps) + 2K * exp(-2n * eta^2)

This decomposes into:
- Approximation error (eps): controlled by K and witness mass p_eps.
- Estimation error (eta): controlled by n via Hoeffding.

See README.md for full specification.
"""

import os
import sys
import csv
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as gamma_fn

# ---------------------------------------------------------------------------
# Import shared utilities from parent experiments/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    compute_true_tvd,
    optimal_critic,
    make_joint_distribution,
    compute_delta,
    fit_power_law,
    COLORS,
    N_REPS,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
D = 20
CORRELATION = 0.5
BETA = 2  # tail thickness parameter for witness mass

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Ground truth distribution
# ---------------------------------------------------------------------------
print("=" * 70)
print("Experiment 11 — Finite-Sample Bounds")
print("=" * 70)
print(f"\nSetting up ground truth distribution (d={D}, correlation={CORRELATION}) ...")
P_UV, P_U, P_V = make_joint_distribution(d=D, correlation=CORRELATION)
tau_c = compute_true_tvd(P_UV, P_U, P_V)
print(f"  True TVD tau_c = {tau_c:.6f}")

h_star = optimal_critic(P_UV, P_U, P_V)
delta_star = compute_delta(h_star, P_UV, P_U, P_V)
print(f"  Optimal critic delta_star = {delta_star:.6f}")

# Precompute flat arrays for efficient computation
P_indep = np.outer(P_U, P_V)
flat_joint = P_UV.flatten().astype(np.float64)
flat_indep = P_indep.flatten().astype(np.float64)
h_star_flat = h_star.flatten().astype(np.float32)
diff_flat = (P_UV - P_indep).flatten().astype(np.float32)


def build_judges(K):
    """Build K random-witness judges. Returns h_judges of shape (K, D*D)."""
    q_vals = np.random.beta(BETA, 1, size=K)
    masks = np.random.rand(K, D * D) < q_vals[:, None]
    rand_bits = (np.random.rand(K, D * D) < 0.5).astype(np.float32)
    return np.where(masks, h_star_flat[None, :], rand_bits)


def true_deltas(h_judges):
    """Compute exact delta for each judge. Shape: (K,)."""
    return h_judges @ diff_flat


def empirical_deltas_multinomial(h_judges, n):
    """Estimate delta for each judge from n samples using multinomial counts.

    Uses a single shared evaluation set (one multinomial draw per distribution)
    across all judges per rep — matching the realistic setting where judges
    evaluate the same dataset.
    """
    counts_j = np.random.multinomial(n, flat_joint).astype(np.float32)
    counts_i = np.random.multinomial(n, flat_indep).astype(np.float32)
    delta_joint = (h_judges @ counts_j) / n
    delta_indep = (h_judges @ counts_i) / n
    return delta_joint - delta_indep


# ===================================================================
# PART A: Best-of-K Bounds (Coverage Regime)
# ===================================================================
print("\n" + "=" * 70)
print("Part A: Best-of-K Bounds (Coverage Regime)")
print("=" * 70)

K_VALS_A = [2, 5, 10, 20, 50, 100, 200]
N_REPS_A = 500

results_A = []
t0 = time.time()

for K in K_VALS_A:
    gaps = np.empty(N_REPS_A)

    for rep in range(N_REPS_A):
        h_judges = build_judges(K)
        deltas = true_deltas(h_judges)
        gaps[rep] = max(0.0, delta_star - np.max(deltas))

    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)
    p95_gap = np.percentile(gaps, 95)

    # Theoretical bound: E[gap] ~ Gamma(1 + 1/beta) * K^{-1/beta}
    theoretical_bound = gamma_fn(1 + 1.0 / BETA) * K ** (-1.0 / BETA)

    results_A.append({
        'K': K,
        'theoretical_bound': theoretical_bound,
        'empirical_mean': mean_gap,
        'empirical_std': std_gap,
        'empirical_p95': p95_gap,
    })
    print(f"  K={K:4d}: theory={theoretical_bound:.6f}, "
          f"empirical_mean={mean_gap:.6f}, p95={p95_gap:.6f}")

print(f"  Part A time: {time.time() - t0:.1f}s")

# Save CSV
csv_path_A = os.path.join(RESULTS_DIR, "exp11_bestofK_results.csv")
with open(csv_path_A, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["K", "theoretical_bound",
                                           "empirical_mean", "empirical_p95"])
    writer.writeheader()
    for r in results_A:
        writer.writerow({k: r[k] for k in ["K", "theoretical_bound",
                                            "empirical_mean", "empirical_p95"]})
print(f"  Saved: {csv_path_A}")


# ===================================================================
# PART B: Estimation Error Bounds (Hoeffding)
# ===================================================================
print("\n" + "=" * 70)
print("Part B: Estimation Error Bounds (Hoeffding)")
print("=" * 70)

N_VALS_B = [50, 100, 200, 500, 1000, 2000, 5000]
K_VALS_B = [4, 16, 64]
N_REPS_B = 500

t0 = time.time()
results_B = []

# For Part B, we fix h = h_star and only vary n.
# K appears in labels but doesn't affect the single-judge estimation.
# We still run separate experiments per K for consistency with the spec,
# but the theoretical bound is the same (since it's a single judge).

for K in K_VALS_B:
    for n in N_VALS_B:
        # Vectorize across all reps: draw n samples for each rep at once
        errors = np.empty(N_REPS_B)
        for rep in range(N_REPS_B):
            counts_j = np.random.multinomial(n, flat_joint).astype(np.float32)
            counts_i = np.random.multinomial(n, flat_indep).astype(np.float32)
            delta_hat = (np.dot(h_star_flat, counts_j)
                         - np.dot(h_star_flat, counts_i)) / n
            errors[rep] = abs(delta_hat - delta_star)

        mean_error = np.mean(errors)
        p95_error = np.percentile(errors, 95)

        # Theoretical bound: E[|error|] ~ sqrt(1/(pi*n))
        # (from half-normal approximation of |N(0, sigma^2)|
        #  where sigma^2 ~ 1/(2n) for bounded [0,1] critic)
        theoretical_bound = np.sqrt(1.0 / (np.pi * n))

        results_B.append({
            'K': K,
            'n': n,
            'theoretical_bound': theoretical_bound,
            'empirical_mean': mean_error,
            'empirical_p95': p95_error,
        })
        print(f"  K={K:3d}, n={n:5d}: theory={theoretical_bound:.6f}, "
              f"empirical_mean={mean_error:.6f}, p95={p95_error:.6f}")

print(f"  Part B time: {time.time() - t0:.1f}s")

# Save CSV
csv_path_B = os.path.join(RESULTS_DIR, "exp11_hoeffding_results.csv")
with open(csv_path_B, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["K", "n", "theoretical_bound",
                                           "empirical_mean", "empirical_p95"])
    writer.writeheader()
    for r in results_B:
        writer.writerow({k: r[k] for k in ["K", "n", "theoretical_bound",
                                            "empirical_mean", "empirical_p95"]})
print(f"  Saved: {csv_path_B}")


# ===================================================================
# PART C: Combined Two-Axis Bound
# ===================================================================
print("\n" + "=" * 70)
print("Part C: Combined Two-Axis Bound")
print("=" * 70)

K_VALS_C = [4, 16, 64, 256]
N_VALS_C = [50, 100, 500, 1000, 5000]
N_REPS_C = 200

t0 = time.time()
results_C = []

for K in K_VALS_C:
    for n in N_VALS_C:
        total_gaps = np.empty(N_REPS_C)

        for rep in range(N_REPS_C):
            h_judges = build_judges(K)  # (K, D*D)

            # Estimate delta from n samples (shared evaluation set)
            delta_hats = empirical_deltas_multinomial(h_judges, n)
            best_delta_hat = np.max(delta_hats)
            total_gaps[rep] = max(0.0, delta_star - best_delta_hat)

        mean_total_gap = np.mean(total_gaps)

        # Theoretical combined bound:
        # E[total gap] <= a * K^{-1/beta} + b * sqrt(log(K)/n)
        approx_error = gamma_fn(1 + 1.0 / BETA) * K ** (-1.0 / BETA)
        estim_error = np.sqrt(np.log(K) / n)
        theoretical_bound = approx_error + estim_error

        results_C.append({
            'K': K,
            'n': n,
            'theoretical_bound': theoretical_bound,
            'empirical_mean': mean_total_gap,
        })
        print(f"  K={K:4d}, n={n:5d}: theory={theoretical_bound:.6f}, "
              f"empirical_mean={mean_total_gap:.6f}")

print(f"  Part C time: {time.time() - t0:.1f}s")

# Save CSV
csv_path_C = os.path.join(RESULTS_DIR, "exp11_combined_results.csv")
with open(csv_path_C, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["K", "n", "theoretical_bound",
                                           "empirical_mean"])
    writer.writeheader()
    for r in results_C:
        writer.writerow(r)
print(f"  Saved: {csv_path_C}")


# ===================================================================
# PART D: Majority Vote (Consensus) Bounds
# ===================================================================
print("\n" + "=" * 70)
print("Part D: Majority Vote (Consensus) Bounds")
print("=" * 70)

K_VALS_D = [1, 2, 4, 8, 16, 32, 64, 128, 256]
GAMMA_VALS = [0.05, 0.1, 0.2]
N_REPS_D = 300

t0 = time.time()
results_D = []

for gamma_val in GAMMA_VALS:
    p_correct = 0.5 + gamma_val
    for K in K_VALS_D:
        # Vectorize: generate (N_REPS_D, K) votes at once
        votes = (np.random.rand(N_REPS_D, K) < p_correct).astype(int)
        majority_correct = (votes.mean(axis=1) > 0.5).astype(int)
        empirical_error = 1.0 - np.mean(majority_correct)

        # Theoretical bound: Pr(majority wrong) <= 2 * exp(-2 * K * gamma^2)
        theoretical_bound = min(1.0, 2.0 * np.exp(-2.0 * K * gamma_val ** 2))

        results_D.append({
            'K': K,
            'gamma': gamma_val,
            'theoretical_bound': theoretical_bound,
            'empirical_mean': empirical_error,
        })
        print(f"  gamma={gamma_val:.2f}, K={K:4d}: "
              f"theory={theoretical_bound:.6f}, empirical={empirical_error:.6f}")

print(f"  Part D time: {time.time() - t0:.1f}s")

# Save CSV
csv_path_D = os.path.join(RESULTS_DIR, "exp11_consensus_results.csv")
with open(csv_path_D, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["K", "gamma", "theoretical_bound",
                                           "empirical_mean"])
    writer.writeheader()
    for r in results_D:
        writer.writerow(r)
print(f"  Saved: {csv_path_D}")


# ===================================================================
# PLOTS
# ===================================================================
print("\n" + "=" * 70)
print("Generating plots ...")
print("=" * 70)

# ---------------------------------------------------------------------------
# Plot 1: Best-of-K bound vs empirical (Part A)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

K_arr_A = np.array([r['K'] for r in results_A])
theory_A = np.array([r['theoretical_bound'] for r in results_A])
emp_mean_A = np.array([r['empirical_mean'] for r in results_A])
emp_std_A = np.array([r['empirical_std'] for r in results_A])
emp_p95_A = np.array([r['empirical_p95'] for r in results_A])

ax.loglog(K_arr_A, theory_A, 's--', color=COLORS[0], linewidth=2,
          markersize=7, label=r'Theoretical: $\Gamma(1+1/\beta)\,K^{-1/\beta}$')
ax.loglog(K_arr_A, emp_mean_A, 'o-', color=COLORS[1], linewidth=2,
          markersize=6, label='Empirical mean gap')
ax.fill_between(K_arr_A,
                np.maximum(emp_mean_A - 1.96 * emp_std_A / np.sqrt(N_REPS_A), 1e-8),
                emp_mean_A + 1.96 * emp_std_A / np.sqrt(N_REPS_A),
                alpha=0.2, color=COLORS[1], label='95% CI (mean)')
ax.loglog(K_arr_A, emp_p95_A, '^:', color=COLORS[2], linewidth=1.5,
          markersize=5, label='Empirical 95th percentile')

ax.set_xlabel(r'Number of judges $K$')
ax.set_ylabel(r'Gap $\tau_c^* - \max_j \Delta_c(h_j)$')
ax.set_title(r'Best-of-$K$ Bound vs Empirical ($\beta=2$)')
ax.legend(fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.4)

for ext in ['pdf', 'png']:
    path = os.path.join(RESULTS_DIR, f"exp11_bestofK_bound_vs_empirical.{ext}")
    fig.savefig(path)
    print(f"  Saved: {path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2: Hoeffding bound vs empirical (Part B)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for idx, K in enumerate(K_VALS_B):
    color = COLORS[idx % len(COLORS)]
    subset = [r for r in results_B if r['K'] == K]
    n_arr = np.array([r['n'] for r in subset])
    emp_arr = np.array([r['empirical_mean'] for r in subset])
    theory_arr = np.array([r['theoretical_bound'] for r in subset])
    p95_arr = np.array([r['empirical_p95'] for r in subset])

    ax.loglog(n_arr, theory_arr, 's--', color=color, linewidth=1.5,
              markersize=6, alpha=0.7,
              label=rf'Theory $K={K}$')
    ax.loglog(n_arr, emp_arr, 'o-', color=color, linewidth=2,
              markersize=5, label=rf'Empirical mean $K={K}$')
    ax.loglog(n_arr, p95_arr, '^:', color=color, linewidth=1,
              markersize=4, alpha=0.6,
              label=rf'Empirical p95 $K={K}$')

ax.set_xlabel(r'Samples per judge $n$')
ax.set_ylabel(r'Estimation error $|\hat\Delta - \Delta|$')
ax.set_title('Hoeffding Bound vs Empirical Estimation Error')
ax.legend(fontsize=7, ncol=3, loc='upper right')
ax.grid(True, which='both', ls=':', alpha=0.4)

for ext in ['pdf', 'png']:
    path = os.path.join(RESULTS_DIR, f"exp11_hoeffding_bound_vs_empirical.{ext}")
    fig.savefig(path)
    print(f"  Saved: {path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 3: Combined bound heatmap (Part C)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Build grids
theory_grid = np.zeros((len(K_VALS_C), len(N_VALS_C)))
emp_grid = np.zeros((len(K_VALS_C), len(N_VALS_C)))

for r in results_C:
    i = K_VALS_C.index(r['K'])
    j = N_VALS_C.index(r['n'])
    theory_grid[i, j] = r['theoretical_bound']
    emp_grid[i, j] = r['empirical_mean']

K_labels = [str(k) for k in K_VALS_C]
N_labels = [str(n) for n in N_VALS_C]

# Shared color limits
vmin = min(theory_grid.min(), emp_grid.min())
vmax = max(theory_grid.max(), emp_grid.max())

im0 = axes[0].imshow(theory_grid, cmap='YlOrRd', aspect='auto',
                      vmin=vmin, vmax=vmax)
axes[0].set_xticks(range(len(N_VALS_C)))
axes[0].set_xticklabels(N_labels, fontsize=9)
axes[0].set_yticks(range(len(K_VALS_C)))
axes[0].set_yticklabels(K_labels, fontsize=9)
axes[0].set_xlabel(r'Samples per judge $n$')
axes[0].set_ylabel(r'Number of judges $K$')
axes[0].set_title('Theoretical Bound')
# Annotate cells
for i in range(len(K_VALS_C)):
    for j in range(len(N_VALS_C)):
        axes[0].text(j, i, f'{theory_grid[i, j]:.3f}',
                     ha='center', va='center', fontsize=8,
                     color='white' if theory_grid[i, j] > 0.5 * vmax else 'black')

im1 = axes[1].imshow(emp_grid, cmap='YlOrRd', aspect='auto',
                      vmin=vmin, vmax=vmax)
axes[1].set_xticks(range(len(N_VALS_C)))
axes[1].set_xticklabels(N_labels, fontsize=9)
axes[1].set_yticks(range(len(K_VALS_C)))
axes[1].set_yticklabels(K_labels, fontsize=9)
axes[1].set_xlabel(r'Samples per judge $n$')
axes[1].set_ylabel(r'Number of judges $K$')
axes[1].set_title('Empirical Mean Gap')
for i in range(len(K_VALS_C)):
    for j in range(len(N_VALS_C)):
        axes[1].text(j, i, f'{emp_grid[i, j]:.3f}',
                     ha='center', va='center', fontsize=8,
                     color='white' if emp_grid[i, j] > 0.5 * vmax else 'black')

fig.colorbar(im1, ax=axes, shrink=0.8, label='Error')
fig.suptitle(r'Combined Bound: $K^{-1/\beta} + \sqrt{\log K / n}$', fontsize=14)
plt.tight_layout()

for ext in ['pdf', 'png']:
    path = os.path.join(RESULTS_DIR, f"exp11_combined_bound_heatmap.{ext}")
    fig.savefig(path)
    print(f"  Saved: {path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 4: Consensus bound (Part D) — semi-log
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for idx, gamma_val in enumerate(GAMMA_VALS):
    color = COLORS[idx % len(COLORS)]
    subset = [r for r in results_D if r['gamma'] == gamma_val]
    K_arr = np.array([r['K'] for r in subset])
    theory_arr = np.array([r['theoretical_bound'] for r in subset])
    emp_arr = np.array([r['empirical_mean'] for r in subset])

    ax.semilogy(K_arr, theory_arr, 's--', color=color, linewidth=2,
                markersize=6,
                label=rf'Theory $\gamma={gamma_val}$')
    ax.semilogy(K_arr, np.maximum(emp_arr, 1e-6), 'o-', color=color,
                linewidth=2, markersize=5,
                label=rf'Empirical $\gamma={gamma_val}$')

ax.set_xlabel(r'Number of judges $K$')
ax.set_ylabel(r'Majority vote error probability')
ax.set_title(r'Consensus Bound: $2\exp(-2K\gamma^2)$ vs Empirical')
ax.legend(fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.4)
ax.set_ylim(bottom=1e-4)

for ext in ['pdf', 'png']:
    path = os.path.join(RESULTS_DIR, f"exp11_consensus_bound.{ext}")
    fig.savefig(path)
    print(f"  Saved: {path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 5: Bound tightness ratio (theoretical / empirical) vs K
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Best-of-K tightness
ratio_A = theory_A / np.maximum(emp_mean_A, 1e-10)
axes[0].semilogx(K_arr_A, ratio_A, 'o-', color=COLORS[0], linewidth=2,
                 markersize=6)
axes[0].axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
axes[0].set_xlabel(r'Number of judges $K$')
axes[0].set_ylabel('Ratio (theoretical / empirical)')
axes[0].set_title(r'Best-of-$K$ Bound Tightness ($\beta=2$)')
axes[0].grid(True, which='both', ls=':', alpha=0.4)

# (b) Consensus tightness for each gamma
for idx, gamma_val in enumerate(GAMMA_VALS):
    color = COLORS[idx + 1]
    subset = [r for r in results_D if r['gamma'] == gamma_val]
    K_arr_d = np.array([r['K'] for r in subset])
    theory_d = np.array([r['theoretical_bound'] for r in subset])
    emp_d = np.array([r['empirical_mean'] for r in subset])

    # Only compute ratio where empirical > 0
    valid = emp_d > 0
    if valid.sum() > 0:
        ratio_d = theory_d[valid] / emp_d[valid]
        axes[1].semilogx(K_arr_d[valid], ratio_d, 'o-', color=color,
                         linewidth=2, markersize=5,
                         label=rf'$\gamma={gamma_val}$')

axes[1].axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
axes[1].set_xlabel(r'Number of judges $K$')
axes[1].set_ylabel('Ratio (theoretical / empirical)')
axes[1].set_title('Consensus Bound Tightness')
axes[1].legend(fontsize=9)
axes[1].grid(True, which='both', ls=':', alpha=0.4)

fig.suptitle('How Tight Are the Theoretical Bounds?', fontsize=14)
plt.tight_layout()

for ext in ['pdf', 'png']:
    path = os.path.join(RESULTS_DIR, f"exp11_bound_tightness.{ext}")
    fig.savefig(path)
    print(f"  Saved: {path}")
plt.close(fig)


# ===================================================================
# Final summary
# ===================================================================
print("\n" + "=" * 70)
print("Experiment 11 complete.")
print("=" * 70)
print(f"\nAll results saved to: {RESULTS_DIR}")
print("CSV files:")
print(f"  - {csv_path_A}")
print(f"  - {csv_path_B}")
print(f"  - {csv_path_C}")
print(f"  - {csv_path_D}")
print("Plots: exp11_bestofK_bound_vs_empirical, exp11_hoeffding_bound_vs_empirical,")
print("       exp11_combined_bound_heatmap, exp11_consensus_bound, exp11_bound_tightness")
print("  (each in .pdf and .png)")
