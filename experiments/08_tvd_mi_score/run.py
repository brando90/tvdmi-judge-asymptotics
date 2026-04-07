"""Experiment 08 — TVD-MI Score Estimator.

Implements and validates TVD-MI score estimators for finite judge panels.
Compares plug-in vs debiased estimators in terms of bias, variance, and MSE.

Estimators:
  1. Plug-in TVD: empirical distributions -> TVD
  2. Debiased TVD (bootstrap): subtract bootstrap-estimated bias
  3. Plug-in MI: empirical mutual information
  4. Miller-corrected MI: MI_plugin - (|support| - 1) / (2n)

Usage:
    python run.py
"""

import os
import sys
import csv
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Import shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    make_joint_distribution,
    compute_true_tvd,
    COLORS,
)

# Re-apply matplotlib styling (already set in utils but be explicit)
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
    'font.family': 'serif',
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Experiment parameters ---
D = 20              # grid dimension
CORRELATION = 0.5   # correlation for the bivariate Gaussian
N_SWEEP = [50, 100, 200, 500, 1000, 2000, 5000]
K_SWEEP = [1, 2, 4, 8, 16, 32, 64]
N_MC = 200          # Monte Carlo repetitions
B_BOOTSTRAP = 100   # bootstrap resamples for debiased TVD

# Fixed slices for plots
K_FIXED = 16
N_FIXED = 500


# ============================================================
# True MI for reference
# ============================================================

def compute_true_mi(P_UV, P_U, P_V):
    """Compute true MI = sum P_UV(u,v) * log(P_UV(u,v) / (P_U(u)*P_V(v)))."""
    P_indep = np.outer(P_U, P_V)
    p_uv = P_UV.ravel()
    p_ind = P_indep.ravel()
    mask = (p_uv > 0) & (p_ind > 0)
    return float(np.sum(p_uv[mask] * np.log(p_uv[mask] / p_ind[mask])))


# ============================================================
# Vectorized estimators
# ============================================================

def mi_from_counts(counts_joint, d):
    """Plug-in MI and support size from joint counts. Returns (mi, support)."""
    n = counts_joint.sum()
    if n == 0:
        return 0.0, 0
    p = counts_joint / n
    P2d = p.reshape(d, d)
    pU = P2d.sum(axis=1)
    pV = P2d.sum(axis=0)
    p_ind = np.outer(pU, pV).ravel()
    mask = (p > 0) & (p_ind > 0)
    mi = float(np.sum(p[mask] * np.log(p[mask] / p_ind[mask])))
    support = int(np.sum(p > 0))
    return mi, support


# ============================================================
# Fully vectorized Monte Carlo evaluation
# ============================================================

def run_monte_carlo(P_UV_flat, P_indep_flat, d, K, n, n_mc, B, rng):
    """Run n_mc Monte Carlo reps for all four estimators.

    Generates data via multinomial, computes plug-in TVD and MI
    vectorized, and uses Dirichlet-Multinomial (Bayesian) bootstrap
    for TVD debiasing — O(d^2 * B) per rep regardless of sample size.
    """
    d_sq = d * d
    total_n = K * n

    # --- Generate all MC data at once via multinomial ---
    # Shape: (n_mc, d_sq) integer counts
    all_cj = rng.multinomial(total_n, P_UV_flat, size=n_mc).astype(np.float64)
    all_ci = rng.multinomial(total_n, P_indep_flat, size=n_mc).astype(np.float64)

    # --- Plug-in TVD: fully vectorized over MC reps ---
    res_plugin_tvd = 0.5 * np.sum(
        np.abs(all_cj / total_n - all_ci / total_n), axis=1
    )

    # --- Plug-in MI & Miller-corrected MI (loop over reps) ---
    res_plugin_mi = np.empty(n_mc)
    res_miller_mi = np.empty(n_mc)
    for rep in range(n_mc):
        mi_val, support = mi_from_counts(all_cj[rep], d)
        res_plugin_mi[rep] = mi_val
        res_miller_mi[rep] = mi_val - (support - 1) / (2 * total_n)

    # --- Debiased TVD via Bayesian bootstrap (Dirichlet-Multinomial) ---
    # Instead of the standard nonparametric bootstrap (which requires
    # multinomial(total_n, p_emp) and is O(total_n * B)), we use the
    # Dirichlet-Multinomial bootstrap: draw weights from
    # Dirichlet(counts + 1) to get bootstrap "distributions", then compute
    # TVD from those. This is O(d^2 * B) regardless of total_n.
    #
    # Equivalently, for the standard bootstrap, counts_b ~ Multinomial(n, c/n)
    # has expected value = c and covariance structure matching bootstrap.
    # The Dirichlet approach: p_b ~ Dir(c + 1), then TVD_b = 0.5*sum|p_b_j - p_b_i|.
    # For large n, Dir(c+1) ≈ Dir(c) ≈ bootstrap distribution.
    res_debiased_tvd = np.empty(n_mc)
    for rep in range(n_mc):
        cj = all_cj[rep]
        ci = all_ci[rep]

        # Dirichlet weights = normalized Gamma(count_k + 1, 1) draws
        # shape: (B, d_sq) — each row is a probability vector
        gj = rng.standard_gamma(cj + 1.0, size=(B, d_sq))
        gi = rng.standard_gamma(ci + 1.0, size=(B, d_sq))
        pj_boot = gj / gj.sum(axis=1, keepdims=True)
        pi_boot = gi / gi.sum(axis=1, keepdims=True)

        boot_tvds = 0.5 * np.sum(np.abs(pj_boot - pi_boot), axis=1)
        bias_hat = boot_tvds.mean() - res_plugin_tvd[rep]
        res_debiased_tvd[rep] = res_plugin_tvd[rep] - bias_hat

    return {
        'plugin_tvd': res_plugin_tvd,
        'debiased_tvd': res_debiased_tvd,
        'plugin_mi': res_plugin_mi,
        'miller_mi': res_miller_mi,
    }


def compute_statistics(estimates, true_value):
    """Compute bias, variance, MSE for an array of estimates."""
    bias = float(estimates.mean() - true_value)
    variance = float(estimates.var())
    mse = bias**2 + variance
    return bias, variance, mse


# ============================================================
# Main experiment
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("Experiment 08 — TVD-MI Score Estimator")
    print("=" * 60)

    # --- Step 1: Ground truth ---
    print("\n[1] Computing ground truth distributions...")
    P_UV, P_U, P_V = make_joint_distribution(d=D, correlation=CORRELATION)
    true_tvd = compute_true_tvd(P_UV, P_U, P_V)
    true_mi = compute_true_mi(P_UV, P_U, P_V)
    print(f"    d={D}, correlation={CORRELATION}")
    print(f"    True TVD = {true_tvd:.6f}")
    print(f"    True MI  = {true_mi:.6f}")

    # Precompute flattened distributions
    P_UV_flat = P_UV.ravel().copy()
    P_indep_flat = np.outer(P_U, P_V).ravel().copy()

    # Map estimator names to their ground-truth values
    true_values = {
        'plugin_tvd': true_tvd,
        'debiased_tvd': true_tvd,
        'plugin_mi': true_mi,
        'miller_mi': true_mi,
    }

    estimator_labels = {
        'plugin_tvd': 'Plug-in TVD',
        'debiased_tvd': 'Debiased TVD (bootstrap)',
        'plugin_mi': 'Plug-in MI',
        'miller_mi': 'Miller-corrected MI',
    }

    estimator_colors = {
        'plugin_tvd': COLORS[0],
        'debiased_tvd': COLORS[1],
        'plugin_mi': COLORS[2],
        'miller_mi': COLORS[3],
    }

    # --- Step 2: Full (K, n) sweep ---
    print(f"\n[2] Running full (K, n) sweep: K in {K_SWEEP}, n in {N_SWEEP}")
    print(f"    {N_MC} MC reps, B={B_BOOTSTRAP} bootstrap resamples")

    rng = np.random.default_rng(42)

    # Storage: {estimator: {(K, n): (bias, var, mse)}}
    all_stats = {est: {} for est in estimator_labels}

    total_combos = len(K_SWEEP) * len(N_SWEEP)
    combo_idx = 0

    for K in K_SWEEP:
        for n in N_SWEEP:
            combo_idx += 1
            t1 = time.time()
            print(f"    [{combo_idx}/{total_combos}] K={K:>3d}, n={n:>5d} "
                  f"(Kn={K*n:>7d}) ...", end='', flush=True)

            mc_results = run_monte_carlo(
                P_UV_flat, P_indep_flat, D, K, n, N_MC, B_BOOTSTRAP, rng
            )

            for est_name, estimates in mc_results.items():
                bias, var, mse = compute_statistics(
                    estimates, true_values[est_name]
                )
                all_stats[est_name][(K, n)] = (bias, var, mse)

            dt = time.time() - t1
            print(f" {dt:.1f}s  (plugin_tvd MSE="
                  f"{all_stats['plugin_tvd'][(K, n)][2]:.6f})")

    # --- Step 3: Save CSV ---
    csv_path = os.path.join(RESULTS_DIR, 'exp08_results.csv')
    print(f"\n[3] Writing CSV to {csv_path}")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['estimator', 'K', 'n', 'bias', 'variance', 'mse'])
        for est_name in estimator_labels:
            for K in K_SWEEP:
                for n in N_SWEEP:
                    bias, var, mse = all_stats[est_name][(K, n)]
                    writer.writerow([est_name, K, n,
                                     f'{bias:.8f}', f'{var:.8f}',
                                     f'{mse:.8f}'])

    # --- Step 4: Generate plots ---
    print("\n[4] Generating plots...")

    def save_fig(fig, name):
        for ext in ['pdf', 'png']:
            path = os.path.join(RESULTS_DIR, f'{name}.{ext}')
            fig.savefig(path)
            print(f"    Saved {path}")
        plt.close(fig)

    # ---- Plot 1: Bias vs n (fixed K=K_FIXED) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for est_name in estimator_labels:
        biases = [all_stats[est_name][(K_FIXED, n)][0] for n in N_SWEEP]
        ax.plot(N_SWEEP, biases, 'o-', label=estimator_labels[est_name],
                color=estimator_colors[est_name], linewidth=2, markersize=5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('n (samples per judge)')
    ax.set_ylabel('Bias')
    ax.set_title(f'Bias vs n (K={K_FIXED})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.set_facecolor('white')
    save_fig(fig, 'exp08_bias_vs_n')

    # ---- Plot 2: Variance vs n (fixed K=K_FIXED) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for est_name in estimator_labels:
        variances = [all_stats[est_name][(K_FIXED, n)][1] for n in N_SWEEP]
        ax.plot(N_SWEEP, variances, 'o-', label=estimator_labels[est_name],
                color=estimator_colors[est_name], linewidth=2, markersize=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('n (samples per judge)')
    ax.set_ylabel('Variance')
    ax.set_title(f'Variance vs n (K={K_FIXED})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.set_facecolor('white')
    save_fig(fig, 'exp08_variance_vs_n')

    # ---- Plot 3: MSE vs n (fixed K=K_FIXED) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for est_name in estimator_labels:
        mses = [all_stats[est_name][(K_FIXED, n)][2] for n in N_SWEEP]
        ax.plot(N_SWEEP, mses, 'o-', label=estimator_labels[est_name],
                color=estimator_colors[est_name], linewidth=2, markersize=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('n (samples per judge)')
    ax.set_ylabel('MSE')
    ax.set_title(f'MSE vs n — Plug-in vs Debiased (K={K_FIXED})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.set_facecolor('white')
    save_fig(fig, 'exp08_mse_vs_n')

    # ---- Plot 4: MSE vs K (fixed n=N_FIXED) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for est_name in estimator_labels:
        mses = [all_stats[est_name][(K, N_FIXED)][2] for K in K_SWEEP]
        ax.plot(K_SWEEP, mses, 'o-', label=estimator_labels[est_name],
                color=estimator_colors[est_name], linewidth=2, markersize=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('K (number of judges)')
    ax.set_ylabel('MSE')
    ax.set_title(f'MSE vs K (n={N_FIXED})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.set_facecolor('white')
    save_fig(fig, 'exp08_mse_vs_K')

    # ---- Plot 5: MSE heatmap for plug-in TVD ----
    fig, ax = plt.subplots(figsize=(9, 6))
    mse_grid = np.zeros((len(K_SWEEP), len(N_SWEEP)))
    for i, K in enumerate(K_SWEEP):
        for j, n in enumerate(N_SWEEP):
            mse_grid[i, j] = all_stats['plugin_tvd'][(K, n)][2]

    im = ax.imshow(np.log10(mse_grid), aspect='auto', origin='lower',
                   cmap='viridis')
    ax.set_xticks(range(len(N_SWEEP)))
    ax.set_xticklabels([str(n) for n in N_SWEEP])
    ax.set_yticks(range(len(K_SWEEP)))
    ax.set_yticklabels([str(K) for K in K_SWEEP])
    ax.set_xlabel('n (samples per judge)')
    ax.set_ylabel('K (number of judges)')
    ax.set_title('log10(MSE) — Plug-in TVD Estimator')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('log10(MSE)')

    # Annotate cells with MSE values
    for i in range(len(K_SWEEP)):
        for j in range(len(N_SWEEP)):
            val = mse_grid[i, j]
            text_color = ('white'
                          if np.log10(val) < np.log10(mse_grid).mean()
                          else 'black')
            ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                    fontsize=7, color=text_color)

    fig.set_facecolor('white')
    save_fig(fig, 'exp08_mse_heatmap')

    # ---- Plot 6: Bias-variance decomposition ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Bias-Variance Decomposition vs n (K={K_FIXED})', fontsize=16)

    for idx, est_name in enumerate(estimator_labels):
        ax = axes[idx // 2][idx % 2]
        biases = [all_stats[est_name][(K_FIXED, n)][0] for n in N_SWEEP]
        variances = [all_stats[est_name][(K_FIXED, n)][1] for n in N_SWEEP]
        bias_sq = [b**2 for b in biases]
        mses = [b2 + v for b2, v in zip(bias_sq, variances)]

        x = np.arange(len(N_SWEEP))
        width = 0.35

        ax.bar(x - width / 2, bias_sq, width, label='Bias$^2$',
               color=estimator_colors[est_name], alpha=0.7)
        ax.bar(x + width / 2, variances, width, label='Variance',
               color=estimator_colors[est_name], alpha=0.35, hatch='//')
        ax.plot(x, mses, 'ko-', label='MSE', linewidth=1.5, markersize=4)

        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in N_SWEEP], fontsize=9)
        ax.set_xlabel('n')
        ax.set_ylabel('Value')
        ax.set_title(estimator_labels[est_name])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.set_facecolor('white')
    plt.tight_layout()
    save_fig(fig, 'exp08_bias_variance_decomposition')

    # --- Summary ---
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"True TVD = {true_tvd:.6f}")
    print(f"True MI  = {true_mi:.6f}")
    print(f"\nAt K={K_FIXED}, n={N_FIXED}:")
    for est_name in estimator_labels:
        bias, var, mse = all_stats[est_name][(K_FIXED, N_FIXED)]
        print(f"  {estimator_labels[est_name]:30s}: "
              f"bias={bias:+.6f}, var={var:.6f}, MSE={mse:.6f}")

    print(f"\nTotal elapsed time: {elapsed:.1f}s")
    print(f"All results saved to {RESULTS_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
