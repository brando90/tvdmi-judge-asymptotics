"""Experiment 09 — Pairwise Comparisons (Bradley-Terry Judge Ensembles).

Extends the judge ensemble framework to pairwise (Bradley-Terry) judgments.
Measures ranking error (Kendall tau distance) as a function of K (judges)
and m (comparisons per judge).

Three aggregation methods:
  1. Bradley-Terry MLE (pooled comparisons, fitted via MM algorithm)
  2. Spectral ranking (stationary distribution of comparison Markov chain)
  3. Simple win-rate (wins / total comparisons per item)
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# Defer heavy imports (matplotlib is slow to load on some systems)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _setup_plotting():
    """Import matplotlib and shared utilities on first use."""
    global plt, COLORS, fit_power_law, fit_exponential_decay
    from utils import (fit_power_law as _fpl, fit_exponential_decay as _fed,
                       COLORS as _C)
    import matplotlib.pyplot as _plt
    plt = _plt
    COLORS = _C
    fit_power_law = _fpl
    fit_exponential_decay = _fed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ITEMS = 20
K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
M_VALUES = [10, 20, 50, 100, 200, 500]
GAMMA_VALUES = [1, 3, 5, 10]
N_MC = 200  # Monte Carlo repetitions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# True quality scores — evenly spaced in [0, 1]
THETA_TRUE = np.arange(N_ITEMS, dtype=float) / N_ITEMS  # 0/20, 1/20, ..., 19/20

# Precompute pairwise differences of true scores for fast Kendall tau
_TRIU = np.triu_indices(N_ITEMS, k=1)
_TRUE_DIFF = THETA_TRUE[_TRIU[0]] - THETA_TRUE[_TRIU[1]]
_TRUE_SIGN = np.sign(_TRUE_DIFF)


def _fast_kendall_tau(scores):
    """Compute Kendall tau correlation between scores and THETA_TRUE.

    Uses precomputed pairwise structure. O(n^2) but very fast for n=20
    since it's fully vectorized with no Python loops.
    """
    diff = scores[_TRIU[0]] - scores[_TRIU[1]]
    signs = np.sign(diff)
    concordant = np.sum(signs == _TRUE_SIGN)
    discordant = np.sum(signs == -_TRUE_SIGN)
    total = concordant + discordant
    if total == 0:
        return 0.0
    return float(concordant - discordant) / float(total)


# ---------------------------------------------------------------------------
# Comparison generation
# ---------------------------------------------------------------------------

def _generate_single(K, m, gamma, rng):
    """Generate one wins matrix for a single MC rep."""
    n = N_ITEMS
    total = K * m
    items_i = rng.integers(0, n, size=total)
    items_j = rng.integers(0, n - 1, size=total)
    items_j = np.where(items_j >= items_i, items_j + 1, items_j)

    gd = gamma * (THETA_TRUE[items_i] - THETA_TRUE[items_j])
    prob = np.empty_like(gd)
    pos = gd >= 0
    prob[pos] = 1.0 / (1.0 + np.exp(-gd[pos]))
    prob[~pos] = np.exp(gd[~pos]) / (1.0 + np.exp(gd[~pos]))
    outcomes = rng.random(total) < prob

    nn = n * n
    w_flat = items_i[outcomes] * n + items_j[outcomes]
    l_flat = items_j[~outcomes] * n + items_i[~outcomes]
    counts = np.bincount(np.concatenate([w_flat, l_flat]), minlength=nn)
    return counts[:nn].reshape(n, n).astype(np.int32)


def generate_wins_batch(K, m, gamma, n_mc, rng):
    """Generate wins matrices for n_mc Monte Carlo reps.

    Returns:
        wins_batch: (n_mc, N_ITEMS, N_ITEMS) int array.
    """
    n = N_ITEMS
    total = K * m

    # For small total comparisons, batch all reps for speed.
    # For large totals, generate per-rep to avoid memory pressure.
    if total * n_mc <= 500_000:
        all_total = n_mc * total
        items_i = rng.integers(0, n, size=all_total)
        items_j = rng.integers(0, n - 1, size=all_total)
        items_j = np.where(items_j >= items_i, items_j + 1, items_j)

        gd = gamma * (THETA_TRUE[items_i] - THETA_TRUE[items_j])
        prob = np.empty_like(gd)
        pos = gd >= 0
        prob[pos] = 1.0 / (1.0 + np.exp(-gd[pos]))
        prob[~pos] = np.exp(gd[~pos]) / (1.0 + np.exp(gd[~pos]))
        outcomes = rng.random(all_total) < prob

        rep_idx = np.repeat(np.arange(n_mc, dtype=np.int32), total)
        flat_size = n_mc * n * n
        w_flat = rep_idx[outcomes] * (n * n) + items_i[outcomes] * n + items_j[outcomes]
        l_flat = rep_idx[~outcomes] * (n * n) + items_j[~outcomes] * n + items_i[~outcomes]
        counts = np.bincount(np.concatenate([w_flat, l_flat]),
                             minlength=flat_size)
        return counts[:flat_size].reshape(n_mc, n, n).astype(np.int32)
    else:
        # Per-rep generation for large configs
        batch = np.empty((n_mc, n, n), dtype=np.int32)
        for r in range(n_mc):
            batch[r] = _generate_single(K, m, gamma, rng)
        return batch


# ---------------------------------------------------------------------------
# Aggregation Method 1: Bradley-Terry MLE (MM algorithm, Hunter 2004)
# ---------------------------------------------------------------------------

def bt_mle(wins, max_iter=20, tol=1e-3):
    """Fit Bradley-Terry model via the MM algorithm.

    Returns estimated quality scores (length N_ITEMS).
    """
    n = wins.shape[0]
    w = wins.sum(axis=1).astype(np.float64)  # total wins per item
    n_ij = (wins + wins.T).astype(np.float64)  # total comparisons per pair

    # Items with zero wins get a small pseudocount to avoid degeneracy
    if np.any(w == 0):
        w = w + 0.5
        n_ij = n_ij + 1.0
        np.fill_diagonal(n_ij, 0.0)

    p = np.ones(n, dtype=np.float64)

    for _ in range(max_iter):
        p_old = p.copy()
        denom = p[:, None] + p[None, :]  # shape (n, n)
        np.fill_diagonal(denom, 1.0)
        q = (n_ij / denom).sum(axis=1)
        q = np.maximum(q, 1e-12)
        p = w / q
        # Normalize so geometric mean = 1
        p = p / p.mean()
        if np.max(np.abs(p - p_old)) < tol:
            break

    return p  # scores (higher = better)


# ---------------------------------------------------------------------------
# Aggregation Method 2: Spectral Ranking
# ---------------------------------------------------------------------------

def spectral_ranking(wins):
    """Rank items via stationary distribution of comparison Markov chain."""
    n = wins.shape[0]
    total = (wins + wins.T).astype(np.float64)

    # Transition matrix: P[i,j] = wins[j,i] / total[i,j]
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(total > 0, wins.T / total, 1.0 / (n - 1))
    np.fill_diagonal(P, 0.0)

    # Normalize rows
    rs = P.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    P /= rs

    # Power iteration for stationary distribution
    pi = np.ones(n) / n
    for _ in range(300):
        pi_new = pi @ P
        s = pi_new.sum()
        if s > 0:
            pi_new /= s
        if np.max(np.abs(pi_new - pi)) < 1e-12:
            break
        pi = pi_new

    return np.abs(pi_new)


# ---------------------------------------------------------------------------
# Aggregation Method 3: Simple Win-Rate
# ---------------------------------------------------------------------------

def win_rate_ranking(wins):
    """Rank items by simple win rate."""
    total_wins = wins.sum(axis=1).astype(np.float64)
    total_games = total_wins + wins.sum(axis=0).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        rates = np.where(total_games > 0, total_wins / total_games, 0.5)
    return rates


# ---------------------------------------------------------------------------
# Batch error computation
# ---------------------------------------------------------------------------

def compute_errors_batch(wins_batch, method_func):
    """Apply a ranking method to each wins matrix and compute Kendall tau errors.

    Returns array of errors (1 - tau) of length n_mc.
    """
    n_mc = wins_batch.shape[0]
    errors = np.empty(n_mc)
    for i in range(n_mc):
        scores = method_func(wins_batch[i])
        tau = _fast_kendall_tau(scores)
        errors[i] = 1.0 - tau
    return errors


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_experiment():
    """Run the full experiment sweep and collect results."""
    print("=" * 70, flush=True)
    print("Experiment 09: Pairwise Comparisons (Bradley-Terry Judge Ensembles)",
          flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()
    all_results = []

    methods = {
        'bt_mle': bt_mle,
        'spectral': spectral_ranking,
        'win_rate': win_rate_ranking,
    }

    # Total number of (gamma, K, m) combos
    total_combos = len(GAMMA_VALUES) * len(K_VALUES) * len(M_VALUES)
    combo_count = 0

    for gamma in GAMMA_VALUES:
        for K in K_VALUES:
            for m in M_VALUES:
                combo_count += 1
                elapsed = time.time() - t_start
                if combo_count % 10 == 0 or combo_count <= 2:
                    print(f"  [{elapsed:6.1f}s] Combo {combo_count}/{total_combos}: "
                          f"gamma={gamma}, K={K}, m={m}", flush=True)

                # Generate all MC reps at once for this (K, m, gamma)
                rng = np.random.default_rng(42 + combo_count)
                wins_batch = generate_wins_batch(K, m, gamma, N_MC, rng)

                for method_name, method_func in methods.items():
                    errors = compute_errors_batch(wins_batch, method_func)
                    all_results.append({
                        'method': method_name,
                        'gamma': gamma,
                        'K': K,
                        'm': m,
                        'mean_kendall_tau': 1.0 - errors.mean(),
                        'std_kendall_tau': errors.std(),
                        'mean_error': errors.mean(),
                        'std_error': errors.std(),
                    })

    elapsed = time.time() - t_start
    print(f"\nSimulation complete in {elapsed:.1f}s", flush=True)

    df = pd.DataFrame(all_results)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'exp09_results.csv')
    df[['method', 'gamma', 'K', 'm', 'mean_kendall_tau', 'std_kendall_tau']].to_csv(
        csv_path, index=False)
    print(f"Results saved to {csv_path}", flush=True)

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_error_vs_K(df):
    """Plot 1: Ranking error vs K for fixed m=100, one curve per gamma, BT-MLE."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df[(df['method'] == 'bt_mle') & (df['m'] == 100)]

    for idx, gamma in enumerate(GAMMA_VALUES):
        data = sub[sub['gamma'] == gamma].sort_values('K')
        ax.errorbar(data['K'], data['mean_error'], yerr=data['std_error'],
                    marker='o', color=COLORS[idx % len(COLORS)],
                    label=rf'$\gamma = {gamma}$', capsize=3, linewidth=2)

    ax.set_xscale('log', base=2)
    ax.set_xlabel(r'Number of judges $K$')
    ax.set_ylabel(r'Ranking error $(1 - \tau)$')
    ax.set_title(r'BT-MLE: Ranking error vs $K$ (fixed $m = 100$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(RESULTS_DIR, f'exp09_error_vs_K.{ext}'))
    plt.close(fig)
    print("  Saved exp09_error_vs_K.{pdf,png}", flush=True)


def plot_error_vs_m(df):
    """Plot 2: Ranking error vs m for fixed K=16, one curve per gamma."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df[(df['method'] == 'bt_mle') & (df['K'] == 16)]

    for idx, gamma in enumerate(GAMMA_VALUES):
        data = sub[sub['gamma'] == gamma].sort_values('m')
        ax.errorbar(data['m'], data['mean_error'], yerr=data['std_error'],
                    marker='s', color=COLORS[idx % len(COLORS)],
                    label=rf'$\gamma = {gamma}$', capsize=3, linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel(r'Comparisons per judge $m$')
    ax.set_ylabel(r'Ranking error $(1 - \tau)$')
    ax.set_title(r'BT-MLE: Ranking error vs $m$ (fixed $K = 16$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(RESULTS_DIR, f'exp09_error_vs_m.{ext}'))
    plt.close(fig)
    print("  Saved exp09_error_vs_m.{pdf,png}", flush=True)


def plot_heatmap_Km(df):
    """Plot 3: Heatmap of ranking error over (K, m) grid for gamma=5, BT-MLE."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sub = df[(df['method'] == 'bt_mle') & (df['gamma'] == 5)]

    K_vals = sorted(sub['K'].unique())
    m_vals = sorted(sub['m'].unique())
    grid = np.full((len(m_vals), len(K_vals)), np.nan)

    for i, m_val in enumerate(m_vals):
        for j, K_val in enumerate(K_vals):
            row = sub[(sub['K'] == K_val) & (sub['m'] == m_val)]
            if len(row) > 0:
                grid[i, j] = row['mean_error'].values[0]

    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn_r', origin='lower')
    ax.set_xticks(range(len(K_vals)))
    ax.set_xticklabels(K_vals)
    ax.set_yticks(range(len(m_vals)))
    ax.set_yticklabels(m_vals)
    ax.set_xlabel(r'Number of judges $K$')
    ax.set_ylabel(r'Comparisons per judge $m$')
    ax.set_title(r'Ranking error heatmap ($\gamma = 5$, BT-MLE)')

    # Annotate cells with values
    valid = grid[~np.isnan(grid)]
    mid = 0.5 * (valid.max() + valid.min()) if len(valid) > 0 else 0.5
    for i in range(len(m_vals)):
        for j in range(len(K_vals)):
            if not np.isnan(grid[i, j]):
                color = 'white' if grid[i, j] > mid else 'black'
                ax.text(j, i, f'{grid[i, j]:.3f}', ha='center', va='center',
                        fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label=r'Ranking error $(1 - \tau)$')

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(RESULTS_DIR, f'exp09_heatmap_Km.{ext}'))
    plt.close(fig)
    print("  Saved exp09_heatmap_Km.{pdf,png}", flush=True)


def plot_methods_comparison(df):
    """Plot 4: BT-MLE vs spectral vs win-rate for gamma=5, fixed m=100."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df[(df['gamma'] == 5) & (df['m'] == 100)]

    method_labels = {'bt_mle': 'BT-MLE', 'spectral': 'Spectral', 'win_rate': 'Win-Rate'}
    markers = {'bt_mle': 'o', 'spectral': 's', 'win_rate': '^'}

    for idx, method in enumerate(['bt_mle', 'spectral', 'win_rate']):
        data = sub[sub['method'] == method].sort_values('K')
        ax.errorbar(data['K'], data['mean_error'], yerr=data['std_error'],
                    marker=markers[method], color=COLORS[idx],
                    label=method_labels[method], capsize=3, linewidth=2)

    ax.set_xscale('log', base=2)
    ax.set_xlabel(r'Number of judges $K$')
    ax.set_ylabel(r'Ranking error $(1 - \tau)$')
    ax.set_title(r'Method comparison ($\gamma = 5$, $m = 100$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(RESULTS_DIR, f'exp09_methods_comparison.{ext}'))
    plt.close(fig)
    print("  Saved exp09_methods_comparison.{pdf,png}", flush=True)


def plot_budget_tradeoff(df):
    """Plot 5: Error vs total budget (K*m) showing different K/m splits."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df[(df['method'] == 'bt_mle') & (df['gamma'] == 5)]

    K_highlight = [2, 8, 32, 128]
    for idx, K_val in enumerate(K_highlight):
        data = sub[sub['K'] == K_val].sort_values('m')
        budget = data['K'] * data['m']
        ax.plot(budget, data['mean_error'],
                marker='o', color=COLORS[idx % len(COLORS)],
                label=rf'$K = {K_val}$', linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel(r'Total budget $K \times m$')
    ax.set_ylabel(r'Ranking error $(1 - \tau)$')
    ax.set_title(r'Budget tradeoff: error vs total comparisons ($\gamma = 5$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(RESULTS_DIR, f'exp09_budget_tradeoff.{ext}'))
    plt.close(fig)
    print("  Saved exp09_budget_tradeoff.{pdf,png}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    df = run_experiment()

    print("\nLoading plotting libraries...", flush=True)
    _setup_plotting()

    print("Generating plots...", flush=True)
    plot_error_vs_K(df)
    plot_error_vs_m(df)
    plot_heatmap_Km(df)
    plot_methods_comparison(df)
    plot_budget_tradeoff(df)

    print("\nDone! All results in:", RESULTS_DIR, flush=True)

    # Print summary statistics
    print("\n--- Summary ---", flush=True)
    for method in ['bt_mle', 'spectral', 'win_rate']:
        sub = df[(df['method'] == method) & (df['gamma'] == 5) & (df['m'] == 100)]
        best = sub.loc[sub['mean_error'].idxmin()]
        print(f"  {method:10s} (gamma=5, m=100): best error = {best['mean_error']:.4f} "
              f"at K={int(best['K'])}", flush=True)
