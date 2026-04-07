#!/usr/bin/env python3
"""Experiment 07 — Aggregation Rules Comparison.

Compares 6 aggregation methods (majority vote, simple mean, weighted mean,
median, Dawid-Skene EM, spectral) across three judge-ensemble scenarios:
  A) Independent honest judges
  B) Correlated judges (rho = 0.3)
  C) Adversarial contamination (delta = 0.2)

For each scenario, measures aggregated error vs K, fits power-law exponents,
records wall-clock time, and produces publication-quality plots.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Import shared utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (  # noqa: E402
    COLORS,
    K_SWEEP,
    compute_delta,
    compute_true_tvd,
    fit_power_law,
    make_joint_distribution,
    optimal_critic,
)
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

D = 20  # grid dimension
GAMMA = 0.1  # per-judge competence margin
RHO = 0.3  # pairwise correlation (Scenario B)
DELTA_ADV = 0.2  # adversarial fraction (Scenario C)
N_REPS = 200  # Monte Carlo repetitions (may be reduced for large K)
CORRELATION = 0.5  # joint distribution correlation

METHOD_NAMES = [
    "Majority Vote",
    "Simple Mean",
    "Weighted Mean",
    "Median",
    "Dawid-Skene EM",
    "Spectral",
]
METHOD_COLORS = COLORS[:6]

# Use a reduced sweep for expensive methods at large K
K_SWEEP_FULL = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


# ===================================================================
# Aggregation methods
# ===================================================================

def agg_majority_vote(votes):
    """Majority vote over binary votes. votes: (K, d, d) array of 0/1."""
    return (votes.mean(axis=0) >= 0.5).astype(float)


def agg_simple_mean(votes):
    """Simple mean, thresholded at 0.5. votes: (K, d, d)."""
    return (votes.mean(axis=0) >= 0.5).astype(float)


def agg_weighted_mean(votes):
    """Weighted mean using inverse-variance weights estimated from
    inter-judge agreement. votes: (K, d, d)."""
    K = votes.shape[0]
    if K == 1:
        return votes[0]
    # Estimate each judge's accuracy from agreement with others (vectorized)
    mean_votes = votes.mean(axis=0)  # (d, d)
    # For each judge j, leave-one-out mean = (K*mean - votes[j]) / (K-1)
    loo_mean = (mean_votes[np.newaxis, :, :] * K - votes) / (K - 1)  # (K, d, d)
    loo_label = (loo_mean >= 0.5).astype(float)  # (K, d, d)
    agreements = np.mean(votes == loo_label, axis=(1, 2))  # (K,)
    # Weights proportional to estimated accuracy (clipped away from 0.5)
    weights = np.clip(agreements, 0.51, 1.0)
    weights = weights / weights.sum()
    weighted = np.tensordot(weights, votes, axes=([0], [0]))
    return (weighted >= 0.5).astype(float)


def agg_median(votes):
    """Median of votes, thresholded at 0.5. votes: (K, d, d)."""
    return (np.median(votes, axis=0) >= 0.5).astype(float)


def agg_dawid_skene(votes, n_iter=20):
    """Dawid-Skene EM for binary labels. votes: (K, d, d) of 0/1.
    Returns aggregated binary labels (d, d). Fully vectorized."""
    K, d1, d2 = votes.shape
    N = d1 * d2
    # Flatten to (K, N) for EM
    V = votes.reshape(K, N)

    # Initialize true labels from majority vote
    T = (V.mean(axis=0) >= 0.5).astype(float)

    for _ in range(n_iter):
        # M-step: estimate judge error rates (vectorized over judges)
        # pi[j, c] = P(judge j says 1 | true = c)
        mask0 = (T == 0)  # (N,)
        mask1 = (T == 1)  # (N,)
        n0 = mask0.sum()
        n1 = mask1.sum()
        pi = np.full((K, 2), 0.5)
        if n0 > 0:
            pi[:, 0] = V[:, mask0].sum(axis=1) / n0
        if n1 > 0:
            pi[:, 1] = V[:, mask1].sum(axis=1) / n1

        # E-step: vectorized log-odds computation
        p1 = np.clip(pi[:, 1], 1e-8, 1 - 1e-8)  # (K,)
        p0 = np.clip(pi[:, 0], 1e-8, 1 - 1e-8)  # (K,)
        # log_ratio_if_vote_1[j] = log(p1[j]/p0[j])
        # log_ratio_if_vote_0[j] = log((1-p1[j])/(1-p0[j]))
        lr1 = np.log(p1) - np.log(p0)  # (K,)
        lr0 = np.log(1 - p1) - np.log(1 - p0)  # (K,)
        # log_odds[i] = sum_j [ V[j,i]*lr1[j] + (1-V[j,i])*lr0[j] ]
        #             = sum_j [ lr0[j] + V[j,i]*(lr1[j]-lr0[j]) ]
        base = lr0.sum()  # scalar
        diff = lr1 - lr0  # (K,)
        log_odds = base + diff @ V  # (N,)
        T_new = (log_odds >= 0).astype(float)

        if np.array_equal(T_new, T):
            break
        T = T_new

    return T.reshape(d1, d2)


def agg_spectral(votes):
    """Spectral aggregation: SVD of the judge response matrix.
    votes: (K, d, d). Flatten each judge's vote to a vector, form
    (K x N) matrix, take leading right singular vector as the
    aggregated label."""
    K, d1, d2 = votes.shape
    N = d1 * d2
    V = votes.reshape(K, N).astype(np.float64)  # (K, N)

    # Center the matrix
    V_centered = V - V.mean(axis=0, keepdims=True)

    # Use power iteration for the leading singular vector (much faster
    # than full eigendecomposition for large K).
    # We want the leading right singular vector of V_centered.
    # If K < N, compute via V^T V on the K-dimensional side.
    if K < N:
        # Leading left singular vector via power iteration on V_c V_c^T (K x K)
        C = V_centered @ V_centered.T  # (K, K)
        # Power iteration
        u = np.random.default_rng(0).standard_normal(K)
        u = u / np.linalg.norm(u)
        for _ in range(30):
            u_new = C @ u
            u_new = u_new / (np.linalg.norm(u_new) + 1e-15)
            if np.abs(np.abs(u_new @ u) - 1.0) < 1e-10:
                u = u_new
                break
            u = u_new
        scores = u @ V  # (N,)
    else:
        # Power iteration on V_c^T V_c (N x N)
        v = np.random.default_rng(0).standard_normal(N)
        v = v / np.linalg.norm(v)
        for _ in range(30):
            v_new = V_centered.T @ (V_centered @ v)
            v_new = v_new / (np.linalg.norm(v_new) + 1e-15)
            if np.abs(np.abs(v_new @ v) - 1.0) < 1e-10:
                v = v_new
                break
            v = v_new
        scores = v

    # Threshold: align sign with majority vote
    maj = (V.mean(axis=0) >= 0.5).astype(float)
    corr = np.dot(scores - scores.mean(), maj - maj.mean())
    if corr < 0:
        scores = -scores
    return (scores >= np.median(scores)).astype(float).reshape(d1, d2)


AGGREGATORS = [
    ("Majority Vote", agg_majority_vote),
    ("Simple Mean", agg_simple_mean),
    ("Weighted Mean", agg_weighted_mean),
    ("Median", agg_median),
    ("Dawid-Skene EM", agg_dawid_skene),
    ("Spectral", agg_spectral),
]


# ===================================================================
# Vote generation
# ===================================================================

def generate_independent_votes(h_star, K, gamma, rng):
    """Scenario A: Independent honest judges.
    Each judge votes h*(u,v) with prob 0.5 + gamma, else flips.
    Returns (K, d, d) binary array."""
    d1, d2 = h_star.shape
    correct_prob = 0.5 + gamma
    # For each cell, draw whether each judge is correct
    correct = rng.random((K, d1, d2)) < correct_prob
    votes = np.where(correct, h_star[np.newaxis, :, :], 1.0 - h_star[np.newaxis, :, :])
    return votes


def generate_correlated_votes(h_star, K, gamma, rho, rng):
    """Scenario B: Correlated judges via equicorrelation construction.
    For each cell, draw Z_common ~ N(0,1), then for each judge
    Z_j ~ N(0,1), form S_j = sqrt(rho)*Z_common + sqrt(1-rho)*Z_j.
    Judge votes h* if S_j > threshold (set so marginal accuracy = 0.5 + gamma).
    Returns (K, d, d) binary array."""
    d1, d2 = h_star.shape
    # Threshold: P(S_j > threshold) = 0.5 + gamma when S_j ~ N(0,1)
    # Since S_j = sqrt(rho)*Z_c + sqrt(1-rho)*Z_j has variance 1
    threshold = norm.ppf(0.5 - gamma)  # votes correct when S_j > threshold

    Z_common = rng.standard_normal((1, d1, d2))  # shared latent
    Z_j = rng.standard_normal((K, d1, d2))  # independent part
    S = np.sqrt(rho) * Z_common + np.sqrt(1 - rho) * Z_j

    # Judge votes h* if S > threshold, else flips
    correct = S > threshold
    votes = np.where(correct, h_star[np.newaxis, :, :], 1.0 - h_star[np.newaxis, :, :])
    return votes


def generate_adversarial_votes(h_star, K, gamma, delta, rng):
    """Scenario C: Adversarial contamination.
    (1-delta) fraction are honest (as in Scenario A),
    delta fraction always output 1 - h*(u,v).
    Returns (K, d, d) binary array."""
    d1, d2 = h_star.shape
    n_adv = int(np.floor(K * delta))
    n_honest = K - n_adv

    votes = np.empty((K, d1, d2))
    # Honest judges
    if n_honest > 0:
        votes[:n_honest] = generate_independent_votes(h_star, n_honest, gamma, rng)
    # Adversarial judges
    if n_adv > 0:
        votes[n_honest:] = 1.0 - h_star[np.newaxis, :, :]

    # Shuffle so adversaries are not always at the end
    perm = rng.permutation(K)
    votes = votes[perm]
    return votes


# ===================================================================
# Main experiment runner
# ===================================================================

def run_scenario(scenario_name, vote_generator, P_UV, P_U, P_V, h_star,
                 tau_c, k_sweep, n_reps, rng):
    """Run all 6 aggregation methods for a given scenario.

    Returns:
        errors: dict mapping method_name -> list of mean errors (one per K)
        stds: dict mapping method_name -> list of std errors (one per K)
        times: dict mapping method_name -> list of mean wall-clock times per rep
    """
    errors = {name: [] for name, _ in AGGREGATORS}
    stds = {name: [] for name, _ in AGGREGATORS}
    times_dict = {name: [] for name, _ in AGGREGATORS}

    for K in k_sweep:
        print(f"  {scenario_name}: K={K}", flush=True)
        # Adjust reps for large K to keep runtime manageable
        if K >= 512:
            effective_reps = min(n_reps, 50)
        elif K >= 128:
            effective_reps = min(n_reps, 100)
        else:
            effective_reps = n_reps

        for method_name, method_fn in AGGREGATORS:
            gaps = []
            elapsed_list = []
            for rep in range(effective_reps):
                # Generate votes
                votes = vote_generator(h_star, K, rng)

                # Aggregate and time it
                t0 = time.perf_counter()
                agg_label = method_fn(votes)
                t1 = time.perf_counter()
                elapsed_list.append(t1 - t0)

                # Compute Delta_c(aggregated)
                delta_c = compute_delta(agg_label, P_UV, P_U, P_V)
                gap = tau_c - delta_c
                gaps.append(max(gap, 0.0))

            errors[method_name].append(np.mean(gaps))
            stds[method_name].append(np.std(gaps))
            times_dict[method_name].append(np.mean(elapsed_list))

    return errors, stds, times_dict


def plot_error_vs_K(k_sweep, errors, stds, scenario_label, filename_base):
    """Plot error vs K for all methods on log-log axes."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, (method_name, _) in enumerate(AGGREGATORS):
        e = np.array(errors[method_name])
        s = np.array(stds[method_name])
        k_arr = np.array(k_sweep, dtype=float)
        # Only plot positive errors on log scale
        mask = e > 0
        if mask.sum() == 0:
            continue
        ax.plot(k_arr[mask], e[mask], "o-", color=METHOD_COLORS[idx],
                label=method_name, linewidth=2, markersize=5)
        ax.fill_between(k_arr[mask],
                        np.maximum(e[mask] - s[mask], 1e-10),
                        e[mask] + s[mask],
                        alpha=0.15, color=METHOD_COLORS[idx])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of judges $K$")
    ax.set_ylabel(r"Aggregation gap $\tau(c) - \Delta_c(\hat{h}_K)$")
    ax.set_title(f"Experiment 07: Aggregation Rules -- {scenario_label}")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="gray")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"{filename_base}.{ext}"))
    plt.close(fig)
    print(f"  Saved {filename_base}.pdf/png")


def plot_pareto(all_times, all_errors, scenario_names):
    """Pareto plot: accuracy (1/error) vs wall-clock time."""
    fig, ax = plt.subplots(figsize=(9, 6))

    markers = ["o", "s", "^"]
    for sc_idx, sc_name in enumerate(scenario_names):
        for m_idx, (method_name, _) in enumerate(AGGREGATORS):
            # Use the error at the largest K as the "final accuracy"
            err = all_errors[sc_name][method_name][-1]
            t = all_times[sc_name][method_name][-1]
            if err <= 0:
                err = 1e-8
            ax.scatter(t, err, marker=markers[sc_idx],
                       color=METHOD_COLORS[m_idx], s=100, zorder=5,
                       edgecolors="black", linewidths=0.5)

    # Legend: methods
    for m_idx, (method_name, _) in enumerate(AGGREGATORS):
        ax.scatter([], [], color=METHOD_COLORS[m_idx], s=80,
                   label=method_name, edgecolors="black", linewidths=0.5)
    # Legend: scenarios
    for sc_idx, sc_name in enumerate(scenario_names):
        ax.scatter([], [], marker=markers[sc_idx], color="gray", s=80,
                   label=sc_name, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Wall-clock time per rep (s) at largest $K$")
    ax.set_ylabel(r"Aggregation gap at largest $K$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Experiment 07: Accuracy vs. Computation Pareto")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="gray",
              fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"exp07_pareto_accuracy_time.{ext}"))
    plt.close(fig)
    print("  Saved exp07_pareto_accuracy_time.pdf/png")


def build_comparison_table(all_errors, k_sweep, scenario_names):
    """Build and save CSV of fitted power-law exponents."""
    rows = []
    for sc_name in scenario_names:
        for method_name, _ in AGGREGATORS:
            e = all_errors[sc_name][method_name]
            alpha, a, b, R2 = fit_power_law(k_sweep, e)
            rows.append({
                "Scenario": sc_name,
                "Method": method_name,
                "alpha": round(alpha, 4) if not np.isnan(alpha) else "N/A",
                "a": round(a, 4) if not np.isnan(a) else "N/A",
                "b": round(b, 6) if not np.isnan(b) else "N/A",
                "R2": round(R2, 4) if not np.isnan(R2) else "N/A",
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "exp07_comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved exp07_comparison_table.csv")
    print(df.to_string(index=False))
    return df


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Experiment 07: Aggregation Rules Comparison")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Build ground-truth distribution
    print("\nBuilding joint distribution (d=%d, correlation=%.1f)..." % (D, CORRELATION))
    P_UV, P_U, P_V = make_joint_distribution(d=D, correlation=CORRELATION)
    h_star = optimal_critic(P_UV, P_U, P_V)
    tau_c = compute_true_tvd(P_UV, P_U, P_V)
    delta_star = compute_delta(h_star, P_UV, P_U, P_V)
    print(f"  tau(c) = {tau_c:.6f}")
    print(f"  Delta_c(h*) = {delta_star:.6f}")
    print(f"  Optimality gap h*: {tau_c - delta_star:.2e}")

    k_sweep = K_SWEEP_FULL

    all_errors = {}
    all_stds = {}
    all_times = {}

    # ------------------------------------------------------------------
    # Scenario A: Independent honest judges
    # ------------------------------------------------------------------
    print("\n--- Scenario A: Independent honest judges (gamma=%.2f) ---" % GAMMA)

    def gen_A(h_star, K, rng):
        return generate_independent_votes(h_star, K, GAMMA, rng)

    errors_A, stds_A, times_A = run_scenario(
        "Independent", gen_A, P_UV, P_U, P_V, h_star, tau_c,
        k_sweep, N_REPS, rng,
    )
    all_errors["Independent"] = errors_A
    all_stds["Independent"] = stds_A
    all_times["Independent"] = times_A
    plot_error_vs_K(k_sweep, errors_A, stds_A,
                    "Independent Judges ($\\gamma=%.2f$)" % GAMMA,
                    "exp07_error_vs_K_independent")

    # ------------------------------------------------------------------
    # Scenario B: Correlated judges
    # ------------------------------------------------------------------
    print("\n--- Scenario B: Correlated judges (rho=%.2f) ---" % RHO)

    def gen_B(h_star, K, rng):
        return generate_correlated_votes(h_star, K, GAMMA, RHO, rng)

    errors_B, stds_B, times_B = run_scenario(
        "Correlated", gen_B, P_UV, P_U, P_V, h_star, tau_c,
        k_sweep, N_REPS, rng,
    )
    all_errors["Correlated"] = errors_B
    all_stds["Correlated"] = stds_B
    all_times["Correlated"] = times_B
    plot_error_vs_K(k_sweep, errors_B, stds_B,
                    "Correlated Judges ($\\rho=%.2f$)" % RHO,
                    "exp07_error_vs_K_correlated")

    # ------------------------------------------------------------------
    # Scenario C: Adversarial contamination
    # ------------------------------------------------------------------
    print("\n--- Scenario C: Adversarial contamination (delta=%.2f) ---" % DELTA_ADV)

    def gen_C(h_star, K, rng):
        return generate_adversarial_votes(h_star, K, GAMMA, DELTA_ADV, rng)

    errors_C, stds_C, times_C = run_scenario(
        "Adversarial", gen_C, P_UV, P_U, P_V, h_star, tau_c,
        k_sweep, N_REPS, rng,
    )
    all_errors["Adversarial"] = errors_C
    all_stds["Adversarial"] = stds_C
    all_times["Adversarial"] = times_C
    plot_error_vs_K(k_sweep, errors_C, stds_C,
                    "Adversarial Judges ($\\delta=%.2f$)" % DELTA_ADV,
                    "exp07_error_vs_K_adversarial")

    # ------------------------------------------------------------------
    # Pareto plot
    # ------------------------------------------------------------------
    print("\n--- Pareto plot: accuracy vs. computation ---")
    scenario_names = ["Independent", "Correlated", "Adversarial"]
    plot_pareto(all_times, all_errors, scenario_names)

    # ------------------------------------------------------------------
    # Comparison table (CSV)
    # ------------------------------------------------------------------
    print("\n--- Comparison table ---")
    build_comparison_table(all_errors, k_sweep, scenario_names)

    print("\n" + "=" * 60)
    print("Experiment 07 complete. Results in:", RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
