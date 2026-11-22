"""Experiment 03 — Two-Axis Decomposition (K x n).

Tests Theorem 3 / Theorem 3' (split-sample): with n joint and n product
samples per split, Hoeffding gives 2K * exp(-n_sel * eta^2), and disjoint
split-sample evaluation eliminates in-sample maximization bias:

    Pr(|tau_hat_split - tau(c)| > eps + 2*eta_sel + eta_eval)
        <= exp(-K * p_eps) + 2K * exp(-n_sel * eta_sel^2) + 2 * exp(-n_eval * eta_eval^2)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ---------------------------------------------------------------------------
# path setup — import shared utilities from parent experiments/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    make_joint_distribution,
    optimal_critic,
    compute_delta,
    fit_power_law,
    COLORS,
)

# ---------------------------------------------------------------------------
# W&B setup
# ---------------------------------------------------------------------------
WANDB_ENTITY = "brando-su"
WANDB_PROJECT = "tvdmi-judge-asymptotics"
try:
    wandb_key_path = os.path.expanduser("~/keys/brandos_wandb_key.txt")
    os.environ["WANDB_API_KEY"] = open(wandb_key_path).read().strip()
except FileNotFoundError:
    print("[warn] W&B key not found; will run without W&B logging.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[warn] wandb not installed; skipping W&B logging.")

try:
    import wandb_workspaces.reports.v2 as wr
    WR_AVAILABLE = True
except ImportError:
    WR_AVAILABLE = False
    print("[warn] wandb_workspaces not installed; skipping W&B report.")

# ---------------------------------------------------------------------------
# output directories
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SUMMARY_DIR = os.path.join(os.path.dirname(__file__), 'results_summary')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# experiment parameters
# ---------------------------------------------------------------------------
D = 20                      # grid dimension for joint distribution
CORRELATION = 0.5           # dependence strength
ALPHA_SHAPE = 1.0           # Beta(alpha_shape, alpha_shape) for judge quality
BETA_PARAM = 1.0            # (beta = 1.0 => Beta(1,1) = Uniform)
K_GRID = [4, 16, 64, 256, 1024]
N_GRID = [50, 100, 500, 1000, 5000]
N_REPS_3 = 200


def generate_noisy_critic(h_star, quality, d):
    """Create a noisy version of h* for a judge with given quality.

    quality in [0, 1]: fraction of entries that agree with h*.
    With probability (1 - quality), each entry is flipped.
    """
    flip_mask = np.random.rand(d, d) > quality
    h_j = h_star.copy()
    h_j[flip_mask] = 1.0 - h_j[flip_mask]
    return h_j


def sample_from_joint(P_UV, n, d):
    """Draw n samples (u_i, v_i) from P_UV."""
    flat = P_UV.ravel()
    indices = np.random.choice(d * d, size=n, p=flat)
    u_indices, v_indices = np.unravel_index(indices, (d, d))
    return u_indices, v_indices


def sample_from_product(P_U, P_V, n):
    """Draw n samples (u_i', v_i') from P_U x P_V (independent)."""
    u_indices = np.random.choice(len(P_U), size=n, p=P_U)
    v_indices = np.random.choice(len(P_V), size=n, p=P_V)
    return u_indices, v_indices


def run_experiment():
    """Run the full two-axis decomposition experiment."""
    # ------------------------------------------------------------------
    # 1. ground truth
    # ------------------------------------------------------------------
    P_UV, P_U, P_V = make_joint_distribution(d=D, correlation=CORRELATION)
    h_star = optimal_critic(P_UV, P_U, P_V)
    tau_c_critic = compute_delta(h_star, P_UV, P_U, P_V)
    print(f"Ground truth: tau_c(h*) = {tau_c_critic:.6f}")
    print(f"Grid: K in {K_GRID}, n in {N_GRID}, reps = {N_REPS_3}")
    print()

    # ------------------------------------------------------------------
    # 2. sweep over (K, n) grid
    # ------------------------------------------------------------------
    gap_matrix = np.zeros((len(K_GRID), len(N_GRID)))

    for ki, K in enumerate(K_GRID):
        for ni, n in enumerate(N_GRID):
            gaps = np.zeros(N_REPS_3)
            for rep in range(N_REPS_3):
                # draw K judges: quality ~ Beta(1, 1) = Uniform[0,1]
                qualities = np.random.beta(ALPHA_SHAPE, BETA_PARAM, size=K)
                judges = [generate_noisy_critic(h_star, q, D) for q in qualities]

                # Split-sample cross-fitting (n_sel + n_eval = n) eliminates winner's-curse bias
                n_sel = max(1, n // 2)
                n_eval = max(1, n - n_sel)
                delta_sel = np.zeros(K)
                for j_idx, h_j in enumerate(judges):
                    u_pos, v_pos = sample_from_joint(P_UV, n_sel, D)
                    u_neg, v_neg = sample_from_product(P_U, P_V, n_sel)
                    delta_sel[j_idx] = (
                        np.mean(h_j[u_pos, v_pos]) - np.mean(h_j[u_neg, v_neg])
                    )

                j_star = int(np.argmax(delta_sel))
                u_ev_p, v_ev_p = sample_from_joint(P_UV, n_eval, D)
                u_ev_n, v_ev_n = sample_from_product(P_U, P_V, n_eval)
                tau_hat_split = (
                    np.mean(judges[j_star][u_ev_p, v_ev_p])
                    - np.mean(judges[j_star][u_ev_n, v_ev_n])
                )
                gaps[rep] = abs(tau_c_critic - tau_hat_split)

            mean_gap = np.mean(gaps)
            gap_matrix[ki, ni] = mean_gap
            print(f"K={K}, n={n}: mean_gap={mean_gap:.6f}")

    print()
    return gap_matrix, tau_c_critic


def plot_heatmap(gap_matrix):
    """Plot the 5x5 heatmap of mean gap over the (K, n) grid."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        gap_matrix,
        annot=True,
        fmt=".4f",
        xticklabels=[str(n) for n in N_GRID],
        yticklabels=[str(K) for K in K_GRID],
        cmap="YlOrRd_r",
        cbar_kws={'label': 'Mean Gap'},
        ax=ax,
    )
    ax.set_xlabel('n (samples per judge)')
    ax.set_ylabel('K (number of judges)')
    ax.set_title('Two-Axis Decomposition: Mean Gap (tau_c - tau_hat)')
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(RESULTS_DIR, f'exp3_heatmap.{ext}'))
    plt.close(fig)
    print("Saved heatmap.")


def plot_marginals(gap_matrix):
    """Plot marginal curves: gap vs K (fixed n) and gap vs n (fixed K)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- gap vs K for fixed n = 1000 ---
    n_idx = N_GRID.index(1000)
    gaps_vs_K = gap_matrix[:, n_idx]
    ax1.plot(K_GRID, gaps_vs_K, 'o-', color=COLORS[0], linewidth=2,
             markersize=6, label=f'n = {N_GRID[n_idx]}')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('K (number of judges)')
    ax1.set_ylabel('Mean Gap')
    ax1.set_title('Gap vs K (fixed n = 1000)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # fit power law for the K marginal
    alpha, a, b, R2 = fit_power_law(K_GRID, gaps_vs_K)
    if not np.isnan(alpha):
        K_fit = np.logspace(np.log10(min(K_GRID)), np.log10(max(K_GRID)), 100)
        ax1.plot(K_fit, a * K_fit ** (-alpha) + b, '--', color=COLORS[1],
                 label=f'fit: alpha={alpha:.2f}, R2={R2:.3f}')
        ax1.legend()

    # --- gap vs n for fixed K = 64 ---
    k_idx = K_GRID.index(64)
    gaps_vs_n = gap_matrix[k_idx, :]
    ax2.plot(N_GRID, gaps_vs_n, 's-', color=COLORS[2], linewidth=2,
             markersize=6, label=f'K = {K_GRID[k_idx]}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('n (samples per judge)')
    ax2.set_ylabel('Mean Gap')
    ax2.set_title('Gap vs n (fixed K = 64)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # fit power law for the n marginal
    alpha_n, a_n, b_n, R2_n = fit_power_law(N_GRID, gaps_vs_n)
    if not np.isnan(alpha_n):
        n_fit = np.logspace(np.log10(min(N_GRID)), np.log10(max(N_GRID)), 100)
        ax2.plot(n_fit, a_n * n_fit ** (-alpha_n) + b_n, '--', color=COLORS[3],
                 label=f'fit: alpha={alpha_n:.2f}, R2={R2_n:.3f}')
        ax2.legend()

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(RESULTS_DIR, f'exp3_marginals.{ext}'))
    plt.close(fig)
    print("Saved marginals.")


def plot_contour(gap_matrix):
    """Plot contour of the gap surface and mark the gap=0.05 contour."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # use log-spaced axes for smoother contours
    log_K = np.log10(K_GRID)
    log_n = np.log10(N_GRID)
    N_mesh, K_mesh = np.meshgrid(log_n, log_K)

    levels = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    cs = ax.contourf(N_mesh, K_mesh, gap_matrix, levels=20, cmap='YlOrRd_r')
    plt.colorbar(cs, ax=ax, label='Mean Gap')

    # highlight the 0.05 contour
    cs_line = ax.contour(N_mesh, K_mesh, gap_matrix, levels=[0.05],
                         colors='black', linewidths=2, linestyles='--')
    ax.clabel(cs_line, fmt='%.2f', fontsize=10)

    ax.set_xlabel('log10(n)')
    ax.set_ylabel('log10(K)')
    ax.set_title('K-n Tradeoff Contour (gap = 0.05 highlighted)')

    # tick labels showing actual values
    ax.set_xticks(log_n)
    ax.set_xticklabels([str(n) for n in N_GRID])
    ax.set_yticks(log_K)
    ax.set_yticklabels([str(K) for K in K_GRID])

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(RESULTS_DIR, f'exp3_tradeoff_contour.{ext}'))
    plt.close(fig)
    print("Saved contour plot.")


def log_to_wandb(gap_matrix, tau_c_critic):
    """Log experiment results to W&B."""
    if not WANDB_AVAILABLE:
        print("[warn] Skipping W&B logging (wandb not available).")
        return None

    try:
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name="exp03-two-axis-decomposition",
            config={
                "experiment": "03_two_axis_decomposition",
                "d": D,
                "correlation": CORRELATION,
                "alpha_shape": ALPHA_SHAPE,
                "beta_param": BETA_PARAM,
                "K_grid": K_GRID,
                "n_grid": N_GRID,
                "N_REPS": N_REPS_3,
                "tau_c_critic": float(tau_c_critic),
            },
        )

        # log gap matrix as a table
        columns = ["K"] + [f"n={n}" for n in N_GRID]
        table = wandb.Table(columns=columns)
        for ki, K in enumerate(K_GRID):
            row = [K] + [float(gap_matrix[ki, ni]) for ni in range(len(N_GRID))]
            table.append(row)
        wandb.log({"gap_matrix": table})

        # log heatmap image
        heatmap_path = os.path.join(RESULTS_DIR, 'exp3_heatmap.png')
        if os.path.exists(heatmap_path):
            wandb.log({"heatmap": wandb.Image(heatmap_path)})

        # log marginal curves
        # gap vs K (fixed n=1000)
        n_idx = N_GRID.index(1000)
        for ki, K in enumerate(K_GRID):
            wandb.log({
                "marginal_K/gap": gap_matrix[ki, n_idx],
                "marginal_K/K": K,
            })

        # gap vs n (fixed K=64)
        k_idx = K_GRID.index(64)
        for ni, n in enumerate(N_GRID):
            wandb.log({
                "marginal_n/gap": gap_matrix[k_idx, ni],
                "marginal_n/n": n,
            })

        # log marginals and contour images
        for fname in ['exp3_marginals.png', 'exp3_tradeoff_contour.png']:
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(fpath):
                key = fname.replace('.png', '').replace('exp3_', '')
                wandb.log({key: wandb.Image(fpath)})

        wandb.finish()
        print("W&B logging complete.")
        return run
    except Exception as e:
        print(f"[warn] W&B logging failed: {e}")
        return None


def create_wandb_report(gap_matrix, tau_c_critic):
    """Create a W&B report for this experiment."""
    if not WANDB_AVAILABLE or not WR_AVAILABLE:
        print("[warn] Skipping W&B report (dependencies not available).")
        return

    try:
        report = wr.Report(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            title="Exp03: Two-Axis Decomposition (K x n)",
            description=(
                "Tests Theorem 3: error decomposes into judge approximation "
                "error (K) and finite-sample estimation error (n). "
                f"tau_c(h*) = {tau_c_critic:.6f}"
            ),
        )
        report.blocks = [
            wr.H1("Experiment 03 - Two-Axis Decomposition"),
            wr.P(
                "This experiment sweeps over a (K, n) grid to show that the "
                "gap between the true TVD critic value and the empirical "
                "maximum decomposes along two axes: number of judges K and "
                "sample size n."
            ),
            wr.H2("Heatmap"),
            wr.P("Mean gap across the (K, n) grid."),
            wr.H2("Marginals"),
            wr.P("Gap vs K (fixed n=1000) and gap vs n (fixed K=64)."),
            wr.H2("Tradeoff Contour"),
            wr.P("Contour plot showing K-n combinations for target gap."),
        ]
        report.save()
        print(f"W&B report saved: {report.url}")
    except Exception as e:
        print(f"[warn] W&B report creation failed: {e}")


def write_local_report(gap_matrix, tau_c_critic):
    """Write a local markdown report summarizing results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # fit power laws for marginals
    n_idx = N_GRID.index(1000)
    k_idx = K_GRID.index(64)
    alpha_K, a_K, b_K, R2_K = fit_power_law(K_GRID, gap_matrix[:, n_idx])
    alpha_n, a_n, b_n, R2_n = fit_power_law(N_GRID, gap_matrix[k_idx, :])

    report_lines = [
        f"# Experiment 03 - Two-Axis Decomposition Results",
        f"",
        f"**Generated:** {timestamp}",
        f"",
        f"## Parameters",
        f"- d = {D}, correlation = {CORRELATION}",
        f"- alpha_shape = {ALPHA_SHAPE}, beta = {BETA_PARAM}",
        f"- K_grid = {K_GRID}",
        f"- n_grid = {N_GRID}",
        f"- N_REPS = {N_REPS_3}",
        f"- tau_c(h*) = {tau_c_critic:.6f}",
        f"",
        f"## Gap Matrix (K x n)",
        f"",
    ]

    # table header
    header = "| K \\ n | " + " | ".join(str(n) for n in N_GRID) + " |"
    sep = "|-------|" + "|".join(["-------"] * len(N_GRID)) + "|"
    report_lines.append(header)
    report_lines.append(sep)
    for ki, K in enumerate(K_GRID):
        row = f"| {K:>5} | " + " | ".join(
            f"{gap_matrix[ki, ni]:.4f}" for ni in range(len(N_GRID))
        ) + " |"
        report_lines.append(row)

    report_lines += [
        f"",
        f"## Power-Law Fits",
        f"",
        f"### Gap vs K (fixed n = 1000)",
        f"- alpha = {alpha_K:.4f}" if not np.isnan(alpha_K) else "- fit failed",
        f"- R^2 = {R2_K:.4f}" if not np.isnan(R2_K) else "",
        f"",
        f"### Gap vs n (fixed K = 64)",
        f"- alpha = {alpha_n:.4f}" if not np.isnan(alpha_n) else "- fit failed",
        f"- R^2 = {R2_n:.4f}" if not np.isnan(R2_n) else "",
        f"",
        f"## Plots",
        f"- Heatmap: `results/exp3_heatmap.pdf`",
        f"- Marginals: `results/exp3_marginals.pdf`",
        f"- Contour: `results/exp3_tradeoff_contour.pdf`",
    ]

    report_path = os.path.join(SUMMARY_DIR, 'report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines) + '\n')
    print(f"Local report saved to {report_path}")

    # also save raw data as JSON
    data = {
        "timestamp": timestamp,
        "tau_c_critic": float(tau_c_critic),
        "K_grid": K_GRID,
        "n_grid": N_GRID,
        "gap_matrix": gap_matrix.tolist(),
        "power_law_K": {"alpha": float(alpha_K), "a": float(a_K),
                        "b": float(b_K), "R2": float(R2_K)},
        "power_law_n": {"alpha": float(alpha_n), "a": float(a_n),
                        "b": float(b_n), "R2": float(R2_n)},
    }
    json_path = os.path.join(SUMMARY_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Raw data saved to {json_path}")


def main():
    print("=" * 60)
    print("Experiment 03 — Two-Axis Decomposition (K x n)")
    print("=" * 60)
    print()

    # run simulation
    gap_matrix, tau_c_critic = run_experiment()

    # generate plots
    print("Generating plots...")
    plot_heatmap(gap_matrix)
    plot_marginals(gap_matrix)
    plot_contour(gap_matrix)

    # W&B logging
    print("\nLogging to W&B...")
    log_to_wandb(gap_matrix, tau_c_critic)
    create_wandb_report(gap_matrix, tau_c_critic)

    # local report
    print("\nWriting local report...")
    write_local_report(gap_matrix, tau_c_critic)

    print("\nDone.")


if __name__ == "__main__":
    main()
