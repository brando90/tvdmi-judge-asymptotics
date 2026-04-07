"""Experiment 02 — Consensus (Majority-Vote) Scaling Law.

Tests Theorem 2 (consensus convergence): each judge votes b_j(z) in {0,1}
with Pr(b_j = h*(z) | z) >= 1/2 + gamma. Majority vote converges
exponentially:

    tau(c) - Delta_c(b_K) <= 2 * exp(-2*K*gamma^2)

We sweep over gamma in {0.01, 0.05, 0.1, 0.2, 0.4} and K in K_SWEEP,
measuring the gap tau(c) - Delta_c(b_K), and compare against the
theoretical Hoeffding-type bound.
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------------------------
# Import shared utilities from parent experiments/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    compute_true_tvd,
    optimal_critic,
    make_joint_distribution,
    compute_delta,
    fit_exponential_decay,
    COLORS,
    K_SWEEP,
    N_REPS,
)

# ---------------------------------------------------------------------------
# W&B setup
# ---------------------------------------------------------------------------
WANDB_ENTITY = "brando-su"
WANDB_PROJECT = "tvdmi-judge-asymptotics"

try:
    os.environ["WANDB_API_KEY"] = open(
        os.path.expanduser("~/keys/brandos_wandb_key.txt")
    ).read().strip()
except FileNotFoundError:
    print("[WARN] W&B API key file not found; W&B logging will be skipped.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARN] wandb not installed; skipping W&B logging.")

try:
    import wandb_workspaces.reports.v2 as wr
    WR_AVAILABLE = True
except ImportError:
    WR_AVAILABLE = False
    print("[WARN] wandb_workspaces not installed; skipping W&B report.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
SUMMARY_DIR = os.path.join(SCRIPT_DIR, "results_summary")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
D = 20
CORRELATION = 0.5
GAMMAS = [0.01, 0.05, 0.1, 0.2, 0.4]


def run_experiment():
    """Run the full consensus scaling experiment."""
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Ground truth
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Experiment 02 — Consensus (Majority-Vote) Scaling Law")
    print("=" * 60)

    print("\n[1/4] Building ground truth distribution (d=%d, rho=%.1f)..." % (D, CORRELATION))
    P_UV, P_U, P_V = make_joint_distribution(d=D, correlation=CORRELATION)
    h_star = optimal_critic(P_UV, P_U, P_V)
    tau_c_critic = compute_delta(h_star, P_UV, P_U, P_V)
    true_tvd = compute_true_tvd(P_UV, P_U, P_V)

    print("  True TVD          = %.6f" % true_tvd)
    print("  tau_c(h*)         = %.6f" % tau_c_critic)
    print("  h* nonzero cells  = %d / %d" % (int(h_star.sum()), D * D))

    # ------------------------------------------------------------------
    # 2. Simulation: sweep gamma x K
    # ------------------------------------------------------------------
    print("\n[2/4] Running simulations (gammas=%s, K_SWEEP up to %d, N_REPS=%d)..."
          % (GAMMAS, max(K_SWEEP), N_REPS))

    # results[gamma_idx][k_idx] = list of N_REPS gap values
    all_mean_gaps = {}   # gamma -> list of mean_gap per K
    all_std_gaps = {}    # gamma -> list of std_gap per K

    for gi, gamma in enumerate(GAMMAS):
        print("  gamma = %.2f ..." % gamma, end=" ", flush=True)
        mean_gaps = []
        std_gaps = []

        for ki, K in enumerate(K_SWEEP):
            gaps = np.empty(N_REPS)
            for rep in range(N_REPS):
                # Each of K judges: correct with prob 0.5+gamma, else flip
                # votes[k, i, j] = h_star[i,j] w.p. 0.5+gamma, else 1-h_star[i,j]
                rand_vals = np.random.rand(K, D, D)
                correct = rand_vals < (0.5 + gamma)
                # If correct, vote = h_star; else vote = 1 - h_star
                votes = correct * h_star[np.newaxis, :, :] + (~correct) * (1.0 - h_star[np.newaxis, :, :])
                # Majority vote: sum votes along axis 0, threshold at K/2
                vote_sum = votes.sum(axis=0)
                b_K = (vote_sum > K / 2).astype(float)
                # Compute delta for majority vote critic
                delta_majority = compute_delta(b_K, P_UV, P_U, P_V)
                gaps[rep] = tau_c_critic - delta_majority

            mean_gaps.append(gaps.mean())
            std_gaps.append(gaps.std())

        all_mean_gaps[gamma] = np.array(mean_gaps)
        all_std_gaps[gamma] = np.array(std_gaps)
        print("done.")

    K_arr = np.array(K_SWEEP, dtype=float)

    # ------------------------------------------------------------------
    # 3. Plotting
    # ------------------------------------------------------------------
    print("\n[3/4] Generating plots...")

    fig, ax = plt.subplots(figsize=(10, 6))

    for gi, gamma in enumerate(GAMMAS):
        color = COLORS[gi % len(COLORS)]
        mean_gap = all_mean_gaps[gamma]

        # Empirical curve
        # Clip to positive for log scale
        mean_gap_pos = np.clip(mean_gap, 1e-15, None)
        ax.semilogy(K_arr, mean_gap_pos, 'o-', color=color, linewidth=2,
                    markersize=5, label=r"$\gamma = %.2f$ (sim)" % gamma)

        # Theoretical bound: 2*exp(-2*K*gamma^2)
        theoretical = 2.0 * np.exp(-2.0 * K_arr * gamma ** 2)
        ax.semilogy(K_arr, theoretical, '--', color=color, linewidth=1.5,
                    alpha=0.6, label=r"$2\exp(-2K\gamma^2)$, $\gamma=%.2f$" % gamma)

    ax.set_xlabel(r"Number of judges $K$")
    ax.set_ylabel(r"Gap $\tau(c) - \Delta_c(b_K)$")
    ax.set_title("Consensus Scaling: Majority-Vote Gap vs K (semi-log)")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(left=0)

    fig.tight_layout()
    pdf_path = os.path.join(RESULTS_DIR, "exp2_semilog.pdf")
    png_path = os.path.join(RESULTS_DIR, "exp2_semilog.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print("  Saved: %s" % pdf_path)
    print("  Saved: %s" % png_path)

    # ------------------------------------------------------------------
    # 4. Fit exponential decay and write CSV
    # ------------------------------------------------------------------
    print("\n[4/4] Fitting exponential decay and writing CSV...")

    csv_path = os.path.join(RESULTS_DIR, "exp2_decay_rates.csv")
    fit_results = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma_true", "gamma_fitted", "R²"])

        for gamma in GAMMAS:
            mean_gap = all_mean_gaps[gamma]
            b, a, c, R2 = fit_exponential_decay(K_SWEEP, mean_gap)

            # From theory: decay rate b should equal 2*gamma^2
            # So gamma_fitted = sqrt(b / 2)
            if np.isnan(b) or b < 0:
                gamma_fitted = np.nan
            else:
                gamma_fitted = np.sqrt(b / 2.0)

            writer.writerow(["%.4f" % gamma, "%.6f" % gamma_fitted, "%.6f" % R2])
            fit_results.append((gamma, gamma_fitted, R2))
            print("  gamma_true=%.4f  gamma_fitted=%.6f  R²=%.6f" % (gamma, gamma_fitted, R2))

    print("  Saved: %s" % csv_path)

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            print("\n[W&B] Initializing run...")
            wandb_run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name="exp02-consensus-scaling",
                config={
                    "experiment": "02_consensus_scaling",
                    "d": D,
                    "correlation": CORRELATION,
                    "gammas": GAMMAS,
                    "K_sweep": K_SWEEP,
                    "N_reps": N_REPS,
                    "true_tvd": float(true_tvd),
                    "tau_c_critic": float(tau_c_critic),
                },
            )

            # Log the semi-log plot
            wandb.log({"exp2_semilog": wandb.Image(png_path)})

            # Log per-gamma curves as tables
            for gi, gamma in enumerate(GAMMAS):
                table = wandb.Table(
                    columns=["K", "mean_gap", "std_gap", "theoretical_bound"],
                )
                for ki, K in enumerate(K_SWEEP):
                    table.add_data(
                        K,
                        float(all_mean_gaps[gamma][ki]),
                        float(all_std_gaps[gamma][ki]),
                        float(2.0 * np.exp(-2.0 * K * gamma ** 2)),
                    )
                wandb.log({"gap_vs_K_gamma_%.2f" % gamma: table})

            # Log fit results
            fit_table = wandb.Table(columns=["gamma_true", "gamma_fitted", "R²"])
            for gamma, gamma_fitted, R2 in fit_results:
                fit_table.add_data(gamma, gamma_fitted, R2)
            wandb.log({"decay_rate_fits": fit_table})

            # Log summary metrics
            for gamma, gamma_fitted, R2 in fit_results:
                wandb.run.summary["gamma_%.2f_fitted" % gamma] = gamma_fitted
                wandb.run.summary["gamma_%.2f_R2" % gamma] = R2

            print("[W&B] Logged all metrics and artifacts.")
        except Exception as e:
            print("[WARN] W&B logging failed: %s" % e)

    # ------------------------------------------------------------------
    # W&B Report
    # ------------------------------------------------------------------
    if WANDB_AVAILABLE and WR_AVAILABLE and wandb_run is not None:
        try:
            print("\n[W&B] Creating report...")
            report = wr.Report(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                title="Exp 02 — Consensus (Majority-Vote) Scaling Law",
                description=(
                    "Tests Theorem 2: majority-vote gap decays exponentially "
                    "as tau(c) - Delta_c(b_K) <= 2*exp(-2*K*gamma^2). "
                    "Generated %s." % datetime.now().strftime("%Y-%m-%d %H:%M")
                ),
            )
            report.blocks = [
                wr.H1("Experiment 02: Consensus Scaling"),
                wr.P(
                    "Each of K judges independently votes b_j(u,v) = h*(u,v) "
                    "with probability 1/2 + gamma, else flips. The majority vote "
                    "b_K converges exponentially to the optimal critic."
                ),
                wr.H2("Semi-log Plot: Gap vs K"),
                wr.P(
                    "Linear K on x-axis, log gap on y-axis. Dashed lines show "
                    "the theoretical Hoeffding bound 2*exp(-2*K*gamma^2)."
                ),
                wr.Image(png_path, caption="Consensus scaling semi-log plot"),
                wr.H2("Fitted Decay Rates"),
                wr.P(
                    "Exponential decay fit: gap ~ a*exp(-b*K) + c. "
                    "gamma_fitted = sqrt(b/2) should match gamma_true."
                ),
            ]
            report.save()
            print("[W&B] Report saved: %s" % report.url)
        except Exception as e:
            print("[WARN] W&B report creation failed: %s" % e)

    # Finish W&B run
    if WANDB_AVAILABLE and wandb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Local markdown report
    # ------------------------------------------------------------------
    print("\n[Local] Writing summary report...")
    report_path = os.path.join(SUMMARY_DIR, "exp02_report.md")
    with open(report_path, "w") as f:
        f.write("# Experiment 02 — Consensus (Majority-Vote) Scaling Law\n\n")
        f.write("**Date:** %s\n\n" % datetime.now().strftime("%Y-%m-%d %H:%M"))
        f.write("## Setup\n\n")
        f.write("- Grid size d = %d\n" % D)
        f.write("- Correlation = %.1f\n" % CORRELATION)
        f.write("- True TVD = %.6f\n" % true_tvd)
        f.write("- tau_c(h*) = %.6f\n" % tau_c_critic)
        f.write("- K sweep: %s\n" % K_SWEEP)
        f.write("- N_REPS = %d\n" % N_REPS)
        f.write("- Gammas: %s\n\n" % GAMMAS)

        f.write("## Theory\n\n")
        f.write("Majority-vote gap bound: tau(c) - Delta_c(b_K) <= 2*exp(-2*K*gamma^2)\n\n")

        f.write("## Results: Fitted Decay Rates\n\n")
        f.write("| gamma_true | gamma_fitted | R^2 |\n")
        f.write("|------------|-------------|------|\n")
        for gamma, gamma_fitted, R2 in fit_results:
            f.write("| %.4f | %.6f | %.6f |\n" % (gamma, gamma_fitted, R2))
        f.write("\n")

        f.write("## Plots\n\n")
        f.write("- Semi-log plot: `results/exp2_semilog.pdf` / `.png`\n")
        f.write("- Decay rates CSV: `results/exp2_decay_rates.csv`\n")

    print("  Saved: %s" % report_path)
    print("\n" + "=" * 60)
    print("Experiment 02 complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
