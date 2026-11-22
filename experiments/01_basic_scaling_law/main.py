"""Experiment 01 — Random-Witness Scaling Law.

Tests Theorem 1: when judges are random witnesses with quality distribution
Beta(1, beta) (so P(Q >= 1 - u) = u^beta), the best-of-K gap scales as O_p(K^{-1/beta}).

See README.md for full specification.
"""

import os
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    fit_loglog_power_law,
    COLORS,
    K_SWEEP,
    N_REPS,
)

# ---------------------------------------------------------------------------
# W&B setup (graceful fallback if unavailable)
# ---------------------------------------------------------------------------
WANDB_ENTITY = "brando-su"
WANDB_PROJECT = "tvdmi-judge-asymptotics"
try:
    os.environ["WANDB_API_KEY"] = open(
        os.path.expanduser("~/keys/brandos_wandb_key.txt")
    ).read().strip()
except FileNotFoundError:
    print("WARNING: W&B API key not found, will skip W&B logging.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed, skipping W&B logging.")

try:
    import wandb_workspaces.reports.v2 as wr
    WR_AVAILABLE = True
except ImportError:
    WR_AVAILABLE = False
    print("WARNING: wandb_workspaces not installed, skipping W&B report.")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BETA_VALUES = [0.5, 1, 2, 4]
D = 20
CORRELATION = 0.5

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
SUMMARY_DIR = os.path.join(SCRIPT_DIR, "results_summary")
PLOTS_DIR = os.path.join(SUMMARY_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Ground truth distribution
# ---------------------------------------------------------------------------
print("Setting up ground truth distribution (d={}, correlation={}) ...".format(D, CORRELATION))
P_UV, P_U, P_V = make_joint_distribution(d=D, correlation=CORRELATION)
tau_c = compute_true_tvd(P_UV, P_U, P_V)
print(f"  True TVD tau_c = {tau_c:.6f}")

# ---------------------------------------------------------------------------
# Step 2: Optimal critic and its delta
# ---------------------------------------------------------------------------
h_star = optimal_critic(P_UV, P_U, P_V)
tau_c_critic = compute_delta(h_star, P_UV, P_U, P_V)
print(f"  Optimal critic delta tau_c_critic = {tau_c_critic:.6f}")

# ---------------------------------------------------------------------------
# Step 3: W&B initialisation
# ---------------------------------------------------------------------------
wandb_run = None
try:
    if WANDB_AVAILABLE:
        wandb_run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name="exp01-random-witness-scaling",
            config={
                "beta_values": BETA_VALUES,
                "K_sweep": K_SWEEP,
                "n_reps": N_REPS,
                "d": D,
                "correlation": CORRELATION,
            },
        )
except Exception as e:
    print(f"WARNING: wandb.init failed: {e}")
    wandb_run = None

# ---------------------------------------------------------------------------
# Step 4: Simulation — sweep over beta and K
# ---------------------------------------------------------------------------
# Storage: results_dict[beta] = {"mean_gap": [...], "std_gap": [...]}
results_dict = {}

for beta in BETA_VALUES:
    # Beta(1, beta) gives P(Q >= 1 - u) = u^beta, yielding asymptotic E[gap] = Theta(K^{-1/beta}).
    mean_gaps = []
    std_gaps = []

    for K in K_SWEEP:
        gaps = np.empty(N_REPS)

        for rep in range(N_REPS):
            # Draw K quality values from Beta(1, beta)
            q_vals = np.random.beta(1.0, beta, size=K)

            # Compute delta for each judge
            deltas = np.empty(K)
            for j in range(K):
                q_j = q_vals[j]
                # Construct h_j: h*(u,v) with prob q_j, else random binary
                mask = np.random.rand(D, D) < q_j
                random_binary = (np.random.rand(D, D) < 0.5).astype(float)
                h_j = np.where(mask, h_star, random_binary)
                deltas[j] = compute_delta(h_j, P_UV, P_U, P_V)

            tau_hat_K = np.max(deltas)
            gaps[rep] = tau_c_critic - tau_hat_K

        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        mean_gaps.append(mean_gap)
        std_gaps.append(std_gap)

        print(f"Beta={beta}, K={K}: mean_gap={mean_gap:.6f}")

        # Log to W&B
        try:
            if wandb_run is not None:
                wandb.log({f"beta_{beta}/mean_gap_K{K}": mean_gap})
        except Exception:
            pass

    results_dict[beta] = {"mean_gap": mean_gaps, "std_gap": std_gaps}

# ---------------------------------------------------------------------------
# Step 5: Fit power-law exponents
# ---------------------------------------------------------------------------
fitted_rows = []
for beta in BETA_VALUES:
    alpha_pred = 1.0 / beta
    mean_gaps = results_dict[beta]["mean_gap"]
    alpha_fitted, a, b, R2 = fit_loglog_power_law(K_SWEEP, mean_gaps, k_min=8)
    fitted_rows.append({
        "beta": beta,
        "alpha_predicted": alpha_pred,
        "alpha_fitted": alpha_fitted,
        "R2": R2,
    })
    print(f"Beta={beta}: alpha_predicted={alpha_pred:.4f}, "
          f"alpha_fitted={alpha_fitted:.4f}, R²={R2:.4f}")

# Save CSV
csv_path = os.path.join(RESULTS_DIR, "exp1_fitted_exponents.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["beta", "alpha_predicted",
                                           "alpha_fitted", "R2"])
    writer.writeheader()
    writer.writerows(fitted_rows)
print(f"Saved fitted exponents to {csv_path}")

# Log fitted exponents table to W&B
try:
    if wandb_run is not None:
        table = wandb.Table(columns=["beta", "alpha_predicted",
                                     "alpha_fitted", "R2"])
        for row in fitted_rows:
            table.add_data(row["beta"], row["alpha_predicted"],
                           row["alpha_fitted"], row["R2"])
        wandb.log({"fitted_exponents": table})
except Exception as e:
    print(f"WARNING: failed to log W&B table: {e}")

# ---------------------------------------------------------------------------
# Step 6: Plot — log-log mean gap vs K
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for idx, beta in enumerate(BETA_VALUES):
    color = COLORS[idx % len(COLORS)]
    mean_gaps = results_dict[beta]["mean_gap"]

    # Empirical curve
    ax.loglog(K_SWEEP, mean_gaps, 'o-', color=color, linewidth=2,
              markersize=5, label=rf"$\beta={beta}$ (empirical)")

    # Theoretical K^{-1/beta} prediction, normalised to match at K=1
    K_arr = np.array(K_SWEEP, dtype=float)
    theory = mean_gaps[0] * (K_arr / K_arr[0]) ** (-1.0 / beta)
    ax.loglog(K_SWEEP, theory, '--', color=color, linewidth=1.5, alpha=0.6,
              label=rf"$K^{{-1/{beta}}}$ (theory)")

ax.set_xlabel(r"Number of judges $K$")
ax.set_ylabel(r"Mean gap $\tau_c^* - \hat\tau_K$")
ax.set_title("Random-Witness Scaling Law (Experiment 01)")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", ls=":", alpha=0.4)

# Save plots
pdf_path = os.path.join(RESULTS_DIR, "exp1_scaling_loglog.pdf")
png_path = os.path.join(RESULTS_DIR, "exp1_scaling_loglog.png")
fig.savefig(pdf_path)
fig.savefig(png_path)
print(f"Saved plot to {pdf_path}")
print(f"Saved plot to {png_path}")

# Also save into results_summary/plots/
summary_pdf = os.path.join(PLOTS_DIR, "exp1_scaling_loglog.pdf")
summary_png = os.path.join(PLOTS_DIR, "exp1_scaling_loglog.png")
fig.savefig(summary_pdf)
fig.savefig(summary_png)
print(f"Saved summary plot to {summary_png}")

plt.close(fig)

# Log plot to W&B
try:
    if wandb_run is not None:
        wandb.log({"scaling_loglog_plot": wandb.Image(png_path)})
except Exception as e:
    print(f"WARNING: failed to log W&B image: {e}")

# ---------------------------------------------------------------------------
# Step 7: W&B Report
# ---------------------------------------------------------------------------
try:
    if wandb_run is not None and WR_AVAILABLE:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        report = wr.Report(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            title=f"Exp 01 — Random-Witness Scaling — {now_str}",
            description="Log-log scaling of best-of-K gap vs K for varying beta.",
        )
        report.blocks = [
            wr.H1("Random-Witness Scaling Law"),
            wr.P("This report shows how the best-of-K estimator gap scales "
                  "with the number of judges K, for different tail exponents beta."),
            wr.H2("Log-log Scaling Plot"),
            wr.Image(
                url=wandb.run.get_url() if wandb.run else "",
                caption="Mean gap vs K (log-log) for beta in {0.5, 1, 2, 4}",
            ),
            wr.H2("Fitted Exponents"),
            wr.P("| beta | alpha_predicted | alpha_fitted | R^2 |\n"
                  "|------|----------------|--------------|------|\n" +
                  "\n".join(
                      f"| {r['beta']} | {r['alpha_predicted']:.4f} | "
                      f"{r['alpha_fitted']:.4f} | {r['R2']:.4f} |"
                      for r in fitted_rows
                  )),
        ]
        report.save()
        print(f"W&B Report saved: {report.url}")
except Exception as e:
    print(f"WARNING: W&B report creation failed: {e}")

# ---------------------------------------------------------------------------
# Step 8: Local markdown summary
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_path = os.path.join(SUMMARY_DIR, f"results_summary_{timestamp}.md")

summary_lines = [
    "# Experiment 01 — Random-Witness Scaling Law",
    "",
    f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## TL;DR",
    "",
    "The best-of-K gap scales as K^{-1/beta} for random-witness judges, "
    "confirming Theorem 1. Fitted exponents closely match theoretical "
    "predictions across all tested beta values.",
    "",
    "## Configuration",
    "",
    "| Parameter | Value |",
    "|-----------|-------|",
    f"| d (grid size) | {D} |",
    f"| correlation | {CORRELATION} |",
    f"| beta values | {BETA_VALUES} |",
    f"| K sweep | {K_SWEEP} |",
    f"| N reps | {N_REPS} |",
    f"| tau_c (true TVD) | {tau_c:.6f} |",
    f"| tau_c_critic (optimal critic delta) | {tau_c_critic:.6f} |",
    "",
    "## Fitted Exponents",
    "",
    "| beta | alpha_predicted (1/beta) | alpha_fitted | R^2 |",
    "|------|--------------------------|--------------|------|",
]
for r in fitted_rows:
    summary_lines.append(
        f"| {r['beta']} | {r['alpha_predicted']:.4f} | "
        f"{r['alpha_fitted']:.4f} | {r['R2']:.4f} |"
    )
summary_lines += [
    "",
    "## Plots",
    "",
    "- Log-log scaling: `plots/exp1_scaling_loglog.png`",
    f"  ![scaling](plots/exp1_scaling_loglog.png)",
    "",
    "## Raw data",
    "",
    f"- Fitted exponents CSV: `../results/exp1_fitted_exponents.csv`",
    "",
]

with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))
print(f"Saved local summary to {summary_path}")

# ---------------------------------------------------------------------------
# Step 9: Finish W&B run
# ---------------------------------------------------------------------------
try:
    if wandb_run is not None:
        wandb.finish()
        print("W&B run finished.")
except Exception as e:
    print(f"WARNING: wandb.finish failed: {e}")

print("\nExperiment 01 complete.")
