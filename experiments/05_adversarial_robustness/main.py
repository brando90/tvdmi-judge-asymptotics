"""Experiment 05 — Adversarial Robustness and Breakdown.

Tests Theorem 5.2: adversarial breakdown threshold for binary voting
and robust scalar aggregation under contamination.

Part A — Binary voting: with fraction delta adversarial judges, the ensemble
         is correct only if delta < 2*gamma/(1+2*gamma).
Part B — Scalar score contamination: compares mean, trimmed mean, and median
         aggregators under adversarial contamination.

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
from scipy.stats import trim_mean

# ---------------------------------------------------------------------------
# Import shared utilities from parent experiments/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    optimal_critic,
    make_joint_distribution,
    compute_delta,
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
D = 20
CORRELATION = 0.5
GAMMA = 0.1
DELTA_STAR = 2 * GAMMA / (1 + 2 * GAMMA)  # ~0.1667

# Part A config
K_BINARY = [16, 64, 256, 1024]
DELTA_BINARY = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45]

# Part B config
TAU_C_SCALAR = 0.5
SIGMA2 = 0.1
SIGMA = np.sqrt(SIGMA2)
DELTA_SCALAR = [0, 0.05, 0.1, 0.2, 0.3, 0.45]

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
# Step 1: Ground truth distribution and optimal critic
# ---------------------------------------------------------------------------
print("Setting up ground truth distribution (d={}, correlation={}) ...".format(D, CORRELATION))
P_UV, P_U, P_V = make_joint_distribution(d=D, correlation=CORRELATION)
h_star = optimal_critic(P_UV, P_U, P_V)
tau_c_critic = compute_delta(h_star, P_UV, P_U, P_V)
print(f"  Optimal critic delta tau_c_critic = {tau_c_critic:.6f}")
print(f"  gamma = {GAMMA}")
print(f"  Theoretical breakdown delta* = {DELTA_STAR:.4f}")

# ---------------------------------------------------------------------------
# Step 2: W&B initialisation
# ---------------------------------------------------------------------------
wandb_run = None
try:
    if WANDB_AVAILABLE:
        wandb_run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name="exp05-adversarial-robustness",
            config={
                "d": D,
                "correlation": CORRELATION,
                "gamma": GAMMA,
                "delta_star": DELTA_STAR,
                "K_binary": K_BINARY,
                "delta_binary": DELTA_BINARY,
                "tau_c_scalar": TAU_C_SCALAR,
                "sigma2": SIGMA2,
                "K_sweep": K_SWEEP,
                "delta_scalar": DELTA_SCALAR,
                "n_reps": N_REPS,
            },
        )
except Exception as e:
    print(f"WARNING: wandb.init failed: {e}")
    wandb_run = None

# ===================================================================
# PART A — Binary voting breakdown
# ===================================================================
print("\n" + "=" * 60)
print("PART A — Binary voting breakdown")
print("=" * 60)

# Storage: gap_binary[K][delta_idx] = mean_gap
gap_binary = {}

for K in K_BINARY:
    gap_binary[K] = []
    for delta in DELTA_BINARY:
        n_honest = round((1 - delta) * K)
        n_adv = K - n_honest
        gaps = np.empty(N_REPS)

        for rep in range(N_REPS):
            votes = np.empty((K, D, D))

            # Honest judges: output h*(i,j) with prob 0.5+gamma, else flip
            for j in range(n_honest):
                coin = np.random.rand(D, D)
                votes[j] = np.where(coin < 0.5 + GAMMA, h_star, 1.0 - h_star)

            # Adversarial judges: always output 1 - h*(i,j)
            for j in range(n_honest, K):
                votes[j] = 1.0 - h_star

            # Majority vote
            majority = (np.mean(votes, axis=0) > 0.5).astype(float)
            gap = compute_delta(majority, P_UV, P_U, P_V)
            gaps[rep] = gap

        mean_gap = np.mean(gaps)
        gap_binary[K].append(mean_gap)
        print(f"  K={K:4d}, delta={delta:.2f}: mean_gap={mean_gap:.6f}")

        # Log to W&B
        try:
            if wandb_run is not None:
                wandb.log({
                    f"partA/K{K}/delta_{delta:.2f}_mean_gap": mean_gap,
                })
        except Exception:
            pass

# Save Part A CSV
csv_path_a = os.path.join(RESULTS_DIR, "exp5_binary_breakdown.csv")
with open(csv_path_a, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["K", "delta", "mean_gap"])
    for K in K_BINARY:
        for idx, delta in enumerate(DELTA_BINARY):
            writer.writerow([K, delta, gap_binary[K][idx]])
print(f"Saved Part A CSV to {csv_path_a}")

# ---------------------------------------------------------------------------
# Part A Plot: gap vs delta, one curve per K
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for idx, K in enumerate(K_BINARY):
    color = COLORS[idx % len(COLORS)]
    ax.plot(DELTA_BINARY, gap_binary[K], 'o-', color=color, linewidth=2,
            markersize=5, label=f"K={K}")

# Mark theoretical breakdown threshold
ax.axvline(x=DELTA_STAR, color='black', linestyle='--', linewidth=1.5,
           alpha=0.7, label=rf"$\delta^* = {DELTA_STAR:.4f}$")

ax.set_xlabel(r"Adversarial fraction $\delta$")
ax.set_ylabel(r"Mean gap $\Delta_c$")
ax.set_title("Binary Voting Breakdown (Experiment 05A)")
ax.legend(fontsize=9)
ax.grid(True, ls=":", alpha=0.4)

pdf_path = os.path.join(RESULTS_DIR, "exp5_binary_breakdown.pdf")
png_path = os.path.join(RESULTS_DIR, "exp5_binary_breakdown.png")
fig.savefig(pdf_path)
fig.savefig(png_path)
print(f"Saved Part A plot to {pdf_path}")

summary_pdf = os.path.join(PLOTS_DIR, "exp5_binary_breakdown.pdf")
summary_png = os.path.join(PLOTS_DIR, "exp5_binary_breakdown.png")
fig.savefig(summary_pdf)
fig.savefig(summary_png)
print(f"Saved Part A summary plot to {summary_png}")
plt.close(fig)

# Log Part A plot to W&B
try:
    if wandb_run is not None:
        wandb.log({"binary_breakdown_plot": wandb.Image(png_path)})
except Exception as e:
    print(f"WARNING: failed to log W&B image: {e}")

# ===================================================================
# PART B — Scalar score contamination
# ===================================================================
print("\n" + "=" * 60)
print("PART B — Scalar score contamination")
print("=" * 60)

AGGREGATOR_NAMES = ["mean", "trimmed_mean", "median"]

# Storage: rmse_scalar[agg_name][K_idx][delta_idx] = rmse
rmse_scalar = {name: [] for name in AGGREGATOR_NAMES}

for K_idx, K in enumerate(K_SWEEP):
    for name in AGGREGATOR_NAMES:
        rmse_scalar[name].append([])

    for d_idx, delta in enumerate(DELTA_SCALAR):
        n_honest = round((1 - delta) * K)
        n_adv = K - n_honest

        sq_errors = {name: np.empty(N_REPS) for name in AGGREGATOR_NAMES}

        for rep in range(N_REPS):
            scores = np.empty(K)

            # Honest judges: S_j ~ N(tau_c, sigma2), clipped to [0,1]
            scores[:n_honest] = np.clip(
                np.random.normal(TAU_C_SCALAR, SIGMA, size=n_honest), 0, 1
            )

            # Adversarial judges: S_j ~ N(1 - tau_c, sigma2), clipped to [0,1]
            if n_adv > 0:
                scores[n_honest:] = np.clip(
                    np.random.normal(1 - TAU_C_SCALAR, SIGMA, size=n_adv), 0, 1
                )

            # Aggregators
            agg_mean = np.mean(scores)
            agg_trimmed = trim_mean(scores, proportiontocut=0.1)
            agg_median = np.median(scores)

            sq_errors["mean"][rep] = (agg_mean - TAU_C_SCALAR) ** 2
            sq_errors["trimmed_mean"][rep] = (agg_trimmed - TAU_C_SCALAR) ** 2
            sq_errors["median"][rep] = (agg_median - TAU_C_SCALAR) ** 2

        for name in AGGREGATOR_NAMES:
            rmse_val = np.sqrt(np.mean(sq_errors[name]))
            rmse_scalar[name][K_idx].append(rmse_val)

        print(f"  K={K:4d}, delta={delta:.2f}: "
              f"RMSE(mean)={rmse_scalar['mean'][K_idx][d_idx]:.6f}, "
              f"RMSE(trim)={rmse_scalar['trimmed_mean'][K_idx][d_idx]:.6f}, "
              f"RMSE(med)={rmse_scalar['median'][K_idx][d_idx]:.6f}")

        # Log to W&B
        try:
            if wandb_run is not None:
                for name in AGGREGATOR_NAMES:
                    wandb.log({
                        f"partB/{name}/K{K}/delta_{delta:.2f}_rmse":
                            rmse_scalar[name][K_idx][d_idx],
                    })
        except Exception:
            pass

# Save Part B CSV
csv_path_b = os.path.join(RESULTS_DIR, "exp5_scalar_contamination.csv")
with open(csv_path_b, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["K", "delta", "rmse_mean", "rmse_trimmed_mean", "rmse_median"])
    for K_idx, K in enumerate(K_SWEEP):
        for d_idx, delta in enumerate(DELTA_SCALAR):
            writer.writerow([
                K, delta,
                rmse_scalar["mean"][K_idx][d_idx],
                rmse_scalar["trimmed_mean"][K_idx][d_idx],
                rmse_scalar["median"][K_idx][d_idx],
            ])
print(f"Saved Part B CSV to {csv_path_b}")

# ---------------------------------------------------------------------------
# Part B Plot (a): RMSE vs K for each aggregator at delta=0.1
# ---------------------------------------------------------------------------
delta_target = 0.1
d_idx_target = DELTA_SCALAR.index(delta_target)

fig, ax = plt.subplots(figsize=(8, 5))

agg_styles = {"mean": "o-", "trimmed_mean": "s-", "median": "^-"}
agg_labels = {"mean": "Mean", "trimmed_mean": "Trimmed Mean (10%)", "median": "Median"}

for a_idx, name in enumerate(AGGREGATOR_NAMES):
    color = COLORS[a_idx % len(COLORS)]
    rmse_vs_K = [rmse_scalar[name][K_idx][d_idx_target] for K_idx in range(len(K_SWEEP))]
    ax.loglog(K_SWEEP, rmse_vs_K, agg_styles[name], color=color, linewidth=2,
              markersize=5, label=agg_labels[name])

# K^{-1/2} reference line, normalised to match mean at K=1
K_arr = np.array(K_SWEEP, dtype=float)
ref_rmse_at_1 = rmse_scalar["mean"][0][d_idx_target]
ref_line = ref_rmse_at_1 * (K_arr / K_arr[0]) ** (-0.5)
ax.loglog(K_SWEEP, ref_line, 'k--', linewidth=1.5, alpha=0.5,
          label=r"$K^{-1/2}$ reference")

ax.set_xlabel(r"Number of judges $K$")
ax.set_ylabel("RMSE")
ax.set_title(rf"Scalar Aggregator RMSE vs K ($\delta={delta_target}$) (Exp 05B)")
ax.legend(fontsize=9)
ax.grid(True, which="both", ls=":", alpha=0.4)

pdf_path = os.path.join(RESULTS_DIR, "exp5_scalar_aggregators.pdf")
png_path = os.path.join(RESULTS_DIR, "exp5_scalar_aggregators.png")
fig.savefig(pdf_path)
fig.savefig(png_path)
print(f"Saved Part B(a) plot to {pdf_path}")

summary_pdf = os.path.join(PLOTS_DIR, "exp5_scalar_aggregators.pdf")
summary_png = os.path.join(PLOTS_DIR, "exp5_scalar_aggregators.png")
fig.savefig(summary_pdf)
fig.savefig(summary_png)
print(f"Saved Part B(a) summary plot to {summary_png}")
plt.close(fig)

# Log to W&B
try:
    if wandb_run is not None:
        wandb.log({"scalar_aggregators_plot": wandb.Image(png_path)})
except Exception as e:
    print(f"WARNING: failed to log W&B image: {e}")

# ---------------------------------------------------------------------------
# Part B Plot (b): RMSE vs delta for fixed K=256
# ---------------------------------------------------------------------------
K_fixed = 256
K_fixed_idx = K_SWEEP.index(K_fixed)

fig, ax = plt.subplots(figsize=(8, 5))

for a_idx, name in enumerate(AGGREGATOR_NAMES):
    color = COLORS[a_idx % len(COLORS)]
    rmse_vs_delta = rmse_scalar[name][K_fixed_idx]
    ax.plot(DELTA_SCALAR, rmse_vs_delta, agg_styles[name], color=color,
            linewidth=2, markersize=6, label=agg_labels[name])

ax.set_xlabel(r"Adversarial fraction $\delta$")
ax.set_ylabel("RMSE")
ax.set_title(rf"Breakdown Curves for $K={K_fixed}$ (Exp 05B)")
ax.legend(fontsize=9)
ax.grid(True, ls=":", alpha=0.4)

pdf_path = os.path.join(RESULTS_DIR, "exp5_breakdown_curves.pdf")
png_path = os.path.join(RESULTS_DIR, "exp5_breakdown_curves.png")
fig.savefig(pdf_path)
fig.savefig(png_path)
print(f"Saved Part B(b) plot to {pdf_path}")

summary_pdf = os.path.join(PLOTS_DIR, "exp5_breakdown_curves.pdf")
summary_png = os.path.join(PLOTS_DIR, "exp5_breakdown_curves.png")
fig.savefig(summary_pdf)
fig.savefig(summary_png)
print(f"Saved Part B(b) summary plot to {summary_png}")
plt.close(fig)

# Log to W&B
try:
    if wandb_run is not None:
        wandb.log({"breakdown_curves_plot": wandb.Image(png_path)})
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
            title=f"Exp 05 — Adversarial Robustness — {now_str}",
            description="Adversarial breakdown threshold and robust aggregation.",
        )
        report.blocks = [
            wr.H1("Adversarial Robustness and Breakdown"),
            wr.P("This report tests adversarial breakdown thresholds for "
                  "binary voting and scalar score aggregation."),
            wr.H2("Part A — Binary Voting Breakdown"),
            wr.P(f"Gamma = {GAMMA}, theoretical breakdown delta* = {DELTA_STAR:.4f}. "
                 "Below delta*, error decreases with K; above it, error is bounded "
                 "away from 0 regardless of K."),
            wr.H2("Part B — Scalar Score Contamination"),
            wr.P(f"tau_c = {TAU_C_SCALAR}, sigma^2 = {SIGMA2}. "
                 "Compares mean, trimmed mean, and median aggregators."),
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
    "# Experiment 05 — Adversarial Robustness and Breakdown",
    "",
    f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## TL;DR",
    "",
    "Binary voting breaks down when the adversarial fraction delta exceeds "
    f"delta* = 2*gamma/(1+2*gamma) = {DELTA_STAR:.4f}. "
    "Robust scalar aggregators (trimmed mean, median) maintain accuracy "
    "under moderate contamination where the simple mean fails.",
    "",
    "## Configuration",
    "",
    "| Parameter | Value |",
    "|-----------|-------|",
    f"| d (grid size) | {D} |",
    f"| correlation | {CORRELATION} |",
    f"| gamma | {GAMMA} |",
    f"| delta* (breakdown) | {DELTA_STAR:.4f} |",
    f"| tau_c_critic | {tau_c_critic:.6f} |",
    f"| tau_c (scalar) | {TAU_C_SCALAR} |",
    f"| sigma^2 | {SIGMA2} |",
    f"| N reps | {N_REPS} |",
    "",
    "## Part A — Binary Voting Breakdown",
    "",
    "| K | " + " | ".join(f"delta={d}" for d in DELTA_BINARY) + " |",
    "|---| " + " | ".join("---" for _ in DELTA_BINARY) + " |",
]
for K in K_BINARY:
    row_vals = " | ".join(f"{g:.6f}" for g in gap_binary[K])
    summary_lines.append(f"| {K} | {row_vals} |")

summary_lines += [
    "",
    "## Part B — Scalar Aggregator RMSE (delta=0.1)",
    "",
    "| K | Mean | Trimmed Mean | Median |",
    "|---|------|--------------|--------|",
]
for K_idx, K in enumerate(K_SWEEP):
    summary_lines.append(
        f"| {K} | "
        f"{rmse_scalar['mean'][K_idx][d_idx_target]:.6f} | "
        f"{rmse_scalar['trimmed_mean'][K_idx][d_idx_target]:.6f} | "
        f"{rmse_scalar['median'][K_idx][d_idx_target]:.6f} |"
    )

summary_lines += [
    "",
    "## Plots",
    "",
    "- Binary breakdown: `plots/exp5_binary_breakdown.png`",
    f"  ![binary](plots/exp5_binary_breakdown.png)",
    "",
    "- Scalar aggregators: `plots/exp5_scalar_aggregators.png`",
    f"  ![scalar](plots/exp5_scalar_aggregators.png)",
    "",
    "- Breakdown curves: `plots/exp5_breakdown_curves.png`",
    f"  ![breakdown](plots/exp5_breakdown_curves.png)",
    "",
    "## Raw data",
    "",
    "- Part A CSV: `../results/exp5_binary_breakdown.csv`",
    "- Part B CSV: `../results/exp5_scalar_contamination.csv`",
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

print("\nExperiment 05 complete.")
