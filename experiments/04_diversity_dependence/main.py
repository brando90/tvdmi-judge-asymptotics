"""Experiment 04 — Diversity and Dependence (rho Effect).

Tests Proposition 4.2: for equicorrelated scalar scores,
    Var(S_bar_K | c) = sigma^2 * ((1 - rho)/K + rho),
so RMSE(K, rho) = sigma * sqrt((1 - rho)/K + rho).

When rho > 0 there is an irreducible error floor of sqrt(sigma^2 * rho).

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
    compute_local_elasticity,
    COLORS,
    K_SWEEP,
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
TAU_C = 0.5          # true quality
SIGMA2 = 0.1         # variance
SIGMA = np.sqrt(SIGMA2)
N_REPS_4 = 1000      # repetitions per (K, rho) pair
RHO_VALUES = [0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8]

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
# Step 1: W&B initialisation
# ---------------------------------------------------------------------------
wandb_run = None
try:
    if WANDB_AVAILABLE:
        wandb_run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name="exp04-diversity-dependence",
            config={
                "tau_c": TAU_C,
                "sigma2": SIGMA2,
                "rho_values": RHO_VALUES,
                "K_sweep": K_SWEEP,
                "n_reps": N_REPS_4,
            },
        )
except Exception as e:
    print(f"WARNING: wandb.init failed: {e}")
    wandb_run = None

# ---------------------------------------------------------------------------
# Step 2: Simulation — sweep over rho and K
# ---------------------------------------------------------------------------
# Storage: results_dict[rho] = list of RMSE values, one per K in K_SWEEP
results_dict = {}

for rho in RHO_VALUES:
    rmse_vals = []
    sqrt_rho = np.sqrt(rho)
    sqrt_1mrho = np.sqrt(1 - rho)

    for K in K_SWEEP:
        sq_errors = np.empty(N_REPS_4)

        for rep in range(N_REPS_4):
            # Equicorrelation construction
            Z_common = np.random.randn() * SIGMA
            Z_indiv = np.random.randn(K) * SIGMA
            S = sqrt_rho * Z_common + sqrt_1mrho * Z_indiv + TAU_C
            S = np.clip(S, 0, 1)
            S_bar = np.mean(S)
            sq_errors[rep] = (S_bar - TAU_C) ** 2

        rmse = np.sqrt(np.mean(sq_errors))
        rmse_vals.append(rmse)

        print(f"rho={rho:.2f}, K={K:>4d}: RMSE={rmse:.6f}")

        # Log to W&B
        try:
            if wandb_run is not None:
                wandb.log({
                    f"rho_{rho}/RMSE_K{K}": rmse,
                    "rho": rho,
                    "K": K,
                    "RMSE": rmse,
                })
        except Exception:
            pass

    results_dict[rho] = rmse_vals

# ---------------------------------------------------------------------------
# Step 3: Theoretical predictions
# ---------------------------------------------------------------------------
def rmse_theory(K, rho):
    """RMSE_theory(K, rho) = sigma * sqrt((1-rho)/K + rho)."""
    return SIGMA * np.sqrt((1 - rho) / K + rho)

# ---------------------------------------------------------------------------
# Step 4: Fit power-law exponents
# ---------------------------------------------------------------------------
# For rho > 0, the RMSE curve flattens at a floor, so we only fit the first
# 6-8 K values (before the floor dominates). For rho = 0, use all K values.
fitted_rows = []
for rho in RHO_VALUES:
    rmse_vals = results_dict[rho]
    # Measure exact unclipped local log-log elasticity across K in [16, 256]
    # instead of free-offset power-law fitting (which fits the 1/K remainder toward 1.0).
    alpha_hat = compute_local_elasticity(K_SWEEP, rmse_vals, k_lo=16, k_hi=256)
    theory_vals = [rmse_theory(K, rho) for K in K_SWEEP]
    alpha_theory = compute_local_elasticity(K_SWEEP, theory_vals, k_lo=16, k_hi=256)
    fitted_rows.append({
        "rho": rho,
        "alpha_hat": alpha_hat,
        "alpha_theory": alpha_theory,
        "floor_theory": np.sqrt(SIGMA2 * rho),
    })
    print(f"rho={rho:.2f}: local_elasticity_16_256={alpha_hat:.4f} "
          f"(theory={alpha_theory:.4f}), floor_theory={np.sqrt(SIGMA2 * rho):.4f}")

# Save CSV
csv_path = os.path.join(RESULTS_DIR, "exp4_fitted_exponents.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["rho", "alpha_hat", "alpha_theory",
                                           "floor_theory"])
    writer.writeheader()
    writer.writerows(fitted_rows)
print(f"Saved fitted exponents to {csv_path}")

# Log fitted exponents table to W&B
try:
    if wandb_run is not None:
        table = wandb.Table(columns=["rho", "alpha_hat", "alpha_theory", "floor_theory"])
        for row in fitted_rows:
            table.add_data(row["rho"], row["alpha_hat"], row["alpha_theory"],
                           row["floor_theory"])
        wandb.log({"fitted_exponents": table})
except Exception as e:
    print(f"WARNING: failed to log W&B table: {e}")

# ---------------------------------------------------------------------------
# Step 5: Plot (a) — Log-log RMSE vs K
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

K_arr = np.array(K_SWEEP, dtype=float)

for idx, rho in enumerate(RHO_VALUES):
    color = COLORS[idx % len(COLORS)]
    rmse_vals = results_dict[rho]

    # Empirical curve (solid)
    ax.loglog(K_SWEEP, rmse_vals, 'o-', color=color, linewidth=2,
              markersize=5, label=rf"$\rho={rho}$ (empirical)")

    # Theoretical overlay (dashed)
    theory_vals = [rmse_theory(K, rho) for K in K_SWEEP]
    ax.loglog(K_SWEEP, theory_vals, '--', color=color, linewidth=1.5,
              alpha=0.6, label=rf"$\rho={rho}$ (theory)")

ax.set_xlabel(r"Number of judges $K$")
ax.set_ylabel(r"RMSE$(\bar{S}_K)$")
ax.set_title("Diversity and Dependence: RMSE vs K (Experiment 04)")
ax.legend(fontsize=7, ncol=2, loc="upper right")
ax.grid(True, which="both", ls=":", alpha=0.4)

# Save plots
pdf_path = os.path.join(RESULTS_DIR, "exp4_rmse_vs_K.pdf")
png_path = os.path.join(RESULTS_DIR, "exp4_rmse_vs_K.png")
fig.savefig(pdf_path)
fig.savefig(png_path)
print(f"Saved plot to {pdf_path}")
print(f"Saved plot to {png_path}")

# Also save into results_summary/plots/
summary_pdf = os.path.join(PLOTS_DIR, "exp4_rmse_vs_K.pdf")
summary_png = os.path.join(PLOTS_DIR, "exp4_rmse_vs_K.png")
fig.savefig(summary_pdf)
fig.savefig(summary_png)
print(f"Saved summary plot to {summary_png}")

plt.close(fig)

# Log plot to W&B
try:
    if wandb_run is not None:
        wandb.log({"rmse_vs_K_plot": wandb.Image(png_path)})
except Exception as e:
    print(f"WARNING: failed to log W&B image: {e}")

# ---------------------------------------------------------------------------
# Step 6: Plot (b) — Fitted exponent alpha_hat vs rho
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))

rhos_plot = [r["rho"] for r in fitted_rows]
alphas_plot = [r["alpha_hat"] for r in fitted_rows]

ax2.plot(rhos_plot, alphas_plot, 'o-', color=COLORS[0], linewidth=2,
         markersize=8, label=r"Fitted $\hat{\alpha}$")
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7,
            label=r"$\alpha = 0.5$ (independent)")
ax2.set_xlabel(r"Inter-judge correlation $\rho$")
ax2.set_ylabel(r"Fitted exponent $\hat{\alpha}$")
ax2.set_title(r"Scaling Exponent vs Dependence (Experiment 04)")
ax2.legend(fontsize=10)
ax2.grid(True, ls=":", alpha=0.4)
ax2.set_xlim(-0.02, 0.85)

# Save plots
pdf_path2 = os.path.join(RESULTS_DIR, "exp4_exponent_vs_rho.pdf")
png_path2 = os.path.join(RESULTS_DIR, "exp4_exponent_vs_rho.png")
fig2.savefig(pdf_path2)
fig2.savefig(png_path2)
print(f"Saved plot to {pdf_path2}")
print(f"Saved plot to {png_path2}")

# Also save into results_summary/plots/
summary_pdf2 = os.path.join(PLOTS_DIR, "exp4_exponent_vs_rho.pdf")
summary_png2 = os.path.join(PLOTS_DIR, "exp4_exponent_vs_rho.png")
fig2.savefig(summary_pdf2)
fig2.savefig(summary_png2)
print(f"Saved summary plot to {summary_png2}")

plt.close(fig2)

# Log plot to W&B
try:
    if wandb_run is not None:
        wandb.log({"exponent_vs_rho_plot": wandb.Image(png_path2)})
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
            title=f"Exp 04 — Diversity & Dependence — {now_str}",
            description="RMSE scaling with K under varying inter-judge "
                        "correlation rho. Tests the equicorrelation variance "
                        "formula and error floor.",
        )
        report.blocks = [
            wr.H1("Diversity and Dependence (rho Effect)"),
            wr.P("This report shows how inter-judge correlation rho affects "
                  "the scaling of RMSE with ensemble size K. Theory predicts "
                  "RMSE(K, rho) = sigma * sqrt((1-rho)/K + rho), implying an "
                  "irreducible floor when rho > 0."),
            wr.H2("RMSE vs K (Log-Log)"),
            wr.Image(
                url=wandb.run.get_url() if wandb.run else "",
                caption="RMSE vs K for various rho values. Solid = empirical, "
                        "dashed = theory.",
            ),
            wr.H2("Fitted Exponents vs rho"),
            wr.Image(
                url=wandb.run.get_url() if wandb.run else "",
                caption="Fitted power-law exponent alpha_hat vs rho.",
            ),
            wr.H2("Exponent Table"),
            wr.P("| rho | alpha_hat | alpha_theory | floor_theory |\n"
                  "|-----|-----------|--------------|--------------|\n" +
                  "\n".join(
                      f"| {r['rho']} | {r['alpha_hat']:.4f} | "
                      f"{r['alpha_theory']:.4f} | {r['floor_theory']:.4f} |"
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
    "# Experiment 04 — Diversity and Dependence (rho Effect)",
    "",
    f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## TL;DR",
    "",
    "RMSE of the ensemble mean follows the equicorrelation formula "
    "RMSE(K, rho) = sigma * sqrt((1-rho)/K + rho). When rho = 0, the "
    "scaling exponent is ~0.5 (standard 1/sqrt(K) rate). As rho increases, "
    "the RMSE curve flattens to an irreducible floor of sqrt(sigma^2 * rho), "
    "and the effective exponent degrades.",
    "",
    "## Configuration",
    "",
    "| Parameter | Value |",
    "|-----------|-------|",
    f"| tau_c (true quality) | {TAU_C} |",
    f"| sigma^2 (variance) | {SIGMA2} |",
    f"| sigma | {SIGMA:.4f} |",
    f"| rho values | {RHO_VALUES} |",
    f"| K sweep | {K_SWEEP} |",
    f"| N reps | {N_REPS_4} |",
    "",
    "## Fitted Exponents",
    "",
    "| rho | alpha_hat | alpha_theory | floor_theory |",
    "|-----|-----------|--------------|--------------|",
]
for r in fitted_rows:
    summary_lines.append(
        f"| {r['rho']} | {r['alpha_hat']:.4f} | "
        f"{r['alpha_theory']:.4f} | {r['floor_theory']:.4f} |"
    )
summary_lines += [
    "",
    "## Plots",
    "",
    "- RMSE vs K (log-log): `plots/exp4_rmse_vs_K.png`",
    f"  ![rmse_vs_K](plots/exp4_rmse_vs_K.png)",
    "",
    "- Exponent vs rho: `plots/exp4_exponent_vs_rho.png`",
    f"  ![exponent_vs_rho](plots/exp4_exponent_vs_rho.png)",
    "",
    "## Raw data",
    "",
    f"- Fitted exponents CSV: `../results/exp4_fitted_exponents.csv`",
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

print("\nExperiment 04 complete.")
