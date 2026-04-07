"""Experiment 06 — End-to-End Scaling Law with LLM Judges (Synthetic Proxy).

Since we do not have real LLM API access, this experiment creates a SYNTHETIC
PROXY that mimics what real LLM judges would produce.  Each synthetic judge
has its own bias and noise level, plus a shared noise component that models
inter-judge correlation (analogous to all LLMs being biased by similar
training data).

We sweep over the number of judges K, aggregate scores via mean / median /
max, and measure Spearman correlation with a ground-truth quality vector.
Both power-law and exponential-decay models are fitted to the error curves.

See README.md for full specification.
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Import shared utilities from parent experiments/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    fit_power_law,
    fit_exponential_decay,
    COLORS,
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
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ITEMS = 100                # number of items to rate
MAX_JUDGES = 50              # total pool of synthetic LLM judges
RHO = 0.3                   # inter-judge correlation (shared noise fraction)
N_SUBSETS = 200              # random subsets per K value
K_VALUES = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
AGGREGATORS = ["mean", "median", "max"]

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
SUMMARY_DIR = os.path.join(SCRIPT_DIR, "results_summary")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


def generate_synthetic_judges(gold_scores, n_judges, rho):
    """Generate a score matrix for synthetic LLM judges.

    Parameters
    ----------
    gold_scores : np.ndarray, shape (n_items,)
        Ground-truth quality scores in [1, 10].
    n_judges : int
        Number of judges to simulate.
    rho : float
        Inter-judge correlation (fraction of shared noise).

    Returns
    -------
    scores : np.ndarray, shape (n_judges, n_items)
        Clipped to [1, 10].
    biases : np.ndarray, shape (n_judges,)
    sigmas : np.ndarray, shape (n_judges,)
    """
    n_items = len(gold_scores)

    # Per-judge parameters
    biases = np.random.normal(0, 0.5, size=n_judges)
    sigmas = np.random.uniform(0.5, 2.0, size=n_judges)

    # Shared noise component (same across all judges for a given item)
    Z_common = np.random.randn(n_items)

    scores = np.empty((n_judges, n_items))
    for j in range(n_judges):
        Z_indep = np.random.randn(n_items)
        noise = sigmas[j] * (np.sqrt(rho) * Z_common + np.sqrt(1 - rho) * Z_indep)
        scores[j] = gold_scores + biases[j] + noise

    # Clip to valid range
    scores = np.clip(scores, 1, 10)
    return scores, biases, sigmas


def aggregate_scores(score_matrix, method):
    """Aggregate a (K, n_items) score matrix along axis 0.

    Parameters
    ----------
    score_matrix : np.ndarray, shape (K, n_items)
    method : str
        One of "mean", "median", "max".

    Returns
    -------
    aggregated : np.ndarray, shape (n_items,)
    """
    if method == "mean":
        return score_matrix.mean(axis=0)
    elif method == "median":
        return np.median(score_matrix, axis=0)
    elif method == "max":
        return score_matrix.max(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def run_experiment():
    """Run the full Experiment 06."""
    print("=" * 65)
    print("Experiment 06 - End-to-End Scaling Law with LLM Judges")
    print("              (SYNTHETIC PROXY - not real LLM data)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 1: Generate ground-truth quality scores
    # ------------------------------------------------------------------
    print("\n[1/7] Generating ground-truth quality scores (n_items=%d) ..." % N_ITEMS)
    gold_scores = np.random.uniform(1, 10, size=N_ITEMS)
    print("  gold_scores: min=%.2f, max=%.2f, mean=%.2f"
          % (gold_scores.min(), gold_scores.max(), gold_scores.mean()))

    # ------------------------------------------------------------------
    # Step 2: Simulate synthetic LLM judges
    # ------------------------------------------------------------------
    print("\n[2/7] Simulating %d synthetic LLM judges (rho=%.2f) ..."
          % (MAX_JUDGES, RHO))
    all_scores, biases, sigmas = generate_synthetic_judges(gold_scores, MAX_JUDGES, RHO)
    print("  Judge biases:  min=%.3f, max=%.3f, mean=%.3f"
          % (biases.min(), biases.max(), biases.mean()))
    print("  Judge sigmas:  min=%.3f, max=%.3f, mean=%.3f"
          % (sigmas.min(), sigmas.max(), sigmas.mean()))

    # ------------------------------------------------------------------
    # Step 3: Estimate empirical inter-judge correlation
    # ------------------------------------------------------------------
    print("\n[3/7] Estimating empirical inter-judge correlation ...")
    pairwise_corrs = []
    for i in range(MAX_JUDGES):
        for j in range(i + 1, MAX_JUDGES):
            r, _ = spearmanr(all_scores[i], all_scores[j])
            pairwise_corrs.append(r)
    empirical_rho = np.mean(pairwise_corrs)
    print("  Average pairwise Spearman correlation (empirical rho): %.4f" % empirical_rho)
    print("  (Set rho = %.2f)" % RHO)

    # ------------------------------------------------------------------
    # Step 4: Sweep over K — subsample judges and compute correlations
    # ------------------------------------------------------------------
    print("\n[4/7] Sweeping K in %s (N_SUBSETS=%d per K) ..." % (K_VALUES, N_SUBSETS))

    # results[agg][k_idx] = {"mean_corr": ..., "std_corr": ...}
    results = {agg: {"mean_corr": [], "std_corr": []} for agg in AGGREGATORS}

    for ki, K in enumerate(K_VALUES):
        print("  K=%d ..." % K, end=" ", flush=True)

        for agg in AGGREGATORS:
            corrs = []
            for _ in range(N_SUBSETS):
                # Sample K judges from the pool
                idx = np.random.choice(MAX_JUDGES, size=K, replace=False)
                subset_scores = all_scores[idx]
                agg_scores = aggregate_scores(subset_scores, agg)
                r, _ = spearmanr(agg_scores, gold_scores)
                corrs.append(r)

            results[agg]["mean_corr"].append(np.mean(corrs))
            results[agg]["std_corr"].append(np.std(corrs))

        print("mean_corr(mean_agg)=%.4f" % results["mean"]["mean_corr"][-1])

    # Convert to arrays
    for agg in AGGREGATORS:
        results[agg]["mean_corr"] = np.array(results[agg]["mean_corr"])
        results[agg]["std_corr"] = np.array(results[agg]["std_corr"])

    K_arr = np.array(K_VALUES, dtype=float)

    # ------------------------------------------------------------------
    # Step 5: Fit scaling laws to error = 1 - correlation
    # ------------------------------------------------------------------
    print("\n[5/7] Fitting scaling laws (power-law and exponential) ...")

    fit_rows = []
    fit_data = {}  # for plotting fitted curves

    for agg in AGGREGATORS:
        errors = 1.0 - results[agg]["mean_corr"]

        # Power-law fit: error = a * K^{-alpha} + b
        alpha_pl, a_pl, b_pl, R2_pl = fit_power_law(K_VALUES, errors)

        # Exponential fit: error = a * exp(-b*K) + c
        rate_exp, a_exp, c_exp, R2_exp = fit_exponential_decay(K_VALUES, errors)

        better = "power_law" if R2_pl >= R2_exp else "exponential"

        fit_rows.append({
            "aggregator": agg,
            "power_alpha": alpha_pl,
            "power_R2": R2_pl,
            "exp_rate": rate_exp,
            "exp_R2": R2_exp,
            "better_fit": better,
        })

        fit_data[agg] = {
            "power_law": (a_pl, alpha_pl, b_pl, R2_pl),
            "exponential": (a_exp, rate_exp, c_exp, R2_exp),
        }

        print("  %6s: power-law alpha=%.4f R2=%.4f | exp rate=%.4f R2=%.4f | better=%s"
              % (agg, alpha_pl, R2_pl, rate_exp, R2_exp, better))

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp6_scaling_fit.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "aggregator", "power_alpha", "power_R2", "exp_rate", "exp_R2", "better_fit"
        ])
        writer.writeheader()
        writer.writerows(fit_rows)
    print("  Saved: %s" % csv_path)

    # ------------------------------------------------------------------
    # Step 6a: Plot — Correlation vs K
    # ------------------------------------------------------------------
    print("\n[6/7] Generating plots ...")

    agg_colors = {"mean": COLORS[0], "median": COLORS[1], "max": COLORS[2]}
    agg_markers = {"mean": "o", "median": "s", "max": "^"}

    fig1, ax1 = plt.subplots(figsize=(9, 6))
    for agg in AGGREGATORS:
        mc = results[agg]["mean_corr"]
        sc = results[agg]["std_corr"]
        color = agg_colors[agg]
        ax1.errorbar(K_arr, mc, yerr=sc, fmt=agg_markers[agg] + "-",
                     color=color, linewidth=2, markersize=6, capsize=3,
                     label="Aggregator: %s" % agg)

    ax1.set_xlabel(r"Number of judges $K$")
    ax1.set_ylabel("Spearman correlation with gold")
    ax1.set_title("Correlation with Gold vs K (SYNTHETIC PROXY)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    fig1.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(RESULTS_DIR, "exp6_correlation_vs_K.%s" % ext)
        fig1.savefig(path)
        print("  Saved: %s" % path)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Step 6b: Plot — Error scaling (log-log) with fitted curves
    # ------------------------------------------------------------------
    K_fine = np.linspace(K_arr.min(), K_arr.max(), 200)

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for agg in AGGREGATORS:
        errors = 1.0 - results[agg]["mean_corr"]
        color = agg_colors[agg]
        marker = agg_markers[agg]

        # Empirical points
        ax2.loglog(K_arr, errors, marker + "-", color=color, linewidth=2,
                   markersize=6, label="%s (empirical)" % agg)

        # Fitted power-law curve
        a_pl, alpha_pl, b_pl, R2_pl = fit_data[agg]["power_law"]
        if not np.isnan(alpha_pl):
            fitted_pl = a_pl * K_fine ** (-alpha_pl) + b_pl
            fitted_pl = np.clip(fitted_pl, 1e-15, None)
            ax2.loglog(K_fine, fitted_pl, "--", color=color, linewidth=1.5,
                       alpha=0.6,
                       label=r"%s PL: $\alpha$=%.2f, $R^2$=%.3f"
                             % (agg, alpha_pl, R2_pl))

        # Fitted exponential curve
        a_exp, rate_exp, c_exp, R2_exp = fit_data[agg]["exponential"]
        if not np.isnan(rate_exp):
            fitted_exp = a_exp * np.exp(-rate_exp * K_fine) + c_exp
            fitted_exp = np.clip(fitted_exp, 1e-15, None)
            ax2.loglog(K_fine, fitted_exp, ":", color=color, linewidth=1.5,
                       alpha=0.6,
                       label=r"%s Exp: $b$=%.3f, $R^2$=%.3f"
                             % (agg, rate_exp, R2_exp))

    ax2.set_xlabel(r"Number of judges $K$")
    ax2.set_ylabel(r"Error = $1 - \rho_{\mathrm{Spearman}}$")
    ax2.set_title("Error Scaling (log-log) with Fitted Curves (SYNTHETIC PROXY)")
    ax2.legend(fontsize=7, ncol=2, loc="upper right")
    ax2.grid(True, which="both", ls=":", alpha=0.4)

    fig2.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(RESULTS_DIR, "exp6_error_scaling.%s" % ext)
        fig2.savefig(path)
        print("  Saved: %s" % path)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Step 7: W&B logging and report
    # ------------------------------------------------------------------
    print("\n[7/7] W&B logging and reports ...")

    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name="exp06-llm-judges-synthetic",
                config={
                    "experiment": "06_llm_judges",
                    "synthetic_proxy": True,
                    "n_items": N_ITEMS,
                    "max_judges": MAX_JUDGES,
                    "rho": RHO,
                    "empirical_rho": float(empirical_rho),
                    "n_subsets": N_SUBSETS,
                    "K_values": K_VALUES,
                    "aggregators": AGGREGATORS,
                },
            )

            # Log plots
            corr_png = os.path.join(RESULTS_DIR, "exp6_correlation_vs_K.png")
            error_png = os.path.join(RESULTS_DIR, "exp6_error_scaling.png")
            wandb.log({"correlation_vs_K": wandb.Image(corr_png)})
            wandb.log({"error_scaling": wandb.Image(error_png)})

            # Log scaling curves as tables
            for agg in AGGREGATORS:
                table = wandb.Table(columns=["K", "mean_corr", "std_corr", "error"])
                for ki, K in enumerate(K_VALUES):
                    table.add_data(
                        K,
                        float(results[agg]["mean_corr"][ki]),
                        float(results[agg]["std_corr"][ki]),
                        float(1.0 - results[agg]["mean_corr"][ki]),
                    )
                wandb.log({"scaling_%s" % agg: table})

            # Log fit results
            fit_table = wandb.Table(columns=[
                "aggregator", "power_alpha", "power_R2", "exp_rate", "exp_R2", "better_fit"
            ])
            for row in fit_rows:
                fit_table.add_data(
                    row["aggregator"], row["power_alpha"], row["power_R2"],
                    row["exp_rate"], row["exp_R2"], row["better_fit"],
                )
            wandb.log({"scaling_fits": fit_table})

            # Summary metrics
            wandb.run.summary["empirical_rho"] = float(empirical_rho)
            for row in fit_rows:
                agg = row["aggregator"]
                wandb.run.summary["%s_power_alpha" % agg] = row["power_alpha"]
                wandb.run.summary["%s_power_R2" % agg] = row["power_R2"]
                wandb.run.summary["%s_exp_rate" % agg] = row["exp_rate"]
                wandb.run.summary["%s_exp_R2" % agg] = row["exp_R2"]

            print("  [W&B] Logged all metrics and artifacts.")
        except Exception as e:
            print("  [WARN] W&B logging failed: %s" % e)

    # W&B Report
    if WANDB_AVAILABLE and WR_AVAILABLE and wandb_run is not None:
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            report = wr.Report(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                title="Exp 06 - LLM Judges Scaling (Synthetic Proxy) - %s" % now_str,
                description=(
                    "SYNTHETIC PROXY: Simulates LLM judge ensembles with "
                    "bias, noise, and inter-judge correlation. Tests scaling "
                    "of Spearman correlation with gold as K grows."
                ),
            )

            # Build fit table for markdown
            fit_md_rows = []
            for row in fit_rows:
                fit_md_rows.append(
                    "| %s | %.4f | %.4f | %.4f | %.4f | %s |" % (
                        row["aggregator"], row["power_alpha"], row["power_R2"],
                        row["exp_rate"], row["exp_R2"], row["better_fit"],
                    )
                )
            fit_md = (
                "| aggregator | power_alpha | power_R2 | exp_rate | exp_R2 | better_fit |\n"
                "|------------|-------------|----------|----------|--------|------------|\n"
                + "\n".join(fit_md_rows)
            )

            report.blocks = [
                wr.H1("Experiment 06: LLM Judges Scaling Law"),
                wr.P(
                    "**NOTE: This is a SYNTHETIC PROXY experiment.** "
                    "No real LLM APIs were called. Instead, we simulate judges "
                    "with individual biases, noise levels, and shared noise "
                    "(inter-judge correlation rho=%.2f)." % RHO
                ),
                wr.H2("Setup"),
                wr.P(
                    "- %d items with gold scores ~ Uniform(1, 10)\n"
                    "- %d synthetic judges with bias ~ N(0, 0.5), "
                    "sigma ~ Uniform(0.5, 2.0)\n"
                    "- Inter-judge correlation rho = %.2f "
                    "(empirical: %.4f)\n"
                    "- %d random subsets per K value"
                    % (N_ITEMS, MAX_JUDGES, RHO, empirical_rho, N_SUBSETS)
                ),
                wr.H2("Correlation with Gold vs K"),
                wr.P("One curve per aggregator (mean, median, max). "
                     "Error bars show std across subsets."),
                wr.H2("Error Scaling (log-log)"),
                wr.P("Dashed = power-law fit, dotted = exponential fit."),
                wr.H2("Fitted Scaling Parameters"),
                wr.P(fit_md),
            ]
            report.save()
            print("  [W&B] Report saved: %s" % report.url)
        except Exception as e:
            print("  [WARN] W&B report creation failed: %s" % e)

    # Finish W&B run
    if WANDB_AVAILABLE and wandb_run is not None:
        try:
            wandb.finish()
            print("  [W&B] Run finished.")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Local markdown report
    # ------------------------------------------------------------------
    print("\n[Local] Writing summary report ...")
    report_path = os.path.join(SUMMARY_DIR, "exp06_report.md")
    with open(report_path, "w") as f:
        f.write("# Experiment 06 - End-to-End Scaling Law with LLM Judges\n\n")
        f.write("**NOTE: SYNTHETIC PROXY - No real LLM APIs were used.**\n\n")
        f.write("**Date:** %s\n\n" % datetime.now().strftime("%Y-%m-%d %H:%M"))

        f.write("## Setup\n\n")
        f.write("- n_items = %d (gold scores ~ Uniform(1, 10))\n" % N_ITEMS)
        f.write("- max_judges = %d\n" % MAX_JUDGES)
        f.write("- Inter-judge correlation rho = %.2f\n" % RHO)
        f.write("- Empirical pairwise Spearman correlation = %.4f\n" % empirical_rho)
        f.write("- Judge biases ~ N(0, 0.5)\n")
        f.write("- Judge noise levels ~ Uniform(0.5, 2.0)\n")
        f.write("- K values: %s\n" % K_VALUES)
        f.write("- N_SUBSETS = %d per K\n" % N_SUBSETS)
        f.write("- Aggregators: %s\n\n" % AGGREGATORS)

        f.write("## Synthetic Judge Model\n\n")
        f.write("Score for judge j on item i:\n\n")
        f.write("    S_{ji} = gold_i + b_j + sigma_j * "
                "(sqrt(rho) * Z_common_i + sqrt(1-rho) * Z_{ji})\n\n")
        f.write("where b_j ~ N(0, 0.5), sigma_j ~ U(0.5, 2.0), "
                "Z ~ N(0, 1), clipped to [1, 10].\n\n")

        f.write("## Results: Scaling Law Fits\n\n")
        f.write("| Aggregator | Power alpha | Power R^2 | Exp rate | Exp R^2 | Better fit |\n")
        f.write("|------------|-------------|-----------|----------|---------|------------|\n")
        for row in fit_rows:
            f.write("| %s | %.4f | %.4f | %.4f | %.4f | %s |\n" % (
                row["aggregator"], row["power_alpha"], row["power_R2"],
                row["exp_rate"], row["exp_R2"], row["better_fit"],
            ))
        f.write("\n")

        f.write("## Correlation with Gold at Selected K\n\n")
        f.write("| K |")
        for agg in AGGREGATORS:
            f.write(" %s corr |" % agg)
        f.write("\n")
        f.write("|---|")
        for _ in AGGREGATORS:
            f.write("---------|")
        f.write("\n")
        for ki, K in enumerate(K_VALUES):
            f.write("| %d |" % K)
            for agg in AGGREGATORS:
                f.write(" %.4f |" % results[agg]["mean_corr"][ki])
            f.write("\n")
        f.write("\n")

        f.write("## Plots\n\n")
        f.write("- Correlation vs K: `../results/exp6_correlation_vs_K.pdf` / `.png`\n")
        f.write("- Error scaling (log-log): `../results/exp6_error_scaling.pdf` / `.png`\n")
        f.write("- Scaling fits CSV: `../results/exp6_scaling_fit.csv`\n")

    print("  Saved: %s" % report_path)

    print("\n" + "=" * 65)
    print("Experiment 06 complete (SYNTHETIC PROXY).")
    print("=" * 65)


if __name__ == "__main__":
    run_experiment()
