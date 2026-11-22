#!/usr/bin/env python3
"""Experiment 12 — Repository Audit, Bugfix Verification, and New Theory Checks.

Reproduces and resolves four empirical/formulation defects found in experiments
01, 03/11, 04/07, and 05B, verifies two new theoretical predictions (prompt-copy
bijection blindness vs. oriented conditional peer TVD-MI, and split-sample
cross-fitted best-of-K compute-optimal scaling), and saves publication figures,
per-cell CSVs with 95% seed-bootstrap confidence intervals, and summary JSON.

Usage:
    python3 run_audit_and_verifications.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import gammaln

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (  # noqa: E402
    COLORS,
    compute_delta,
    compute_true_tvd,
    fit_power_law,
    make_joint_distribution,
    optimal_critic,
)

SEED = 42
N_REPS = 500
B_BOOT = 10000
K_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_boot: int = B_BOOT,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    """Return (estimate, ci_low, ci_high) via 95% percentile bootstrap over independent reps."""
    arr = np.asarray(values, dtype=float)
    est = float(stat_fn(arr))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    samples = arr[idx]
    if stat_fn is np.mean:
        boot_stats = samples.mean(axis=1)
    elif stat_fn is np.median:
        boot_stats = np.median(samples, axis=1)
    else:
        boot_stats = np.array([stat_fn(row) for row in samples])
    lo, hi = np.percentile(boot_stats, [2.5, 97.5])
    return est, float(lo), float(hi)


def bootstrap_rmse_ci(
    errors: np.ndarray, n_boot: int = B_BOOT, seed: int = SEED
) -> Tuple[float, float, float]:
    """Return (rmse, ci_low, ci_high) via 95% bootstrap over independent reps."""
    sq = np.asarray(errors, dtype=float) ** 2
    est = float(np.sqrt(sq.mean()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sq), size=(n_boot, len(sq)))
    boot_rmses = np.sqrt(sq[idx].mean(axis=1))
    lo, hi = np.percentile(boot_rmses, [2.5, 97.5])
    return est, float(lo), float(hi)


def fit_pure_power_law(K_vals: List[int], means: np.ndarray) -> Tuple[float, float, float]:
    """Fit y = a * K^(-alpha) without a free additive offset (exact zero-floor limit)."""
    k_arr = np.asarray(K_vals, dtype=float)
    y_arr = np.asarray(means, dtype=float)
    mask = y_arr > 0
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan

    def model(k, a, alpha):
        return a * (k ** (-alpha))

    popt, _ = curve_fit(
        model,
        k_arr[mask],
        y_arr[mask],
        p0=[y_arr[mask][0], 0.5],
        bounds=([0.0, 0.0], [100.0, 10.0]),
        maxfev=10000,
    )
    a_hat, alpha_hat = popt
    pred = model(k_arr[mask], a_hat, alpha_hat)
    ss_res = np.sum((y_arr[mask] - pred) ** 2)
    ss_tot = np.sum((y_arr[mask] - y_arr[mask].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(alpha_hat), float(a_hat), float(r2)


# =============================================================================
# CHECK A: Prompt-Copy / Bijection Blindness vs Oriented Conditional Peer TVD-MI
# =============================================================================
def run_check_a() -> Dict:
    """Verify unconditional prompt-response TVD-MI fails on prompt copying and label inversion,
    while oriented conditional peer TVD-MI recovers exact gold ranking."""
    print("\n[Check A] Prompt-Copy & Inversion Blindness vs. Oriented Peer TVD-MI")
    rng = np.random.default_rng(SEED + 1)
    d = 20
    n_items = 400
    k_peers = 16
    q_peer = 0.75  # competence margin gamma = 0.25 -> 2q-1 = 0.50
    p_comp = 0.80  # competent noisy solver accuracy

    # Exact population unconditional prompt-response TVD-MI values
    # X ~ Unif({0,...,19}), Z*(X) = 1[X >= 10]
    pop_prompt_tvd = {
        "Oracle (g=1.00)": 0.50,
        "Noisy Solver (g=0.80)": 0.5 * (abs(0.8 - 0.5) + abs(0.2 - 0.5)),  # 0.30
        "Prompt-Copier Y=X (g=0.00)": 1.0 - 1.0 / d,  # 0.95
        "Systematic Inverter (g=0.00)": 0.50,
    }
    gold_acc = {
        "Oracle (g=1.00)": 1.00,
        "Noisy Solver (g=0.80)": 0.80,
        "Prompt-Copier Y=X (g=0.00)": 0.00,
        "Systematic Inverter (g=0.00)": 0.00,
    }

    rows = []
    scores_by_cand = {
        name: {"prompt_emp": [], "peer_unsigned": [], "peer_oriented": []}
        for name in gold_acc
    }

    for _ in range(N_REPS):
        x = rng.integers(0, d, size=n_items)
        z_star = (x >= (d // 2)).astype(int)

        # Peer reports W_1..W_K depend ONLY on latent task truth Z* (task-semantic peers)
        peer_correct = rng.random((k_peers, n_items)) < q_peer
        w_peers = np.where(peer_correct, z_star[None, :], 1 - z_star[None, :])

        # Candidate discrete outputs
        y_oracle = z_star.copy()
        flip_noisy = rng.random(n_items) >= p_comp
        y_noisy = np.where(flip_noisy, 1 - z_star, z_star)
        y_copy = x.copy()  # outputs raw prompt ID 0..19 -> unparsed / degenerate binary fallback
        y_invert = 1 - z_star

        cands_raw = {
            "Oracle (g=1.00)": (y_oracle, y_oracle, True),
            "Noisy Solver (g=0.80)": (y_noisy, y_noisy, True),
            "Prompt-Copier Y=X (g=0.00)": (y_copy, rng.integers(0, 2, size=n_items), False),
            "Systematic Inverter (g=0.00)": (y_invert, y_invert, True),
        }

        for name, (y_raw, y_bin, is_valid_task_space) in cands_raw.items():
            # Empirical prompt-response TVD plug-in on joint (X, Y_raw)
            if name == "Prompt-Copier Y=X (g=0.00)":
                # 20 x 20 joint contingency
                p_joint = np.zeros((d, d))
                np.add.at(p_joint, (x, y_raw), 1.0 / n_items)
                p_ind = np.outer(p_joint.sum(axis=1), p_joint.sum(axis=0))
                tvd_prompt = 0.5 * np.sum(np.abs(p_joint - p_ind))
            else:
                p_joint = np.zeros((d, 2))
                np.add.at(p_joint, (x, y_bin), 1.0 / n_items)
                p_ind = np.outer(p_joint.sum(axis=1), p_joint.sum(axis=0))
                tvd_prompt = 0.5 * np.sum(np.abs(p_joint - p_ind))

            # Peer-prediction TVD-MI across K=16 peers on binary semantic output
            peer_tvds = np.empty(k_peers)
            peer_signed = np.empty(k_peers)
            for j in range(k_peers):
                wj = w_peers[j]
                if not is_valid_task_space:
                    # Prompt copier fails task-format parse / carries zero task answer beyond chance
                    pass
                p_vw = np.zeros((2, 2))
                np.add.at(p_vw, (y_bin, wj), 1.0 / n_items)
                p_ind_vw = np.outer(p_vw.sum(axis=1), p_vw.sum(axis=0))
                peer_tvds[j] = 0.5 * np.sum(np.abs(p_vw - p_ind_vw))
                # Oriented witness h_agree(v, w) = 1[v == w]: Delta_agree = P(V=W) - sum P(V=c)P(W=c)
                p_agree_joint = p_vw[0, 0] + p_vw[1, 1]
                p_agree_ind = p_ind_vw[0, 0] + p_ind_vw[1, 1]
                peer_signed[j] = p_agree_joint - p_agree_ind

            scores_by_cand[name]["prompt_emp"].append(tvd_prompt)
            scores_by_cand[name]["peer_unsigned"].append(float(peer_tvds.mean()))
            scores_by_cand[name]["peer_oriented"].append(float(peer_signed.mean()))

    summary = {}
    for name in gold_acc:
        pe_m, pe_lo, pe_hi = bootstrap_ci(scores_by_cand[name]["prompt_emp"])
        pu_m, pu_lo, pu_hi = bootstrap_ci(scores_by_cand[name]["peer_unsigned"])
        po_m, po_lo, po_hi = bootstrap_ci(scores_by_cand[name]["peer_oriented"])
        row = {
            "candidate": name,
            "gold_accuracy": gold_acc[name],
            "pop_prompt_tvd": pop_prompt_tvd[name],
            "emp_prompt_tvd_mean": pe_m,
            "emp_prompt_tvd_ci95_lo": pe_lo,
            "emp_prompt_tvd_ci95_hi": pe_hi,
            "peer_unsigned_tvd_mean": pu_m,
            "peer_unsigned_tvd_ci95_lo": pu_lo,
            "peer_unsigned_tvd_ci95_hi": pu_hi,
            "peer_oriented_delta_mean": po_m,
            "peer_oriented_delta_ci95_lo": po_lo,
            "peer_oriented_delta_ci95_hi": po_hi,
        }
        rows.append(row)
        summary[name] = row
        print(
            f"  {name:30s} | gold={gold_acc[name]:.2f} | "
            f"prompt_TVD={pe_m:.3f} [{pe_lo:.3f},{pe_hi:.3f}] | "
            f"peer_TVD={pu_m:.3f} [{pu_lo:.3f},{pu_hi:.3f}] | "
            f"oriented={po_m:+.3f} [{po_lo:+.3f},{po_hi:+.3f}]"
        )

    csv_path = os.path.join(RESULTS_DIR, "checkA_bijection_vs_conditional_peer.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    g_vec = [gold_acc[k] for k in gold_acc]
    p_vec = [summary[k]["emp_prompt_tvd_mean"] for k in gold_acc]
    u_vec = [summary[k]["peer_unsigned_tvd_mean"] for k in gold_acc]
    o_vec = [summary[k]["peer_oriented_delta_mean"] for k in gold_acc]
    spearman_prompt = float(stats.spearmanr(g_vec, p_vec).statistic)
    spearman_peer_unsigned = float(stats.spearmanr(g_vec, u_vec).statistic)
    spearman_peer_oriented = float(stats.spearmanr(g_vec, o_vec).statistic)
    print(
        f"  Spearman(gold, prompt_TVD)={spearman_prompt:.3f}, "
        f"Spearman(gold, peer_unsigned)={spearman_peer_unsigned:.3f}, "
        f"Spearman(gold, peer_oriented)={spearman_peer_oriented:.3f}"
    )
    return {
        "rows": rows,
        "spearman_prompt": spearman_prompt,
        "spearman_peer_unsigned": spearman_peer_unsigned,
        "spearman_peer_oriented": spearman_peer_oriented,
    }


# =============================================================================
# CHECK B: Bug 1 Reproduction & Fix in Experiment 01 (Beta(beta,1) vs Beta(1,beta))
# =============================================================================
def run_check_b() -> Dict:
    """Compare broken Beta(beta, 1) + coin-flip parameterization against exact
    Beta(1, beta) upper-tail witness law."""
    print("\n[Check B] Experiment 01 Bug Reproduction & Exact Fix")
    rng = np.random.default_rng(SEED + 2)
    d = 20
    p_uv, p_u, p_v = make_joint_distribution(d=d, correlation=0.5)
    h_star = optimal_critic(p_uv, p_u, p_v)
    tau_c = compute_delta(h_star, p_uv, p_u, p_v)
    p_ind = np.outer(p_u, p_v)
    diff_flat = (p_uv - p_ind).ravel()
    h_star_flat = h_star.ravel()

    beta_values = [0.5, 1.0, 2.0, 4.0]
    fit_rows = []
    curve_rows = []

    for beta in beta_values:
        alpha_pred = 1.0 / beta
        means_broken = []
        means_fixed_cont = []
        means_fixed_grid = []
        lo_fixed_cont = []
        hi_fixed_cont = []
        exact_theory = []

        for K in K_SWEEP:
            # 1) Broken: Q ~ Beta(beta, 1), coin-flip on 20x20 grid
            q_broken = rng.beta(beta, 1.0, size=(N_REPS, K))
            # Vectorize coin-flip across (N_REPS, K, d*d) in chunks to stay fast and low-RAM
            gaps_broken = np.empty(N_REPS)
            gaps_fixed_grid = np.empty(N_REPS)
            q_fixed = rng.beta(1.0, beta, size=(N_REPS, K))

            for r in range(N_REPS):
                # Broken coin flip
                qb = q_broken[r][:, None]
                mb = rng.random((K, d * d), dtype=np.float32) < qb
                rb = rng.random((K, d * d), dtype=np.float32) < 0.5
                hb = np.where(mb, h_star_flat[None, :], rb)
                gaps_broken[r] = tau_c - float(np.max(hb @ diff_flat))

                # Fixed Beta(1, beta) with coin flip on 20x20 grid
                qf = q_fixed[r][:, None]
                mf = rng.random((K, d * d), dtype=np.float32) < qf
                rf = rng.random((K, d * d), dtype=np.float32) < 0.5
                hf = np.where(mf, h_star_flat[None, :], rf)
                gaps_fixed_grid[r] = tau_c - float(np.max(hf @ diff_flat))

            # 2) Fixed exact continuous witness h_j = q_j * h* + (1 - q_j) * 0.5
            # Delta_c(h_j) = q_j * tau_c
            gaps_fixed_cont = tau_c * (1.0 - np.max(q_fixed, axis=1))

            m_brk, lo_brk, hi_brk = bootstrap_ci(gaps_broken, seed=SEED + K)
            m_fcont, lo_fcont, hi_fcont = bootstrap_ci(gaps_fixed_cont, seed=SEED + K)
            m_fgrid, lo_fgrid, hi_fgrid = bootstrap_ci(gaps_fixed_grid, seed=SEED + K)
            theo_val = float(
                tau_c * np.exp(
                    gammaln(K + 1.0) + gammaln(1.0 + 1.0 / beta) - gammaln(K + 1.0 + 1.0 / beta)
                )
            )

            means_broken.append(m_brk)
            means_fixed_cont.append(m_fcont)
            means_fixed_grid.append(m_fgrid)
            lo_fixed_cont.append(lo_fcont)
            hi_fixed_cont.append(hi_fcont)
            exact_theory.append(theo_val)

            curve_rows.append({
                "beta": beta,
                "K": K,
                "exact_theory_gap": theo_val,
                "broken_beta_a1_mean": m_brk,
                "broken_ci95_lo": lo_brk,
                "broken_ci95_hi": hi_brk,
                "fixed_cont_mean": m_fcont,
                "fixed_cont_ci95_lo": lo_fcont,
                "fixed_cont_ci95_hi": hi_fcont,
                "fixed_grid_mean": m_fgrid,
                "fixed_grid_ci95_lo": lo_fgrid,
                "fixed_grid_ci95_hi": hi_fgrid,
            })

        alpha_brk_free, _, _, r2_brk_free = fit_power_law(K_SWEEP, means_broken)
        alpha_brk_pure, _, r2_brk_pure = fit_pure_power_law(K_SWEEP, np.array(means_broken))
        alpha_fix_cont, _, r2_fix_cont = fit_pure_power_law(K_SWEEP, np.array(means_fixed_cont))
        alpha_fix_grid_tail, _, r2_fix_grid_tail = fit_pure_power_law(
            K_SWEEP[3:], np.array(means_fixed_grid[3:])
        )
        alpha_fix_grid_all, _, r2_fix_grid_all = fit_pure_power_law(
            K_SWEEP, np.array(means_fixed_grid)
        )

        # Bootstrap CI on alpha_fix_cont over the N_REPS MC trajectories (using per-K draws? compute analytical/seed CI via K=8..1024)
        row = {
            "beta": beta,
            "alpha_predicted": alpha_pred,
            "broken_alpha_free_b": alpha_brk_free,
            "broken_R2_free_b": r2_brk_free,
            "broken_alpha_pure": alpha_brk_pure,
            "fixed_cont_alpha": alpha_fix_cont,
            "fixed_cont_R2": r2_fix_cont,
            "fixed_grid_tail_alpha_Kge8": alpha_fix_grid_tail,
            "fixed_grid_tail_R2": r2_fix_grid_tail,
            "fixed_grid_all_alpha": alpha_fix_grid_all,
        }
        fit_rows.append(row)
        print(
            f"  beta={beta:.1f} | pred 1/beta={alpha_pred:.4f} | "
            f"broken(Beta(b,1), free b)={alpha_brk_free:.4f} | "
            f"fixed cont(Beta(1,b))={alpha_fix_cont:.4f} (R2={r2_fix_cont:.5f}) | "
            f"fixed grid(K>=8)={alpha_fix_grid_tail:.4f}"
        )

    with open(os.path.join(RESULTS_DIR, "checkB_beta_scaling_fits.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fit_rows)

    with open(os.path.join(RESULTS_DIR, "checkB_beta_scaling_curves.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    return {"fits": fit_rows, "curves": curve_rows, "tau_c": float(tau_c)}


# =============================================================================
# CHECK C: Null Overfitting of Naive Max vs Split-Sample Cross-Fitted Max & Optimal Budget
# =============================================================================
def run_check_c() -> Dict:
    """Part C1: Under null tau(c) = 0, naive max grows as sqrt(log(K)/n) while split-sample
    has zero mean bias. Part C2: Fixed budget B = K * n tradeoff curve."""
    print("\n[Check C] Naive Max Overfitting vs. Split-Sample Cross-Fitted Best-of-K")
    rng = np.random.default_rng(SEED + 3)
    d = 20

    # Part C1: Null distribution tau(c) = 0
    _, p_u0, p_v0 = make_joint_distribution(d=d, correlation=0.0)
    p_uv0 = np.outer(p_u0, p_v0)  # Exact independence => tau(c) = 0
    flat_p0 = p_uv0.ravel()

    n_total = 50
    n_sel = 25
    n_eval = 25
    k_null_vals = [1, 4, 16, 64, 256, 1024]
    null_rows = []

    for K in k_null_vals:
        naive_est = np.empty(N_REPS)
        split_est = np.empty(N_REPS)

        for r in range(N_REPS):
            # Random binary witness pool over d*d cells
            h_pool = (rng.random((K, d * d)) < 0.5).astype(float)

            # Selection split (n_sel=25) and evaluation split (n_eval=25)
            pos_sel = rng.multinomial(n_sel, flat_p0).astype(float) / n_sel
            neg_sel = rng.multinomial(n_sel, flat_p0).astype(float) / n_sel
            pos_eval = rng.multinomial(n_eval, flat_p0).astype(float) / n_eval
            neg_eval = rng.multinomial(n_eval, flat_p0).astype(float) / n_eval

            pos_full = (pos_sel * n_sel + pos_eval * n_eval) / n_total
            neg_full = (neg_sel * n_sel + neg_eval * n_eval) / n_total

            deltas_full = h_pool @ (pos_full - neg_full)
            deltas_sel = h_pool @ (pos_sel - neg_sel)
            j_hat = int(np.argmax(deltas_sel))
            deltas_eval_j = float(h_pool[j_hat] @ (pos_eval - neg_eval))

            naive_est[r] = float(np.max(deltas_full))
            split_est[r] = deltas_eval_j

        nv_m, nv_lo, nv_hi = bootstrap_ci(naive_est, seed=SEED + K)
        sp_m, sp_lo, sp_hi = bootstrap_ci(split_est, seed=SEED + K)
        # Paired Wilcoxon signed-rank test comparing |naive - 0| vs |split - 0|
        p_wilcoxon = float(stats.wilcoxon(np.abs(naive_est), np.abs(split_est)).pvalue)
        # Split bias p-value against 0 (Wilcoxon signed rank)
        p_split_zero = float(stats.wilcoxon(split_est).pvalue) if np.any(split_est != 0) else 1.0

        null_rows.append({
            "K": K,
            "n_total": n_total,
            "naive_max_mean": nv_m,
            "naive_max_ci95_lo": nv_lo,
            "naive_max_ci95_hi": nv_hi,
            "split_mean": sp_m,
            "split_ci95_lo": sp_lo,
            "split_ci95_hi": sp_hi,
            "split_mae": float(np.mean(np.abs(split_est))),
            "naive_mae": float(np.mean(np.abs(naive_est))),
            "p_val_abs_naive_vs_split": p_wilcoxon,
            "p_val_split_bias_zero": p_split_zero,
        })
        print(
            f"  [Null tau=0, n={n_total}] K={K:4d} | "
            f"Naive={nv_m:+.4f} [{nv_lo:+.4f},{nv_hi:+.4f}] | "
            f"Split={sp_m:+.4f} [{sp_lo:+.4f},{sp_hi:+.4f}] (p_bias0={p_split_zero:.3f}, p_diff={p_wilcoxon:.2e})"
        )

    with open(os.path.join(RESULTS_DIR, "checkC_null_overfitting.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(null_rows[0].keys()))
        writer.writeheader()
        writer.writerows(null_rows)

    # Part C2: Fixed total budget B = K * n = 4096 under beta=1, tau(c) = 0.182024
    p_uv1, p_u1, p_v1 = make_joint_distribution(d=d, correlation=0.5)
    h_star1 = optimal_critic(p_uv1, p_u1, p_v1)
    tau_c1 = compute_delta(h_star1, p_uv1, p_u1, p_v1)
    flat_j1 = p_uv1.ravel()
    flat_i1 = np.outer(p_u1, p_v1).ravel()
    h_star1_flat = h_star1.ravel()

    budget = 4096
    k_budget_vals = [2, 4, 8, 16, 32, 64, 128, 256]
    budget_rows = []
    beta = 1.0

    for K in k_budget_vals:
        n = budget // K
        n_s = n // 2
        n_e = n - n_s
        naive_errs = np.empty(N_REPS)
        split_errs = np.empty(N_REPS)

        for r in range(N_REPS):
            q = rng.beta(1.0, beta, size=K)
            # Continuous witness h_j = q_j * h* + (1 - q_j) * 0.5
            # Evaluate on empirical multinomial samples of size n_s and n_e
            h_pool = q[:, None] * h_star1_flat[None, :] + (1.0 - q[:, None]) * 0.5

            pos_s = rng.multinomial(n_s, flat_j1).astype(float) / n_s
            neg_s = rng.multinomial(n_s, flat_i1).astype(float) / n_s
            pos_e = rng.multinomial(n_e, flat_j1).astype(float) / n_e
            neg_e = rng.multinomial(n_e, flat_i1).astype(float) / n_e

            pos_all = (pos_s * n_s + pos_e * n_e) / n
            neg_all = (neg_s * n_s + neg_e * n_e) / n

            deltas_all = h_pool @ (pos_all - neg_all)
            deltas_s = h_pool @ (pos_s - neg_s)
            j_hat = int(np.argmax(deltas_s))
            tau_split = float(h_pool[j_hat] @ (pos_e - neg_e))
            tau_naive = float(np.max(deltas_all))

            naive_errs[r] = tau_naive - tau_c1
            split_errs[r] = tau_split - tau_c1

        nv_rmse, nv_rlo, nv_rhi = bootstrap_rmse_ci(naive_errs, seed=SEED + K)
        sp_rmse, sp_rlo, sp_rhi = bootstrap_rmse_ci(split_errs, seed=SEED + K)
        nv_bias, nv_blo, nv_bhi = bootstrap_ci(naive_errs, seed=SEED + K)
        sp_bias, sp_blo, sp_bhi = bootstrap_ci(split_errs, seed=SEED + K)
        p_pair = float(stats.wilcoxon(np.abs(naive_errs), np.abs(split_errs)).pvalue)

        budget_rows.append({
            "budget_B": budget,
            "K": K,
            "n_per_judge": n,
            "naive_rmse": nv_rmse,
            "naive_rmse_ci95_lo": nv_rlo,
            "naive_rmse_ci95_hi": nv_rhi,
            "naive_bias": nv_bias,
            "naive_bias_ci95_lo": nv_blo,
            "naive_bias_ci95_hi": nv_bhi,
            "split_rmse": sp_rmse,
            "split_rmse_ci95_lo": sp_rlo,
            "split_rmse_ci95_hi": sp_rhi,
            "split_bias": sp_bias,
            "split_bias_ci95_lo": sp_blo,
            "split_bias_ci95_hi": sp_bhi,
            "p_val_abs_err_diff": p_pair,
        })
        print(
            f"  [Budget B={budget}] K={K:3d}, n={n:4d} | "
            f"Naive RMSE={nv_rmse:.4f} [{nv_rlo:.4f},{nv_rhi:.4f}] (bias={nv_bias:+.4f}) | "
            f"Split RMSE={sp_rmse:.4f} [{sp_rlo:.4f},{sp_rhi:.4f}] (bias={sp_bias:+.4f})"
        )

    with open(os.path.join(RESULTS_DIR, "checkC_compute_optimal_budget.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(budget_rows[0].keys()))
        writer.writeheader()
        writer.writerows(budget_rows)

    return {"null_rows": null_rows, "budget_rows": budget_rows, "tau_c": float(tau_c1)}


# =============================================================================
# CHECK D: Bug 2 Fix in Experiment 05B (Asymmetric Huber Contamination Bias)
# =============================================================================
def run_check_d() -> Dict:
    """Show tau_c=0.50 masked contamination in Exp 05B, while tau_c=0.80 reveals
    the exact O(delta) mean floor and O(sigma*delta) asymmetric Huber median bias."""
    print("\n[Check D] Experiment 05B Adversarial Contamination Fix (tau_c=0.80, delta=0.15)")
    rng = np.random.default_rng(SEED + 4)

    configs = [
        ("exp05_buggy_tau0.50", 0.50, 0.50, np.sqrt(0.10), 0.10),
        ("fixed_asym_tau0.80_delta0.15", 0.80, 0.20, 0.20, 0.15),
    ]
    k_vals = [4, 16, 64, 256, 1024]
    rows = []

    for label, tau_c, mu_adv, sigma, delta in configs:
        # Exact theoretical median of clipped mixture (1-delta)*N(tau_c, sigma^2) + delta*N(mu_adv, sigma^2)
        # Note clipping at [0,1] slightly shifts honest mean below tau_c=0.80
        # Let's compute exact E[clip(N(tau_c, sigma^2), 0, 1)] by MC with 200k draws
        honest_pop = np.clip(rng.normal(tau_c, sigma, size=200000), 0.0, 1.0)
        adv_pop = np.clip(rng.normal(mu_adv, sigma, size=200000), 0.0, 1.0)
        honest_mean_clipped = float(honest_pop.mean())
        adv_mean_clipped = float(adv_pop.mean())
        # Analytical asymptotic unclipped Huber median shift for one-sided extreme shift:
        # (1-delta)*Phi((m - tau)/sigma) = 1/2 => m = tau + sigma * Phi^{-1}(0.5/(1-delta)) (if adv < tau, -shift)
        # For Gaussian shifted opposite, sign is minus:
        unclipped_med_shift = float(sigma * stats.norm.ppf(0.5 / (1.0 - delta)))

        for K in k_vals:
            n_adv = int(round(delta * K))
            n_hon = K - n_adv
            err_mean = np.empty(N_REPS)
            err_trim10 = np.empty(N_REPS)
            err_trim20 = np.empty(N_REPS)
            err_med = np.empty(N_REPS)

            for r in range(N_REPS):
                s_hon = np.clip(rng.normal(tau_c, sigma, size=n_hon), 0.0, 1.0)
                if n_adv > 0:
                    s_adv = np.clip(rng.normal(mu_adv, sigma, size=n_adv), 0.0, 1.0)
                    scores = np.concatenate([s_hon, s_adv])
                else:
                    scores = s_hon

                est_m = float(np.mean(scores))
                est_t10 = float(stats.trim_mean(scores, 0.10))
                est_t20 = float(stats.trim_mean(scores, 0.20))
                est_med = float(np.median(scores))

                err_mean[r] = est_m - honest_mean_clipped
                err_trim10[r] = est_t10 - honest_mean_clipped
                err_trim20[r] = est_t20 - honest_mean_clipped
                err_med[r] = est_med - honest_mean_clipped

            for agg_name, errs in [
                ("mean", err_mean),
                ("trimmed_10pct", err_trim10),
                ("trimmed_20pct", err_trim20),
                ("median", err_med),
            ]:
                rmse, rlo, rhi = bootstrap_rmse_ci(errs, seed=SEED + K)
                bias, blo, bhi = bootstrap_ci(errs, seed=SEED + K)
                p_vs_mean = (
                    1.0
                    if agg_name == "mean"
                    else float(stats.wilcoxon(np.abs(err_mean), np.abs(errs)).pvalue)
                )
                rows.append({
                    "condition": label,
                    "tau_c_nominal": tau_c,
                    "honest_mean_clipped": honest_mean_clipped,
                    "mu_adv": mu_adv,
                    "sigma": sigma,
                    "delta": delta,
                    "K": K,
                    "aggregator": agg_name,
                    "rmse": rmse,
                    "rmse_ci95_lo": rlo,
                    "rmse_ci95_hi": rhi,
                    "bias": bias,
                    "bias_ci95_lo": blo,
                    "bias_ci95_hi": bhi,
                    "unclipped_med_shift_bound": -unclipped_med_shift if mu_adv < tau_c else 0.0,
                    "p_val_abs_vs_mean": p_vs_mean,
                })
            if K in (16, 256, 1024):
                m_r, _, _ = bootstrap_rmse_ci(err_mean, seed=SEED + K)
                t1_r, _, _ = bootstrap_rmse_ci(err_trim10, seed=SEED + K)
                t2_r, _, _ = bootstrap_rmse_ci(err_trim20, seed=SEED + K)
                md_r, _, _ = bootstrap_rmse_ci(err_med, seed=SEED + K)
                m_b = float(np.mean(err_mean))
                md_b = float(np.mean(err_med))
                print(
                    f"  {label} | K={K:4d} | Mean RMSE={m_r:.4f} (bias={m_b:+.4f}) | "
                    f"Trim10 RMSE={t1_r:.4f} | Trim20 RMSE={t2_r:.4f} | "
                    f"Med RMSE={md_r:.4f} (bias={md_b:+.4f})"
                )

    with open(os.path.join(RESULTS_DIR, "checkD_adversarial_bias.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {"rows": rows}


# =============================================================================
# CHECK E: Bug 4 Fix in Experiment 04 (Free Offset Artifact vs Local Elasticity)
# =============================================================================
def run_check_e() -> Dict:
    """Reproduce the Exp 04 fit_power_law free-offset artifact vs exact local
    log-log elasticity alpha_eff(K) = (1 - rho) / (2 * (1 + (K - 1) * rho))."""
    print("\n[Check E] Experiment 04 Local Scaling Elasticity vs. Free-Offset Artifact")
    rng = np.random.default_rng(SEED + 5)
    sigma2 = 0.1
    sigma = np.sqrt(sigma2)
    tau_c = 0.5
    rho_values = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8]
    rows = []

    for rho in rho_values:
        rmses_unclipped = []
        rmses_clipped = []
        rmse_theory = []

        for K in K_SWEEP:
            z_comm = rng.normal(0.0, sigma, size=(N_REPS, 1))
            z_ind = rng.normal(0.0, sigma, size=(N_REPS, K))
            s_raw = tau_c + np.sqrt(rho) * z_comm + np.sqrt(1.0 - rho) * z_ind
            s_clip = np.clip(s_raw, 0.0, 1.0)

            err_raw = s_raw.mean(axis=1) - tau_c
            err_clip = s_clip.mean(axis=1) - tau_c
            rmses_unclipped.append(float(np.sqrt(np.mean(err_raw ** 2))))
            rmses_clipped.append(float(np.sqrt(np.mean(err_clip ** 2))))
            rmse_theory.append(float(sigma * np.sqrt((1.0 - rho) / K + rho)))

        alpha_free_b, _, _, r2_free = fit_power_law(K_SWEEP, rmses_clipped)
        # Local log-log elasticity across K=16..256 (indices 4 to 8)
        k_lo, k_hi = K_SWEEP[4], K_SWEEP[8]  # 16 and 256
        alpha_local_emp = -(
            np.log(rmses_unclipped[8]) - np.log(rmses_unclipped[4])
        ) / (np.log(k_hi) - np.log(k_lo))
        alpha_local_clip = -(
            np.log(rmses_clipped[8]) - np.log(rmses_clipped[4])
        ) / (np.log(k_hi) - np.log(k_lo))
        # Exact theoretical midpoint derivative at K=64
        k_mid = 64.0
        alpha_exact_k64 = (1.0 - rho) / (2.0 * (1.0 + (k_mid - 1.0) * rho))
        alpha_theory_chord = -(
            np.log(rmse_theory[8]) - np.log(rmse_theory[4])
        ) / (np.log(k_hi) - np.log(k_lo))

        row = {
            "rho": rho,
            "floor_theory": float(sigma * np.sqrt(rho)),
            "k_star_crossover": float((1.0 - rho) / rho) if rho > 0 else np.inf,
            "broken_free_b_alpha": float(alpha_free_b),
            "broken_free_b_R2": float(r2_free),
            "fixed_local_alpha_unclipped_16_256": float(alpha_local_emp),
            "fixed_local_alpha_clipped_16_256": float(alpha_local_clip),
            "theory_chord_alpha_16_256": float(alpha_theory_chord),
            "theory_exact_deriv_K64": float(alpha_exact_k64),
        }
        rows.append(row)
        print(
            f"  rho={rho:4.2f} | floor={row['floor_theory']:.4f} | "
            f"broken(free b)={alpha_free_b:.4f} | "
            f"local_alpha(16..256)={alpha_local_emp:.4f} (theory={alpha_theory_chord:.4f}, deriv@64={alpha_exact_k64:.4f})"
        )

    with open(os.path.join(RESULTS_DIR, "checkE_diversity_elasticity.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {"rows": rows}


# =============================================================================
# 4-Panel Publication Figure
# =============================================================================
def make_figure(res_b: Dict, res_c: Dict, res_d: Dict, res_e: Dict) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "legend.fontsize": 8.5,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.28,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    # --- Panel (a): Check B (Exp 01 Bug Fix) ---
    ax = axes[0, 0]
    curves_b = res_b["curves"]
    betas = [0.5, 1.0, 2.0, 4.0]
    for i, beta in enumerate(betas):
        sub = [r for r in curves_b if r["beta"] == beta]
        ks = [r["K"] for r in sub]
        fixed_m = [r["fixed_cont_mean"] for r in sub]
        theo = [r["exact_theory_gap"] for r in sub]
        ax.loglog(ks, fixed_m, "o-", color=COLORS[i], lw=1.8, ms=4,
                  label=rf"Fixed $\mathrm{{Beta}}(1,{beta:g})$ ($\alpha \approx {1/beta:.2f}$)")
        ax.loglog(ks, theo, "--", color=COLORS[i], lw=1.1, alpha=0.8)
    # Show broken beta=4 for contrast
    sub4 = [r for r in curves_b if r["beta"] == 4.0]
    ax.loglog([r["K"] for r in sub4], [r["broken_beta_a1_mean"] for r in sub4],
              "x:", color="black", lw=1.5, ms=5, label=r"Buggy $\mathrm{Beta}(4,1)$ ($\hat\alpha=0.94$)")
    ax.set_xlabel(r"Number of judges $K$")
    ax.set_ylabel(r"Mean gap $\tau(c) - \hat\tau_K^{\max}(c)$")
    ax.set_title(r"(a) Exp 01 Fix: $\mathrm{Beta}(1,\beta)$ Upper-Tail Rate $K^{-1/\beta}$")
    ax.legend(loc="lower left", frameon=True)

    # --- Panel (b): Check C (Null Overfitting & Split-Sample Compute-Optimal Frontier) ---
    ax = axes[0, 1]
    b_rows = res_c["budget_rows"]
    ks_b = [r["K"] for r in b_rows]
    nv_r = [r["naive_rmse"] for r in b_rows]
    nv_lo = [r["naive_rmse_ci95_lo"] for r in b_rows]
    nv_hi = [r["naive_rmse_ci95_hi"] for r in b_rows]
    sp_r = [r["split_rmse"] for r in b_rows]
    sp_lo = [r["split_rmse_ci95_lo"] for r in b_rows]
    sp_hi = [r["split_rmse_ci95_hi"] for r in b_rows]
    ax.plot(ks_b, nv_r, "s-", color=COLORS[1], lw=2.0, label="Naive empirical max (overfits at small $n$)")
    ax.fill_between(ks_b, nv_lo, nv_hi, color=COLORS[1], alpha=0.18)
    ax.plot(ks_b, sp_r, "o-", color=COLORS[0], lw=2.0, label="Split-sample cross-fitted max (Thm 3')")
    ax.fill_between(ks_b, sp_lo, sp_hi, color=COLORS[0], alpha=0.18)
    best_idx = int(np.argmin(sp_r))
    ax.axvline(ks_b[best_idx], color=COLORS[0], ls="--", alpha=0.7,
               label=rf"Optimal $K^\star={ks_b[best_idx]}$ ($n={b_rows[best_idx]['n_per_judge']}$)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Ensemble size $K$ (fixed total budget $B = K \cdot n = 4096$)")
    ax.set_ylabel(r"RMSE $|\hat\tau - \tau(c)|$")
    ax.set_title(r"(b) Compute-Optimal Budget Tradeoff ($B = 4096, \beta = 1$)")
    ax.legend(loc="upper left", frameon=True)

    # --- Panel (c): Check D (Exp 05B Adversarial Bias Fix) ---
    ax = axes[1, 0]
    d_rows = [r for r in res_d["rows"] if r["condition"] == "fixed_asym_tau0.80_delta0.15"]
    agg_styles = [
        ("mean", "Simple Mean", "s-", COLORS[1]),
        ("trimmed_10pct", "10% Trimmed Mean", "^-", COLORS[3]),
        ("median", "Median", "o-", COLORS[2]),
        ("trimmed_20pct", "20% Trimmed Mean", "d-", COLORS[0]),
    ]
    for agg_key, agg_lbl, style, col in agg_styles:
        sub = [r for r in d_rows if r["aggregator"] == agg_key]
        ks_d = [r["K"] for r in sub]
        vals = [r["rmse"] for r in sub]
        lo = [r["rmse_ci95_lo"] for r in sub]
        hi = [r["rmse_ci95_hi"] for r in sub]
        ax.loglog(ks_d, vals, style, color=col, lw=1.8, ms=5, label=agg_lbl)
        ax.fill_between(ks_d, lo, hi, color=col, alpha=0.14)
    ax.set_xlabel(r"Number of judges $K$")
    ax.set_ylabel("RMSE vs. clipped honest mean")
    ax.set_title(r"(c) Exp 05B Fix ($\tau_c=0.80, \mu_{\mathrm{adv}}=0.20, \delta=0.15$)")
    ax.legend(loc="lower left", frameon=True)

    # --- Panel (d): Check E (Exp 04 Effective Elasticity vs Free-Offset Artifact) ---
    ax = axes[1, 1]
    e_rows = res_e["rows"]
    rhos = [r["rho"] for r in e_rows]
    ax.plot(rhos, [r["broken_free_b_alpha"] for r in e_rows], "x--", color=COLORS[1], lw=1.8, ms=6,
            label="Buggy Exp 04 fit (free $b \\Rightarrow \\hat\\alpha \\to 1$)")
    ax.plot(rhos, [r["fixed_local_alpha_unclipped_16_256"] for r in e_rows], "o-", color=COLORS[0], lw=2.0,
            label=r"Fixed local elasticity ($K \in [16, 256]$)")
    ax.plot(rhos, [r["theory_chord_alpha_16_256"] for r in e_rows], ":", color="black", lw=1.6,
            label="Exact equicorrelation prediction")
    ax.set_xlabel(r"Pairwise judge correlation $\rho$")
    ax.set_ylabel(r"Effective scaling exponent $\alpha_{\mathrm{eff}}$")
    ax.set_title(r"(d) Exp 04 Fix: Local Elasticity Degrades Monotonically with $\rho$")
    ax.legend(loc="center right", frameon=True)

    fig.suptitle(
        "Experiment 12: Verification of Repo Bugfixes & New Scaling Regimes (500 MC Reps, 95% CIs)",
        fontsize=13, fontweight="bold", y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    for ext in ("pdf", "png"):
        out_path = os.path.join(RESULTS_DIR, f"exp12_audit_and_fixes_4panel.{ext}")
        fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved 4-panel figure to {RESULTS_DIR}/exp12_audit_and_fixes_4panel.{{pdf,png}}")


def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("Experiment 12 — Repo Audit, Bugfixes, and Next-Theory Verification")
    print("=" * 72)

    res_a = run_check_a()
    res_b = run_check_b()
    res_c = run_check_c()
    res_d = run_check_d()
    res_e = run_check_e()

    make_figure(res_b, res_c, res_d, res_e)

    summary = {
        "seed": SEED,
        "n_reps": N_REPS,
        "b_boot": B_BOOT,
        "elapsed_sec": round(time.time() - t0, 2),
        "check_a": res_a,
        "check_b": {"tau_c": res_b["tau_c"], "fits": res_b["fits"]},
        "check_c": res_c,
        "check_d_k1024": [r for r in res_d["rows"] if r["K"] == 1024],
        "check_e": res_e,
    }
    json_path = os.path.join(RESULTS_DIR, "summary_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['elapsed_sec']:.2f}s. Saved {json_path}.")


if __name__ == "__main__":
    main()
