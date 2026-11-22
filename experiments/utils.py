"""utils.py — shared utilities across all TVD-MI scaling experiments.

Provides:
- Joint distribution construction (discretized bivariate Gaussian)
- True TVD computation
- Optimal critic computation
- Power-law and exponential decay fitting
- Consistent matplotlib styling
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib

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

COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
          '#00BCD4', '#795548', '#607D8B']

K_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_REPS = 500


def compute_true_tvd(P_UV, P_U, P_V):
    """Compute TV(P_UV, P_U ⊗ P_V) on a discrete grid."""
    P_indep = np.outer(P_U, P_V)
    return 0.5 * np.sum(np.abs(P_UV - P_indep))


def optimal_critic(P_UV, P_U, P_V):
    """h*(u,v) = 1[p_uv(u,v) > p_u(u)*p_v(v)]."""
    P_indep = np.outer(P_U, P_V)
    return (P_UV > P_indep).astype(float)


def make_joint_distribution(d=20, correlation=0.5):
    """Create a discretized bivariate distribution with known dependence.

    Returns (P_UV, P_U, P_V) as numpy arrays on a d×d grid.
    """
    cov = [[1, correlation], [correlation, 1]]
    rv = multivariate_normal(mean=[0, 0], cov=cov)
    edges = np.linspace(-3, 3, d + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    P_UV = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            P_UV[i, j] = rv.pdf([centers[i], centers[j]])
    P_UV /= P_UV.sum()
    P_U = P_UV.sum(axis=1)
    P_V = P_UV.sum(axis=0)
    return P_UV, P_U, P_V


def compute_delta(h, P_UV, P_U, P_V):
    """Compute Δ_c(h) = E_{P_UV}[h] - E_{P_U⊗P_V}[h] for a critic h."""
    P_indep = np.outer(P_U, P_V)
    return np.sum(h * P_UV) - np.sum(h * P_indep)


def fit_power_law(K_vals, errors, allow_negative_alpha=False):
    """Fit error = a * K^(-alpha) + b. Return (alpha, a, b, R²)."""
    K_arr = np.array(K_vals, dtype=float)
    e_arr = np.array(errors, dtype=float)
    mask = e_arr > 0
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan

    def model(K, a, alpha, b):
        return a * K ** (-alpha) + b

    alpha_lo = -5.0 if allow_negative_alpha else 0.0
    try:
        popt, _ = curve_fit(model, K_arr[mask], e_arr[mask],
                            p0=[1.0, 0.5, 0.0], maxfev=10000,
                            bounds=([0, alpha_lo, -1], [100, 5, 1]))
        a, alpha, b = popt
        ss_res = np.sum((e_arr[mask] - model(K_arr[mask], *popt))**2)
        ss_tot = np.sum((e_arr[mask] - e_arr[mask].mean())**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return alpha, a, b, R2
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def fit_loglog_power_law(K_vals, errors, k_min=8):
    """Fit log(error) = log(a) - alpha * log(K) for K >= k_min (no free offset artifact).
    Returns (alpha, a, 0.0, R²)."""
    K_arr = np.asarray(K_vals, dtype=float)
    e_arr = np.asarray(errors, dtype=float)
    mask = (K_arr >= k_min) & (e_arr > 0)
    if mask.sum() < 3:
        mask = e_arr > 0
    if mask.sum() < 2:
        return np.nan, np.nan, 0.0, np.nan
    lk = np.log(K_arr[mask])
    le = np.log(e_arr[mask])
    slope, intercept = np.polyfit(lk, le, 1)
    le_hat = slope * lk + intercept
    ss_res = np.sum((le - le_hat) ** 2)
    ss_tot = np.sum((le - le.mean()) ** 2)
    R2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
    return float(-slope), float(np.exp(intercept)), 0.0, R2


def compute_local_elasticity(K_vals, rmse_vals, k_lo=16, k_hi=256):
    """Exact unclipped local scaling elasticity -d(log RMSE)/d(log K) across [k_lo, k_hi]."""
    K_arr = np.asarray(K_vals, dtype=float)
    e_arr = np.asarray(rmse_vals, dtype=float)
    idx_lo = int(np.argmin(np.abs(K_arr - k_lo)))
    idx_hi = int(np.argmin(np.abs(K_arr - k_hi)))
    if idx_hi == idx_lo or e_arr[idx_lo] <= 0 or e_arr[idx_hi] <= 0:
        return np.nan
    return float(-np.log(e_arr[idx_hi] / e_arr[idx_lo]) / np.log(K_arr[idx_hi] / K_arr[idx_lo]))


def fit_exponential_decay(K_vals, errors):
    """Fit error = a * exp(-b*K) + c. Return (b, a, c, R²)."""
    K_arr = np.array(K_vals, dtype=float)
    e_arr = np.array(errors, dtype=float)

    def model(K, a, b, c):
        return a * np.exp(-b * K) + c

    try:
        popt, _ = curve_fit(model, K_arr, e_arr,
                            p0=[e_arr[0], 0.01, 0.0], maxfev=10000,
                            bounds=([0, 0, -1], [100, 10, 1]))
        a, b, c = popt
        ss_res = np.sum((e_arr - model(K_arr, *popt))**2)
        ss_tot = np.sum((e_arr - e_arr.mean())**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return b, a, c, R2
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
