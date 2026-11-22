"""Render 220 DPI standalone PNG equation cards for Experiment 15 Google Doc & PDF."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "doc_equation_images_11-21-2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
})

CARDS = [
    (
        "eq01_srb_spec_deconfounding.png",
        "1. Stage-2 Signal Deconfounding (MultiFactorContinuous.lean: marginal_cov_confounds_signal_and_cloning)",
        r"$\mathrm{Cov}(S_j, S_k) = a_j a_k + \mathbf{1}_{\mathrm{fam}(j)=\mathrm{fam}(k)}\,\tau_{\mathrm{fam}(j)}^2 \quad\Longrightarrow\quad E_{\mathrm{res}} = C_{\mathrm{std}} - z_0\hat{a}^\top,\;\; \mathrm{Cov}(E_{\mathrm{res},j},E_{\mathrm{res},k})=\mathbf{1}_{\mathrm{fam}(j)=\mathrm{fam}(k)}\tau_{\mathrm{fam}(j)}^2$",
        "Off-family signal covariance a_j*a_k is removed before clone mass m_res is computed.",
        (11.0, 1.85),
    ),
    (
        "eq02_shrinkage_dominance.png",
        "2. Tempered Precision & Interior Shrinkage Dominance (SRB-Spec, interior_shrinkage_strictly_dominates_both)",
        r"$w_j(\alpha) = \frac{\hat{a}_j}{((1-\alpha)+\alpha\max(1-\hat{a}_j^2,\tau_0))\,m_{\mathrm{res},j}\,s_j},\qquad \lambda^\star = \frac{v_U - c_{UB}}{v_U+v_B-2c_{UB}} \;\Longrightarrow\; R(\lambda^\star) < \min(v_U, v_B)$",
        "Closes clean RewardBench 2 gap (0.831-0.836) while preserving clone/polarity robustness (0.832-0.837).",
        (11.0, 1.95),
    ),
    (
        "eq03_partial_proxy_floor.png",
        "3. Imperfect Proxy Covariate Law (partial_proxy_residual_confounder_variance, partial_proxy_strict_floor_when_imperfect)",
        r"$\mathrm{MSE}(\bar{S}^{\mathrm{CR-RB}}(r)) = (1 - r^2)\,c^2 v_W + \frac{1}{F^2}\sum_{f=1}^{F}(\tau_f^2 + \frac{\sigma_f^2}{K_f}) \;\leq\; (1 - r^2)\,c^2 v_W + \frac{B}{F}$",
        "When observed proxy correlation r^2 < 1, residual floor (1 - r^2) c^2 v_W remains strictly positive unless r^2 -> 1.",
        (11.0, 1.90),
    ),
    (
        "eq04_multifactor_rank_r_dk.png",
        "4. Multi-Factor Rank-r Spectral Perturbation (multifactor_rayleigh_trace_sin_sq_bound, multifactor_subspace_rate_vanishes)",
        r"$\sum_{k=1}^{r}\sin^2\theta_k = r - \left\|U_r^\top \widehat{U}_r\right\|_F^2 \;\leq\; \frac{2\,r\,\varepsilon}{\lambda_r - \lambda_{r+1}} = O\!\left(\frac{r}{\sqrt{n}}\right) \longrightarrow 0$",
        "Reduces identically to rank-1 bound 2*eps/(lambda_1 - lambda_2) at r = 1; 0 violations across 10,500 RB2 draws.",
        (11.0, 1.85),
    ),
    (
        "eq05_continuous_measure_tvd.png",
        "5. Continuous Measure-Theoretic TVD Attainment (measure_tvd_event_witness_attainment)",
        r"$\left|\int_S 1\,\mathrm{d}P - \int_S 1\,\mathrm{d}Q\right| = \left|P_{\mathbb{R}}(S) - Q_{\mathbb{R}}(S)\right|, \qquad P_{\mathbb{R}}(S) - Q_{\mathbb{R}}(S) = Q_{\mathbb{R}}(S^c) - P_{\mathbb{R}}(S^c)$",
        "Machine-checked in Lean 4 Mathlib MeasureTheory on arbitrary measurable spaces (alpha, MeasurableSpace alpha).",
        (11.0, 1.85),
    ),
    (
        "eq06_partition_refinement_tvd.png",
        "6. Finite Partition Refinement Monotonicity (partition_tvd_refinement_monotone)",
        r"$\frac{1}{2}\sum_{b=1}^{B}\left|\sum_{j=1}^{M}p_{b,j} - \sum_{j=1}^{M}q_{b,j}\right| \;\leq\; \frac{1}{2}\sum_{b=1}^{B}\sum_{j=1}^{M}\left|p_{b,j} - q_{b,j}\right|$",
        "Coarser judge discretizations never overestimate fine-grained or continuous Total Variation Distance.",
        (11.0, 1.80),
    ),
]


def main() -> None:
    for fname, title, formula, note, figsize in CARDS:
        fig = plt.figure(figsize=figsize, dpi=220)
        fig.patch.set_facecolor("#fbfcfe")
        fig.text(0.03, 0.80, title, fontsize=10.5, fontweight="bold", color="#0b3d91", family="sans-serif")
        fig.text(0.50, 0.45, formula, fontsize=14.0, ha="center", va="center", color="#111111")
        fig.text(0.03, 0.12, note, fontsize=8.8, color="#333333", family="sans-serif")
        fig.savefig(OUT_DIR / fname, dpi=220, facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"Rendered {len(CARDS)} equation cards to {OUT_DIR}")


if __name__ == "__main__":
    main()
