#!/usr/bin/env python3
"""Render typeset LaTeX equation PNGs for Google Doc companion (Trigger Rule 91)."""

import os
import subprocess
import tempfile
from PIL import Image, ImageOps

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "doc_equation_images_11-21-2026"
)
os.makedirs(OUT_DIR, exist_ok=True)

PDFLATEX = "/usr/local/texlive/2024/bin/universal-darwin/pdflatex"
PDFTOPPM = "/opt/homebrew/bin/pdftoppm"

EQUATIONS = {
    "eq01_beta_witness_exact": r"""
\mathbb{P}\!\left(Q_j \ge 1 - u\right) = u^\beta \quad \big(Q_j \sim \mathrm{Beta}(1,\beta)\big), \\
\mathbb{E}\!\left[\tau(c) - \hat\tau_K^{\max}(c)\right]
= \tau(c)\,\frac{\Gamma(K+1)\,\Gamma(1 + 1/\beta)}{\Gamma(K + 1 + 1/\beta)}
\;\sim\; \tau(c)\,\Gamma(1+1/\beta)\,K^{-1/\beta}
""",
    "eq02_hoeffding_null_overfit": r"""
\mathbb{P}\!\left(\left|\widehat\Delta_c(h) - \Delta_c(h)\right| > \eta\right) \le 2 e^{-n\eta^2}, \\
\tau(c)=0 \;\implies\; \max_{1 \le j \le K} \widehat\Delta_c(h_j) = \Omega_p\!\left(\sqrt{\frac{\log K}{n}}\right)
""",
    "eq03_split_sample_thm3prime": r"""
\mathbb{P}\!\left(\left|\hat\tau^{\mathrm{split}}_{K,n}(c) - \tau(c)\right| > \varepsilon + 2\eta_{\mathrm{sel}} + \eta_{\mathrm{eval}}\right)
\le e^{-K p_\varepsilon(c)} + 2K e^{-n_{\mathrm{sel}}\eta_{\mathrm{sel}}^2} + 2 e^{-n_{\mathrm{eval}}\eta_{\mathrm{eval}}^2}, \\
K^\star(B) \asymp \left(\frac{B}{\log B}\right)^{\frac{\beta}{\beta+2}}, \qquad
n^\star(B) \asymp B^{\frac{2}{\beta+2}}(\log B)^{\frac{\beta}{\beta+2}}, \qquad
\mathrm{Error}^\star(B) \asymp \left(\frac{\log B}{B}\right)^{\frac{1}{\beta+2}}
""",
    "eq04_local_diversity_elasticity": r"""
\mathrm{RMSE}(\bar S_K) = \sigma\sqrt{\rho + \frac{1-\rho}{K}}
= \sigma\sqrt{\rho} + \frac{\sigma(1-\rho)}{2\sqrt{\rho}}\,K^{-1} + O(K^{-2}), \\
\alpha_{\mathrm{eff}}(K) := -\frac{d \log \mathrm{RMSE}}{d \log K}
= \frac{1-\rho}{2\big(1 + (K-1)\rho\big)}
""",
    "eq05_huber_median_bias": r"""
\mathrm{Bias}\!\left(\bar S_K\right) = \delta\left(\mu_{\mathrm{adv}} - \tau(c)\right), \\
\mathrm{Bias}\!\left(\mathrm{Med}(S_{1:K})\right)
\xrightarrow{K\to\infty} -\sigma\,\Phi^{-1}\!\left(\frac{1}{2(1-\delta)}\right)
= -\sqrt{\frac{\pi}{2}}\,\frac{\sigma\,\delta}{2} + O(\delta^2) \quad (\mu_{\mathrm{adv}} \ll \tau(c))
""",
    "eq06_bijection_peer_recovery": r"""
\mathrm{TV}\!\left(P_{X, f(X)},\, P_X \otimes P_{f(X)}\right) = 1 - \sum_y P_Y(y)^2, \\
\Delta^{\mathrm{orient}}_{c,j} := \mathbb{E}_{P_{V_c, W_j}}\!\left[\mathbf{1}(V_c = W_j)\right] - \mathbb{E}_{P_{V_c}\otimes P_{W_j}}\!\left[\mathbf{1}(V_c = W_j)\right]
= 2(2q_j - 1)(2p_c - 1)\,\mathrm{Var}(Z^\star)
""",
    "eq07_confounded_consensus_floor": r"""
Q(W) := \mathbb{P}\!\left(b_j(Z) = h^\star(Z) \mid Z, W\right) \;\; (\text{i.i.d.\ across } j \text{ given } (Z,W)), \\
\lim_{K \to \infty} \mathbb{P}\!\left(b_K(Z) \neq h^\star(Z)\right)
= \mathbb{P}\!\left(Q(W) < \tfrac{1}{2}\right) + \tfrac{1}{2}\mathbb{P}\!\left(Q(W) = \tfrac{1}{2}\right)
""",
}


def render_eq(name: str, latex_body: str) -> str:
    tex = rf"""\documentclass[border=10pt,12pt]{{standalone}}
\usepackage{{amsmath,amssymb,mathtools}}
\begin{{document}}
$\displaystyle \begin{{gathered}}
{latex_body.strip()}
\end{{gathered}}$
\end{{document}}
"""
    with tempfile.TemporaryDirectory() as tmp:
        tex_path = os.path.join(tmp, f"{name}.tex")
        pdf_path = os.path.join(tmp, f"{name}.pdf")
        png_base = os.path.join(tmp, name)
        with open(tex_path, "w") as f:
            f.write(tex)
        subprocess.run(
            [PDFLATEX, "-interaction=nonstopmode", f"{name}.tex"],
            cwd=tmp,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            [PDFTOPPM, "-r", "220", "-png", "-singlefile", pdf_path, png_base],
            cwd=tmp,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        raw_png = png_base + ".png"
        out_png = os.path.join(OUT_DIR, f"{name}.png")
        img = Image.open(raw_png).convert("RGB")
        bordered = ImageOps.expand(img, border=8, fill="white")
        bordered.save(out_png)
        return out_png


def main() -> None:
    for name, body in EQUATIONS.items():
        path = render_eq(name, body)
        print(f"Rendered {name} -> {path}")


if __name__ == "__main__":
    main()
