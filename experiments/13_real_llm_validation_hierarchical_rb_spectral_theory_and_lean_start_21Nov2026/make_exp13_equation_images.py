"""Render standalone LaTeX equations (220 DPI) for Experiment 13 Google Doc companion."""

from pathlib import Path
import shutil
import subprocess

EXP13_DIR = Path(__file__).resolve().parent
BUILD_DIR = EXP13_DIR / "_eq_build"
OUT_DIR = EXP13_DIR / "doc_equation_images_11-21-2026"
PDFLATEX = "/usr/local/texlive/2024/bin/universal-darwin/pdflatex"
PDFTOPPM = "pdftoppm"

EQUATIONS = {
    "eq01_variational_tvd_and_injection": r"""
\mathrm{TV}(P_{X,Y}, P_X \otimes P_Y)
  = \max_{h:\,\mathcal{U}\times\mathcal{V}\to[0,1]}
    \Big(\mathbb{E}_{P_{X,Y}}[h] - \mathbb{E}_{P_X \otimes P_Y}[h]\Big),
\\[6pt]
Y = f(X)\;\;(f\text{ injective})
\;\Longrightarrow\;
\mathrm{TV}(P_{X,f(X)}, P_X \otimes P_{f(X)})
  = 1 - \sum_{x \in \mathcal{U}} P_X(x)^2.
""",
    "eq02_double_centering_invariance": r"""
h(x,y) = a(x) + b(y) + \gamma\,\psi(x,y)
\;\Longrightarrow\;
\Delta(h) := \mathbb{E}_{P_{XY}}[h] - \mathbb{E}_{P_X \otimes P_Y}[h]
= \gamma\,\Delta(\psi).
""",
    "eq03_oriented_peer_witness": r"""
\Delta^{\mathrm{orient}}_{c,j}
  := \mathbb{E}\!\big[
      \big(\mathbf{1}\{W_j = 1\} - \mathbf{1}\{W_j = 0\}\big)
      \big(Y_c - \widetilde{Y}_c\big)\big]
  = 2(2q_j - 1)(2p_c - 1)\,\mathrm{Var}(Z^\star),
""",
    "eq04_clone_hijacking_variance": r"""
\mathrm{Var}\!\left(\bar{S}^{\mathrm{unif}}_{K_1, K_2}\right)
  = \frac{K_1^2 \tau_1^2 + K_1 \sigma_1^2 + K_2^2 \tau_2^2 + K_2 \sigma_2^2}
         {(K_1 + K_2)^2}
  \;\xrightarrow[K_1 \to \infty]{}\; \tau_1^2
\quad\text{(drowns diverse family 2)}.
""",
    "eq05_rb_spec_algorithm_and_1_over_F": r"""
m_j = \sum_{j'=1}^K \big|\widehat{R}_{j j'}\big|^6, \quad
d_j = m_j^{-1/2}, \quad
Z^{\mathrm{bal}}_{i,j} = d_j \widetilde{Z}_{i,j}, \quad
w_j^{\mathrm{RB}} \propto d_j \, v_{1,j}\!\left(Z^{\mathrm{bal}}\right),
\\[6pt]
\mathrm{Var}\!\left(\bar{S}^{\mathrm{bal}}_{F,\{K_f\}}\right)
  = \frac{1}{F^2}\sum_{f=1}^F \left(\tau_f^2 + \frac{\sigma_f^2}{K_f}\right)
  \le \frac{B}{F} \;\xrightarrow[F\to\infty]{}\; 0
  \quad\forall\, K_1,\dots,K_F \ge 1.
""",
    "eq06_split_sample_thm3prime": r"""
\mathbb{P}\!\left(\left|\hat{\tau}^{\mathrm{split}}_{K,n}(c) - \tau(c)\right|
  > \varepsilon + 2\eta_{\mathrm{sel}} + \eta_{\mathrm{eval}}\right)
  \le e^{-K p_\varepsilon(c)}
    + 2K e^{-n_{\mathrm{sel}}\eta_{\mathrm{sel}}^2}
    + 2 e^{-n_{\mathrm{eval}}\eta_{\mathrm{eval}}^2},
\\[6pt]
K^\star(B) \asymp \left(\frac{B}{\log B}\right)^{\frac{\beta}{\beta+2}},
\qquad
\mathrm{Error}^\star(B) \asymp \left(\frac{\log B}{B}\right)^{\frac{1}{\beta+2}}.
""",
    "eq07_effective_family_breakdown": r"""
\text{Balanced family vote:}\quad
(1 - \delta_{\mathrm{fam}})\!\left(\tfrac{1}{2}+\gamma\right)
  + \delta_{\mathrm{fam}}\!\left(\tfrac{1}{2}-\gamma\right) > \tfrac{1}{2}
\;\Longleftrightarrow\;
\delta_{\mathrm{fam}} < \frac{1}{2},
\\[6pt]
\text{Confounded consensus floor:}\quad
\lim_{K \to \infty}\mathbb{P}\big(b_K(Z)\neq h^\star(Z)\big)
  = \mathbb{P}\!\left(Q(W) < \tfrac{1}{2}\right)
    + \tfrac{1}{2}\mathbb{P}\!\left(Q(W) = \tfrac{1}{2}\right).
""",
    "eq08_cauchy_schwarz_optimal_weights": r"""
\sum_{j=1}^K w_j \mu_j = 1,\;\; v_j > 0
\;\Longrightarrow\;
\sum_{j=1}^K w_j^2 v_j \;\ge\; \left(\sum_{j=1}^K \frac{\mu_j^2}{v_j}\right)^{-1},
\qquad
w_j^\star \propto \frac{\mu_j}{v_j}.
""",
}

TEMPLATE = r"""\documentclass[border=10pt,varwidth=18cm]{standalone}
\usepackage{amsmath,amssymb,amsthm,bm}
\begin{document}
\[
\begin{gathered}
__BODY__
\end{gathered}
\]
\end{document}
"""


def main() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, body in EQUATIONS.items():
        tex_path = BUILD_DIR / f"{name}.tex"
        pdf_path = BUILD_DIR / f"{name}.pdf"
        prefix = OUT_DIR / name
        tex_path.write_text(TEMPLATE.replace("__BODY__", body.strip()), encoding="utf-8")
        res = subprocess.run(
            [PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"LaTeX failed for {name}:\n{res.stdout}\n{res.stderr}")
        res2 = subprocess.run(
            [PDFTOPPM, "-png", "-r", "220", "-singlefile", str(pdf_path), str(prefix)],
            capture_output=True,
            text=True,
        )
        if res2.returncode != 0:
            raise RuntimeError(f"pdftoppm failed for {name}: {res2.stderr}")
        png_path = Path(f"{prefix}.png")
        print(f"Rendered {png_path.name} ({png_path.stat().st_size} bytes)")

    shutil.rmtree(BUILD_DIR)


if __name__ == "__main__":
    main()
