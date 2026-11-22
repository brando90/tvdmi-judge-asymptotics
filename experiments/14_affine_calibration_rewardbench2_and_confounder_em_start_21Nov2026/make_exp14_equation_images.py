"""Render standalone LaTeX equations (220 DPI) for Experiment 14 Google Doc companion."""

from pathlib import Path
import shutil
import subprocess

EXP14_DIR = Path(__file__).resolve().parent
BUILD_DIR = EXP14_DIR / "_eq_build"
OUT_DIR = EXP14_DIR / "doc_equation_images_11-21-2026"
PDFLATEX = "/usr/local/texlive/2024/bin/universal-darwin/pdflatex"
PDFTOPPM = "pdftoppm"

EQUATIONS = {
    "eq01_icc_gap_and_affine_calibration": r"""
\mathrm{ICC}(A,1)
  = \frac{2\,\mathrm{Cov}(\widehat{Y}, Y)}{\mathrm{Var}(\widehat{Y}) + \mathrm{Var}(Y) + (\mu_{\widehat{Y}} - \mu_Y)^2}
  < \frac{2\,\mathrm{Cov}(\widehat{Y}, Y)}{\mathrm{Var}(\widehat{Y}) + \mathrm{Var}(Y)}
  = \mathrm{ICC}(C,1)
  \quad (\mu_{\widehat{Y}} \neq \mu_Y),
\\[6pt]
\widehat{Y}_i^{(m)} = \mathrm{clip}\!\big(\hat{a}_m Z^{\mathrm{RB}}_i + \hat{b}_m,\, 1, 5\big),
\qquad
\mathrm{Excess\,MSE}\!\left(\hat{b}_m\right)
  = \frac{\sigma_{\mathrm{cal}}^2}{m}
  \le \frac{B}{m} \xrightarrow[m\to\infty]{} 0.
""",
    "eq02_platt_probability_calibration": r"""
\hat{p}_m(y \succ y' \mid x)
  = \sigma\!\big(\hat{\alpha}_m \, s_{\mathrm{RB}}(x, y, y') + \hat{\beta}_m\big),
\qquad
\sigma(z) = \frac{1}{1 + e^{-z}},
\\[6pt]
\mathrm{ECE}_B = \sum_{b=1}^B \frac{|I_b|}{N}\big|\mathrm{acc}(I_b) - \mathrm{conf}(I_b)\big|,
\qquad
\mathrm{Brier} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2.
""",
    "eq03_cross_family_confounder_floor": r"""
S_{f,j} = \mu + c\,W + u_f + e_{f,j}
\qquad
\big(\mathrm{Var}(W) = v_W > 0,\;\; c \neq 0\big),
\\[6pt]
\mathrm{Var}\!\left(\bar{S}^{\mathrm{bal}}_{F,\{K_f\}}\right)
  = c^2 v_W + \frac{1}{F^2}\sum_{f=1}^F \left(\tau_f^2 + \frac{\sigma_f^2}{K_f}\right)
  \;\ge\; c^2 v_W \;>\; 0
  \qquad \forall\, F \ge 1.
""",
    "eq04_confounder_residualization_cr_rbspec": r"""
\widetilde{Z}^{\,\mathrm{CR}}
  = \big(I - P_{W,\mathrm{train}}\big)\,\widetilde{Z},
\qquad
P_{W,\mathrm{train}} = W_{\mathrm{tr}}\!\left(W_{\mathrm{tr}}^\top W_{\mathrm{tr}}\right)^{+} W_{\mathrm{tr}}^\top,
\\[6pt]
\mathrm{Var}\!\left(\bar{S}^{\mathrm{CR-RB}}\right)
  = (c - \hat{c})^2 v_W + \frac{1}{F^2}\sum_{f=1}^F \left(\tau_f^2 + \frac{\sigma_f^2}{K_f}\right)
  \;\xrightarrow{\hat{c} = c}\;
  \frac{1}{F^2}\sum_{f=1}^F \left(\tau_f^2 + \frac{\sigma_f^2}{K_f}\right) \le \frac{B}{F}.
""",
    "eq05_davis_kahan_spectral_perturbation": r"""
Q(u_1) = \lambda_1,\quad
Q(\hat{u}) \le \alpha^2 \lambda_1 + \sin^2\theta\,\lambda_2,
\quad
\big|\widehat{Q}(u) - Q(u)\big| \le \varepsilon,\quad
\widehat{Q}(\hat{u}) \ge \widehat{Q}(u_1),
\\[6pt]
(\lambda_1 - \lambda_2)\,\sin^2\theta \le 2\varepsilon
\quad\Longrightarrow\quad
\sin^2\theta(\hat{u}, u_1)
  = \sum_{j=1}^K w_j^2
  \;\le\; \frac{2\varepsilon}{\lambda_1 - \lambda_2}.
""",
    "eq06_davis_kahan_sample_rate": r"""
\varepsilon_n \le \frac{C}{\sqrt{n}},\;\; \gamma_{\mathrm{gap}} := \lambda_1 - \lambda_2 > 0
\;\Longrightarrow\;
\sin^2\theta(\hat{u}_n, u_1)
  \le \frac{2C}{\gamma_{\mathrm{gap}}\sqrt{n}}
  \xrightarrow[n\to\infty]{} 0,
\quad
\text{spiked covariance rate: }\sin^2\theta = O_p(1/n).
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
        tex_path.write_text(TEMPLATE.replace("__BODY__", body.strip()), encoding="utf-8")
        subprocess.run(
            [PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=BUILD_DIR,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pdf_path = BUILD_DIR / f"{name}.pdf"
        out_prefix = OUT_DIR / name
        subprocess.run(
            [PDFTOPPM, "-png", "-r", "220", "-singlefile", str(pdf_path), str(out_prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        out_png = OUT_DIR / f"{name}.png"
        print(f"Rendered {out_png.name} ({out_png.stat().st_size} bytes)")

    shutil.rmtree(BUILD_DIR)


if __name__ == "__main__":
    main()
