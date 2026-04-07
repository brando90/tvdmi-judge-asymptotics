# Paper LaTeX Sources

| Submission | Main file |
|---|---|
| NeurIPS 2025 | `neurips/main.tex` |

## Overleaf-style workflow in Cursor

1. Install **MacTeX**: `brew install --cask mactex`
2. Install the **LaTeX Workshop** extension in Cursor
3. Open a `.tex` file — **save to compile**, `Cmd+Alt+V` to open PDF preview side-by-side

That's it. Edit → save → PDF updates automatically, just like Overleaf.

> **Tip:** If LaTeX Workshop doesn't find the root file, open the root `.tex` first or add `% !TEX root = main.tex` at the top of any sub-file.

## Manual compilation

```bash
cd neurips
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
