# Manuscript: Inverse-Designed SiN Nanobeam Cavity for Rb Atom–Photon Transduction

PRX Quantum manuscript built on the RevTeX 4.2 template. Self-contained source +
figures, ready to upload to Overleaf.

## Files
- `main.tex` — the manuscript (`revtex4-2`, `prxquantum` journal substyle).
- `refs.bib` — bibliography (BibTeX, `apsrev4-2` style, selected by the class).
- `figures/fig1_device.pdf` … `fig4_performance.pdf` — the four figures.
- `make_paper_figures.py` — regenerates the figures from the project datasets.
- `build.sh` — local compile script.

## Compile locally
```
bash build.sh        # pdflatex -> bibtex -> pdflatex x2  ->  main.pdf
```
Requires a TeX distribution providing `revtex4-2.cls` and `apsrev4-2.bst`
(TeX Live `texlive-publishers`).

## Upload to Overleaf
Create a new project and upload `main.tex`, `refs.bib`, and the `figures/`
folder (Overleaf's RevTeX 4.2 supports the `prxquantum` substyle). Set the
compiler to pdfLaTeX. The author block, title, and any final wording are
placeholders to be edited.

## Regenerate figures
```
cd ..                # repo root
PYTHONPATH=. python paper/make_paper_figures.py
```
Reuses the verified datasets in `publication_ready/` and the optimization record
in `invDesResults/full_seed/`; no cloud simulations.

## Note on the journal substyle
If a local TeX install lacks the `prxquantum` substyle, change the
`\documentclass` options in `main.tex` from `prxquantum` to `prx` (or remove it
for generic APS). Overleaf and current TeX Live include `prxquantum`.
