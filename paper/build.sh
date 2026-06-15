#!/usr/bin/env bash
# Compile the manuscript: pdflatex -> bibtex -> pdflatex x2.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
echo "==> built main.pdf"
