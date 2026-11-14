# QuorumGuard — *Elsevier FGCS / IEEE TPDS* Archival Manuscript

Contains the two-column archival manuscript (`main.tex` & compiled `main.pdf`), Elsevier *Future Generation Computer Systems (FGCS)* Research Highlights (`highlights.tex` & `highlights.pdf`), verified BibTeX database (`references.bib`), 7 vector PDF figures (`figures/fig0..fig6.pdf`), static validator (`validate_latex.py`), and one-click Overleaf submission bundle (`quorumguard_fgcs_overleaf.zip`).

## Quick Commands

### 1. Regenerate All 7 Vector PDF Figures from Hardware Telemetry JSONs
```bash
python3 generate_figures.py
```

### 2. Run Static LaTeX, Citation, Label & Figure Validator
```bash
python3 validate_latex.py
```

### 3. Compile Manuscript & Highlights PDFs
* **Overleaf (1-click):** Upload `quorumguard_fgcs_overleaf.zip` directly to Overleaf (`pdfLaTeX`, main document `main.tex`).
* **TeX Live / MacTeX:**
  ```bash
  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
  pdflatex highlights.tex
  ```
