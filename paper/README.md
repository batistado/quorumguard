# QuorumGuard — 2027 IEEE International Systems Conference (SysCon) Manuscript

Contains the 8-page IEEE double-column conference manuscript (`main.tex` & compiled `main.pdf`, prepared for the SysCon 2027 Special Session on *Agentic AI for Autonomous Network Systems: Architecture, Verification, and Governance*), verified BibTeX database (`references.bib`), vector PDF figures (`figures/fig0..fig6.pdf`), static validator (`validate_latex.py`), and Overleaf submission bundle (`quorumguard_syscon2027_overleaf.zip`).

## Quick Commands

### 1. Regenerate All Vector PDF Figures from Hardware Telemetry JSONs
```bash
python3 generate_figures.py
```

### 2. Run Static LaTeX, Citation, Label & Figure Validator
```bash
python3 validate_latex.py
```

### 3. Compile Conference Manuscript PDF
* **Overleaf (1-click):** Upload `quorumguard_syscon2027_overleaf.zip` directly to Overleaf (`pdfLaTeX`, main document `main.tex`).
* **TeX Live / MacTeX:**
  ```bash
  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
  ```
