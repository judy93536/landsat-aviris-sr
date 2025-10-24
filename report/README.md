# Technical Report

SPIE-formatted technical report on deep learning-based Landsat-to-AVIRIS hyperspectral super-resolution.

## Prerequisites

You need to have the SPIE LaTeX template files:
- `spieman.cls` - SPIE manuscript class file
- `spiejour.bst` - SPIE journal BibTeX style

Download from: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/authors/author-instructions

Place these files in the `report/` directory.

## Building the Document

### Using pdflatex (recommended)

```bash
cd report/

# First pass
pdflatex main.tex

# Generate bibliography
bibtex main

# Second pass (resolve references)
pdflatex main.tex

# Third pass (resolve citations)
pdflatex main.tex
```

### Using latexmk (automatic)

```bash
cd report/
latexmk -pdf main.tex
```

### Clean auxiliary files

```bash
latexmk -c  # Clean most files
# or
latexmk -C  # Clean everything including PDF
```

## File Structure

```
report/
├── main.tex           # Main document
├── references.bib     # BibTeX references
├── figures/          # Figures directory (create as needed)
│   ├── architecture.pdf
│   ├── results_stage1.pdf
│   └── results_stage2.pdf
├── README.md         # This file
└── .gitignore       # LaTeX auxiliary files
```

## Adding Results

As training progresses, add:

1. **Architecture Diagrams**: Create figures showing network architectures
2. **Training Curves**: Export from TensorBoard
3. **Results Tables**: Quantitative metrics (PSNR, SSIM, SAM)
4. **Qualitative Comparisons**: Side-by-side visualizations

Use TikZ for architecture diagrams or export from draw.io/Inkscape as PDF.

## Key Sections to Complete

- [ ] Section 4: Results - Add training curves and metrics
- [ ] Section 4: Results - Add qualitative comparisons
- [ ] Section 4: Results - Ablation studies
- [ ] Section 5: Discussion - Analyze results
- [ ] Section 6: Conclusion - Summarize findings

## References

All key references are included in `references.bib`:
- RCAN architecture (Zhang et al., ECCV 2018)
- Spectral SR methods (Shi et al., 2019)
- SAM loss (Kruse et al., 1993)
- AVIRIS-NG instrument (Green et al., 2015)
- Landsat-8 OLI (Roy et al., 2014)

## Quick Reference Research

To find a specific paper:
```bash
grep -i "keyword" references.bib
```

To see all citations used:
```bash
grep "\\cite{" main.tex | sort | uniq
```

## Notes

- Document uses SPIE manuscript format (not proceedings format)
- Single-column layout
- Author affiliations use superscript letters
- Figures should be referenced as Fig. 1, Fig. 2, etc.
- Tables should be referenced as Table 1, Table 2, etc.
- All units in equations should be defined
