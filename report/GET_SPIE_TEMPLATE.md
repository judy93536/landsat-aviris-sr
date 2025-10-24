# Getting SPIE LaTeX Template Files

You need two files from SPIE to compile the document:

## Required Files

1. **spieman.cls** - SPIE manuscript document class
2. **spiejour.bst** - SPIE journal bibliography style

## Download Instructions

### Option 1: Official SPIE Website

1. Visit: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/authors
2. Scroll to "Templates and Style Guides"
3. Download "LaTeX Manuscript Template"
4. Extract `spieman.cls` and `spiejour.bst`
5. Place them in this `report/` directory

### Option 2: Direct Links (if available)

- https://www.spiedigitallibrary.org/documents/SPIE_LaTeX_class_file.zip

### Verification

After downloading, your `report/` directory should contain:

```
report/
├── main.tex
├── references.bib
├── spieman.cls       ← SPIE class file
├── spiejour.bst      ← SPIE bib style
├── README.md
├── .gitignore
└── figures/
```

## Alternative: Use Standard Article Class

If you can't get the SPIE template, you can temporarily use the standard article class:

Change line 2 in `main.tex` from:
```latex
\documentclass[]{spieman}
```

To:
```latex
\documentclass[11pt,letterpaper]{article}
```

Then compile with:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Note**: This will use standard article formatting, not SPIE format. You'll need to reformat before submission.

## License Note

The SPIE template files are copyrighted by SPIE. They are provided for preparing manuscripts for SPIE publications. Do not redistribute the template files - users should download them directly from SPIE.
