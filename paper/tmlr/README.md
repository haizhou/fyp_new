# TMLR-format FYP draft

This directory contains a new code-and-artifact-driven dissertation draft. It does not use
`docs/paper_draft_v1.md` as evidence or source text.

## Build

```bash
cd paper/tmlr
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The project vendors the official TMLR `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, and licence. The
paper uses `\usepackage[preprint]{tmlr}` because this is an attributed FYP draft, not an accepted
TMLR paper. Replace the placeholder author, affiliation, and email before submission.

## Draft status

- The abstract, introduction, related work, problem setup, methodology, experiments, results,
  analysis, limitations, conclusion, ethics statement, references, and five appendices are
  drafted.
- Results are tied to `results_macros.tex` and repository artifacts listed in Appendix A.
- Figures are intentionally placeholders. Earlier generated figure experiments in this working
  tree are not referenced by `main.tex`.
- `FIGURE_TABLE_BRIEF.zh-CN.md` tells the author exactly what each figure/table should contain and
  which official paper figures are useful references.
- Before a final submission, create a clean code tag, apply the strict typed-answer comparator,
  export one complete current-runtime bridge trace, generate tables from artifacts automatically,
  and replace every `TODO`/placeholder.

## Source layout

```text
main.tex
results_macros.tex
sections/01_introduction.tex ... 09_conclusion.tex
appendix/a_reproducibility.tex ... e_examples.tex
references.bib
FIGURE_TABLE_BRIEF.zh-CN.md
```
