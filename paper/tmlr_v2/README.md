# TMLR-format FYP draft — restructured revision (v2)

A revision of `../tmlr` on the standard TMLR section layout. No number, claim, or artifact
reference was changed anywhere; the work was structural and editorial.

## Build

```bash
cd paper/tmlr_v2
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Layout

| § | Title | Pages | Source in `../tmlr` |
|---|---|---|---|
| — | Abstract | — | unchanged |
| 1 | Introduction | 1–3 | §1; contributions as prose, not a bullet list |
| 2 | Related Work | 3 | §2; no sub-headings, one continuous argument |
| 3 | Background and Preliminaries | 3–4 | §3; no subsections, continuous |
| 4 | Method | 4–9 | §4; prose unchanged, subsections merged |
| 5 | Experiments | 9–11 | §5 + §6 + empirical half of §7 |
| 6 | Discussion and Limitations | 11–13 | discussion half of §7 + §8 + Broader Impact |
| 7 | Conclusion | 13 | §9 |
| | References | 13–15 | |
| A–E | Appendix | 15–20 | as before, plus material moved down from the body |

Main text ≈ 12.4 pages, total 20 pages. The original draft was 23 pages with an 18-page body.

## Section 4

Prose is byte-identical to `../tmlr/sections/04_method.tex` (verified by diff); only the headers
changed. The four `\subsubsection`s under 4.2 became running paragraphs, algebra and
deterministic execution share one subsection, understanding/planning/grounding share one, and
release checks with bounded repair share one — six subsections, no subsubsections, down from nine
plus four.

Reaching 12.0 pages would mean moving streaming ingestion, role-aware entity resolution, and fact
extraction into an appendix (~1.4 pages). Not applied: Section 4 is kept complete in the body.

## What moved to the appendix

- Graph node/edge statistics table → Appendix A, same label `tab:dataset-stats`.
- The `2x2` verifier-vs-oracle diagnostic and the family-by-depth recommendation → Appendix D.
- The type-strict comparator rule (Boolean vs numeric) → Appendix B.
- PACS sealing seed and gate implementation stay in Appendix C; the body keeps only the two
  weaknesses that change how the results may be read.
- The per-category full-runtime table stays in Appendix D; the body reports it inline.

## Editorial pass

The guarantee-boundary caveat (static valid ≠ runtime verified ≠ oracle match) appeared in seven
places in the previous draft. It is defined once in Section 3, applied where it is measurable in
Section 5.2, and closed once in Section 7.

Section 5.1 now describes the **training** export (size, acceptance rules, its duplicate/overlap
defect) before the benchmark, so the sealing checks in the next paragraph have a referent. The
previous draft described only PACS.

Limitations is continuous prose rather than `\paragraph` tags. Across the rewritten sections,
`therefore` fell 13→1, `consequently`/`nevertheless`/`not merely`/em-dash asides to 0, and the
"X is not P, it is Q" cadence 9→3; sentence length now varies (sd 9.1→10.7, longest 89w→60w)
instead of sitting flat. The counts that remain are all inside the untouched Section 4.

## Draft status

- Results are tied to `results_macros.tex` and the artifacts in Appendix A.
- Figures are placeholders; `FIGURE_TABLE_BRIEF.zh-CN.md` specifies their content.
- Before submission: clean code tag, strict typed-answer comparator, one exported current-runtime
  bridge trace, tables generated from artifacts, every `TODO`/placeholder replaced.
