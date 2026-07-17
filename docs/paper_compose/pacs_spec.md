# PACS — Procurement Analytics Challenge Set (specification, pre-registered before generation)

Decision (user review, 2026-07-17): the legacy 2,285 paired run is DEMOTED to backward
compatibility (Set A). The paper's primary result comes from PACS (Set B): an
intent-first benchmark stratified by real analytical task families and composition
depth, with surfaces not produced by the training renderer. Sets C (structural OOD)
and D (robustness/boundary) retain their existing artifacts as mechanism and
shortcut audits. Headline numbers in abstract/conclusion cite Set B only.

## Two axes

### Axis 1 — task families (8, intent-first)
| family | archetype question |
|---|---|
| F1 spending & volume | how much did buyer X spend on category C in year Y? |
| F2 temporal change | was 2024 spending higher than 2023? by how much? |
| F3 supplier concentration | which supplier won the most awards / highest total? |
| F4 cross-buyer comparison | which of two buyers spent more on C? |
| F5 overlap & exclusion | suppliers serving both A and B; only A and never B |
| F6 relational composition | which other buyers do suppliers of buyer-class K serve? |
| F7 disclosure & compliance | which buyers never disclosed field D on their notices? |
| F8 unanswerable | ambiguous / empty / unsupported / requires-missing-operator |

### Axis 2 — composition depth
- L1 atomic: one filter + one reduction
- L2 simple: 2–3 operators (filtered sum; groupby+argext)
- L3 nested: semijoin/anti-join; per-side filtered comparison; set operations
- L4 structural OOD: full shape absent from training; constituent operators all seen
  (shape exclusion coordinated with the training exporter via shape signatures)

## Size and quotas
600–1,000 rows total. Per cell (family × applicable level): >= 25 answerable rows.
F8 = 120+ rows across its four subtypes. Entities/years/CPV/answer sizes varied per
cell; no anchor reused across more than 3 rows.

## Generation protocol (per row)
1. Intent instantiated from the family archetype (parameter sampling from the KG).
2. Gold tree authored per family×level template; dual-evaluator execution; any
   disagreement or degeneracy discards the row (existing machinery).
3. Surfaces, THREE channels per row:
   a. canonical-independent: second surface grammar, independently authored
      (no shared stems/connectors/ordering policy with the training renderer);
   b. naturalized: LLM rewrite behind deterministic fidelity gates (entities,
      numbers, logical slots verbatim-checked; abstention cues preserved) —
      ch4 gate machinery reused;
   c. training-renderer surface kept ONLY as a diagnostic column, never scored
      as the primary number.
4. Human audit: every template family reviewed; >= 5% of instances read; iron rule
   (10 deterministic-random rows per gate) applies to every acceptance gate.

## Scoring
Answerable: type-aware match vs dual-verified oracle (guided decoding, single call,
no repair). F8: dual metric — Status Exact Match primary; faithfulness-gated Safe
Semantic Outcome supplementary (implemented in run_compose_probe_eval.py).

## Main table of the paper (replaces iteration table as headline)
Rows = task families + macro average. Columns = L1 atomic | seen composition |
unseen composition (L4) | independent surface (channels a/b vs diagnostic c) |
abstention. Iteration curves and holdout cells move to Set C (mechanism section).

## Experiment sets after this decision
- A Legacy Regression (2,285 paired vs old champion; McNemar; retirement question)
- B PACS (THIS set; primary)
- C Structural OOD (B_clean, rotated construction holdouts, B_anchor when built)
- D Robustness & boundary (reorder/paraphrase/masked; out-of-grammar; abstention)

## Build order
1. Freeze this spec (+ any user edits) BEFORE generation.
2. Family×level tree templates (intent-first; reuse algebra + dual evaluators).
3. Independent surface grammar (second author-voice; no shared assets).
4. Naturalization gates; generation; audits; freeze PACS v1.
5. Evaluate compose-v3 (and base) on PACS; paper main table.
GPU needed only at step 5; steps 1–4 proceed during the driver outage.
