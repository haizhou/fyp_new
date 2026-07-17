# PACS — Procurement Analytics Challenge Set — specification v2.1 — FROZEN BEFORE GENERATION (2026-07-17; v2.1 size/naming amendment approved pre-generation)

Decision (user review, 2026-07-17): the legacy 2,285 paired run is DEMOTED to backward
compatibility (Set A). The paper's primary result comes from PACS (Set B). Sets C
(structural OOD) and D (robustness/boundary) retain existing artifacts as mechanism
and shortcut audits. Headline numbers in abstract/conclusion cite PACS-test only.

**Naming discipline.** PACS is *domain-motivated and intent-balanced*, not a measured
user distribution: the seven families are derived from procurement-oversight demands
(aggregation, change, concentration, comparison, overlap, relational chains,
disclosure), and per-family quotas are design choices. No claim of matching real
query frequencies is made; grounding quotas in an empirical needs analysis (e.g.
FOI-request or oversight-report mining) is listed as future work.

## Axis 1 — task families (7, intent-balanced)
| family | archetype |
|---|---|
| F1 spending & volume | total spend / notice count for buyer, category, year |
| F2 temporal change | year-over-year comparison and magnitude |
| F3 supplier concentration | top supplier by count or total value |
| F4 cross-buyer comparison | which of two buyers spent more on C |
| F5 overlap & exclusion | suppliers serving both A and B; only A, never B |
| F6 relational composition | which other buyers do suppliers of class K serve |
| F7 disclosure & compliance | buyers with no disclosed field D on any notice |

## Axis 2 — answerability status (crosses every family; F8 abolished as a family)
Each family's cells include answerable rows AND status variants where meaningful:
`answerable | ambiguous | empty_result | unsupported_field | requires_missing_operator`.
Status rows are built from the same intent templates (e.g. F1 with an ambiguous buyer
anchor; F5 with a pair matching zero suppliers; F2 requesting a median — missing
operator). Scoring for non-answerable rows: Status Exact Match primary,
faithfulness-gated Safe Semantic Outcome supplementary. Quota: >= 20% of each
family's rows are status variants, spanning >= 2 status types per family.

## Axis 3 — program complexity, SPLIT from structural exposure (two labels per row)
- `depth`: L1 atomic (one filter + one reduction) | L2 simple (2–3 operators) |
  L3 nested (semijoin/anti-join, per-side filtered comparison, set operations,
  group-wise combination).
- `exposure`: `seen` (shape signature present in training) | `unseen` (whole shape
  signature excluded from training via the exporter's shape-holdout coordination).
Exposure is orthogonal to depth: unseen cells exist at every depth, so "deeper"
and "never-shown" are separately attributable. (Replaces the former L4, which
conflated the two.)

## Surfaces — three PAIRED channels per row (same tree, same oracle)
Every row is ONE program rendered three ways; channel comparisons are within-row
paired measurements:
- a `canonical_independent`: second surface grammar, independently authored — no
  shared stems, connectors, or ordering policy with the training renderer;
- b `naturalized`: LLM rewrite behind deterministic fidelity gates. Literal
  gates (entities, numbers, abstention cues verbatim; ch4 machinery) PLUS logic
  gates: every gold tree exports a logical signature (operation kind, negation
  flags, aggregation, comparison direction, quantifier, left/right scopes) and
  the naturalized text must pass checks against it — operation-relation,
  negation-preservation, comparison-direction, quantifier, and scope-side checks
  (guards against both→either, only-A→A-or-B, greater→at-least, all→any,
  dropped negation, swapped comparison sides);
- c `training_renderer`: diagnostic column only, never a primary number.
Primary metric: channel a. Reported alongside: b, and paired deltas a−c, b−c
(per family and overall) as the surface-transfer measurement.

## Split discipline — dev/test, written hard
PACS is split at generation time by intent-instance (all three surfaces and all
status variants of one instance stay together): **PACS-dev ~20% (~220–270
clusters)** and **PACS-test ~80% (~880–1,080 clusters)**. Binding rules: (1) each frozen system configuration may produce one complete
confirmatory PACS-test result; technical reruns are permitted only when the
previous run is incomplete because of documented infrastructure or software
failure, and must use identical checkpoint, prompt, decoding, evaluator, and
scoring configuration (interruption cause recorded); PACS-test is never
inspected row-wise before freeze of the paper's numbers; (2) every diagnostic, error analysis, renderer fix, recipe change, or
model selection uses PACS-dev (or Sets C/D) only; (3) any post-hoc PACS-test
re-evaluation requires a new versioned system and is reported as such. This
extends the dissertation's dev/confirmatory discipline to the new benchmark.

## Size and quotas (v2.1)
Quotas are defined over applicable family × depth cells. Exposure is a
stratification label within each cell, NOT a second multiplicative quota. Each
applicable cell contains 40–50 answerable intent instances, including 15–20
unseen-shape instances where constructible. Arithmetic: 21 cells × 40–50 =
~840–1,050 answerable intents; plus ~250–300 status variants → PACS totals
~1,100–1,350 intent clusters. Rationale: the benchmark is multi-axis
(family × depth × exposure × surface × status); cell-level conclusions need
cell-level n. Anchors varied; no anchor in more than 3 instances;
entity/year/CPV/answer-size diversity enforced per cell.

## Statistical unit and analysis
The unit of analysis is the gold intent instance (equivalently its gold program
tree), NOT the rendered text row. The three surface channels are paired
measurements of the same item; status variants derived from one intent belong to
the same cluster. Consequences, binding: confidence intervals use intent-level
cluster bootstrap; channel differences (canonical_independent − training_renderer,
naturalized − training_renderer) are computed as within-intent paired deltas —
channels are never counted as independent samples; answerable accuracy and
status-recognition results are aggregated and reported separately.

## Generation protocol (per instance)
1. Intent instantiated from the family archetype (KG-grounded parameter sampling).
2. Gold tree authored from family×depth templates; dual-evaluator execution; any
   disagreement or degeneracy discards the instance.
3. Status variants derived from the same instance where scheduled.
4. Three surfaces rendered/gated as above.
5. Audits: every template reviewed; >= 5 naturalized instances human-read per
   applicable family × depth cell, with elevated sampling for set operations,
   negation, anti-joins, comparisons, universals, and ALL status types (user
   included among readers); iron rule (10 deterministic-random rows per
   acceptance gate) applies.

## Main table of the paper (PACS-test, channel a)
Rows: F1–F7 + macro average. Columns: L1 | L2-seen | L3-seen | unseen (pooled
depths, reported with per-depth breakdown in appendix) | naturalized channel b |
abstention (status axis). Iteration curves, construction holdouts, and perturbation
series move to Sets C/D sections.

## Experiment sets (paper naming: one PRIMARY BENCHMARK + three COMPANION EVALUATION SUITES)
- PRIMARY BENCHMARK — B, PACS (this spec): the final comprehensive exam.
- Companion suite A, Legacy Regression: 2,285 paired vs old champion (McNemar;
  the retirement question) — checks old knowledge.
- Companion suite C, Structural OOD: B_clean, rotated construction holdouts,
  B_anchor when built — checks compositional mechanism.
- Companion suite D, Robustness & boundary: reorder/paraphrase/masked;
  out-of-grammar; abstention — checks shortcut resistance and limits.

## Model freeze rule
The first official PACS results MUST come from the current frozen compose-v3
checkpoint (and the untrained base as control). No retraining or data iteration
may be informed by PACS before those results are reported; shortfalls PACS
exposes feed Discussion and future work, not a v4 trained against it.

## Build order
1. Spec FROZEN (this version, user-approved).
2. Family×depth tree templates (reuse algebra + dual evaluators).
3. Independent surface grammar (second author-voice; no shared assets).
4. Naturalization gates; generation; dev/test split; audits; freeze PACS v1.
5. Evaluate compose-v3 (and base) on PACS-dev, then once on PACS-test.
GPU needed only at step 5; steps 1–4 proceed during the driver outage.
