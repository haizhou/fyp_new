# Chapter 4 — Benchmark Construction and Split Discipline

## Chapter thesis (what this chapter must convince the reader of)

Every number in this dissertation is only as credible as the benchmark beneath it, so the
benchmark is treated as an instrument with measured error: 12,828 questions born from
fully-parameterised gold programs, oracles validated by an independent second implementation
at 99.88% agreement, three abstain classes designed as traps rather than decoration, and
plan-level split isolation with zero train/test plan overlap. The chapter must also convince
the reader of our honesty protocol: compare_set_v4 is a DEV set (the v2 pipeline fixes were
derived from its v1 errors), final_test is confirmatory and dual-reported including/excluding
the 19 cue-matched rows, and every adaptive-overfitting risk found in review is disclosed, not
patched over.

## Section outline

### 4.1 Plan-first generation (L1) and persona rewriting (L2)
L1: 33 template families instantiate parameterised plans against the KG, compute oracles by
execution, then verbalise. L2: LLM rewrites under four personas with a deterministic
per-surface accept/reject gate (9,762 accepted / 1,170 rejected, dominated by invented
temporal relations, 890); v4 adds six explicit style axes with mechanical acceptance
(trigram-Jaccard median 0.429 as the distributional evidence). Why plan-first: every question
is born with an executable gold program — the training target.

### 4.2 Quality audit v1 → v2: six measured defects, six mechanical fixes
Boolean answers 100% True → 155 mutated False twins (oracle recomputed independently); factoid
small-value-domain shortcut → 1,183 rows re-bucketed to categorical; lexically detectable
unsupported questions → answerable contrast twins; L2 drift invisible to an LLM checker →
position-aware mechanical drift detector; under-specified gold programs → parameter backfill;
oracle circularity → the independent second implementation (§4.4).

### 4.3 Splits, balance, and isolation hygiene
Frozen v4 arithmetic: **12,828 = train 9,267 + dev_tune 556 + dev_select 671 + dev_smoke 49 +
final_test 2,285**. Three hard gates in every build: no plan straddles train/eval; global id
uniqueness; row conservation. plan_id overlap train-vs-test = 0 (6,674 vs 2,184 plans); 34
shared template families + 1 test-only family. final_test is deliberately harder
(hard-composite 54.7% + compositional 8.4% vs train 45.0%/1.7%; L2 73.3% vs 65.9%; abstain
15.8% vs 7.7% for statistical power on the abstention claim).

### 4.4 Independent oracle verification (the 99.88% audit)
A pure-pandas evaluator sharing no code with generator or executor recomputes every answerable
oracle. First pass 92.7%, mismatches clustered by family → convention gaps diagnosed (flat
first-party universe, additive-only money, an empty-string party name matching every record —
a constant 1.61B excess in every bridge sum; 230 degenerate constraints). Final:
14,752/14,770 = 99.88%, residual 18 characterised (top-k parameter inference 15; one boolean
surface-parsing family 3).

### 4.5 The three abstain types as blind-region probes
abstain_unsupported / abstain_no_results / abstain_ambiguous (120 each in final_test). These
are the abstention layer's completeness test (ch. 5), not benchmark decoration; diversified
rewrites must preserve the cue words (rule added after three surfaces lost their cues and
became answerable).

### 4.6 Dev/confirmatory discipline and the 19-cue disclosure
compare_set_v4 (260) ⊂ final_test and is relabelled DEV after the v2 fixes were derived from
its v1 errors (adaptive-overfitting finding, adopted from independent review). The 5
dev-derived _UNSUPPORTED_CUES string-match 19/2,285 final_test rows; every headline is
dual-reported (deltas −0.08 to −0.21pt; rankings and significance unchanged) WITH the
mandatory note: the cue rows were mostly answered correctly by ALL systems, which is why the
deltas are negative. Also disclosed: 2 verbatim train↔eval duplicate questions (flagged and
excluded in final stats).

### 4.7 v4.1 curation and the eval-protocol footnote
System-blind malformation sweep (text patterns only, plans/oracles untouched): 2 final_test
rows fixed, 0 in compare_set_v4; impact bound ≤0.09pt. Eval protocol (repair budget 1) vs
data-engine harvest (repair budget 2) are two deliberate configurations, cited separately.

## Evidence manifest

| Number | Where used | Source / artifact |
|---|---|---|
| 12,828 = 9,267/556/671/49/2,285; compare_set_v4 (260) ⊂ final_test | §4.3, §4.6 | [TABLE-SOURCED] benchmark arithmetic line; data/qa/cicada_core_v4/ |
| Cue-split dual report: 85.65→85.57 (−0.08) / 78.03→77.89 (−0.14) / 69.76→69.55 (−0.21) / 75.97→75.82 (−0.16) / 83.33→83.23 (−0.10); bound ≤0.21pt | §4.6 | [TABLE-SOURCED] cue-split table |
| 99.88% (14,752/14,770), first pass 92.7%, residual 18 | §4.4 | [DOC-SOURCED: thesis_draft.md §4.4; worklog 2026-07-04 — promote to master table] |
| plan_id overlap 0 (6,674 vs 2,184); 34+1 families; hardness mix 54.7%/8.4%, L2 73.3%, abstain 15.8% | §4.3 | [WORKLOG-SOURCED: 2026-07-05 data-tables entry; docs/data_distribution_v4.md — promote] |
| L2 gate: 9,762 accepted / 1,170 rejected (890 temporal); trigram-Jaccard 0.429; abstain floors 120 each; every final_test bucket ≥ 93 (top_k 128) | §4.1, §4.3, §4.5 | [DOC-SOURCED: thesis_draft.md §4.1/§4.5 — promote] |
| v4.1: 2 rows fixed / 0 in DEV; bound ≤0.09pt | §4.7 | [WORKLOG-SOURCED: 2026-07-06 v4.1 entry — promote] |
| 2 verbatim train↔eval duplicates disclosed | §4.6 | [WORKLOG-SOURCED: 2026-07-06 review-verdicts entry — promote] |
| Figure slot: benchmark construction funnel + split diagram | §4.1–4.3 | [PENDING: render; no F-number assigned yet] |

## Claims discipline (this chapter must NOT)

- MUST NOT cite any compare_set_v4 number as a headline result — it is the DEV set; its only
  chapter-4 role is defining the discipline. final_test numbers appear here only inside the
  cue-split disclosure.
- MUST NOT present the cue-split table without the "mostly answered correctly by ALL systems"
  sentence — omitted, the negative deltas read as evidence of contamination in the wrong
  direction.
- MUST NOT claim the benchmark measures real-user language — surfaces are LLM-generated under
  six style axes; distributional evidence (0.429) is reported, naturalness is not claimed.
- MUST NOT call the hard-composite slice "OOD" in split descriptions — the slice exists in
  both train and test (plans disjoint, surfaces novel); the single strict-holdout family is
  small-n and lives in limitations.
- MUST NOT claim oracle correctness beyond the audit's scope — 99.88% is convention-relative
  agreement between two implementations, with the residual 18 characterised, not zero.
- MUST NOT blur the two repair budgets — harvest (2) and eval (1) numbers are never mixed in
  one comparison.
