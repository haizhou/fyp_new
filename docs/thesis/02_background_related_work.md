# Chapter 2 — Background and Related Work

## Chapter thesis (what this chapter must convince the reader of)

CICADA sits at the junction of four literatures — KGQA benchmarks/semantic parsing,
verifier-guided generation and self-correction, bootstrapped self-training/distillation, and
LLM-over-KG agents — and its generate–filter–retrain loop shape is deliberately NOT claimed as
novel. The dividing line against the nearest neighbours (ExeSQL, SCD-style self-consistency
distillation, STaR, ReST, RoG, ChatKBQA) is drawn on two axes only: **filter strength** (a
deterministic, executable, multi-stage verifier plus a training-time-only oracle gate, instead
of gold-answer agreement, self-consistency votes, or execution-success-only filters) and
**blind-region handling** (what passes the filter wrongly is routed into labelled hard
negatives and abstention supervision instead of silently poisoning the pool). The reader must
finish the chapter able to predict, for any related system, where its filter is weaker and
what it does with its blind region.

## Section outline

### 2.1 KGQA benchmarks and semantic parsing
GrailQA, LC-QuAD 2.0, KQA Pro: template-first construction, i.i.d./compositional/zero-shot
splits. We adopt the recipe with LLM stages replacing crowd stages, and replace crowd
validation with mechanical fidelity checks (motivated by a measured failure: an LLM checker
approved 9,762/9,762 rewrites including known drifts). Our dual-implementation oracle audit
extends this line: benchmark correctness is usually asserted; we measure it.

### 2.2 Verifier-guided generation and self-correction
Intrinsic self-correction degrades reasoning (Huang et al., ICLR 2024); environment-feedback
repair works when errors are typed (Self-Debugging, CRITIC, DIN-SQL/MAC-SQL, QueryAgent's
ERASER). Our reflector follows the typed-error prescription pattern and adds provenance-gated
empty-result repair (an empty result whose literals all trace to the question is an answer,
not a defect). Position the reflector here as a CONSUMER of verifier signal.

### 2.3 Self-training, rejection sampling, and distillation — the divide-line section
STaR/ReST-style loops accept model outputs on a pass criterion; ExeSQL bootstraps text-to-SQL
with execution filtering; SCD-style pipelines filter by self-consistency. Same loop shape as
ours. The divide-line sentence (verbatim target): our contribution is not the loop but (a) the
FILTER: deterministic compile–ground–execute–verify with closed slot enums plus an oracle that
gates and never authors, and (b) the BLIND REGION: verifier-passing-but-wrong outputs become
labelled hard negatives for preference learning, and correct abstentions become supervision,
rather than both being invisible.

### 2.4 LLM-over-KG agents
RoG plans relation paths; ChatKBQA generates then grounds logical forms; Pangu uses
discrimination-guided search. Contrast: in CICADA the KG executor is the sole answer
authority, the plan is a checkable artifact (typed graph plan, deterministic 12-transform
compile), and grounding failures are diagnosable events that feed repair or abstention.

### 2.5 LLM-generated evaluation data
Naive paraphrase collapses diversity; we condition on six style axes with mechanical
acceptance and report distributional evidence (trigram-Jaccard median 0.429) instead of
asserting diversity. Abstain-cue preservation is a benchmark-integrity device (ch. 4).

### 2.6 Positioning summary table
One row per neighbour system: filter type, filter strength, blind-region handling, abstention
support. The table operationalises the divide-line; no novelty claims about loop shape.

## Evidence manifest

| Item | Where used | Source / artifact |
|---|---|---|
| LLM checker approved 9,762/9,762 rewrites incl. known drifts | §2.1 | [DOC-SOURCED: thesis_draft.md §2 — promote to master table before citing] |
| Dual-oracle agreement 99.88% (14,752/14,770) | §2.1 | [DOC-SOURCED: thesis_draft.md §4.4; worklog 2026-07-04 — promote] |
| Naive repair loop net-negative: 84%→66%, hallucinations 1→6 (dev_smoke, directional) | §2.2 | [DOC-SOURCED: thesis_draft.md §5.3 — dev-slice, motivation-only, never a claim] |
| Trigram-Jaccard median 0.429 | §2.5 | [DOC-SOURCED: thesis_draft.md §4.5 — promote] |
| Hard negatives exist at scale: bridge verified 86.7% vs oracle-correct 52.3% (34.4pt verifier-blind) | §2.3 | [WORKLOG-SOURCED: 2026-07-05 self-harvest entry — promote] |
| Figure slot: none (positioning table 2.1 instead) | §2.6 | table, not figure |

## Claims discipline (this chapter must NOT)

- MUST NOT claim novelty of the generate–filter–retrain LOOP SHAPE — the divide-line targets
  filter strength and blind-region handling only; any sentence implying "we invented
  verifier-filtered self-training" is banned.
- MUST NOT describe the reflector as a verifier or as part of the filter when contrasting with
  self-correction work — it consumes verifier signal and its outputs re-enter the same four
  checks; this chapter must not re-open the "who verifies the verifier/PRM" loop.
- MUST NOT cite dev_smoke measurements (84%→66% repair regression; −12pt Step-1 removal) as
  evidence of superiority over any cited system — they are design-motivation measurements on a
  ≤50-question development slice, and must be labelled as such.
- MUST NOT claim our benchmark is more "natural" than crowd-built benchmarks — LLM-generated
  surfaces are a documented limitation (ch. 8); the claim is measured correctness and measured
  diversity, nothing more.
- MUST NOT imply the oracle participates at inference time when contrasting with
  execution-filtered pipelines — check 4 (answer ↔ oracle) exists only during training-data
  harvest; in evaluation it would be cheating.
- MUST NOT attach numbers to neighbour systems from memory — related-work numbers require
  paper citations, and no number from our master table may be juxtaposed as if measured on
  their benchmarks.
