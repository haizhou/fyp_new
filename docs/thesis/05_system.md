# Chapter 5 — System: Hard Verification, Abstention, and the Movable Projection Boundary

## Chapter thesis (what this chapter must convince the reader of)

The system is organised as three layers. First, a hard-verification layer: every property on
the semantic-judgement chain that can be projected onto structure or execution is pinned by a
deterministic check. Second, an abstention layer: the unprojectable residue (pure semantic
correspondence) forms a blind region, and abstention catches it — the three abstain traps of
chapter 4 are this layer's completeness test. Third, a learning layer: signal flowing out of
the verifiable region drives distillation and bootstrapping (ch. 6), and the learned
competence covers part of the blind region (the one-sentence thesis). The chapter's deepest
claim is that the projection boundary is MOVABLE: engineering effort carries semantic
properties across the projection line, demonstrated live by the v2 fixes; and the reflector
sits outside the verification chain as a gated consumer of its signal, never a fifth check.

## Section outline

### 5.1 Architecture walk-through
Step-1 understanding briefing (nano → distilled local adapter) → Step-2 typed graph plan
(teacher → student LoRA, strict JSON) → schema grounding → entity grounding → deterministic
12-transform compile with three gates → levelled execution → verification → gated reflection.
One figure, one paragraph per stage.

### 5.2 The four-check chain (hard / hard / seam / hard-training-only)
Verbatim from the narrative core: (1) executor ↔ plan graph — hard (compiled execution);
(2) plan graph ↔ briefing — hard (structural comparison in code); (3) briefing ↔ question —
SEAM: code can check only projections (slots, entities, literals, numbers, roles, cue words),
never "is the meaning right" — the blind-region entrance, where all hard negatives leak in;
(4) answer ↔ oracle — hard, but exists only at training time; in evaluation it would be
cheating. Framing sentence: the four checks squeeze the blind region layer by layer; they do
not eliminate it.

### 5.3 The abstention layer
Where every check-3 residue lands: unsupported concepts (cue vocabulary + unsupported-field
tokens), no-result emptiness (provenance-gated: an empty result whose literals all trace to
the question is an answer), ambiguity. Tie back to the three benchmark traps as the
completeness test of this layer.

### 5.4 The movable projection boundary — v2 as the worked example
The blind region is not a fixed set: engineering effort = carrying semantic properties across
the projection line. The v2 fixes projected three formerly-semantic error classes into
structural checks: count-intermediate-set (return targeting an entity_set on a record-count
question — code-checkable), empty-value lookup (categorical lookups pass the type gate with
empty values), unsupported-cue veto on the fast path. Measured effect: roughly +5pt on every
ladder rung (v1→v2.2). The cautionary half: v2.0/v2.1 over-reach — coverage scoring rejected
88% of 169 REAL field texts from teacher traces — shows over-projection wounds the system;
moving the boundary requires regression against real pipeline intermediates, not synthetic
unit cases. What cannot be carried across stays with abstention and learning.

### 5.5 The reflector: a consumer of verifier signal (strict wording)
The reflector is NOT a fifth check. The four deterministic checks produce diagnostic signal;
the reflector (an LLM, uncertain) consumes the signal to repair plans; its output re-enters
the SAME four checks (gated repair). Motivating measurement (dev, directional): the naive
"no answer → replan" loop was net-negative (84%→66%, hallucinations 1→6) because it repaired
correct abstentions into confident wrong answers. Design intent sentence: this architecture
deliberately refuses the "uncertain model verifies uncertain model" loop.

### 5.6 Interface to the learning layer
Routing of harvest outcomes: verified_sft (oracle-gated; the oracle filters, never authors),
hard_negatives (verifier-passing-but-wrong — check-3 leakage made visible), abstain_sft,
repair_sft, preference pairs. Full treatment in ch. 6.

## Evidence manifest

| Number | Where used | Source / artifact |
|---|---|---|
| v1→v2.2 rung lifts (DEV, n=260): zeroshot 61.5→70.4; SFT 76.1→81.2; RSFT 76.1→81.5; DPO 78.8→83.5 | §5.4 | [TABLE-SOURCED] outputs/eval/matrix/cicada-qwen3-*/ vs outputs/eval/matrix_v2/cicada-qwen3-*/ |
| "roughly +5pt per rung = one boundary move" wording | §5.4 | [SOURCED: thesis_narrative_core.md movable-boundary section] |
| Teacher v2.2 3-run floor 72.6% ± 1.1 (71.9/71.9/73.9) vs v1 single 70.0% | §5.4 | [TABLE-SOURCED] outputs/eval/matrix_v2/teacher* |
| v2.1 over-reach: 88% of 169 real field texts rejected; decision-identical repair in v2.2 | §5.4 | [WORKLOG-SOURCED: 2026-07-06 v2.0→v2.2 entry — promote or cite as process fact] |
| Naive repair 84%→66%, hallucinations 1→6; gated reflector recovers (dev_smoke, directional) | §5.5 | [DOC-SOURCED: thesis_draft.md §5.3/§5.4 — dev-slice, motivation-only] |
| v1 error modes fixed by v2: 21 abstained-on-answerable / 12 answered-on-unsupported / 22 wrong-value | §5.4 | [WORKLOG-SOURCED: 2026-07-06 pipeline-v2 entry — promote] |
| Figure slot F3: pipeline v1→v2.2 ablation dumbbell (same adapters, scaffolding only) | §5.4 | outputs/figures/F3_pipeline_ablation.pdf (DEV v2.2 render; re-render check [PENDING]) |
| Figure slot F8: abstention frontier / calibration scatter | §5.3 | outputs/figures/F8_abstention_frontier.pdf (DEV render; re-render check [PENDING]) |
| Figure slot: architecture diagram (three layers + four checks) | §5.1–5.2 | [PENDING: render] |

## Claims discipline (this chapter must NOT)

- 禁写 (banned verbatim): "deterministic checks detect everything / detect uncertainty".
  应写 (required framing): everything hard-verifiable is hard-verified; what cannot be hard
  is compressed to a minimum, then abstention and learning catch it; the four checks squeeze
  the blind region layer by layer — they do not eliminate it.
- 禁写: calling the reflector a verifier / "the fifth verification". 应写: the verifier
  produces signal (deterministic); the reflector consumes signal (uncertain, but gated) and
  its output re-passes the same four checks. Never invite the "who verifies the reflector"
  PRM regress — the design's point is to not enter that loop.
- MUST NOT present check 4 (oracle) as part of the inference pipeline — training-time only;
  a sentence stating "in evaluation it would be cheating" is required, not optional.
- MUST NOT cite v2.0/v2.1 intermediate matrices (66.5/66.1 teacher legs etc.) as results —
  they are archived non-citable intermediates; only v1 and v2.2 are citable versions.
- MUST NOT report the +5pt boundary-move effect as a training gain — it is a scaffolding
  (floor) effect measured across ALL systems including zero-shot and teacher; training gains
  are measured on top of the raised floor (ch. 6 owns that decomposition).
- MUST NOT use dev_smoke numbers (84%, 66%, 10/10 abstains) as claims — directional
  motivation only, ≤50-question development slice, always labelled.
