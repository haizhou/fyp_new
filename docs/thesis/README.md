# FYP Dissertation Skeleton — CICADA

Scaffolded 2026-07-06 from the frozen single sources of truth. Every chapter file contains:
(1) a chapter thesis, (2) a section-by-section outline with stubs, (3) an evidence manifest
with exact numbers and artifact paths, (4) a claims-discipline note (what the chapter must
NOT claim).

## Frozen sources (do not contradict; wording of claim sentences is copied, never paraphrased)

- `docs/thesis_narrative_core.md` — one-sentence thesis, three-layer system framing, 禁写/应写 rules
- `docs/results_master_table.md` — the ONLY source of citable numbers
- `docs/abstract_variants.md` — abstract wording (verdict A executed), fixed sentences, naming discipline
- `docs/ood_probe_v1_prereg.md` — pre-registered compositional probe (pending)
- `docs/cicada_worklog.md` (2026-07-05 onward) — decision log; provenance for process facts

## File index

| File | Content |
|---|---|
| `00_abstract.md` | Abstract skeleton (Variant A, superiority — locked by the 2026-07-07 adjudication) |
| `01_introduction.md` | One-sentence thesis + four-pillar contributions + results preview |
| `02_background_related_work.md` | Four literatures + the divide-line (filter strength & blind-region handling) |
| `03_data_and_kg.md` | OCDS data engineering + KG (215,221 contracts / 131,502 canonical orgs) |
| `04_benchmark.md` | 12,828-question benchmark, dual-oracle 99.88%, three abstain types, split discipline |
| `05_system.md` | Three-layer system: hard verification / abstention / learning; movable projection boundary |
| `06_bootstrapping_experiments.md` | Ladder + five-pairing final_test + bootstrap yields + Step-1 distillation |
| `07_analysis.md` | Decay mechanism, DPO near-miss toxicity, noise floor, error modes, ceilings |
| `08_conclusion.md` | Conclusion + limitations + future work (RLVR/GRPO, BIRD probe, PhD agenda) |

## The three writing rules (binding for every chapter)

**Rule 1 — numbers only from the master table.** A number may appear in a chapter only if it
is in `docs/results_master_table.md` with dataset, metric, pipeline version, and artifact
path. This skeleton marks provenance explicitly: `[TABLE-SOURCED]` (in the master table now),
`[WORKLOG-SOURCED]` / `[DOC-SOURCED: <file>]` / `[ARTIFACT-DERIVED]` (true, but must be
promoted into the master table before the writing pass), `[PENDING: <what>]` (experiment not
finished; leave the slot empty rather than improvise). 'Data-engine yield' (harvest) and
'system accuracy' (eval) are different metrics and are never compared across.

**Rule 2 — claims discipline.** Aggregated from thesis_narrative_core.md and
abstract_variants.md; each chapter file repeats the subset that applies to it:
- Never call the reflector a verifier or "the fifth check". Verifiers produce signal
  (deterministic); the reflector consumes signal (uncertain, gated).
- Never write "deterministic checks detect everything / detect uncertainty". Write: everything
  hard-verifiable is hard-verified; the residue is squeezed minimal, then caught by abstention
  and learning. The four checks shrink the blind region; they do not eliminate it.
- Never use "OOD" for the hard-composite slice. In-text name: "hard-composite slice"
  (= hard operators ∪ L2 rewrite ∪ abstain classes; both train and test contain this class).
  Compositional-generalisation claims belong exclusively to ood_probe_v1 (pre-registered).
- Dev-set numbers are never cited as headline. compare_set_v4 (260) is the DEV set (model
  selection; the v2 fixes were derived from its v1 errors) and is a subset of final_test.
  final_test (2,285) is confirmatory, always dual-reported incl./excl. the 19 cue-matched rows,
  and the cue table must carry the note that those rows were mostly answered correctly by ALL
  systems (otherwise the negative deltas read backwards).
- Student-beats-teacher wording is two-layer: single-teacher-replicate pairing on final_test;
  robustness to provider nondeterminism established on DEV against three teacher replicates.
  Single-run teacher deltas below ~3pt are not claimable (noise floor 72.6% ± 1.1).
- The llama-FL > qwen-hybrid cross pairing is post-hoc and non-preregistered: one sentence in
  the analysis chapter plus a footnote, never a headline.
- Smoke numbers (35/50, 38/50, any n=50 train slice) are never citable. v2.0/v2.1 intermediate
  matrices are not citable. The oracle filters training data; it never authors targets and it
  never enters the evaluation pipeline.

**Rule 3 — two-document split (conference vs FYP).** The FYP dissertation (this skeleton)
carries the full engineering narrative: audits, incidents, negative results, and
worklog-traceable decisions. The conference paper is a separate, compressed document built on
the same four pillars. Claim sentences and all citable numbers are shared verbatim from the
frozen sources; neither document may improvise new claim wording, and material cut for the
conference page limit must not be deleted from the dissertation.

## Known erratum to fix before the writing pass

`results_master_table.md` row "Teacher harvest" has a mangled value cell ("0.0% (2/8555)")
and its artifact `data/qa/teacher_full_v1/summary.json` currently holds a 3-question stub.
The correct teacher train-pool yield per the 2026-07-05 worklog entry is 65.5% oracle-correct
on answerable (5,605/8,555), abstention 86.8% (618/712). Repair the table row (and the
artifact pointer) before any chapter cites it.

## Pending slots carried by the skeleton

- r2 rung (headline-swap gates pre-committed in abstract_variants.md)
- ood_probe_v1 pilot + results (three-branch criteria pre-committed)
- schema-valid diagnostic results (3-cell interpretation grid pre-committed)
- incl./excl. cue-split is DONE (in master table); four-point decay figure render pending
- OOD-slice (hard-composite) table promotion: student advantage +18-20pt vs teacher
