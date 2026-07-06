# Abstract (skeleton)

Status: Variant A (superiority) is LOCKED — the pre-committed adjudication playbook in
`docs/abstract_variants.md` selected branch A on 2026-07-07 (fully-local vs hybrid on
final_test: Δ=+7.61pt, p<1e-15). No parity/trade-off wording may be used. All numbers below
are [TABLE-SOURCED] from `docs/results_master_table.md` unless marked otherwise.

## Draft skeleton (to be tightened, sentence roles fixed)

1. **Problem sentence.** QA over structured public data faces a trust dilemma: LLMs read
   natural language but answer unverifiably; symbolic executors answer verifiably but cannot
   read. (Framing inherited from `docs/thesis_draft.md`; keep.)

2. **System sentence.** CICADA: an LLM only proposes; a deterministic
   ground–compile–execute–verify stack is the sole answer authority over a 215,221-contract
   UK procurement knowledge graph [DOC-SOURCED: kg_enrichment_plan.md — promote to master
   table before final citation].

3. **One-sentence thesis (verbatim, frozen).** *"Supervision flows only from the verifiable
   region, yet the learned competence extends beyond it."*

4. **Headline result sentences (Variant A frame, numbers filled).** The bootstrapped system
   runs entirely on a single local GPU — both the understanding and planning stages are 8B
   LoRA adapters distilled through the pipeline's own verifier — and **outperforms its
   cloud-hybrid counterpart** on the held-out test set (n=2,285: 85.65% vs 78.03%; +7.61pt,
   95% CI [+6.10,+9.13], McNemar +242/−68, p<1e-15) while exceeding its cloud teacher by
   +15.89pt (69.76% → 85.65%; CI [+14.07,+17.70], +406/−43, p<1e-15; single teacher
   replicate), at zero marginal API cost. De-teachering replicates on a second base:
   Llama fully-local 83.33% vs its hybrid 75.97% (+7.35pt, CI [+5.73,+8.97]); all five
   pairings p<1e-14 — the "on both bases" wording is licensed.

5. **Robustness sentence (fixed two-layer wording, verbatim from abstract_variants.md).**
   "exceeds the teacher on the held-out test set (n=2,285, McNemar p<1e-15, single teacher
   replicate); robustness of this gap to provider nondeterminism was established on the
   development set, where the student beats each of THREE teacher replicates
   (+35/−5, +34/−4, +30/−5; p≤2e-5)."

6. **Mechanism sentence (fixed softened wording, verbatim).** "Across three teacher
   replicates, accuracy decays 6–12 points from the iid to the hard-composite slice; the
   fully-local student shows **no measurable decay (+0.8pt, n=136)** — with the SAME planning
   checkpoint, attributable to the distilled understanding stage."

7. **Abstention sentence.** Abstention closes the verifier's blind region: three abstain
   classes are first-class benchmark citizens and anti-hallucination supervision flows through
   the same verifier gate. (Numbers, if any, from master table only.)

8. **Benchmark credibility clause.** 12,828-question benchmark; gold programs fully
   parameterised; oracles validated by an independent second implementation at 99.88%
   agreement [DOC-SOURCED: thesis_draft.md §4.4 / worklog 2026-07-04 — promote before citing].

9. **Optional closing slot.** [PENDING: r2 rung — if the pre-committed headline-swap gate
   fires, headline numbers change per abstract_variants.md §r2 rules]
   [PENDING: ood_probe_v1 compositional result — add one clause only if branch (i) of the
   pre-committed criteria holds]

## Evidence manifest

| Number | Where used | Source row / artifact |
|---|---|---|
| 85.65% (1957/2285) fully-local Qwen, final_test | sent. 4 | [TABLE-SOURCED] outputs/eval/final_test/fully_local_qwen |
| 78.03% (1783/2285) hybrid Qwen | sent. 4 | [TABLE-SOURCED] outputs/eval/final_test/hybrid_qwen_dpo |
| +7.61pt, CI [+6.10,+9.13], +242/−68, p<1e-15 | sent. 4 | [TABLE-SOURCED] FINAL_TEST headline block |
| 69.76% (1594/2285) teacher, 1 replicate | sent. 4 | [TABLE-SOURCED] outputs/eval/final_test/teacher_r1 |
| +15.89pt, CI [+14.07,+17.70], +406/−43 | sent. 4 | [TABLE-SOURCED] FINAL_TEST headline block |
| Llama FL 83.33% / Llama hybrid 75.97%; +7.35 CI [+5.73,+8.97]; all pairings p<1e-14 | sent. 4 | [TABLE-SOURCED] FINAL scoreboard incl. Llama arms |
| +35/−5, +34/−4, +30/−5; p≤2e-5 (DEV, 3 teacher replicates) | sent. 5 | [TABLE-SOURCED] paired significance table |
| teacher decay 6–12pt; student +0.8pt (n=136) | sent. 6 | [TABLE-SOURCED] iid→ood_candidate table (values 75.7/73.8/78.6 → 65.4/67.6/66.2; FL 84.5→85.3) |
| 12,828 questions | sent. 8 | [TABLE-SOURCED] benchmark arithmetic line |
| 99.88% dual-oracle agreement (14,752/14,770) | sent. 8 | [DOC-SOURCED: thesis_draft.md §4.4; worklog 2026-07-04 — promote to master table] |

## Claims discipline (abstract must NOT)

- MUST NOT use parity or trade-off wording — verdict A is locked; equally, MUST NOT inflate
  beyond the licensed sentences (no "solves", no "verified answers" for the whole system).
- MUST NOT say "OOD" — the in-text name is "hard-composite slice"; compositional claims are
  reserved for ood_probe_v1 and are [PENDING].
- MUST NOT cite any dev-set number as headline; DEV appears only inside the fixed robustness
  sentence (sent. 5) with its role stated.
- MUST NOT call the reflector a verifier; the abstract mentions verifier-gated distillation
  only ("distilled through the pipeline's own verifier" — the filter, not an author).
- MUST NOT merge data-engine yield (65.5%/76.6%) into accuracy sentences — different metric;
  if yields appear at all they are labelled as harvest yield.
- MUST NOT drop "single teacher replicate" from the final_test teacher pairing.
