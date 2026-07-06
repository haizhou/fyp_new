# Chapter 8 — Conclusion, Limitations, and Future Work

## Chapter thesis (what this chapter must convince the reader of)

The dissertation demonstrated its one sentence: supervision flowed only from the verifiable
region — verifier-passed, oracle-gated, with the oracle filtering and never authoring — yet
the learned competence extends beyond it, measured as a fully-local 8B stack that exceeds its
cloud teacher by +15.89pt on a held-out set and stays flat exactly where the teacher decays.
The chapter must also do the opposite work with equal energy: state what the evidence does NOT
cover (one self-built single-domain benchmark; a difficulty-composite slice rather than a true
holdout; pending pre-registered probes), and convert each limitation into a concrete next
experiment, ending with the research agenda that generalises the movable projection boundary.

## Section outline

### 8.1 Summary of contributions (four pillars, one evidence line each)
Pillar 1 bootstrap loop: yields 65.5% → 76.6%/78.1%, three instantiations; pillar 2 gain
decomposition: scaffolding floor (zero-shot 70.4 DEV) vs training gain, v1→v2.2 boundary move;
pillar 3 DPO near-miss toxicity: mechanism-explained negative result with cross-base dose
curve; pillar 4 benchmark methodology: 12,828 questions, 99.88% dual-oracle, split discipline
that survived an adversarial review. Headline recap: 69.76 < 75.97 < 78.03 < 83.33 < 85.65
(n=2,285, all pairings p<1e-14).

### 8.2 Limitations (each with its factual bound)
1. **Single self-built benchmark, single domain.** All claims are within UK-procurement KGQA
   on a benchmark we constructed; architecture claims rest on within-domain ablations.
2. **The hard-composite slice is difficulty-composite, not a holdout.** ood_candidate = hard
   operators ∪ L2 rewrite ∪ abstain classes; both train and test contain the class (plans
   disjoint, surfaces novel). The strict holdout is a single test-only template family
   (small n) — reported here, not in headlines.
3. **Compositional generalisation is unproven pending ood_probe_v1.** [PENDING: ood_probe_v1
   results — 5 novel structural signatures (3 bridge + 2 non-bridge), ~600 rows, three-branch
   pre-committed criteria; ANY branch is reported verbatim, including the difficulty-floor
   branch. Note the probe's symmetry advantage: the first student-teacher comparison where
   both are zero-shot on the tested compositions.]
4. **Teacher few-shot asymmetry.** The teacher operates zero-shot in-domain while students are
   trained on in-domain data; student-beats-teacher claims are claims about the
   pipeline-plus-distillation recipe, not about model quality. The symmetric comparison is
   exactly what ood_probe_v1 provides (limitation 3).
5. **Teacher is a single replicate on final_test** (provider-noise robustness lives on DEV
   with three replicates); teacher noise floor ±1.1 disciplines all teacher deltas.
6. **Convention-relative correctness.** Oracles are defined w.r.t. documented conventions;
   the 99.88% dual audit bounds but does not remove convention risk (residual 18/14,770).
7. **LLM-generated surfaces.** Six style axes widen but do not equal real-user language.
8. **r2 not yet adjudicated.** [PENDING: r2 rung vs pre-committed gates.]

### 8.3 Future work
1. **RLVR / GRPO on the verifier reward.** The verifier is already a dense, deterministic
   reward source; DPO's near-miss toxicity (ch. 7) argues for additive on-policy RL (GRPO-style)
   rather than pairwise suppression — currently future work pending compute.
2. **Cross-schema probe (BIRD).** Port the projection-boundary recipe to text-to-SQL (BIRD)
   to test whether the three-layer decomposition and the yield-surpassing self-harvest are
   schema-portable — the direct answer to limitation 1.
3. **Bootstrap convergence.** r2/r3 rounds as a convergence curve; additive bridge
   re-consolidation as the tested lever.
4. **PhD agenda: the movable projection boundary as a general recipe.** Treat "which semantic
   properties can be projected into deterministic checks, at what engineering cost, and how
   does learned competence scale with the projected fraction" as a measurable research
   object across domains where verification is inherently partial (data QA, code, scientific
   claims). The one-sentence thesis becomes a design law to be tested, not a finding to be
   restated.

### 8.4 Closing paragraph
Return to the opening dilemma: trust neither the fluent guesser nor the mute executor; build
the seam between them out of checks where possible, abstention where not, and let the checked
region teach. Close on the frozen sentence.

## Evidence manifest

| Number | Where used | Source / artifact |
|---|---|---|
| Headline recap 69.76/75.97/78.03/83.33/85.65; +15.89 CI [+14.07,+17.70]; all p<1e-14 | §8.1 | [TABLE-SOURCED] FINAL scoreboard; outputs/eval/final_test/* |
| Yields 65.5% / 76.6% / 78.1% | §8.1 | 76.6/78.1 [TABLE-SOURCED]; 65.5 [WORKLOG-SOURCED — master-table row needs repair first] |
| 12,828; 99.88% (14,752/14,770, residual 18) | §8.1, §8.2.6 | 12,828 [TABLE-SOURCED]; 99.88% [DOC-SOURCED: thesis_draft.md §4.4 — promote] |
| Slice definition + strict-holdout note (1 test-only family) | §8.2.2 | [TABLE-SOURCED] iid→ood_candidate table header note; merge_stratify_l1_l2.py:is_ood_candidate |
| ood_probe_v1 design facts: 5 templates, 3+2 bridge/non-bridge, ~600 rows, 3-branch criteria, symmetry note | §8.2.3 | [SOURCED: docs/ood_probe_v1_prereg.md — frozen prereg; results PENDING] |
| Teacher noise floor 72.6% ± 1.1 | §8.2.5 | [TABLE-SOURCED] outputs/eval/matrix_v2/teacher* |
| GRPO deferred to future work | §8.3.1 | [DOC-SOURCED: cicada_planner_training_plan.md] |
| No figures — text-only chapter | — | — |

## Claims discipline (this chapter must NOT)

- MUST NOT generalise beyond the domain: no "this recipe works for KGQA at large" — the
  licensed claim is within-benchmark with a named future probe (BIRD) as the test.
- MUST NOT rename the hard-composite slice "OOD", even in limitations — the limitation IS the
  naming discipline; compositional claims stay [PENDING] on ood_probe_v1, and if the probe
  lands in the deficit or difficulty-floor branch, that branch is reported verbatim.
- MUST NOT soften limitation 4 (teacher few-shot asymmetry) — the student-beats-teacher
  result must be framed as a recipe claim, with the probe named as the symmetric test.
- MUST NOT claim abstention is "solved" or hallucination "eliminated" — the blind region is
  squeezed and caught, never eliminated (frozen framing from the narrative core).
- MUST NOT propose future work that contradicts recorded evidence — e.g., no "stronger DPO
  suppression" proposal; the recorded consequence is additive (RL/self-distillation) levers.
- MUST NOT cite dev-set numbers in the conclusion recap except the explicitly-roled ones
  (scaffolding floor, noise floor), each labelled DEV.
