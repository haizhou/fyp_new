# Standalone paper outline (user-approved structure, 2026-07-17)

Document split: dissertation (frozen, partial-verifiability story) vs THIS paper
(composition as protagonist, old system as foundation). Numbers below are all
verified in docs/cicada_worklog.md unless marked GAP.

## 1 Introduction
Procurement analytics needs exact aggregation/comparison/relational analysis ->
retrieval+LLM insufficient -> fixed query programs limit complex analysis ->
executable, verifiable compositional programs. Three contributions.

## 2 Related Work (by literature, NOT by pipeline component)
2.1 Procurement data & KGs (OCDS). 2.2 Executable KGQA & semantic parsing (KoPL,
LLM planners). 2.3 Program-first generation & verification-guided learning
(KQA Pro, BYOKG, FlexKBQA, ExeSQL, execution filtering). 2.4 Compositional
generalisation (GrailQA, CFQ, structural holdouts, surface shortcuts).

## 3 Task Construction and Evaluation Environment
3.1 Analytical question classes (exhaustive retrieval, aggregation, arithmetic,
    temporal compare, set reasoning, relational composition, abstention).
3.2 Data & KG (only correctness-relevant: snapshot, ER, additive convention,
    evidence). 3.3 **Typed operator language FIRST** (17 nodes, types, caps).
3.4 Program-first generation — HONEST UNIFICATION WORDING: one language, two
    generation eras, unified by the regression-validated translation (old gold
    plans -> trees, final_test 1823/1823 both evaluators). NOT "one generator".
3.5 Evaluation regions: original fixed-query test / B_clean (re-held-out per
    iteration) / strict construction holdout (rotated: keys_where v1 64.8%,
    intersect v2 39/39, v3 19/19) / probe / reorder+wording perturbations /
    out-of-grammar primitives. Dual oracle + pre-training leakage audit.
3.6 Baselines & metrics: RAG (31.5/28.8 v1-era, note protocol age), original
    fixed planner (85.65 old protocol), zero-shot compositional (base guided
    49.38; free-decode collapse 0-32% = format-channel mechanism), trained.
    GAP: rule/template baseline if reviewers demand.

## 4 Verifiable Compositional Reasoning (method only, ONE language)
4.1 Question->program planning (guided decoding as part of method — justified
    by the 0->45.06 format-channel result). 4.2 Compile/ground/execute.
4.3 Verification, evidence, abstention (dual-metric protocol: Status Exact
    Match primary, faithfulness-gated Safe Semantic Outcome supplementary).
4.4 Verification-grounded supervision (program-authored + dual-verified;
    old-task translation mix; order twins; abstain channel).

## 5 Experiments (RQ-driven)
5.1 RQ1 basic queries reliable? RQ2 unseen compositions learnable? RQ3 surface
    shortcuts? RQ4 backward compat + boundary?
5.2 Fixed-query foundation — HALF PAGE (old ladder, teacher, second base).
    RQ1 anchored by compose-v3 on final_test answerable buckets (PENDING RUN),
    old champion as baseline row.
5.3 Compositional generalisation: B_clean 91.72/98.99/99.17 (+619/-0),
    strict holdouts, probe. Claim rule: trained constructions = coverage only.
5.4 Robustness & diagnostics: reorder 44.7->24.0->8.7 gap series, wording,
    masked control, C4 regression story (battery catches what headline hides),
    lock-in/format-channel experiment, measurement-guided iterative ablation
    NAMING (not "controlled ablation").
5.5 Backward compat & boundary: full final_test PAIRED vs old champion
    (McNemar, per-bucket, cost metrics: calls/tokens/latency/failure taxonomy)
    — THE pivotal experiment, PENDING (server fix in progress). Retirement
    claim scope: single-stage grammar-constrained planner replaces multi-stage
    generative planning layer; grounding/executor/verifier/evidence retained.
    Old-task 87.25 (400-sample) already suggests parity.

## 6 Discussion & Limitations
Operators driven by procurement needs; model-vs-executor capability split;
verification blind region; synthetic surfaces (dual-channel renderer plan:
canonical deterministic + independent naturalization with fidelity gates;
independent surface grammar for test, not paraphrase); single domain/schema;
stage-3: reflector-identified gaps proposing new primitives (future work).

## 7 Conclusion

## Pending experiments for this paper
1. Full final_test paired run (compose-v1/v2/v3 vs old champion per-item) —
   BLOCKED on serve fix (HF cache: good cache at /var/tmp/cicada/hf, use
   HF_HOME explicitly; still debugging).
2. Cost-metric collection during that run.
3. Optional: rule/template baseline; independent surface grammar test set.
