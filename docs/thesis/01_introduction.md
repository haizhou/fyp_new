# Chapter 1 — Introduction

## Chapter thesis (what this chapter must convince the reader of)

Question answering over structured public data cannot be trusted to an LLM's free generation,
and cannot be fully verified either — but a large, precisely characterisable part of failure
IS mechanically verifiable. The dissertation's one-sentence thesis, stated verbatim in this
chapter and defended across chapters 5–7, is: **"Supervision flows only from the verifiable
region, yet the learned competence extends beyond it."** The reader must leave chapter 1
knowing (a) the task and why retrieval-augmented generation structurally fails it, (b) the
three-layer design answer (hard verification / abstention / learning), and (c) the four
contribution pillars with their headline evidence — so that every later chapter is a
deepening, not a surprise.

## Section outline

### 1.1 Task and motivation
Natural-language QA over UK public procurement (215,221 contract-award records): answer with
KG-traceable evidence, or abstain when the question is ambiguous, unsupported by the schema,
or matches no records. Public-money accountability makes unverifiable answers worse than no
answer.

### 1.2 The trust dilemma and why RAG is not enough
LLMs read but answer unverifiably; executors answer verifiably but cannot read. RAG fails
structurally on exhaustive computation (counts, sums, rankings, multi-hop joins): no top-k
window contains "all matching records". Cite RAG baselines vs the zero-shot scaffolding floor.

### 1.3 The one-sentence thesis
State the frozen sentence, then unpack: training signal comes 100% from the verifier-passed,
oracle-gated verifiable region (the oracle filters, never authors); yet the learned competence
extends past the projection line — the teacher decays 6–12pt from iid to the hard-composite
slice while the fully-local student shows no measurable decay (+0.8pt, n=136). This is the
mechanism answer to "why is partial verifiability enough".

### 1.4 Approach overview: the three-layer structure
One paragraph per layer (full treatment in ch. 5): (i) hard-verification layer — every
property projectable to structure/execution is pinned by deterministic checks; (ii) abstention
layer — the unprojectable residue is a blind region caught by abstention (three abstain traps
are its completeness test); (iii) learning layer — signal from the verifiable region drives
distillation and bootstrapping, which in turn covers part of the blind region.

### 1.5 Contributions (the four pillars)
1. **Bootstrap closed loop** — three instantiations of learning from the verifiable region:
   teacher filtering, student self-harvest that overtakes the teacher, Step-1 denoising
   distillation (ch. 6).
2. **Gain decomposition** — separating the hard-verification layer's contribution from the
   learning layer's (scaffolding floor vs training gain; ch. 5–6).
3. **DPO near-miss toxicity** — a mechanism-explained negative result contrasting subtractive
   preference suppression with additive distillation (ch. 7).
4. **Benchmark methodology** — the abstention layer's test apparatus and the credibility
   foundation of every number: dual-implementation oracles, plan-level split isolation,
   dev/confirmatory discipline (ch. 4).

### 1.6 Results preview
The five-pairing final_test scoreboard in one table and one sentence: teacher 69.76 <
llama-hybrid 75.97 < qwen-hybrid 78.03 < llama-fully-local 83.33 < qwen-fully-local 85.65
(n=2,285; all pairings McNemar p<1e-14), at zero marginal API cost for the fully-local stack.

### 1.7 Dissertation structure
One line per chapter, mapped to pillars.

## Evidence manifest

| Number | Where used | Source / artifact |
|---|---|---|
| Scoreboard 69.76 / 75.97 / 78.03 / 83.33 / 85.65 (n=2,285, all p<1e-14) | §1.6 | [TABLE-SOURCED] FINAL scoreboard; outputs/eval/final_test/{teacher_r1,hybrid_llama_sft,hybrid_qwen_dpo,fully_local_llama,fully_local_qwen} |
| Qwen FL vs teacher +15.89pt CI [+14.07,+17.70] | §1.6 | [TABLE-SOURCED] FINAL_TEST headline block |
| Teacher hard-composite decay −10.3/−6.1/−12.5; FL +0.8 (iid n=103, slice n=136) | §1.3 | [TABLE-SOURCED] iid→ood_candidate table |
| RAG naive 31.5% (82/260), RAG strong 28.8% (75/260), v1, DEV | §1.2 | [TABLE-SOURCED] outputs/eval/baselines/rag_{naive,strong}/... — label as v1 DEV context, not headline |
| Zero-shot scaffolding floor 61.5% v1 / 70.4% v2.2 (DEV) | §1.2/§1.5 | [TABLE-SOURCED] outputs/eval/matrix{,_v2}/cicada-qwen3-zeroshot/ |
| 215,221 contracts / 131,502 canonical orgs | §1.1 | [DOC-SOURCED: kg_enrichment_plan.md, PROJECT_STRUCTURE.md — promote to master table] |
| 12,828-question benchmark | §1.5 | [TABLE-SOURCED] benchmark arithmetic line |
| Figure slot: architecture overview (three layers + four checks) | §1.4 | [PENDING: render; no F-number assigned yet] |

## Claims discipline (this chapter must NOT)

- MUST NOT claim "the system verifies its answers" unqualified — write: everything hard-
  verifiable is hard-verified; the residue is squeezed minimal, then abstention and learning
  catch it. The four checks shrink the blind region; they never eliminate it.
- MUST NOT call the reflector a verifier anywhere in the pipeline overview (reflector =
  consumer of verifier signal, gated; the design deliberately avoids "uncertain model verifies
  uncertain model").
- MUST NOT use "OOD" for the hard-composite slice in the thesis statement or preview — the
  slice is difficulty-composite, present in both train and test (plans disjoint, surfaces
  novel); "hard-composite slice" is the only permitted name.
- MUST NOT preview dev-set numbers as results (the DEV ladder appears in §1.5 only as
  "gain decomposition exists", with final_test carrying the claims).
- MUST NOT compare RAG baselines (v1 pipeline, DEV set) directly against v2.2/final_test
  numbers in one sentence — versions and sets must be named.
- MUST NOT state teacher inferiority from the single final_test replicate alone — the fixed
  two-layer wording (final_test single replicate + DEV three replicates) applies from the
  first mention.
