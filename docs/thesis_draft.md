# CICADA: Verifier-Gated Knowledge-Graph Question Answering over UK Public Procurement, and Bootstrapping Planner Training Data Through Partial Verifiability

*Working draft v0.2 — 2026-07-05. All numbers in this draft were recomputed from the artefacts
in the repository on this date; development-slice measurements are labelled as such. Items
marked [TBD] await the full teacher harvest and the local-model training experiments.
Voice is unified to "we" throughout. Figures are indicated as placeholders.*

---

## Abstract (draft)

Question answering over structured public data faces a trust dilemma: large language models
read natural language but answer unverifiably, while symbolic executors answer verifiably but
cannot read. We present CICADA, an end-to-end system for UK public-procurement question
answering in which an LLM only ever *proposes* a plan, and a deterministic
compile–ground–execute–verify stack is the sole answer authority.

Our central claim is methodological: although full answer correctness cannot be verified
without gold labels, a large and useful subset of failure *is* mechanically verifiable —
schema validity, literal groundability, dependency-graph acyclicity, evidential support,
aggregation safety, answer-shape consistency. We call this **partial verifiability** and use it
twice: at inference time as a safety gate, and at training time as the *acceptance function* of
a self-bootstrapping data engine whose blind spots (verifier-passing but externally wrong
plans) are routed into preference-training hard negatives instead of silently poisoning the
supervised pool.

Concretely, we contribute:
1. a knowledge graph of 215,221 UK contract-award records with conservative, audit-first
   entity resolution (204,711 aliases → 131,502 canonical organisations);
2. a 12,828-question benchmark whose gold programs are fully parameterised and whose oracles
   are validated by an independent second implementation at 99.88% agreement;
3. a two-step planning pipeline with a twelve-transform deterministic compiler and a
   repair loop that only acts on diagnosably-defective plans — on our development slice this
   reached 84% accuracy with zero hallucinated answers and 10/10 correct abstentions
   simultaneously ([TBD] full held-out test);
4. a teacher data engine that harvests verified SFT targets, repair trajectories, abstention
   supervision, preference pairs, and verifier-blind hard negatives, used to train local
   8B-class students (Qwen3-8B; Llama-3.1-8B for cross-base generality). [TBD: student results.]

---

## 1. Introduction

**Task.** Given a natural-language question about UK public procurement ("Which suppliers who
worked with Highlands and Islands Enterprise won contracts worth more than £10m in 2024?"),
return an answer traceable to knowledge-graph evidence, or abstain when the question is
ambiguous, unsupported by the schema, or matches no records.

**Why retrieval-augmented generation is not enough.** RAG retrieves text and lets an LLM
compose the answer; the composition step is unauditable, and questions requiring *exhaustive*
computation — counts, sums, rankings, multi-hop joins — fail structurally because no top-k
retrieval window contains "all matching records". Section 6 quantifies this on a controlled
220-question comparison: RAG baselines reach 23.2% (naive) and 25.5% (strengthened), scoring
0% on additive sums and bridge joins *despite* reaching 90% on single-record factoid lookup —
the one slice where retrieval suffices. Our system's figure on the same set is [TBD — being
rerun with the current planner generation; an earlier system generation already measured
91.8%, so this is a floor].

**Design principle.** The LLM proposes; a deterministic executor answers; a verifier judges; a
reflector repairs strictly within the verifier's jurisdiction.

**Contributions** are listed in the abstract; the thesis narrative is that one idea — partial
verifiability — organises all four.

*[Figure 1 placeholder: eight-stage pipeline architecture, §5.1.]*

---

## 2. Related Work

**KGQA benchmarks and semantic parsing.** GrailQA, LC-QuAD 2.0 and KQA Pro established the
template-first construction recipe (logical form → pseudo-language → crowd paraphrase → crowd
validation) and the i.i.d./compositional/zero-shot generalisation split; we adopt the recipe
with LLM stages substituted for crowd stages, and replace crowd validation with *mechanical*
fidelity checks after measuring that an LLM checker approved 9,762/9,762 rewrites including
known semantic drifts. Our dual-implementation oracle audit extends this line: benchmark
correctness is usually asserted; we measure it (99.88%).

**Verifier-guided generation and self-correction.** Intrinsic self-correction without external
signals degrades reasoning (Huang et al., ICLR 2024); execution- or environment-feedback
methods — Self-Debugging, CRITIC, text-to-SQL refiners such as DIN-SQL/MAC-SQL, and
QueryAgent's ERASER — correct selectively on typed error signals. Our reflector follows the
ERASER pattern (typed environmental errors, per-type prescriptions) and adds one element the
prior benchmarks did not need because they lack abstain classes: *provenance-gated
empty-result repair* — an empty result whose literals all trace to the question is an answer,
not a defect. We measured the naive alternative (repair whenever no answer) at −18 points and
6× hallucinations before adopting this design.

**Self-training and distillation.** STaR-style bootstrapping and rejection-sampling fine-tuning
accept model outputs into training data when a criterion passes. Our variant makes the
criterion an explicit runtime verifier (not gold-answer agreement), records *why* each sample
was accepted, and — unlike filter-only pipelines — retains verifier-passing-but-wrong outputs
as labelled hard negatives for preference optimisation, on the argument that the verifier's
decision boundary is exactly what the student must learn to respect.

**LLM-generated evaluation data.** Surface diversity work shows naive LLM paraphrasing
collapses toward near-copies; we condition rewrites on six explicit style axes and gate
acceptance mechanically, reporting distributional evidence (trigram-Jaccard median 0.429)
rather than asserting diversity.

---

## 3. Data and Knowledge-Graph Construction

### 3.1 What was built
Five years of OCDS releases (2022–2026) are ingested from `.jsonl.gz` dumps and deduplicated to
the latest release per OCID (166,277 releases), extracted to award-level records, and built
into typed Parquet node/edge tables: **215,221 contract nodes, 131,502 canonical organisation
nodes, 3,870 CPV nodes**, with buyer/supplier/classification/evidence edges. 176,002 records
carry additive contract values; 193,544 carry signed award dates.

### 3.2 Design decisions and alternatives
- **Latest-release snapshot** vs full amendment-chain modelling vs first release: amendment
  chains multiply schema complexity for questions the benchmark never asks; first-release
  discards corrections; latest-release matches the publisher's own presentation.
- **Columnar tables, not a graph database.** At this scale, vectorised joins outperform
  query-engine round-trips, keep the executor deterministic and unit-testable, and enable the
  independent-oracle audit (§4.4) with plain dataframes. A graph store adds operational surface
  without expressiveness the workload needs.
- **Value semantics as data, not convention.** Framework ceilings and call-off values must not
  be summed together; an explicit `value_is_additive` flag makes "additive-only money
  aggregation" a checkable guard rather than folklore.

### 3.3 Entity resolution: conservative, tiered, audited
204,711 raw organisation aliases resolve to 131,502 canonical organisations through ordered
tiers — registry-ID deterministic merges (26,704), safe deterministic variants (6,332),
normalised name+region (14,875), name-only (1,286), LLM-adjudicated borderline cases (255,
human-reviewable), government lookup (37) — with 82,013 singletons left unmerged and fuzzy
matching restricted to producing human-review candidate files. Notice-scoped identifiers
(GB-FTS) are never canonical.

*Why precision-first:* ER false merges corrupt every downstream count and sum. The cost —
residual case-variant splits of one organisation — is documented as a convention and handled
at query time by a variant equivalence class (`The X` / `X (BEIS)` → IN-filter) and
evidence-driven abbreviation expansion (ICB → Integrated Care Board), both added after the
oracle audit showed these were the only ambiguity classes with practical impact.

---

## 4. Benchmark Construction and Quality Control

### 4.1 Plan-first generation
**L1**: 33 template families instantiate fully-parameterised plans against the KG, compute
oracles by executing them, then verbalise canonical questions (9,044 generated; 6,772 after a
cleaning pass removed hidden-count presence filters and the ill-posed conjunction family).
**L2**: an LLM rewrites each question under four personas with a deterministic per-surface
accept/reject gate (9,762 accepted; 1,170 rejected with machine-readable reasons, dominated by
invented temporal relations, 890).

*Why plan-first:* every question is born with an executable gold program — the property this
project needs most, since programs are the training target. Post-hoc annotation of collected
questions is more natural but unverifiable in coverage and outside FYP cost.

### 4.2 The quality audit (v1 → v2)
A comparison against professional-benchmark practice exposed six defects; each was fixed
mechanically and each fix is itself testable:

| Defect (measured) | Fix |
|---|---|
| boolean answers 100% True (330/330) | mutated False twins, oracle recomputed independently |
| 42% of factoid answers from tiny value domains | 1,183 rows re-bucketed to `categorical` |
| 52% of unsupported questions lexically detectable | answerable contrast twins (cue attribute swapped) |
| L2 rewrite drift invisible to the LLM checker | position-aware mechanical drift detector (final precision: the 1 true defect, 0 false positives) |
| gold programs under-specified (thresholds/sides/params only in surface text) | backfill: 991 additive guards, 690 compare-params, 316 answer-fields, 15 top-k params |
| oracle circularity | independent second implementation (§4.4) |

### 4.3 Curation, balance, and split hygiene (v3)
The initial split was inverted (final_test 10,049 > train 5,534) and saturated easy buckets
dominated (count = 35.9% of train). Curation produced a stratified test core with per-bucket
caps and round-robin sampling across template families (harder generalisation classes
preferred; never selected by model performance, avoiding Goodhart bias), returned surplus
i.i.d. rows to train (leakage-free: nothing had been trained on; whole plan groups move
together), and enforced three hard gates in every subsequent build: **no plan straddles
train/eval; global id uniqueness; row conservation.** A follow-up matrix review corrected
residual supply inversions (comparison, boolean, top_k in test; bridge/factoid material in
train).

### 4.4 Independent oracle verification
A pure-pandas evaluator sharing no code with generator or executor recomputes every answerable
oracle from raw node/edge tables. First pass: 92.7% agreement, mismatches *clustered by
family* — clusters localise convention gaps. Diagnosed and resolved: bridge oracles implicitly
assumed the flat first-party universe and additive-only money (made explicit in gold and the
dataset card); an empty-string party name matched every record lacking that party (a constant
1.61B excess in every bridge sum); 230 degenerate empty constraints. **Final: 14,752/14,770 =
99.88% agreement** (residual 18: top-k parameter inference and one boolean surface-parsing
family). This is our answer to "how do you know the oracles are right".

### 4.5 Surface diversity and scarce buckets (v4)
Rewrites are conditioned on six explicit style axes (terse query, verbose context, embedded
request, syntactic flip, multi-sentence; typo noise in train only) with mechanical acceptance:
all gold literals verbatim, no invented years/CPVs, drift rules, and — for abstain questions —
mandatory preservation of the ambiguity/unsupported cue words (a rule added after three
diversified abstain surfaces lost their cues and were answered). Weak modes were measured and
strengthened (the initial syntactic-flip prompt produced near-copies; a mode-specific
similarity gate now rejects them). Scarce buckets were filled by template generation with
fully-parameterised gold and independent-evaluator oracles.

**Frozen benchmark (v4): 12,828 rows = train 9,267 / final_test 2,285 / dev_tune 556 /
dev_select 671 / dev_smoke 49**, plus a 7,483-row surplus pool held out of all splits.
Diversity coverage: 43.8% of eval-split surfaces rewritten in place (originals retained in
metadata); 2,589 styled train variants; trigram-Jaccard median 0.429. Final composition floors:
every final_test bucket ≥ 93 rows (top_k 128; abstain classes 120 each).

As a closing end-to-end check, running the complete answering system over a stratified train
sample surfaced four residual data defects (a scoring-shape gap, the three abstain-cue losses,
and one inherited degenerate-constraint class affecting 53 derived twins) — all fixed at source
before freezing. The strongest dataset validator available is the full consumer pipeline.

---

## 5. The Reasoning Pipeline

### 5.1 Architecture
*[Figure 1: pipeline diagram — Step-1 briefing → Step-2 graph plan → schema grounding →
entity grounding → deterministic compile (T1–T12 + three gates) → levelled execution →
verification → scoped reflection.]*

1. **Step-1 understanding** (gpt-5.4-nano): a dense natural-language briefing — answer type,
   explicit atoms, reverse reasoning tree, procedure, *named* intermediate target sets, role
   directions, ambiguity notes.
2. **Step-2 planning** (grok-4-1-fast → local student): a `graph_plan` under provider-enforced
   strict JSON — flat variable array with tree-coded ids (`b1a1` feeds `a1`), closed slot
   enums, relations, return spec.
3. **Schema grounding**: field text → canonical slots via candidate generation + type gate +
   confidence threshold + top-2 margin; relation-phrase direction cues override role nouns
   ("awarded TO X" makes X the supplier).
4. **Entity grounding**: threshold+margin resolution (never `hits[0]`), variant equivalence
   classes, abbreviation expansion.
5. **Deterministic compile**: twelve normalisation transforms convert recognisable planner
   idioms into executable structure (variable-reference filters → dependency edges; dependency
   direction conflicts resolved in favour of explicit dataflow; anchor-echo literals dropped;
   singular-question set-returns rewritten to uniqueness-checked selects; compare literals
   folded to thresholds), plus three compile-time gates: whole-program groundability, DAG
   validity, no unconstrained bind sources.
6. **Execution**: dependency-levelled evaluation over the flat KG universe with bind
   propagation, additive and nonzero guards, date-aware comparison.
7. **Verification**: preflight schema checks; runtime checks (uniqueness, additivity,
   population coverage); answer sanity; evidence verdict. Incomparable comparison sides raise
   errors instead of silently returning False.
8. **Reflection**: bounded, verifier-scoped repair (§5.3).

### 5.2 Why two steps, and planner-specific contracts
Removing Step-1 cost 12 points in a controlled development ablation — the briefing separates
semantic reading from structural discipline, letting each be trained and diagnosed
independently. Two further findings (details in Appendix A): the Step-2 prompt and JSON schema
are *model-specific configuration* — a four-cell A/B showed each planner has its own optimum
and giving the stronger model the weaker model's scaffolding costs accuracy; and feeding the
stronger planner the weaker model's *finished structured output* (instead of the prose
briefing) anchors it 16 points downward. Teachers must receive material, not conclusions.

### 5.3 Reflection that respects legitimate absence
The naive repair loop ("no answer yet → replan") measured **net-negative**: 84%→66% with
hallucinations 1→6 on the development slice, because correct abstentions were "repaired" into
confident wrong answers — consistent with the intrinsic-self-correction literature and the
text-to-SQL empty-result trap. The rebuilt reflector is eligibility-gated: repair fires only on
diagnosable defects (compile rejection, executor error, planner-invented literals in an empty
result — *provenance-gated empty repair*); feedback carries the failed plan verbatim plus a
deterministic reason→prescription table; repairs re-run the full ground→execute→verify path;
and a repaired answer replacing a verifier-flagged one must be strictly cleaner.

### 5.4 Development measurements
All numbers below are on the 49–50-question development slice (dev_smoke) and are directional;
held-out results on final_test (2,285) are [TBD]. Configuration matters, so the headline
numbers are disambiguated:

| Run (chronological) | Step-2 config | Repair | Accuracy | Hallucinations | Abstain correct |
|---|---|---|---|---|---|
| prompt-only planning | early | off | 56% | — | — |
| + strict JSON schema | early | off | 62% | — | — |
| + compiler transforms (era of T1–T7) | card/filler | off | 66–70% | 1–2 | 9/10 |
| + measured planner config | lean/optional | off | 80.0% | 2 | 8/10 |
| same, naive repair loop | lean/optional | on (naive) | 66% | 6 | 4/10 |
| same, rebuilt reflector | lean/optional | on (gated) | 79.6% | 3→1 | 7→9/10 |
| **closing run** (all fixes + structural resample) | lean/optional | on (gated) | **84.0%** | **0** | **10/10** |

Replaying archived planner outputs through the final compiler gives an 88% ceiling on the same
slice, i.e. live-sampling variance accounts for ±2–3 questions. The same deterministic
hardening lifted every planner tier: direct compilation of Step-1 intent programs (no Step-2
LLM) 58%→62%; nano-as-planner 64%→76%; grok 70%→88% (replay). Residual misses concentrate in
bridge/comparison structure — the training targets.

---

## 6. Baseline Comparison: Why Not RAG

**Setup.** A 220-question comparison set (20 per category: six L1 operation families + five L2
subsets, deterministic stride sampling; `data/qa/eval/compare_set.jsonl`). All systems are
scored identically and type-aware (§8); abstention is correct on unanswerable rows.
- **Ours**: [TBD — being rerun with the current two-step CICADA system on the same 220
  questions; the table below currently shows the earlier rule-decomposition generation, whose
  91.8% therefore UNDER-states the current system. RAG rows are final.]
- **RAG-naive**: each contract record rendered as one text document; TF-IDF retrieval
  (50k-feature vocabulary), top-10 records; gpt-5.4-nano answers from the retrieved records
  only, with a forced-JSON contract and an explicit unknown/abstain option.
- **RAG-strong**: entity-boosted documents (buyer/supplier/CPV term repetition), sublinear
  TF-IDF with word bigrams (120k features), top-40 records — a credible retrieval upgrade.

**Results** (`data/qa/eval/compare/`):

| System | Overall (220) | additive_sum | filtered_count | bridge_join | unanswerable | contract_factoid |
|---|---|---|---|---|---|---|
| Ours | **91.8%** (202/220) | 100% | 100% | 100% | 100% | 55% |
| RAG-naive | 23.2% | 0% | 10% | 0% | — | **90%** |
| RAG-strong | 25.5% | 0% | 10% | 0% | — | 90% |

Excluding the conjunction family (later shown ill-posed and removed from the benchmark), ours
is 95.5% (191/200). The pattern is the argument: RAG is *good at the one thing retrieval can
do* — locating a single record (factoid 90%, beating our early planner's 55% on that slice) —
and structurally incapable of exhaustive aggregation and joins, where upgrading the retriever
moved nothing (0%→0% on sums and bridges). This motivates the executor-authoritative design
rather than better retrieval.

---

## 7. The Teacher Data Engine

### 7.1 Attempt protocol
For each training question: Step-1 briefs; the teacher plans (with structural resampling on
detectable failure); the pipeline compiles, executes, verifies; up to k verifier-guided repairs
each re-run the full path. Every attempt is recorded. Acceptance into supervision is
**verifier-based** — a teacher output is never treated as gold because the teacher produced it.

### 7.2 Three-tier routing: making partial verifiability concrete
Pilot harvesting exposed the core tension directly: a material fraction of verifier-accepted
plans are externally wrong (the canonical case: a clean-looking list computed from a silently
broadened match set). Routing turns the gap into signal:

*[Figure 2 placeholder: routing diagram.]*

- verified ∧ externally correct ∧ answer-shape-consistent → **verified_sft** (target: the
  *compiler-normalised* plan — canonical shapes, not teacher idiosyncrasies);
- verifier-passing ∧ externally wrong → **hard_negatives**: plans the runtime verifier cannot
  distinguish from correct ones — exactly the decision boundary preference training must
  sharpen; one oracle-*gated* repair attempt (feedback: "failed external validation", content
  hidden) can convert these into chosen/rejected pairs of maximal contrast;
- correct abstentions → **abstain_sft** (anti-hallucination supervision);
- all-fail → **failures** (next-round curriculum);
- plus **repair_sft** (feedback→repaired-plan trajectories) and attempt-protocol DPO pairs.

A mechanical answer-shape gate (boolean question ⇒ boolean answer) requires no oracle and
catches operation drift during repair.

### 7.3 Harvest and training plan
The engine runs over the full 9,267-question training split. [TBD: final yield table —
per-bucket verified rate, verified@1 vs verified@k, five-pool composition.] Students: Qwen3-8B
(primary) and Llama-3.1-8B (cross-base generality), QLoRA SFT → rejection-sampling fine-tuning
(the student replaces the teacher inside the same harness; previously-failed questions it now
solves become new verified data) → DPO over the three pair sources. Local serving uses
grammar-constrained decoding, which enforces the exact plan schema more strongly than the API
teacher's server-side mode — model choice is therefore purely about planning semantics.
Step-2 is distilled first (largest measured gap); Step-1 and the no-LLM intent path second.
[TBD: all student numbers.]

---

## 8. Evaluation Protocol and Metrics

**Correctness** is type-aware (`answers_match`):
numeric answers — relative tolerance 1e-6; dates — ISO `YYYY-MM-DD` prefix equality; strings —
case-folded exact match; sets — order-insensitive string multiset equality; top-k — ordered
name-sequence equality (rank-pair or bare-name forms both accepted); comparisons — equality of
the boolean verdict; a single-element list answering a scalar question is unwrapped (semantic
equality, not leniency).
**Abstention credit**: on rows whose expected status is ambiguous/unsupported/no-results, the
system is correct iff it produces *no* answer; producing any answer counts as a hallucination.
We additionally report whether the abstention carried the *right reason* as a secondary
diagnostic.
**Splits**: dev_tune (556) for checkpoint selection; dev_select (671) for method-level
decisions; final_test (2,285) evaluated once per system generation; dev_smoke (49) as a
regression slice.
**Reporting**: per-bucket matrices alongside overall accuracy, since bucket composition is
controlled; hallucination count and abstention accuracy are always reported next to accuracy
because they trade against each other.

---

## 9. Threats to Validity / Limitations

- **Convention-relative correctness.** Oracles are defined w.r.t. documented conventions
  (flat first-party record universe; additive-only money; latest-release snapshot). The dual
  audit bounds convention risk (18/14,770 unresolved) but a different convention set would
  yield different oracles.
- **LLM-generated language.** Six style axes widen but do not equal real-user distribution;
  no crowd validation was affordable; originals are retained beside every rewrite.
- **Single domain, single KG.** Architecture claims rest on within-domain ablations.
- **Teacher ceiling and verifier coverage.** Students learn from a teacher measured (on the
  development slice) at 84–88%; harvest composition depends on verifier coverage, which is
  itself the object of study.
- **Development-slice inference.** All ±2-point development comparisons used ≤50-question
  slices and are directional; paper-grade claims rest on final_test (2,285). [TBD]

---

## 10. Reproducibility

Every dataset version (v1→v4), every cited run, all audit outputs, and a dated decision log
(`docs/cicada_worklog.md`) are in the repository. Hard integrity gates (plan-level split
isolation, id uniqueness, row conservation) execute inside every data build. The independent
evaluator, drift detectors and shape gates are standalone scripts with no dependency on the
system under test.

---

## Appendix A (planned): engineering findings referenced from §5.2
Model-specific prompt/schema four-cell A/B; structured-Step-1 anchoring experiment;
dependency-direction conflict resolution rules (naming vs explicit vs relation edges); the
heredoc/regex tooling incident log.
