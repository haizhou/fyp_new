# Reasoning Pipeline Design

Status: implemented runtime reasoning core for QAv2 evaluation, including
bounded N-hop decomposition, extended operations, predicate execution, and
auditable abstention handling.

This document describes the runtime reasoning pipeline that will answer user
questions over the procurement KG. It is deliberately aligned with the QA
benchmark pipeline: the same logical semantics used to generate golden answers
should also be used to execute real questions.

## Design Goal

The reasoning system should not be an LLM-only answer generator. The LLM is a
planner and verbaliser; the KG executor remains the authority for retrieval,
aggregation, and final answer values.

Core constraints:

- use the frozen KG v0.1 schema as the structured evidence source;
- prefer deterministic tabular/KG execution over graph-neural path guessing;
- require exhaustive retrieval for count, sum, set, top-k, comparison, and other
  reduction operations;
- use documents only after KG retrieval has identified a specific contract or
  OCID;
- expose no internal/source identifiers in user-facing answers unless the user
  explicitly asks for an identifier and it is a public procurement code such as
  CPV.

## Reference Patterns

The design combines several proven patterns:

- Previous local prototype: `step1_query_understanding.py` to
  `step5_reflector.py` showed a useful sequence of query understanding,
  candidate search, hard verification, evidence verdict, answer card, and
  reflector routing.
- RoG-style KGQA: an LLM proposes relation/query plans, but the KG grounds and
  verifies real paths.
- ToG-style graph traversal: relation/entity expansion can be iterative for
  open-ended questions, but each step must be checked against the KG.
- BioGraphletQA-style benchmark generation: prompt variants and independent LLM
  filtering are part of the QA build process, not ad hoc post-processing.
- Text-to-Cypher / query-spec systems: generate an executable intermediate
  query, run it, and reject invalid or ambiguous outputs before producing an
  answer.

The project should borrow these structures, not copy their exact algorithms.
The procurement KG is custom, tabular-friendly, and rich in structured fields,
so deterministic query execution should be the default.

## Runtime Principle

The runtime pipeline is an auditable search-and-execution system, not an
LLM-answering system.

```text
LLM proposes; KG executes; verifier judges; reflector refines.
```

The planner may interpret the question and propose a query plan, but the final
answer value must come from deterministic KG execution. The verifier and sanity
gates decide whether the result is acceptable. The reflector may repair or
reject a failed attempt, but it must not invent an answer.

Every run should preserve enough trace to diagnose which layer failed:

- raw planner response;
- pre-ground and grounded specs;
- grounding status and reason;
- preflight checks and failed checks;
- execution status, evidence count, dedupe key, and aggregation method;
- evidence verdict and optional document/web status;
- answer-sanity flags;
- reflector action, fallback use, and fallback reason.

## High-Level Flow

```text
User question
  -> LLM planner / deterministic fallback planner
  -> RuntimeQuerySpec candidate
  -> grounding into frozen KG schema
  -> retrieval gating and entity/CPV/time resolution
  -> preflight verifier
  -> deterministic KG executor
  -> runtime verifier checks
  -> evidence verdict, with optional document/web corroboration
  -> answer sanity gate
  -> reflector repair/reject decision
  -> answer card
```

## Stage 1: Planner / Query Understanding

The planner converts the user question into one or more `RuntimeQuerySpec`
candidates. It does not answer the question.

Responsibilities:

- classify the operation family: factoid, count, sum, comparison, temporal,
  CPV/category, role/path, top-k, or unsupported;
- extract visible semantic constraints from the question;
- link organization names, CPV descriptions/codes, years, value terms, and role
  words to KG fields;
- mark unsupported requests early, for example carbon clauses or payment terms
  if the KG lacks the required fields;
- return multiple candidates when the wording is ambiguous.

The first implementation can use a provider-agnostic open-source model adapter
with schema-validated JSON output. It should also support a deterministic dry
run for tests.

Required trace:

- `raw_response`;
- `planner_source`;
- `fallback_used`;
- `fallback_reason`;
- `fallback_planner`;
- pre-ground `RuntimeQuerySpec`.

## Stage 2: Grounding

Grounding locks the planner output into the frozen KG schema. The LLM may emit
near-miss names such as `total_contract_value` or `cpv_code`, but the runtime
must execute only canonical fields.

For sums, grounding enforces the additive-value safety policy:

- `answer_field = value_amount`;
- hidden `value_is_additive = True` constraint;
- exhaustive retrieval required;
- stable `dedupe_key` for contract-level aggregation.

Required trace:

- `pre_ground_spec`;
- `grounded_spec`;
- `grounding.ok`;
- `grounding.reason`;
- `grounding.changes`;
- `grounding.issues`.

## Stage 3: Candidate Retrieval

Runtime retrieval should start with exact KG indexes rather than RGCN or
unconstrained beam search.

Preferred retrieval order:

1. exact structured filters over `contract_nodes`, `org_nodes`, and CPV fields;
2. alias/name lookup for buyer and supplier organizations;
3. CPV hierarchy and category mapping;
4. bounded graph expansion for role/path questions;
5. optional relation/path search only when a query cannot be expressed as a
   direct `RuntimeQuerySpec`.

If exact matching fails, the runtime may invoke a bounded semantic fallback:

1. retrieve top-k KG candidates for the failed text constraint, using either a
   lexical baseline or an embedding index;
2. pass only those candidates to a selector model;
3. allow the selector to choose one candidate or abstain;
4. rewrite the failed constraint only with a selected KG value;
5. retry through the same grounding, preflight, executor, verifier, and
   reflector loop.

The selector must not invent values. The trace records the original mention,
top-k candidates, selector response, selected KG value, and retry result.

Reduction operations must be exhaustive. For `count`, `sum`, `set`, `top_k`,
`compare`, and temporal aggregations, the executor must retrieve the complete
matching evidence set before computing the answer. Beam-pruned evidence sets are
not valid golden evidence.

## Stage 4: Preflight Verifier

Before querying the KG, the verifier performs cheap deterministic checks:

- all referenced fields exist;
- constraints are not self-contradictory;
- reduction operations require exhaustive retrieval;
- unsupported operations are rejected before execution.

Required trace:

- `preflight.checks`;
- `preflight.failed_checks`.

## Stage 5: Deterministic KG Execution

Execution evaluates a `RuntimeQuerySpec` against the KG and returns an
`ExecutionResult`.

The executor should reuse the semantics proven in the QA benchmark:

- `constraints`;
- `answer_operation`;
- `answer_field`;
- `answer_value_type`;
- `dedupe_key`;
- `sampled_evidence_ids` / matched evidence IDs;
- value additive policy for sums.

The runtime executor may support more operations than the benchmark at a later
stage, but the initial version should support the benchmark-covered set first:

- `select_unique`;
- `count`;
- `sum`;
- CPV/category count;
- temporal count;
- conjunction / multi-filter count.

Required trace:

- `execution_status`;
- `answer_raw_value`;
- `evidence_count`;
- `evidence_ids`;
- `applied_constraints`;
- `dedupe_key`;
- `aggregation_method` where relevant.

## Stage 6: Runtime Verification

Verification decides whether a result can be answered safely.

Required checks:

- schema validity: every referenced field and relation exists;
- constraint satisfiability: no contradictory constraints;
- completeness: reduction operations used all matching evidence;
- uniqueness: factoid questions have one deterministic answer;
- additive safety: sums include only additive value rows;
- identifier safety: answer text does not rely on hidden internal IDs;
- coverage boundary: supplier, value, evidence, and region limitations are
  reported rather than silently ignored.

If verification fails, the system should return a structured failure or route to
the reflector. It should not ask the LLM to patch the numeric answer.

An optional LLM verifier analyzer may explain failed checks, but it is advisory
only. It cannot override deterministic checks, cannot mark an answer correct,
and cannot compute answer values. Its output is stored as `verifier_analysis`.

## Stage 7: Evidence Verdict

The KG is the primary evidence source. Documents are passive, opportunistic
support.

Trigger rule:

1. KG execution identifies a contract, award, OCID, or evidence set.
2. If the answer needs clause-level support or contradiction checking, inspect
   document URLs attached to that KG evidence.
3. Make a lightweight live request.
4. If the content is accessible, extract temporary text in memory and select the
   most relevant snippet.
5. If unavailable, login-gated, empty, or unsupported, continue with KG-only
   evidence and record the limitation.

No offline document corpus, global document retrieval index, or default OCR is
part of the runtime path.

Document/web evidence must store source metadata when used:

- `source_url` or `document_id`;
- `retrieved_at`;
- page or section where available;
- `span_text`;
- character offsets where available;
- `supports`, `contradicts`, or `not_found`.

No verbatim span means no claim of document support.

## Stage 8: Answer Sanity

Answer sanity checks values that are executable but suspicious, especially
money totals. It does not rewrite the answer; it downgrades confidence and adds
limitations.

Checks include:

- nominal / placeholder value;
- single dominant contributor;
- framework-ceiling risk;
- non-additive value risk.

Required trace:

- `sanity_flags`;
- `confidence_before`;
- `confidence_after`;
- `limitations`.

## Stage 9: Reflector

The reflector is a query refinement controller. It repairs or rejects failed
attempts; it does not reason up a new factual answer.

Common actions:

- add a missing additive guard;
- reject invalid fields as schema issues;
- report no evidence in strict mode;
- cautiously relax non-answer constraints only when configured;
- mark unsupported operations explicitly;
- report ambiguous factoids instead of choosing one value.

Required trace:

- first-pass status;
- repair type and reason;
- original and repaired specs;
- attempt list;
- repair success;
- fallback use and fallback reason.

An optional LLM reflection analyzer may comment on the deterministic reflector's
decision. It can recommend candidate retrieval, clarification, unsupported, or
manual review, but the deterministic reflector action remains the executed
control signal unless explicitly promoted in a later experimental branch. Its
output is stored as `reflector_analysis`.

## Stage 10: Answer Card

The answer card is the only layer allowed to produce user-facing answer text.
It must not change the computed answer.

Answer cards should include:

- final answer value;
- answer operation and value type;
- natural-language answer;
- evidence IDs and concise KG evidence;
- optional document verdict/snippets;
- confidence/status derived from verification, not model preference;
- limitations and skipped evidence sources;
- trace metadata for audit.

## Bounded Decomposition

Complex questions may be decomposed into smaller executable specs, but each
substep must go through the same planner-ground-execute-verify loop.

Example:

```text
For the buyer of contract X, how many services contracts did they publish in 2025?
```

Substeps:

1. identify contract X;
2. extract buyer;
3. count 2025 services contracts for that buyer.

Each substep records its own grounded spec, execution status, intermediate
answer, and evidence IDs. Multi-hop reasoning is therefore an auditable chain,
not a hidden beam-search path.

## Package Layout

Initial package:

```text
src/procurement_graph/reasoning/
  __init__.py
  models.py
  planner.py
  verifier.py
  executor.py
  answer_card.py
```

Now implemented (all with tests):

```text
src/procurement_graph/reasoning/
  linking.py       # entity, CPV, field, date, value linking (+ pluggable OrgResolver)
  retrieval.py     # exact-filter retrieval planning + expansion gating + entity resolution
  evidence.py      # KG-first evidence verdict + opportunistic DocumentInspector hook
  reflector.py     # deterministic repair routing (relax / additive-guard / dedupe)
  pipeline.py      # ReasoningPipeline: the end-to-end closed loop -> ReasoningTrace
  kg_backend.py    # RuntimeKGBackend adapter over the KG query interface (import on demand)
  llm_planner.py   # LLM planner adapter (any chat client), rule-based fallback
```

`kg_backend.py` and `llm_planner.py` are deliberately excluded from the package `__init__`
so importing the core reasoning logic needs no pandas / KG / LLM dependency.

Keep this separate from `procurement_graph.qa.benchmark`. The runtime package
may reuse concepts from the benchmark, but the benchmark generator should remain
an evaluation/data-construction component rather than a production dependency.

## First Implementation Slice

1. Define model contracts in `reasoning.models`.
2. Implement a deterministic executor for benchmark-supported
   `RuntimeQuerySpec` operations.
3. Implement a schema-only planner dry run for tests.
4. Add an open-source-model planner adapter with strict JSON validation.
5. Add verifier and answer-card tests using a small mock KG.
6. Only then add bounded graph/path expansion for open-ended role questions.

RGCN or learned graph retrievers may be added later as an experiment or
baseline, but they should not be required for the main reasoning path.

## Current Implementation

The runtime now implements the full deterministic closed loop end to end:

- `reasoning.models`: shared data contracts (plans, query specs, execution results,
  evidence verdicts, answer cards, reflector actions, reasoning trace);
- `reasoning.linking`: deterministic year/category/CPV/signed-date/org linking and
  unsupported-concept detection, consumed by the planner;
- `reasoning.planner`: `ReasoningPlanner` protocol, planner JSON schema, payload→plan
  conversion, and a deterministic `RuleBasedDryRunPlanner`;
- `reasoning.llm_planner`: `LLMReasoningPlanner` over any `complete_json` chat client, with
  a rule-based fallback on any parse/schema failure;
- `reasoning.retrieval`: `plan_retrieval` — exact-filter retrieval, capability gating for
  role/path/comparison/top-k, and organisation-name resolution;
- `reasoning.executor` + `reasoning.verifier`: `select_unique`/`count`/`sum` execution with
  schema, constraint-conflict, exhaustiveness, uniqueness, and additive-value checks;
- `reasoning.evidence`: KG-first `build_evidence_verdict` with an opportunistic, non-default
  `DocumentInspector` hook (`NullDocumentInspector` by default);
- `reasoning.reflector`: deterministic repair routing — broaden an over-specific filter,
  add the additive-sum guard, dedupe contradictory constraints, or mark unsupported / ask
  to clarify;
- `reasoning.answer_card`: user-facing cards that never change the computed answer;
- `reasoning.pipeline`: `ReasoningPipeline.run(question) -> ReasoningTrace`, orchestrating
  plan → retrieval gating → execute → (reflector-driven bounded repair) → evidence → card,
  surfacing applied auto-repairs as answer-card limitations;
- `reasoning.kg_backend`: `RuntimeKGBackend` adapter mapping runtime constraints to the KG
  query interface (splitting `between` into `gte`/`lte`), plus a records-based org resolver.

### Runtime usage

```python
from procurement_graph.reasoning import ReasoningPipeline, RuleBasedDryRunPlanner
from procurement_graph.reasoning.kg_backend import RuntimeKGBackend

backend = RuntimeKGBackend.from_directory("data/kg")
pipeline = ReasoningPipeline(backend=backend, planner=RuleBasedDryRunPlanner(),
                             org_resolver=backend.org_resolver())
trace = pipeline.run("How many works contracts were published in 2024 under CPV 45000000?")
print(trace.answer_card.answer_text, trace.evidence_verdict.status)
```

Swap `RuleBasedDryRunPlanner()` for `LLMReasoningPlanner.from_env("gpt-5.4-nano")` to use a
live planner; the deterministic executor remains the answer authority either way.

Tests: `tests/test_reasoning_runtime.py`, `tests/test_reasoning_planner.py`, and
`tests/test_reasoning_pipeline.py` (39 total) cover linking, retrieval gating, the reflector
repair routes, evidence verdicts (incl. document contradiction), the end-to-end pipeline
(count, reflector-repaired sum, no-results relaxation, unsupported, ambiguous), the KG adapter
(constraint translation incl. `between`, org resolution), and the LLM planner adapter
(valid payload executes; malformed payload falls back to rules).

## Original Next Extensions (partly superseded)

The following list was written before the Phase 2 reasoning work. It is kept for
history, but several items have since been implemented or narrowed. See the
"Phase 2 Implementation Update" section below for the current status.

- Bounded graph expansion for `role_path` questions (retrieval currently gates them as
  unsupported rather than guessing);
- comparison / superlative / top-k executors once the benchmark executor gains them;
- a live `DocumentInspector` implementation (opportunistic fetch of contract document URLs);
- an LLM reflection step layered after the deterministic reflector for open-ended failures.

## Phase 2 Implementation Update

The runtime has moved beyond the original skeleton. It now supports the main QAv2
reasoning families needed for targeted benchmark evaluation.

### Bounded N-hop Decomposition

The decomposition layer is not a 2-hop special case. A `DecompositionPlan` is a
bounded sequence of substeps. Each step can bind a prior step's emitted entity set
as an `in` filter on a later step:

- a 2-hop bridge is two steps;
- a 3-hop chain is three steps;
- an N-hop chain is N steps, bounded by `max_hops=4`;
- `compare` is two independent answer steps plus a combiner, not a chain.

The bounded depth is deliberate. The procurement KG is tabular and exact-filter
friendly, so unbounded graph expansion would reintroduce graph-era beam-search
machinery without improving evidence guarantees.

Each answer step runs the shared deterministic executor, so grounding,
exhaustiveness, additive-value safety, evidence counting, and answer-sanity checks
apply per hop. Entity-set steps emit typed distinct values such as buyer names,
supplier names, or CPV IDs.

Implemented decomposition families include:

- buyer -> supplier set -> contracts;
- supplier -> buyer set -> contracts;
- buyer -> CPV set -> contracts;
- compare two independent filtered counts;
- synthetic 3-hop chain tests such as buyer -> suppliers -> those suppliers'
  other buyers -> count.

### Extended Operation Support

The executor now supports the QAv2 extended operation families:

- `exists`: returns `False` for an empty matching set, not `no_results`;
- `count`: returns `0` for an empty matching set, not `no_results`;
- `sum`: still requires additive value evidence and applies the hidden additive
  guard;
- `argmax` / `argmin`: returns the record with the extreme numeric `sort_field`;
- `distinct_set`: returns sorted distinct values for the requested answer field;
- `top_k`: groups the exhaustive match set and ranks by count or sum;
- `predicate`: computes a boolean over a field, count, or sum against a numeric,
  string, or date threshold.

The count/exists empty-set semantics are important for bridge and compare queries:
a sub-count of zero is a valid answer, not a failed query that should be relaxed
into a different question.

### Decomposition-Aware Planning

`DecompositionAwarePlanner` recognises QAv2-style extended operations and bridge
joins while falling back to the base rule planner for direct count/sum/factoid
queries.

It covers:

- filter-only operations such as highest/lowest value contracts and top-k;
- boolean existence questions;
- boolean value, date, and field-equality predicates;
- set/list questions;
- compare questions;
- bridge semijoins over buyer/supplier/CPV intermediate sets.

This planner is intentionally benchmark-tuned and deterministic. A general LLM
decomposition planner remains future work.

### Evidence Materialisation

Large reductions and bridge joins can match tens of thousands of rows. The runtime
keeps the answer exact while capping heavy evidence payloads:

- `count` and `exists` use vectorised backend counts plus a capped evidence sample;
- `sum` projects only the fields needed for aggregation and sanity;
- `evidence_count` remains exact;
- `evidence_ids` and `ocids` are capped to 200 and marked with
  `evidence_ids_capped`;
- postflight checks receive exact matched counts even when evidence rows are sampled;
- sum sanity uses contributor summary metrics when available.

This optimisation preserves semantics while making bridge evaluation practical.

### QAv2 Evaluation Signals

Current QAv2 evaluation is split into executor-ceiling and pipeline modes:

- executor ceiling is perfect for naturalized and coverage-fixed direct questions;
- extended operations improved from essentially unsupported in the base rule planner
  to roughly 89% in pipeline mode after decomposition-aware planning and `predicate`
  execution;
- bridge joins are now measurable after evidence-materialisation optimisation; a
  400-row bridge sample reached roughly 86% in pipeline mode;
- unanswerable evaluation now distinguishes safe abstention from right-reason
  abstention and hallucination, exposing remaining unsupported-field detection
  failures.

### Updated Remaining Work

The old "not yet built" list above is partly obsolete: comparison, top-k, bounded
decomposition, and predicate-style booleans now exist. Remaining work is narrower:

- close the unsupported-field detection gap so non-KG fields such as payment terms,
  bidder counts, evaluation scores, delivery performance, and fairness judgements
  route to `mark_unsupported`;
- improve canonical organisation resolution for bridge and set-list residue where
  surface-name variants still block valid plans;
- recognise the remaining tiny bridge template families;
- build a general LLM decomposition planner for arbitrary natural language;
- move QAv2 runtime eval outputs out of the frozen dataset directory and into a
  dedicated reports/eval directory;
- implement a live `DocumentInspector` only as verdict-time optional corroboration,
  not as a main retrieval system.
