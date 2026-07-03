# Reasoning Architecture Review

Date: 2026-07-01

A senior-architect review of the runtime reasoning pipeline (`procurement_graph.reasoning`)
against the author's previous 5-step procurement KGQA pipeline (`step1_query_understanding` …
`step5_reflector`), and the hardening changes that resulted.

## 1. What the previous pipeline established (and we keep)

The prior pipeline is more mature and encodes several correct, hard-won principles:

- **The LLM plans/verbalises; a deterministic engine is the sole answer authority.**
- **Reductions must be exhaustive** — `count`/`sum`/`set`/`min`/`max`/`rank` retrieve the complete
  matching set before computing, because a pruned set is confidently wrong.
- **Generate-then-ground** — the LLM proposes a plan, but fields, operations and safety guards are
  rebuilt deterministically (`step1.ground_llm_interpretations`).
- **Text output is checkable against verified facts** — evidence spans must occur verbatim
  (`step3_5`), and any LLM verbalisation must preserve verified answer atoms (`step4`).
- **Aggregation answer-sanity** — a sum dominated by one contributor is flagged, not asserted
  (`step5.aggregation_anomaly_features`).
- **Deterministic-first reflector with memory** — repair routing by rule, plus a learning sink.

## 2. Complexity the new design deliberately does NOT inherit

The frozen KG v0.1 is **tabular** (one query record per `contract_node_id`). That single fact lets
us delete the machinery the old pipeline needed for a graph-shaped KG:

| Prior pipeline | ~LOC | New pipeline | Why it collapses |
| --- | ---: | --- | --- |
| `step2` beam search + RGCN + backward reachability + recovery ladders | ~1200 | `backend.query(constraints)` | Exact tabular filters are exhaustive by construction, so "reductions must be exhaustive" is *free* — no beam to prune. |
| `step1` regex/phrase detectors (`is_temporal_supplier_buyer_relationship_question`, …) | ~1000 | `linking.py` (lean) + grounded LLM planner | Brittle phrase matching is replaced by an LLM planner whose output is *grounded*, not trusted. |
| File-staged JSON between 6 scripts | — | in-process dataclasses + `ReasoningTrace` | A runtime library wants typed contracts and one audit object, not disk hand-offs. |
| Duplicated parsers (`parse_money` defined twice) | — | single modules | — |

**Beam search over a tabular KG is solving a problem we do not have.** This is the largest
architectural simplification and it is a place where the new design is strictly better *for this KG*.

## 3. The unifying innovation: one executor, one sanity gate, two consumers

The benchmark generator (`qa.benchmark.executor`, Gate A) and the runtime
(`reasoning.executor`) share identical reduction/additive semantics. Consequences:

- **The benchmark is a self-consistent oracle whose execution is provably identical to the
  runtime's.** Therefore a runtime error on a benchmark item is, by construction, a
  *query-understanding* (planning) error — never an execution mismatch. The benchmark thus
  isolates exactly the capability the LLM is responsible for.
- **"Is this total trustworthy?" has one definition** (`answer_sanity.check_sum_sanity`): the
  same placeholder / dominant-contributor signal that WARN-flags a benchmark sum spec downgrades a
  runtime answer's confidence and attaches a disclosed caveat.

## 4. Hardening added this review (all tested)

- **`grounding.py` — generate-then-ground boundary.** Aliases natural LLM field names
  (`year`→`release_year`, `cpv`→`tender_cpv_id`, …), rejects non-schema fields with a clear reason,
  **proactively adds the `value_is_additive` guard to every sum** (so a sum never silently includes
  non-additive tender-fallback values), forces exhaustive retrieval for reductions, and marks
  unsupported operations (`average`, `top_k`) cleanly. Wired before execution in the pipeline; it
  now pre-empts the reflector's fail-then-repair path for sums (one clean attempt instead of two).
- **`answer_sanity.py` — unified aggregation sanity gate.** Flags placeholder totals and
  dominant-contributor concentration; a flagged sum can never be rated `high` confidence and its
  caveat is surfaced on the answer card.
- **`verbalize.py` — guarded LLM verbalisation.** An optional LLM may rewrite the answer, but the
  output is rejected to the deterministic text unless it preserves every verified answer atom
  (GCR-style). Number atoms are matched with tolerance and by normalised substring.
- **`memory.py` — reflector memory.** Append-only JSONL of non-benign diagnoses for analysis and
  future planning feedback.
- **Confidence decomposition.** The answer card now carries a `confidence_breakdown`
  (planning / execution / evidence / sanity) and repairs apply a penalty, so a broadened or
  suspicious answer is never presented as high confidence.

## 5. Original Future Work (partly superseded)

This list records the state at the first architecture review. Several items have
since been implemented or narrowed by the Phase 2 update below.

- Bounded graph expansion for genuine `role_path` questions (currently gated as unsupported).
- Comparison / superlative / top-k executors — add to the shared executor first, then both the
  benchmark and runtime inherit them.
- A live `DocumentInspector` with verbatim-span verification (the `step3_5` guarantee), behind the
  existing hook.
- An LLM reflection step after the deterministic reflector, for genuinely open-ended failures.

## 6. Empirical hardening from the hard-20 runtime eval (2026-07-01)

Ran the 20 "hard but answerable" eval questions (`data/qa/eval/hard20_*`) through the pipeline with
both the rule planner and the live `gpt-5.4-nano` LLM planner (`scripts/run_hard20_nano.py`). Both
scored 12/20; the *failure modes* were diagnostic and drove three fixes, each mapped to an
established KGQA / text-to-SQL technique.

**Finding 1 — a cosmetic field name killed correct plans (`schema_error`).** Three nano failures
(two sums, one signed-date factoid) had *byte-correct* query plans yet failed with
`execution_status='schema_error'`. Root cause: `preflight` schema-checks every referenced field
(constraints + `answer_field` + `dedupe_key` + `sort_field`), but grounding only aliased/validated
constraint and answer fields. A tiny model echoing the schema placeholder into `dedupe_key` (e.g.
`"optional KG field"`) leaked an invalid field straight to preflight.
*Fix (generate-then-ground, "repair don't reject" — cf. Pangu grounded generation, ChatKBQA):*
grounding is now authoritative over **every** preflight-referenced field — an unknown `dedupe_key`
resets to `contract_node_id`, an unknown `sort_field` is dropped. Verified against the real KG: all
three recover with exact oracle answers (`895398774.06`, `0.0`, `2025-05-24…`).

**Finding 2 — the planner was never shown the answerable columns.** `PLANNER_FIELD_VOCABULARY`
listed only *filterable* fields, so for "what is the signed date?" the model had to invent a column
(`award_signed_date`, `date_signed`).
*Fix (schema linking — cf. text-to-SQL DIN-SQL / RESDSQL):* added `PLANNER_ANSWER_FIELDS` to the
prompt (exact names incl. `award_date_signed`, `value_amount`, `value_source`) and expanded
`FIELD_ALIASES` to snap the common invented variants. The model now selects real columns; grounding
is the backstop.

**Finding 3 — conjunction counts run ~3% high, and the runtime is *right*.** All five
`conjunction_count` items mismatched (e.g. 102 vs 99) because the **golden** specs carry hidden
`supplier_count>=1` / `buyer_count>=1` coverage guards that the natural-language question never
states. The runtime answers the question faithfully; the oracle silently over-constrains. This is a
**benchmark-quality** signal, not a runtime bug — the fix belongs in Stage-1 generation (drop the
coverage guards from conjunction golden answers, or verbalise them), not in the answerer. Left as-is
here to honour the "do not modify existing benchmark files" constraint; flagged for the next
Stage-1 rebuild.

Confirmed with the live nano planner: **12 -> 15/20** (sum 6/6, boundary 2/2 with the £0 placeholder
flagged low-confidence, factoid 3/4). The only remaining misses are the four Finding-3 conjunctions
and the deliberately-unsupported `value_source` provenance factoid.

**Finding 4 — verify checked the *plan*, never the *answer*; the reflector only reacts to execution
*failure*.** Auditing the 20-question trace: `preflight` + `additive_value_check` run every time
(load-bearing — they surfaced Finding 1); `reflect_plan` produced the one abstention; but the
`reflect` repair loop never fired (grounding pre-empts execution failures, so every plan passes
first try) and the evidence verdict is a rubber stamp under the null document inspector. Nothing
inspected whether the matched *population* actually supported the answer — which is exactly why the
conjunction over-count passed silently as `high` confidence.
*Fix (post-execution answer verification):* added `verifier.postflight_checks` — for a passed
count/sum it reports **population coverage** (matched rows with no recorded supplier/buyer); for a
`select_unique` it reports **answer multiplicity**. The pipeline discloses a non-trivial gap on the
answer card ("of 102 matched contracts, 3 with no recorded supplier ..."), so a faithful count now
*explains* its difference from a coverage-filtered figure instead of surprising the reader. The
smoke harness additionally classifies each numeric mismatch (`_gap_diagnosis`) as
`golden_over_constrained` (benchmark artifact) vs `planning_gap`, so the report never misattributes a
benchmark defect to the planner — on the hard-20 all four conjunction misses are labelled artifacts,
zero planning errors.

Tests: `tests/test_reasoning_hardening.py` (grounding dedupe_key/sort_field sanitisation, signed-date
aliasing, junk-bookkeeping end-to-end, postflight coverage + multiplicity, count self-disclosure);
63 reasoning tests pass.

## 7. Verdict

Keep the previous pipeline's *principles*; discard its *graph-era machinery*. The result is a
smaller, typed, in-process runtime with the same safety guarantees, plus a novel property the old
design lacked: the offline oracle and the online answerer are the same deterministic core, so
evaluation measures query understanding in isolation.

## 8. Phase 2 Update: N-hop Runtime Reasoning for QAv2

The reasoning package has since moved from hard-20 hardening into QAv2-targeted
runtime evaluation. The key architectural decision is that multi-hop is no longer
handled as a 2-hop bridge special case. It is implemented as bounded N-hop
decomposition:

- a `DecompositionPlan` is an ordered list of subqueries;
- an entity-set step emits typed distinct KG values;
- a later answer step may bind that emitted set as an `in` filter;
- a 2-hop bridge is two steps, a 3-hop chain is three steps, and N-hop is N steps
  bounded by `max_hops=4`;
- `compare` is represented as two independent answer steps plus a combiner, not as
  a chain.

This preserves the core rule: every answer-producing step still goes through the
same grounded deterministic executor. Additive guards, exhaustive retrieval,
schema checks, evidence counts, and answer sanity are therefore applied per hop.

The practical consequence is that the system gains multi-hop coverage without
reintroducing RGCN/beam-search machinery. For this tabular procurement KG, exact
filters plus bounded set binding are the right abstraction.

## 9. Phase 2 Operation Coverage

The executor now covers the operation families needed by QAv2:

- direct operations: `select_unique`, `count`, `sum`;
- boolean/existence: `exists`;
- extrema: `argmax`, `argmin`;
- set answers: `distinct_set`;
- ranking: `top_k`;
- value predicates: `predicate`;
- decomposed operations: `compare` and bridge-style `in_subquery` through
  `DecompositionPlan`.

One important semantic correction was made: `count` and `exists` over an empty
matching set now return `0` and `False` respectively. They no longer produce
`no_results`. This matters because a bridge or comparison subquery can validly
have zero matches; relaxing that query would silently answer a different
question.

The value-predicate operation closes a large boolean gap by comparing a field,
count, or sum against a threshold. This supports questions such as:

- whether any matching contract exists;
- whether a total exceeds a threshold;
- whether a contract was signed after a date;
- whether a contract field equals an expected category or CPV.

## 10. Planner Boundary After Phase 2

There are now three planner roles:

- the base rule planner for direct count/sum/factoid smoke tests;
- `DecompositionAwarePlanner` for QAv2-style extended operations and bridge joins;
- `LLMReasoningPlanner` for general natural-language planning with explicit fallback
  metadata.

`DecompositionAwarePlanner` is deliberately benchmark-tuned. It recognises
filter-only operations, organisation-anchored operations, compare, and several
bridge templates. It should not be mistaken for the final open-domain planner.
The final general path is still an LLM planner whose output is grounded before
execution.

Fallback accounting is now explicit. If the LLM planner fails to parse or match
the schema, the resulting plan records:

- `planner_source`;
- `fallback_used`;
- `fallback_reason`;
- `fallback_planner`;
- `raw_response`.

This prevents rule-planner capability from being accidentally counted as model
capability in experiments.

## 11. Evidence Materialisation Optimisation

The first bridge implementation was semantically correct but slow because it
materialised large sets of full row dictionaries. The current runtime keeps exact
answers while capping heavy evidence payloads:

- vectorised backend count/sample/project paths avoid unnecessary dict materialisation;
- `count` and `exists` use exact backend counts plus sampled evidence rows;
- `sum` projects only the small set of fields needed for aggregation and sanity;
- `evidence_count` remains exact;
- `evidence_ids` and `ocids` are capped and explicitly marked as capped;
- answer-sanity can use contributor summary metrics instead of rescanning every row.

This is a semantics-preserving optimisation: answers and counts remain exact, but
bulk QAv2 bridge evaluation becomes feasible.

## 12. Current Empirical Signals

The runtime is now evaluated on QAv2 in two modes:

- executor mode: reconstruct the executable spec from each row and measure the
  deterministic executor ceiling;
- pipeline mode: run question -> planner -> grounding -> executor -> verifier and
  measure end-to-end reasoning.

Current signals:

- naturalized and coverage-fixed direct questions have perfect executor ceiling;
- extended operations moved from essentially unsupported in the base rule planner to
  roughly 89% pipeline accuracy after decomposition-aware planning and predicate
  execution;
- bridge joins became measurable after evidence optimisation, with a 400-row sample
  around 86% pipeline accuracy;
- unanswerable scoring now separates safe abstention from right-reason abstention and
  hallucination.

The unanswerable split is especially useful. A row is not simply "correct" because
the system returned no answer; the evaluation now records whether it abstained for
the expected reason (`mark_unsupported`, `ask_clarifying_question`, or
`report_insufficient_evidence`) and whether it hallucinated an answer.

## 13. Updated Open Risks

The remaining risks are no longer broad executor capability gaps. They are narrower
and more diagnosable:

- unsupported-field detection is still weak for some non-KG concepts, so the runtime
  can soft-abstain or accidentally answer where it should `mark_unsupported`;
- bridge and set-list failures are often organisation-resolution residue rather than
  decomposition machinery failures;
- a few tiny bridge template families remain unrecognised;
- the current decomposition-aware planner is benchmark-tuned, so a general LLM
  decomposition planner is still needed for arbitrary user wording;
- QAv2 evaluation outputs should be moved out of the frozen full2k dataset directory
  so the dataset manifest remains clean and reproducible.

The architectural verdict remains the same but is now better supported: keep the
old pipeline's safety principles, keep the deterministic executor as answer
authority, and use LLMs only for planning, candidate selection, diagnostics, and
verbalisation under traceable constraints.
