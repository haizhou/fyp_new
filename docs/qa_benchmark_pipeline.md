# QA Benchmark Construction Pipeline

This project builds QA benchmark examples from KG logic, not from document
retrieval. The current build reads the frozen deterministic KG v0.1 parquet
outputs under `data/kg/` and writes benchmark artifacts under
`data/qa/generated/`.

## Flow

1. Stage 1 samples procurement-specific `AnswerSpec` records from the full KG.
2. Gate A checks the generated logic against an independent full-graph
   reference index.
3. The verified `AnswerSpec` is executed deterministically to produce a golden
   answer.
4. Stage 2 generation turns each accepted spec into a natural-language
   question without exposing internal/source identifiers.
5. Gate B asks an independent verifier model to validate the generated question
   against the same logic/evidence.
6. Assembly writes benchmark items only when generation and Gate B pass; failed
   verification rows go to a regeneration queue.

## Answer Spec

The current framework uses:

- `constraints`
- `answer_operation`
- `answer_field`
- `answer_value_type`
- `dedupe_key`
- `logic_chain`
- `sampled_evidence_ids`

These fields are intentionally independent of the final procurement KG schema.
Real loaders can later map canonical entities, contracts, awards, suppliers, and
buyers into the same interface.

## Gate A

Gate A has two checks:

- Completeness: rerun the `constraints` against the full graph and require the
  full result IDs to exactly match `sampled_evidence_ids`.
- Uniqueness: execute the `AnswerSpec` and require a deterministic answer.

This catches local-subgraph omissions and multi-answer ambiguity before a golden
answer is accepted.

## Gate B

Gate B must use a verifier independent from the question-generation model. It
answers the generated question using only supplied KG evidence. If its answer
does not match the golden answer, the candidate is discarded or regenerated.

The production Stage 2 workflow is decoupled:

- `pipelines/62_qa_generate.py`: nano generation pass, writing
  `questions.jsonl`;
- `pipelines/63_qa_verify.py`: Grok Gate B pass, writing
  `verifications.jsonl`;
- `pipelines/64_qa_assemble.py`: joins both streams into `benchmark.jsonl`,
  `rejected_stage2.jsonl`, and `regen_queue.jsonl`.

The older `pipelines/60_build_qa_stage2.py` one-shot flow remains useful for
small pilots and prompt ablation but is not the preferred full-run entry point.

## Current Status

The package under `procurement_graph.qa.benchmark` now has two backends:

- `TabularQueryBackend` for fast mock unit tests.
- `ParquetKGQueryBackend` for the real deterministic KG v0.1 tables under
  `data/kg/`.

The real backend keeps one query record per `contract_node_id`. Buyer and
supplier fields are stored as tuple-valued attributes on that record, so
supplier multi-edges do not duplicate contract rows during Gate A completeness
checks.

Region fields exposed by the real backend:

- `buyer_regions` and `supplier_regions` are tuple-valued fields derived from
  `org_nodes.address_regions`, the all-observed region set.
- `buyer_name` and `supplier_name` are convenience first values for simple
  `select_unique` specs; samplers that need complete party context should use
  the tuple-valued fields.

Current test coverage includes:

- entity-style starts: one known contract/OCID, selecting one field.
- feature-style starts: constraints that match a set of contracts, with `count`
  and `sum` operations.
- Gate A completeness failure: a sampled subgraph that misses a full-graph
  matching record is rejected.
- Gate A uniqueness failure: a complete evidence set with multiple possible
  `select_unique` answers is rejected.
- Gate B failure: an independent verifier mock that disagrees with the golden
  answer causes the candidate to be discarded.

This validates the framework gates, deterministic execution, Stage 2 prompt/Gate
B logic, and real KG smoke paths.

Real KG smoke coverage includes:

- exact contract query by `contract_node_id`;
- Gate A completeness and uniqueness on a real contract;
- feature-set count over real KG constraints;
- a performance smoke check for a full-graph query over `release_year`,
  `tender_category`, and `value_is_additive`.

Observed performance on the current KG v0.1:

- one-time backend load is the expensive step because parquet tables are read
  and buyer/supplier summaries are grouped in memory;
- once loaded, a full-graph query returning `26,215` rows completed in about
  `1` second in the local environment.

`ParquetKGQueryBackend.from_directory(..., include_evidence=False)` can skip
evidence pointer aggregation for QA generation runs that do not need evidence
counts. Verdict-stage evidence lookup can enable evidence loading or read
`evidence_for` directly.

Current target-scale Stage 1 output:

- `data/qa/generated/answer_specs.jsonl`: `10,000` accepted specs;
- `data/qa/generated/gate_a_report.jsonl`: `10,000` Gate A records;
- `reports/qa/stage1_summary.json`: `9,971 PASS`, `29 WARN`, `0 FAIL`;
- all WARNs are aggregation-sum value sanity flags kept for audit.

Prompt ablation v03 selected the no-ID-fallback policy: factoid questions that
cannot be phrased with a natural contract anchor are rejected as
`rejected_no_anchor`, not generated with `contract_node_id` or other internal
identifiers.

## Targeted QAv2 Freeze

Targeted QAv2 v0.2 is frozen separately from the Stage 1/Stage 2 generated
benchmark.

Frozen root:

```
data/qa/targeted_v2/full2k/
```

Manifest:

```
data/qa/targeted_v2/full2k/manifest.full2k.json
```

Summary:

```
data/qa/targeted_v2/full2k/validation_summary.full2k.json
```

This freeze contains `9,990` accepted targeted rows out of a nominal `10,000`
target. Four subsets reached `2,000/2,000`; `unanswerable` reached `1,990/2,000`
because the duplicate id/question guard rejected 10 rows. All accepted rows
passed row-level validation, and all accepted rows have zero duplicate ids and
zero duplicate questions.

The frozen QAv2 set should be treated as read-only input for reasoning
evaluation. Future targeted-v2 experiments should write a new run directory and
new manifest rather than modifying `full2k` in place.
