# QA Benchmark Design

Status: KG v0.1 field semantics frozen for QA sampler work.

This document records the stable KG fields and known coverage boundaries that
QA benchmark generation must respect. Any future change to these field
semantics requires re-validating already generated benchmark examples.

## Stable KG Inputs

QA generation reads the deterministic KG v0.1 parquet outputs under `data/kg/`.

Primary contract-level backend:

- `ParquetKGQueryBackend`
- one query record per `contract_node_id`
- buyer/supplier relationships are tuple-valued fields on that contract record

Important stable fields:

- `contract_node_id`: primary QA evidence record ID.
- `ocid`, `award_id`: source contract/award identifiers.
- `value_amount`: best available award/contract/tender fallback value.
- `value_source`: one of `award`, `contract`, `tender`, or empty. Data-provenance only
  (which OCDS record the value came from); used as a factoid answer and as evidence, but
  NOT as a question filter, since it is not a procurement concept a user would query on.
- `value_is_additive`: true only for `award` and `contract` sources.
- `award_date_signed`: signed date from matched contract JSON.
- `address_region`: primary/modal organization region.
- `address_regions`: JSON list of all observed organization party regions.
- `buyer_regions`, `supplier_regions`: tuple-valued QA backend fields derived
  from `address_regions`.

## Target QA Design: Three Axes

The benchmark should not be balanced by hop count alone. For procurement QA,
content quality depends on three axes:

1. operation family: what computation the answer requires;
2. domain slice: which procurement concept grounds the question;
3. generalization/difficulty: how far the question is from common templates.

Target final benchmark size: approximately `10,000` QA examples after Gate A
and Gate B. The proportions below are targets, not hard quotas: underfill a
bucket rather than forcing weak or ambiguous questions.

### Axis 1: Operation Family

| Family | Target | Current status | Example |
| --- | ---: | --- | --- |
| Contract factoid | 10-12% | executable | "Who is the buyer for this contract?" |
| Role/path query | 12-15% | planned | "Which buyers has supplier X contracted with?" |
| Filtered count | 15% | executable | "How many contracts were published in 2025 under CPV X?" |
| Additive sum | 15% | executable | "What is the total contract value for 2024 goods contracts?" |
| Conjunction / multi-filter | 15% | executable | "How many 2023 services contracts were published under CPV 79341000?" |
| Comparison | 10% | planned | "Which buyer had the higher total value, A or B?" |
| Superlative / top-k | 8% | planned | "Which supplier had the highest total value under CPV X?" |
| Temporal | 8% | executable, count-only | "How many published contracts have a signed award date?" |
| CPV / region slice | 7% | CPV executable; region planned | "How many goods contracts were under CPV 33000000?" |

Implementation note: Stage 1 v0.2 still emits only executable
`AnswerSpec` operations supported by the deterministic executor:
`select_unique`, `count`, and `sum`. Role/path, comparison, top-k, and richer
region questions are design targets for the next executor extension, not
forced into the current sampler.

### Axis 2: Domain Slice

Each QA item should be labelled with one primary `domain_slice`:

- `contract_identity`: a single contract and one of its fields;
- `buyer_supplier_role`: buyer/supplier participation and role context;
- `value`: monetary value, value source, and additive value semantics;
- `cpv`: CPV code/category grounded questions;
- `temporal`: publication year and signed award date availability;
- `category`: goods/services/works;
- `region`: buyer/supplier region attributes, once region samplers are added.

This prevents the final QA set from being "all CPV/year counts" even when the
operation distribution looks balanced.

### Axis 3: Difficulty And Generalization

Difficulty is assigned from both structure and evidence size:

- Easy: single contract, single answer field, or one semantic filter.
- Medium: two semantic filters, or one aggregation over a moderate evidence
  set.
- Hard: three or more semantic filters, comparison/top-k, role plus CPV/year,
  or large evidence sets requiring reliable full-graph execution.

Each item also receives a `generalization_class` label:

- `iid`: common field combinations seen frequently in the KG;
- `compositional`: multiple known fields combined in less common ways;
- `zero_shot_like`: rare CPV/category/year or entity combinations held out
  for stress testing.

This follows the spirit of GrailQA-style evaluation: a benchmark should measure
more than template memorisation.

### Evidence Floors

Aggregation-like questions should not collapse into disguised factoids.
Default evidence floors for Stage 1 v0.2:

| Family | Minimum evidence rows |
| --- | ---: |
| Contract factoid | 1 |
| Filtered count | 3 |
| Additive sum | 3 |
| Conjunction / multi-filter | 3 |
| Temporal count | 3 |
| CPV / categorical count | 3 |

Items below the floor are not sampled for that family. They may still be useful
as factoids or edge cases, but not as main aggregation benchmark examples.

## Start Selection

### Entity Starts

Entity starts use resolved `canonical_id` values and role context from
`org_nodes`.

Sampling rules:

- stratify across buyer, supplier, and mixed-role organizations;
- prefer middle activity ranges where possible, avoiding only the largest
  national frameworks and the smallest one-off entities;
- avoid supplier-specific starts when `supplier_count == 0`;
- avoid buyer-specific starts when `buyer_count == 0`.

### Feature Starts

Feature starts are precise structured filters over KG fields:

- `release_year`
- `tender_category`
- `tender_cpv_id`
- `value_is_additive`
- `has_award_signed_date`
- `supplier_count`
- `buyer_count`

Feature starts represent node sets, not a single node. They should cover the
full 2022-2026 release span when possible.

## Subgraph Construction

The benchmark does not use local graph traversal with a depth limit.

Instead:

1. choose a contract/entity/feature start;
2. construct explicit `AnswerSpec.constraints`;
3. query the full KG backend with exact tabular filters;
4. record every matched `contract_node_id` as `sampled_evidence_ids`;
5. execute the operation over the full matched set.

This targets a real procurement risk: local traversal can miss evidence that is
structurally farther away but still belongs to the same logical answer set.

## Gate A

Gate A runs four deterministic checks per spec. Each is reported as PASS / WARN / FAIL
in `gate_a_report.jsonl` and rolled up in `stage1_summary.json`. A spec is accepted only
when no check is FAIL; WARN keeps the spec but flags it for review.

Constraint integrity (`gate_a_constraints`):

- constraints are normalised at generation time, dropping a bound made redundant by an
  `eq` on the same field (for example `supplier_count gte 1` when `supplier_count eq 9`);
- FAIL on a contradiction (mutually unsatisfiable constraints) so an always-empty spec
  never reaches output.

Completeness (`gate_a_completeness`):

- the matching contract set is re-derived from the source KG tables (`contract_nodes`
  plus the `buyer_of` / `supplier_of` edges) through `ReferenceKGIndex`, a code path
  independent of the backend's flattened `records_df`, with an independent
  `supplier_count` / `buyer_count` recomputation;
- require the recorded `sampled_evidence_ids` to exactly match that independent
  re-derivation, with no duplicate contract rows and no empty evidence;
- if a constraint op cannot be independently verified, WARN (flag) rather than silently
  pass.

This is a genuine cross-check: agreement is evidence of completeness, and disagreement
means the backend dropped, duplicated, or miscounted contracts. A same-query check is
circular, because `sampled_evidence_ids` is itself produced by the backend query; the
independent re-derivation is what makes the check meaningful.

Deterministic answer (`gate_a_uniqueness`):

- execute `answer_operation` on the complete evidence set;
- reject multi-valued `select_unique` answers;
- reject empty answer fields;
- reject aggregate value specs that omit `value_is_additive == True`.

Aggregate value sanity (`gate_a_value_sanity`, aggregation-sum only):

- WARN on nominal placeholder totals (every summed row value is a `GBP 0`/`GBP 1` placeholder),
  implausibly small totals, or per-evidence-mean magnitude outliers (a log10 Tukey
  `1.5*IQR` fence across the sum cohort);
- WARN keeps the spec but marks it for review; `--exclude-sum-anomalies` turns these into
  rejects.

## Golden Answer

Golden answers are produced by deterministic execution of `AnswerSpec`.

No LLM is used for Stage 1 golden-answer generation.

## Natural-Language Question Generation

Question generation is implemented in Stage 2:

- code: `procurement_graph.qa.benchmark.question_gen`;
- CLI: `pipelines/60_build_qa_stage2.py`;
- generator model: `gpt-5.4-nano` through the user's Azure-compatible OpenAI
  endpoint;
- current API path: `chat.completions`;
- current structured-output guard: `response_format={"type": "json_object"}`
  when `--json-mode` is passed, plus deterministic JSON parsing and rejection
  of empty/invalid questions.

The generation prompt does not include the golden answer. For aggregation and
feature-set questions it receives the target operation plus the semantic
filters that the question must encode. Internal scoping fields such as
`value_is_additive`, `supplier_count`, and `buyer_count` are intentionally not
phrased as user-visible filters.

For factoid questions, Stage 2 tries to build a natural contract anchor from
buyer, supplier, CPV, year, and category, excluding the answer field. If those
attributes do not uniquely identify the answer, the item is rejected with
`rejected_no_anchor`. Stage 2 must not generate user-facing questions that
expose `contract_node_id`, OCID, award IDs, canonical IDs, scheme IDs, or other
internal/source identifiers. CPV codes are the only ID-like values allowed in
question text because they are procurement classification codes users may
reasonably ask about.

Every model-generated accepted record stores:

- `prompt_version`;
- `schema_version`;
- `schema_hash`;
- raw model response;
- parsed question;
- token usage, when returned by the API.

### Strict Schema Note

Responses strict JSON schema is not required for the current Stage 2 pilot to
run: the existing `chat.completions` JSON mode has already produced valid pilot
outputs. Strict schema is an optional hardening step before a large live run.
Its value is that the API enforces required keys and basic types before the
pipeline parses the result, reducing silent shape drift such as missing
`question`, malformed `filters`, or renamed fields.

Do not switch APIs solely for neatness. Switch only if the user's Azure
deployment supports Responses `text.format=json_schema` reliably, and after a
small smoke test confirms that the generation and verification models both
respect the schema.

## Gate B

Gate B is implemented in Stage 2:

- code: `procurement_graph.qa.benchmark.gate_b`;
- verifier model in the current pipeline: `grok-4-1-fast-non-reasoning`;
- generator and verifier are different model families, preserving the
  independent-verification design.

Gate B uses two modes:

- `recompute`: for `select_unique` factoids, the verifier answers from a small
  projected evidence record and the result is compared with the deterministic
  golden answer.
- `faithfulness`: for `count` and `sum`, the verifier extracts the operation
  and semantic filters from the generated question. This avoids asking an LLM
  to count or sum long evidence lists, and directly tests whether the question
  asks for the same structured query as the `AnswerSpec`.

Verification policy:

- pilot runs use 100% verification by setting `--factoid-sample-rate 1.0`;
- after prompt stability, factoids may be sampled, while aggregation and
  temporal/constrained questions remain fully verified;
- if any stratum falls below 95% agreement, raise that stratum to 100%
  verification until fixed.

## Two-Stage Build

### Stage 1: Local Deterministic Build

No LLM calls.

Outputs:

- `data/qa/generated/answer_specs.jsonl`
- `data/qa/generated/gate_a_report.jsonl`
- `reports/qa/stage1_summary.json`

Steps:

1. load KG once;
2. generate procurement-specific answer specs;
3. run Gate A completeness and deterministic-answer checks;
4. execute golden answers;
5. write only Gate-A-passing specs to `answer_specs.jsonl`;
6. write all attempted specs and gate outcomes to `gate_a_report.jsonl`.

Stage 1 sampler v0.2 quality rules:

- ordinary filtered count no longer duplicates the CPV-specific
  `release_year + tender_cpv_id` template; CPV templates are owned by the
  CPV/categorical sampler;
- `supplier_count` / `buyer_count` may be used only as coverage guards such as
  `gte 1`, not as answer-changing `eq N` semantic constraints;
- aggregation-like families use evidence floors by default:
  `count >= 3`, `sum >= 3`, `conjunction >= 3`, `temporal >= 3`, `cpv >= 3`;
- each spec metadata record includes `operation_family`, `domain_slice`,
  `generalization_class`, `evidence_floor`, and `template_fields` so the final
  QA set can be audited along the three design axes.

The 10,000-spec artifact generated before this section was added is a v0.1
Stage 1 artifact. It remains useful for testing Gate B rejection behavior, but
the final benchmark should be regenerated with sampler v0.2.

### Stage 2: LLM Question Generation and Gate B

LLM calls start only after Stage 1 outputs have been inspected.

### Stage 2a: Prompt Ablation

Before choosing the final generation prompt, run a BioGraphletQA-style prompt
ablation over a fixed Stage 1 spec slice. The current implemented variants are:

- `current`: the baseline prompt used in the first pilot;
- `strict_filters`: explicitly requires every semantic filter to be verbalised;
- `natural_procurement`: prefers fluent UK procurement wording while preserving
  every semantic filter.

Entrypoint:

```powershell
python -B pipelines\61_run_qa_prompt_ablation.py --sample-per-type 4 --factoid-sample-rate 1.0 --json-mode --out-prefix prompt_ablation
```

Dry-run mode makes no API calls:

```powershell
python -B pipelines\61_run_qa_prompt_ablation.py --dry-run --sample-per-type 1 --out-prefix dry_check
```

Selection metrics:

- Gate B accept rate;
- generation rejection rate;
- hidden semantic constraint failures;
- factoid natural-anchor vs `rejected_no_anchor` rate;
- ID/source identifier leak failures;
- manual spot-check of question naturalness and procurement wording.

The aggregate report is written to
`reports/qa/prompt_ablation_<out-prefix>.json`, with per-variant benchmark,
rejected, and summary files also written under the usual QA output directories.

Important: prompt ablation should run after Stage 1 specs are regenerated with
the current sampler. Old specs may still contain answer-changing hidden
constraints, and these are now deliberately rejected by Gate B.

Pilot command:

```powershell
python -B pipelines\60_build_qa_stage2.py --sample-per-type 4 --factoid-sample-rate 1.0 --json-mode --out-tag pilot
```

Dry-run mode makes no API calls and is useful for plumbing checks:

```powershell
python -B pipelines\60_build_qa_stage2.py --dry-run --sample-per-type 1 --out-tag dry_check --no-resume
```

Resume is enabled by default. If a tagged output already contains records, the
summary for a resumed rerun may show `attempted=0` and `skipped_resume=N`; in
that case the JSONL output itself is the quality artifact, not the rerun
summary alone.

### Stage 2b: Decoupled Full Run

For target-scale generation, Stage 2 is now split into three resumable passes so
the generator and verifier can run independently under different rate limits:

1. `pipelines/62_qa_generate.py`: nano-only generation from `answer_specs.jsonl`
   or a later `regen_queue.jsonl`, writing `data/qa/generated/questions.jsonl`;
2. `pipelines/63_qa_verify.py`: Grok-only Gate B verification from
   `questions.jsonl`, writing `data/qa/generated/verifications.jsonl`;
3. `pipelines/64_qa_assemble.py`: joins generated questions and verifier
   outcomes into `benchmark.jsonl`, `rejected_stage2.jsonl`, and
   `regen_queue.jsonl`.

This replaces the older per-spec `generate -> verify` loop for full runs. The
one-shot `60_build_qa_stage2.py` entry point remains useful for pilots and
prompt ablation plumbing, but the production workflow should use `62/63/64`.

Generation rejections include `rejected_no_anchor` for factoids that cannot be
phrased without internal/source identifiers. Verifier failures are written to
`regen_queue.jsonl` with feedback; rerun step 62 on that queue to regenerate
only failed items.

## Known Limitations

These are expected KG v0.1 coverage boundaries, not pipeline failures.

### Supplier Coverage

`supplier_of` covers `204,186 / 215,221` contract nodes (`94.9%`).

Implication for QA generation:

- supplier-specific questions must require `supplier_count > 0` or an existing
  `supplier_of` edge;
- samplers must not assume every contract has a supplier;
- Gate A should reject supplier questions whose sampled evidence omits any
  full-graph supplier-bearing records.

### Evidence Pointer Coverage

`evidence_for` covers `215,202 / 215,221` contract nodes.

Implication for QA generation:

- questions that require text evidence should require `evidence_count > 0` or
  an existing `evidence_for` edge;
- contracts without evidence pointers can still support pure structured KG
  questions, but not text-evidence questions;
- evidence pointers are based on extracted OCDS text fields, not live document
  crawling.

### Buyer Coverage

`buyer_of` covers `215,218 / 215,221` contract nodes.

Implication for QA generation:

- buyer-specific questions should require `buyer_count > 0`;
- the missing three buyer edges are expected boundary cases and should not be
  sampled as buyer-question seeds.

### Coverage Guards Are Answer-Changing For Counts (fix in next Stage-1 rebuild)

Stage 1 sampler v0.2 permits `supplier_count`/`buyer_count` `gte 1` as coverage
guards on the assumption they are *not* answer-changing (see "Stage 1 sampler v0.2
quality rules"). The hard-20 runtime eval (`docs/hard20_runtime_eval.md`) shows
this assumption is false for **count** families: because supplier coverage is only
`94.9%`, a `supplier_count >= 1` guard silently removes the matching notices whose
supplier edge is missing, so the golden count is `~1-7%` below a faithful reading
of the natural-language question (which never states the guard).

Observed on the four hard-20 conjunction items: runtime `102/97/94/98` vs golden
`99/93/93/91`; appending the guards reconciles each exactly, confirming the plans
are correct and only the golden population differs.

- **Current handling (no regeneration):** the mismatches are labelled
  `golden_over_constrained` by the eval harness and excluded from planning-error
  accounting; `answer_key` is left unchanged.
- **Next Stage-1 rebuild (Plan A):** for families whose golden **is** the count
  (`count`, `conjunction`, `cpv`, `temporal`), do not apply answer-changing
  coverage guards — either drop them, or verbalise them into the question
  ("... awarded to a supplier on record"). Reserve `gte 1` guards for cases where
  the guarded field is not part of the answer.

### Organisation Name Casing / Canonicalisation (KG entity-resolution known issue)

The same organisation can appear as multiple distinct `buyer_name` / `supplier_name`
strings that differ only by casing or punctuation (e.g. `University of Sheffield`
vs `UNIVERSITY OF SHEFFIELD`). Because the flattened backend filters by the name
string with exact `eq`, a factoid anchored on one surface form misses contracts
stored under another (observed on hard-100 `factoid_0017`, where the planner's
title-case buyer missed the all-caps record).

- This is an **entity-resolution gap**, not a planning or runtime-logic error.
- Handling is deferred to the **canonical org resolver / semantic-repair track**
  (a name -> canonical-id map, or case-insensitive multi-variant matching), which
  should own all name normalisation uniformly.
- No answer_key / benchmark change is made for this; runtime does not apply a
  casing patch.

### Region Semantics

`address_region` is not a complete set. It is the primary/modal region inherited
from ER outputs.

Use `address_regions` for region-set questions.

Implication for QA generation:

- single-region questions may use `address_region` only when asking about the
  primary/modal region;
- coverage or multi-region questions must use `address_regions` through
  `buyer_regions` / `supplier_regions`;
- region fields are attributes for QA and evidence context, not merge keys.

### Value Semantics

`value_amount` can use tender fallback for coverage.

Implication for QA generation:

- row-level questions may use `value_amount` with its `value_source`;
- aggregate value questions must require `value_is_additive == True`;
- aggregate questions must not sum rows where `value_source == "tender"`.

## Frozen Checks

The current frozen KG v0.1 validation baseline:

- KG validation: `23 PASS`, `3 WARN`, `0 FAIL`.
- The three WARNs are the coverage boundaries above.
- `address_regions` JSON invalid rows: `0`.
- primary `address_region` missing from `address_regions`: `0`.
- QA benchmark tests: `20` passing (mock pipeline, independent completeness
  gate, Stage 2 prompt/Gate B logic, and real KG smoke tests).
- real KG QA smoke tests: included in the `20` tests above, covering
  reference/backend agreement.

## Current QA Build State

Stage 1 has been run at the target scale:

- `data/qa/generated/answer_specs.jsonl`: `10,000` accepted specs;
- `data/qa/generated/gate_a_report.jsonl`: `10,000` gate records;
- `reports/qa/stage1_summary.json`: `10,000 / 10,000` accepted, `0`
  rejected;
- status rollup: `9,971 PASS`, `29 WARN`, `0 FAIL`;
- all `29` WARNs are aggregation-sum value sanity flags, kept for audit.

Accepted type distribution:

- factoid: `2,841`;
- aggregation-count: `1,705`;
- aggregation-sum: `1,704`;
- conjunction/constraint: `2,273`;
- temporal: `909`;
- categorical/CPV: `568`.

Stage 2 prompt ablation v03 has been run over a fixed 60-spec slice:

- `strict_filters`: `57 / 60` accepted, `0` Gate B rejects, `3`
  `rejected_no_anchor` factoids;
- `natural_procurement`: `57 / 60` accepted, `0` Gate B rejects, `3`
  `rejected_no_anchor` factoids;
- `current`: `49 / 60` accepted, `8` Gate B rejects, `3`
  `rejected_no_anchor` factoids;
- all variants reported `0` internal-ID factoids and `0` ID-leak failures.

Use `strict_filters` or `natural_procurement` for target-scale generation; both
passed the v03 quality gate, while `current` is kept only as a baseline.

## Stage 1 Pilot And Target Build Result

Implemented local deterministic Stage 1 for the first executable sampler pass.
No LLM/API calls are made.

Code:

- `procurement_graph.qa.benchmark.samplers`
- `procurement_graph.qa.benchmark.constraints` (normalisation + conflict detection)
- `procurement_graph.qa.benchmark.anomaly` (aggregation-sum value sanity)
- `procurement_graph.qa.benchmark.reference_index` (independent completeness oracle)
- `procurement_graph.qa.benchmark.stage1`
- `pipelines/50_build_qa_stage1.py`

Pilot command:

```powershell
python -B pipelines\50_build_qa_stage1.py --target-specs 200 --seed 42 --max-evidence-rows 5000
```

Pilot result (200 specs):

- attempted / accepted / rejected: `200 / 200 / 0`;
- status rollup: PASS `198`, WARN `2`, FAIL `0`;
- the two WARNs were aggregation-sum value sanity flags.

Target build result (10,000 specs):

- attempted / accepted / rejected: `10,000 / 10,000 / 0`;
- status rollup: PASS `9,971`, WARN `29`, FAIL `0`;
- `gate_a_constraints`: `0` contradictions;
- `gate_a_completeness`: `0` failures; the independent `ReferenceKGIndex`
  re-derivation matched the recorded evidence;
- `gate_a_uniqueness`: `10,000 / 10,000` deterministic answers;
- `gate_a_value_sanity`: `29` aggregation-sum WARNs, kept and flagged.

Accepted type distribution for the target build:

- factoid: `2,841`;
- aggregation-count: `1,705`;
- aggregation-sum: `1,704`;
- conjunction/constraint: `2,273`;
- temporal: `909`;
- categorical/CPV: `568`.

Performance:

- fixed load: backend `~47s` + reference index `~1s`;
- per-spec cost is flat (`~0.50s` with the independent completeness gate), so
  the full local build is linear and has no LLM cost.

Known follow-ups:

- factoid questions now try natural anchors and reject examples with no natural
  anchor, rather than falling back to opaque identifiers;
- factoid `select_unique` on `buyer_name` / `supplier_name` uses the first party for
  multi-party contracts; consider restricting those to single-party contracts.
