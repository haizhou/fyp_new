# Handoff: Phase 1 Package Migration

Date: 2026-06-29

## Completed

- Created `src/procurement_graph/` package skeleton.
- Migrated shared normalisation utility to:
  - `src/procurement_graph/common/normalise.py`
- Added domain package placeholders:
  - `reference/`
  - `experiments/`
  - `ingest/`
  - `er/`
  - `extract/`
  - `kg/`
  - `qa/`
- Migrated reference lookup to:
  - `src/procurement_graph/reference/lookup.py`
- Migrated reference fetch/cache logic to:
  - `src/procurement_graph/reference/fetchers.py`
- Migrated reference ablation logic to:
  - `src/procurement_graph/experiments/reference_ablation.py`
- Migrated ER modules to:
  - `src/procurement_graph/er/phase1.py`
  - `src/procurement_graph/er/phase2.py`
  - `src/procurement_graph/er/candidates.py`
- Migrated extraction module to:
  - `src/procurement_graph/extract/tables.py`
- Added package-local ingest loader dependency for extraction:
  - `src/procurement_graph/ingest/loader.py`

## Compatibility

The old flat module paths are still present and keep existing imports working:

- `src/normalise.py`
- `src/reference_lookup.py`
- `src/er_phase1.py`
- `src/er_phase2.py`
- `src/er_candidates.py`
- `src/extract.py`

Pipeline/script entrypoints still work:

- `pipelines/00_fetch_reference.py`
- `scripts/run_entity_ablation.py`

## Validation Run

Commands run with `PYTHONDONTWRITEBYTECODE=1` and `python -B`.

- New package imports: passed.
- Legacy flat imports: passed.
- `python -B pipelines/00_fetch_reference.py --source contracts_finder`: passed.
- `python -B scripts/run_entity_ablation.py`: passed.
- Duplicate checks:
  - `canonical_id` duplicates: `0`
  - `alias_map.raw_id` duplicates: `0`
- Reference ablation counts unchanged:
  - `govuk_only`: `831`
  - `nhs_only`: `1,135`
  - `cf_buyer_only`: `9,235`
  - `all_references`: `10,146`

## Test Note

`pytest` still has the known Windows `tmp_path` permission issue:

- 157 non-`tmp_path` tests passed.
- 8 `tmp_path` fixture tests errored during setup/cleanup because pytest could
  not access its temp base directory.

This is the same environment issue observed before this migration, not a new
code regression.

## Reference LLM Adjudication

Implemented an offline LLM review workflow for reference/API entity candidates:

- Module:
  - `src/procurement_graph/experiments/reference_llm_adjudication.py`
- Entrypoint:
  - `scripts/run_reference_llm_adjudication.py`
- Review queue:
  - `data/ablation/reference/llm_review_queue.jsonl`
- Decisions template:
  - `data/ablation/reference/llm_decisions.template.jsonl`
- Dry-run gate outputs:
  - `data/ablation/reference/llm_validated_decisions.parquet`
  - `data/ablation/reference/llm_approved_merges.parquet`
  - `reports/ablation/reference/llm_adjudication_summary.csv`
  - `reports/ablation/reference/llm_uncertain_cases.csv`
  - `reports/ablation/reference/llm_rejected_or_blocked_cases.csv`

Current queue size:

- `663` total tasks
- `435` reference collision tasks
- `159` inactive/closed/medium-confidence review tasks
- `69` alias/name mismatch review tasks

The workflow does not mutate `data/entities/*`. Contracts Finder evidence is
included only as supporting provenance inside packets, not as an independent
authority for automatic merges.

Template dry-run status:

- `663` decisions validated
- `0` approved merge rows
- `663` blocked/uncertain rows

Once the Azure API call format is available, plug the model output into:

```powershell
python -B scripts/run_reference_llm_adjudication.py --validate-decisions --decisions data\ablation\reference\llm_decisions.jsonl
```

Azure/OpenAI queue runner added:

- Provider helper:
  - `src/procurement_graph/llm/azure_responses.py`
- Generic queue runner:
  - `src/procurement_graph/experiments/llm_queue_runner.py`
- Entrypoint:
  - `scripts/run_azure_llm_queue.py`

Queue schema/prompt versioning:

- Reference queue rows include `prompt_version="reference_v2"` and
  `schema_version="reference_decision_schema_v2"`.
- ER pair queue rows include `prompt_version="er_pair_v2"` and
  `schema_version="er_pair_decision_schema_v2"`.
- Queue rows include `response_format` with strict JSON schema for the
  Responses API `text.format`.
- Raw response audit rows include prompt version, schema version, and schema
  hash.

Reference `merge_subset` validation is strict:

- `approved_entity_ids` must be a non-empty subset of candidate IDs.
- `excluded_entity_ids` must be a non-empty subset of candidate IDs.
- approved and excluded IDs must not overlap.
- approved + excluded must exactly partition all candidate IDs in the task.
- `merge_all` must approve every candidate and exclude none.
- `do_not_merge` / `uncertain` cannot approve any candidate.

Runtime dependencies are listed in `requirements.txt`:

- `openai`
- `azure-identity`

Expected environment variables for live Azure calls:

```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://uceeh01-5458-resource.services.ai.azure.com/openai/v1"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-5.4-nano"
$env:AZURE_OPENAI_SCOPE = "https://ai.azure.com/.default"
```

Auth behavior:

- If `AZURE_OPENAI_API_KEY` or `OPENAI_API_KEY` is set, the runner uses that key.
- Otherwise it uses `DefaultAzureCredential()` with the configured scope.
- A 1-task live smoke reached the auth layer but failed because the current
  machine/session has no usable Azure credential:
  - no service-principal environment variables
  - no Azure CLI on PATH
  - no Azure PowerShell `Az.Account` module
  - no Azure Developer CLI
  - no shared token cache account

Checkpoint/resume behavior:

- The queue runner writes JSONL incrementally after each completed task.
- `--max-tasks N` means "process at most N new pending tasks in this run".
- Resume is enabled by default and skips task IDs already present in the
  decisions JSONL.
- API failures are written to the errors JSONL only by default, so failed tasks
  remain retryable on the next run.
- After a main batch, explicitly retry errors with `--only-errors`:
  ```powershell
  python -B scripts/run_azure_llm_queue.py --only-errors --queue data\ablation\reference\llm_review_queue.jsonl --decisions data\ablation\reference\llm_decisions.jsonl --raw data\ablation\reference\llm_raw_responses.jsonl --errors data\ablation\reference\llm_errors.jsonl
  ```
- Use `--write-error-decisions` only when failed API calls should be frozen as
  conservative `uncertain` decisions.
- Do not run two writers against the same decisions file at the same time.

Small-batch review:

- Entrypoint:
  - `scripts/review_llm_batch.py`
- Outputs:
  - `reports/ablation/llm/batch_review_summary.md`
  - `reports/ablation/llm/batch_decision_review.csv`
  - `reports/ablation/llm/batch_error_review.csv`
- Review goal is not just "the API ran"; manually check:
  - `merge_subset` approved/excluded partition is semantically reasonable.
  - risk flags match true residual risk in approved merges.
  - errors are not mostly enum/schema coverage failures.
  - known hard cases such as trusted ID conflict are marked or rejected.
  - token usage is acceptable before scaling.

Reference queue live call:

```powershell
python -B scripts/run_azure_llm_queue.py --queue data\ablation\reference\llm_review_queue.jsonl --decisions data\ablation\reference\llm_decisions.jsonl --raw data\ablation\reference\llm_raw_responses.jsonl --errors data\ablation\reference\llm_errors.jsonl
```

Then validate and gate:

```powershell
python -B scripts/run_reference_llm_adjudication.py --validate-decisions --decisions data\ablation\reference\llm_decisions.jsonl
```

## Full-Corpus ER Ablation V1

Implemented a read-only full-corpus ER ablation framework so future entity
work is not limited to GOV.UK/NHS/Contracts Finder reference candidates.

- Module:
  - `src/procurement_graph/experiments/er_ablation.py`
- Entrypoint:
  - `scripts/run_er_ablation.py`
- Outputs:
  - `data/ablation/er/entity_context.parquet`
  - `data/ablation/er/candidate_pairs.parquet`
  - `data/ablation/er/pair_features.parquet`
  - `reports/ablation/er/blocking_metrics.csv`
  - `reports/ablation/er/feature_ablation_metrics.csv`
  - `reports/ablation/er/high_risk_pairs.csv`
  - `reports/ablation/er/high_risk_clusters.csv`
  - `reports/ablation/er/summary.md`
  - `data/ablation/er/llm_review_queue.jsonl`

Current V2 diagnostic run:

```powershell
python -B scripts/run_er_ablation.py --max-pairs-total 0 --max-pairs-per-strategy 12000 --max-block-size 80 --max-token-frequency 120 --max-alias-frequency 60 --cluster-threshold 0.88 --auto-merge-threshold 0.88 --llm-max-tasks 0
```

Candidate generation strategies:

- exact normalised name
- normalised name + region
- 3-character prefix
- rare informative token
- alias normalised-name overlap
- acronym block

This is the full-corpus blocking/candidate-generation layer. Current V2 removed
the global `50,000` cap, but still keeps a per-strategy cap for diagnostics.
All-pairs would be about `9.84B`, so blocking is required.

Feature layers:

- name similarity
- token Jaccard
- alias overlap
- region/category/type agreement
- trusted official scheme/id conflict
- buyer/supplier OCID context
- CPV2/year overlap
- heuristic dry-run score

Current V2 diagnostic metrics:

- `128,887` entities with procurement context
- `57,021` candidate pairs
- pair reduction ratio: `0.999994203`
- `47,397` LLM-review pairs from global risk tiering
- `5,792` deterministic auto-merge candidates
- `3,832` middle candidate-only pairs
- `8,319` high-score dry-run clusters
- largest dry-run cluster size: `34`

Important limitation:

- Candidate generation is still incomplete because per-strategy caps were hit:
  - `exact_norm_name`
  - `prefix3`
  - `rare_token`
  - `alias_norm_overlap`
  - `acronym`
- `name_region` did not hit the cap.
- Do not treat current `57,021` pairs or `47,397` LLM tasks as final full-corpus
  totals until these strategy caps are resolved or deliberately replaced by
  strategy-specific rules.

Risk tier outputs:

- `data/ablation/er/risk_tiers.parquet`
- `data/ablation/er/deterministic_auto_merge_candidates.parquet`
- `data/ablation/er/middle_candidate_pairs.parquet`
- `reports/ablation/er/risk_tier_metrics.csv`

Current global tiering rule:

- `risk_flags != []`: `llm_review`
- `risk_flags == []` and `heuristic_score >= 0.88`:
  `deterministic_auto_merge_candidate`
- otherwise: `candidate_only`

Important safety rule added during this work:

- GB-FTS vs trusted official IDs are flagged as `fts_official_mixed`, not as
  trusted official scheme conflicts.
- Same trusted scheme but different canonical IDs are flagged as
  `trusted_id_conflict` and penalised.

This experiment does not mutate `data/entities/*`.

ER queue live call:

```powershell
python -B scripts/run_azure_llm_queue.py --queue data\ablation\er\llm_review_queue.jsonl --decisions data\ablation\er\llm_decisions.jsonl --raw data\ablation\er\llm_raw_responses.jsonl --errors data\ablation\er\llm_errors.jsonl
```

## Next Suggested Work

1. Convert flat modules into smaller true wrappers once comfortable.
2. Move pipeline files from old numbering to the v2 numbering plan:
   - `00_reference_refresh.py`
   - `10_ingest.py`
   - `20_er_phase1.py`
   - `21_er_phase2.py`
   - `30_extract.py`
3. Start KG implementation directly in `src/procurement_graph/kg/`.

## 2026-06-30 Update: Full-Corpus ER Ablation Completed

The full-corpus ER ablation has now been completed and safely applied to the
live entity tables.

Final ER workflow completed:

- full blocking diagnostics without global candidate truncation
- risk-tier construction over the full candidate set
- Azure LLM pair adjudication for the filtered high-risk queue
- deterministic rule decisions for pure official-scheme conflicts
- final LLM decision de-duplication by `task_id`
- merge / do-not-merge / uncertain distribution report
- cluster-level conflict check before any writeback
- safe merge staging
- backup of pre-merge entity tables
- writeback to `data/entities/*`
- post-write validation

Final LLM decision coverage:

- filtered LLM queue tasks: `71,377`
- missing decision task IDs: `0`
- final LLM decisions:
  - `do_not_merge`: `35,084`
  - `uncertain`: `35,820`
  - `merge`: `473`
- schema-valid LLM decisions: `71,331`
- schema-invalid LLM decisions: `46`

Cluster-level merge gate:

- deterministic auto-merge edges before cluster check: `11,504`
- valid LLM merge edges before cluster check: `472`
- proposed merge edges total: `11,976`
- merge components total: `7,260`
- components with conflict: `673`
- safe merge edges after cluster check: `8,895`
  - deterministic safe edges: `8,635`
  - LLM safe edges: `260`

Applied entity outputs:

- input `canonical_orgs`: `140,263`
- output `canonical_orgs`: `131,502`
- canonical row delta: `-8,761`
- input `alias_map`: `198,897`
- output `alias_map`: `204,711`
- alias row delta: `+5,814`

The alias table grew because old canonical IDs from merged components are
preserved as aliases that resolve to the surviving canonical ID.

Written live files:

- `data/entities/canonical_orgs.parquet`
- `data/entities/alias_map.parquet`
- `data/entities/er_audit.csv`

Backup path:

- `data/entities/backups/er_llm_merge_20260630_121448/`

Key audit/report files:

- `scripts/analyse_er_llm_decisions.py`
- `scripts/apply_safe_er_merges.py`
- `reports/ablation/er/llm_decision_audit_summary.md`
- `reports/ablation/er/safe_merge_apply_summary.md`
- `reports/ablation/er/merge_component_conflicts.csv`
- `reports/ablation/er/llm_merge_sample.csv`
- `reports/ablation/er/llm_merge_blocked_by_cluster_sample.csv`

Post-merge validation:

- duplicate `canonical_orgs.canonical_id`: `0`
- duplicate `alias_map.raw_id`: `0`
- alias targets missing from `canonical_orgs`: `0`
- safe merge edges not joined through `alias_map`: `0`
- non-singleton `GB-FTS-*` canonical IDs: `0`
- `scripts/post_extract_check.py`: PASS WITH WARNINGS before award enrichment;
  later resolved to ALL PASS on 2026-06-30.

Status: entity ablation / ER merge phase is complete. The next phase can move
to KG enrichment and KG construction.

## 2026-06-30 Update: Award Date/Value Enrichment Warnings Resolved

The two post-extract award warnings were resolved by enriching
`data/extracted/awards.parquet` from `data/interim/releases.parquet`
`contracts_json`, without changing award row cardinality.

Written live file:

- `data/extracted/awards.parquet`

Backup path:

- `data/extracted/backups/award_enrichment_20260630_122716/`

Code paths updated:

- `src/procurement_graph/extract/tables.py`
- `src/extract.py`
- `scripts/enrich_awards_from_contracts.py`
- `scripts/post_extract_check.py`
- `tests/test_extract.py`

Award table after enrichment:

- rows preserved: `215,221`
- `award_date_signed`: `193,544 / 215,221` non-empty (`89.9%`)
- raw `award_value_amount`: `15,640 / 215,221` non-null (`7.3%`)
- `award_value_best_amount`: `195,625 / 215,221` non-null (`90.9%`)
- value source counts:
  - `contract`: `160,362`
  - `tender`: `19,623`
  - `award`: `15,640`
  - missing: `19,596`

Important value semantics:

- `award_value_amount` remains the raw OCDS `award.value.amount`.
- `contract_value_amount` is extracted from matching contracts by
  `(ocid, award_id)`.
- `award_value_best_amount` uses fallback order:
  `award.value -> contract.value -> tender.value`.
- Tender fallback improves coverage but is not additive at award-row level,
  because the same tender value can repeat across multiple award rows.
  Aggregate value summaries therefore exclude `award_value_source == "tender"`
  unless the analysis explicitly wants repeated tender-level fallback values.

Validation:

- `scripts/post_extract_check.py`: `72 PASS`, `0 WARN`, `0 FAIL`
- `python -B -m unittest tests.test_qa_benchmark_pipeline`: `7` tests OK
- package and legacy extract imports both succeed

## 2026-06-30 Update: Deterministic KG v0 Built

The first deterministic KG build is complete. It uses only stable structured
inputs and does not perform any new LLM enrichment.

Implemented package modules:

- `src/procurement_graph/kg/schema.py`
- `src/procurement_graph/kg/nodes.py`
- `src/procurement_graph/kg/edges.py`
- `src/procurement_graph/kg/validate.py`
- `src/procurement_graph/kg/build.py`

Pipeline entrypoints:

- `pipelines/40_build_kg.py`
- `pipelines/41_validate_kg.py`

Written KG files:

- `data/kg/nodes/org_nodes.parquet`: `131,502` rows
- `data/kg/nodes/contract_nodes.parquet`: `215,221` rows
- `data/kg/nodes/cpv_nodes.parquet`: `3,870` rows
- `data/kg/edges/buyer_of.parquet`: `215,218` rows
- `data/kg/edges/supplier_of.parquet`: `334,063` rows
- `data/kg/edges/categorized_by.parquet`: `164,691` rows

Validation report:

- `reports/kg/kg_validation_summary.json`
- result: `17 PASS`, `2 WARN`, `0 FAIL`

Validation style matches the ER closeout checks:

- duplicate org node primary keys: `0`
- duplicate contract node primary keys: `0`
- duplicate `(ocid, award_id)`: `0`
- duplicate CPV node primary keys: `0`
- alias targets missing from org nodes: `0`
- edge endpoints missing from org/contract/CPV nodes: `0`
- duplicate edge IDs: `0`
- additive value semantics valid:
  - `award` and `contract` value sources are additive
  - `tender` fallback values are non-additive

The two warnings are coverage signals:

- buyer edge coverage: `215,218 / 215,221` contract nodes
- supplier award coverage: `204,186 / 215,221` contract nodes

Build safety note:

- `pipelines/40_build_kg.py` validates in memory before writing parquet files.
  If validation has any `FAIL`, outputs are not written.

## 2026-06-30 Update: KG v0.1 Deterministic Enrichment Added

The deterministic enrichment layers from `docs/kg_enrichment_plan.md` have been
added to the KG build. This still uses structured local tables only; no LLM
enrichment and no document crawling are involved.

Layer 2: organization context in `org_nodes`:

- role flags: `is_buyer`, `is_supplier`, `is_mixed_role`
- role counts: `buyer_contract_count`, `supplier_contract_count`
- additive value summaries: `buyer_additive_value_sum`,
  `supplier_additive_value_sum`
- activity range: `first_activity_year`, `last_activity_year`
- alias count: `alias_raw_id_count`
- region set: `address_regions`

Region semantics are now stable:

- `address_region` is the primary/modal region from ER output.
- `address_regions` is a JSON list of all observed party regions mapped through
  `alias_map` to the canonical organization.
- `address_regions` is context for QA/evidence, not a merge key.

Observed organization role counts:

- buyers: `7,125`
- suppliers: `114,281`
- mixed-role organizations: `1,269`

Observed region-set counts:

- empty region set: `4,374`
- single-region organizations: `116,063`
- multi-region organizations: `11,065`

Layer 3: CPV enrichment in `cpv_nodes`:

- `contract_count`
- `additive_value_sum`
- `first_activity_year`
- `last_activity_year`

Layer 4: temporal enrichment in `contract_nodes`:

- `has_award_signed_date`
- `has_contract_period`
- `days_release_to_tender_end`
- `days_release_to_award_signed`
- `contract_duration_days`

Observed temporal coverage:

- signed award dates: `193,544`
- release-to-tender-end duration: `106,692`
- release-to-award-signed duration: `193,544`
- contract duration: `46`

Layer 5: text evidence pointers:

- `data/kg/nodes/evidence_nodes.parquet`: `535,731` rows
- `data/kg/edges/evidence_for.parquet`: `1,326,240` rows

Updated KG row counts:

- `org_nodes`: `131,502`
- `contract_nodes`: `215,221`
- `cpv_nodes`: `3,870`
- `evidence_nodes`: `535,731`
- `buyer_of`: `215,218`
- `supplier_of`: `334,063`
- `categorized_by`: `164,691`
- `evidence_for`: `1,326,240`

Updated validation:

- `reports/kg/kg_validation_summary.json`
- result: `23 PASS`, `3 WARN`, `0 FAIL`

Additional region checks:

- invalid `address_regions` JSON rows: `0`
- rows where primary `address_region` is missing from `address_regions`: `0`

The warnings remain coverage-only:

- buyer edge coverage: `215,218 / 215,221`
- supplier award coverage: `204,186 / 215,221`
- evidence pointer coverage: `215,202 / 215,221`

## 2026-06-30 Update: QA Benchmark Interface Connected to KG v0.1

The QA benchmark abstraction now has a real KG backend:

- `procurement_graph.qa.benchmark.kg_interface.ParquetKGQueryBackend`

It reads `data/kg/` parquet outputs and exposes one query record per
`contract_node_id`. Buyer and supplier relationships are represented as
tuple-valued fields on the contract record, so Gate A counts contract evidence
without duplicating rows for multi-supplier awards.

Added test coverage:

- `tests/test_qa_real_kg_backend.py`

Validated behavior:

- exact real-contract `select_unique` spec passes Gate A
- real feature-set `count` spec passes Gate A
- full-graph query performance smoke passes
- existing mock QA tests still pass

Performance note:

- backend initialization is the expensive step because it reads parquet tables
  and groups buyer/supplier summaries in memory
- once initialized, a full-graph query over
  `release_year == 2025`, `tender_category == "services"`, and
  `value_is_additive == True` returned `26,215` rows in about `1` second locally
- `include_evidence=False` skips evidence pointer aggregation for QA generation
  runs that do not need evidence counts

## 2026-06-30 Update: KG v0.1 Field Semantics Frozen for QA

Final pre-freeze checks completed:

- `org_nodes.address_region` confirmed as primary/modal single region.
- `org_nodes.address_regions` added and validated as the all-observed region
  JSON list.
- Layer 2 organization metrics were rechecked after the region-set change:
  - buyers: `7,125`
  - suppliers: `114,281`
  - mixed-role organizations: `1,269`
  - buyer contract count sum: `215,218`
  - supplier contract count sum: `334,063`
  - negative role counts: `0`
- region-set distribution:
  - empty: `4,374`
  - single-region: `116,063`
  - multi-region: `11,065`

QA design limitations documented in:

- `docs/qa_benchmark_design.md`

Sampler rules now explicitly account for:

- supplier coverage: `204,186 / 215,221` contract nodes
- evidence pointer coverage: `215,202 / 215,221` contract nodes
- buyer coverage: `215,218 / 215,221` contract nodes
- aggregate value questions must require `value_is_additive == True`

Decision: KG v0.1 field semantics are frozen for QA sampler development.
Future semantic changes require re-validation of any generated QA benchmark
examples.

## 2026-06-30 Update: QA Stage 1 Local Sampler Pilot

Implemented local deterministic QA Stage 1 for procurement-specific answer
spec generation. No LLM/API calls are made in this stage.

Code added:

- `src/procurement_graph/qa/benchmark/samplers.py`
- `src/procurement_graph/qa/benchmark/stage1.py`
- `src/procurement_graph/qa/benchmark/serialization.py`
- `pipelines/50_build_qa_stage1.py`

Important implementation detail:

- Stage 1 now queries the full KG once per spec and executes Gate A from that
  result, avoiding repeated full-graph queries for completeness and answer
  execution.

Pilot command:

- `python -B pipelines\50_build_qa_stage1.py --target-specs 200 --seed 42 --max-evidence-rows 5000`

Outputs:

- `data/qa/generated/answer_specs.jsonl`
- `data/qa/generated/gate_a_report.jsonl`
- `reports/qa/stage1_summary.json`
- `reports/qa/stage1_answer_spec_sample.csv`

Pilot result:

- attempted: `200`
- accepted: `200`
- rejected: `0`
- Gate A completeness passed: `200 / 200`
- Gate A deterministic-answer check passed: `200 / 200`

Accepted type distribution:

- factoid: `74`
- aggregation-count: `30`
- aggregation-sum: `30`
- conjunction/constraint: `40`
- temporal: `16`
- categorical/CPV: `10`

Evidence set size:

- min: `1`
- max: `736`
- average: `13.37`

Final benchmark target remains approximately `10,000` examples. Before running
the full 10k Stage 1 build, add batching/caching or another performance pass;
the current pilot is correct but still too slow for casual full-scale reruns.
