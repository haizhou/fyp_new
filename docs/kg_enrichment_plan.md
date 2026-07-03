# KG Enrichment Plan

Status: planning note after ER merge and award date/value warning resolution.

The immediate goal is not to add many speculative fields. The goal is to turn
the already-cleaned extraction tables into stable KG-ready semantics: clear
node keys, clear edge keys, clear value/date meanings, and traceable fallback
provenance.

## Completed Pre-Enrichment Fix

`data/extracted/awards.parquet` has been enriched from
`data/interim/releases.parquet` `contracts_json`.

Current award semantics:

- `award_value_amount`: raw OCDS `award.value.amount`.
- `contract_value_amount`: value from matching contract JSON by `(ocid, award_id)`.
- `tender_value_amount`: release-level tender value fallback.
- `award_value_best_amount`: fallback order
  `award.value -> contract.value -> tender.value`.
- `award_value_source`: one of `award`, `contract`, `tender`, or empty.
- `award_date_signed`: contract `date_signed` from matching contract JSON.

Important caveat:

- `award_value_best_amount` is good for coverage and single-award evidence.
- It is not always additive. Rows with `award_value_source == "tender"` can
  repeat the same tender-level value across multiple awards, so aggregate value
  summaries should normally exclude tender fallback unless the analysis
  explicitly wants repeated tender-level fallback values.

Validation after this fix:

- `scripts/post_extract_check.py`: `72 PASS`, `0 WARN`, `0 FAIL`.
- `tests/test_extract.py`: `73 passed`.
- QA benchmark unit tests: `7 passed`.

## Enrichment Principles

1. Preserve raw fields.
   Enriched fields should sit next to raw extraction fields, not overwrite them.

2. Always store provenance.
   Any fallback or inferred field needs a source column, for example
   `award_value_source`.

3. Keep aggregation semantics explicit.
   Fields intended for row-level evidence are not automatically valid for
   graph-level totals.

4. Prefer deterministic enrichment before LLM enrichment.
   LLMs should adjudicate evidence or ambiguity, not repair obvious structured
   fields.

5. Do not build an offline document corpus now.
   Document evidence remains verdict-time and opportunistic: KG first, live
   document fetch only if a relevant `document_url` exists and is accessible.

## Recommended Enrichment Layers

### Layer 1: Contract/Award Semantics

Purpose: make contract and award nodes queryable without losing OCDS nuance.

Fields to expose in KG:

- `contract_node.value_amount`: prefer `award_value_best_amount`.
- `contract_node.value_source`: from `award_value_source`.
- `contract_node.value_is_additive`: false when source is `tender`, true for
  `award` or `contract`.
- `contract_node.award_date_signed`: from enriched awards.
- `contract_node.award_status`, `award_title`, `award_period_start`,
  `award_period_end`.

Useful validation:

- no duplicate `(ocid, award_id)` award nodes.
- every supplier edge points to an existing canonical organization.
- value totals only sum rows with `value_is_additive == true` by default.

### Layer 2: Organization Context

Purpose: make entity nodes useful for KGQA and evidence verdicts.

Fields to derive from existing entity/extraction data:

- role counts: buyer count, supplier count, mixed-role flag.
- source identifier schemes seen: GB-COH, GB-NHS, GB-CHC, GB-FTS, etc.
- region set and primary region, stored as attributes rather than merge keys.
- procurement activity counts by year.
- total additive award value as buyer/supplier, using additive value rows only.

This layer should not reopen ER merges. It should annotate canonical entities
after the safe ER result.

### Layer 3: CPV and Category Normalisation

Purpose: support category questions and path queries.

Fields/nodes:

- CPV code node from `tender_core.tender_cpv_id`.
- CPV division/group/class/subclass prefixes where code length allows.
- procurement method and main category as controlled attributes.

Implementation should be deterministic. Missing or malformed CPV values should
be kept as null, not guessed.

### Layer 4: Temporal Features

Purpose: support questions about timing, delays, and active periods.

Candidate fields:

- tender published/release date from `data/interim/releases.parquet.date`.
- tender period end.
- award signed date.
- award/contract period start/end.
- derived durations where both endpoints are present.

All derived duration fields need endpoint provenance and null-safe handling.

### Layer 5: Text Evidence Pointers

Purpose: connect KG paths to exact textual evidence without building a document
retrieval system.

Use existing `data/extracted/text_evidence.parquet`:

- attach short text snippets to contract/tender nodes by `ocid`.
- store `field_path` and `lot_id` so evidence remains traceable.
- for long text, select top snippets at verdict time using keyword overlap with
  the claim/question.

Do not pre-chunk the whole corpus yet. Chunk only when evidence verdict needs a
long text field.

## Not Recommended Now

- Full offline document crawling or OCR.
- HTML portal parsers for authenticated procurement platforms.
- LLM-based enrichment of every node.
- Region-driven organization merging after ER closure.
- Aggregate totals over `award_value_best_amount` without checking
  `award_value_source`.

## Suggested Implementation Order

Status: implemented for KG v0 on 2026-06-30.

1. Add `procurement_graph.kg.schema` with node/edge data contracts.
2. Add `procurement_graph.kg.nodes` for organization, contract/award, and CPV
   node tables.
3. Add `procurement_graph.kg.edges` for buyer, supplier, and category edges.
4. Add `procurement_graph.kg.validate` with row counts, key uniqueness,
   referential integrity, and additive-value checks.
5. Add thin pipeline entrypoints:
   - `pipelines/40_build_kg.py`
   - `pipelines/41_validate_kg.py`
6. Only after KG validation passes, connect QA benchmark interfaces to the real
   KG tables.

Implementation note: KG v0 combines `data/interim/releases.parquet` with
`data/extracted/awards.parquet` because the first extracted tender table is a
rich tender-extension table, while release-level buyer/title/value/CPV fields
still live in the interim snapshot.

## First Build Target

The first KG build should be intentionally small and deterministic:

- organization nodes from `data/entities/canonical_orgs.parquet`.
- contract/award nodes from enriched `data/extracted/awards.parquet` joined to
  `data/interim/releases.parquet`.
- buyer edges from `releases.buyer_raw_id -> alias_map -> canonical_id`.
- supplier edges from `awards.supplier_raw_ids -> alias_map -> canonical_id`.
- CPV edges from `releases.tender_cpv_id`.

This is enough to support early KGQA and evidence-verdict experiments while
keeping enrichment logic auditable.

## KG v0 Build Result

Generated files:

- `data/kg/nodes/org_nodes.parquet`: `131,502` rows.
- `data/kg/nodes/contract_nodes.parquet`: `215,221` rows.
- `data/kg/nodes/cpv_nodes.parquet`: `3,870` rows.
- `data/kg/edges/buyer_of.parquet`: `215,218` rows.
- `data/kg/edges/supplier_of.parquet`: `334,063` rows.
- `data/kg/edges/categorized_by.parquet`: `164,691` rows.
- `reports/kg/kg_validation_summary.json`.

Validation result:

- `17 PASS`
- `2 WARN`
- `0 FAIL`

The two warnings are coverage signals, not structural failures:

- buyer edge coverage: `215,218 / 215,221` contract nodes.
- supplier award coverage: `204,186 / 215,221` contract nodes.

Build safety:

- `pipelines/40_build_kg.py` builds tables in memory and runs validation before
  writing outputs. If any validation check has `FAIL`, parquet outputs are not
  written.
- `pipelines/41_validate_kg.py` re-validates already-written KG outputs from
  disk.

## KG v0.1 Deterministic Enrichment Result

Implemented on 2026-06-30.

Layer 2 organization context added to `org_nodes`:

- `alias_raw_id_count`
- `address_regions`
- `is_buyer`
- `is_supplier`
- `is_mixed_role`
- `buyer_contract_count`
- `supplier_contract_count`
- `buyer_additive_value_sum`
- `supplier_additive_value_sum`
- `first_activity_year`
- `last_activity_year`

Region semantics:

- `address_region` is the primary/modal region inherited from ER outputs.
- `address_regions` is a JSON list of all observed non-empty party
  `address_region` values mapped through `alias_map` to the canonical
  organization.
- `address_regions` is an attribute for QA/evidence context, not a merge key.

Observed role coverage:

- buyer organizations: `7,125`
- supplier organizations: `114,281`
- mixed buyer/supplier organizations: `1,269`

Observed region-set coverage:

- empty region set: `4,374`
- single-region organizations: `116,063`
- multi-region organizations: `11,065`

Layer 3 CPV enrichment added to `cpv_nodes`:

- `contract_count`
- `additive_value_sum`
- `first_activity_year`
- `last_activity_year`

Layer 4 temporal enrichment added to `contract_nodes`:

- `has_award_signed_date`
- `has_contract_period`
- `days_release_to_tender_end`
- `days_release_to_award_signed`
- `contract_duration_days`

Observed temporal coverage:

- `has_award_signed_date == true`: `193,544`
- `days_release_to_tender_end` non-null: `106,692`
- `days_release_to_award_signed` non-null: `193,544`
- `contract_duration_days` non-null: `46`

Layer 5 text evidence pointers added:

- `data/kg/nodes/evidence_nodes.parquet`: `535,731` rows.
- `data/kg/edges/evidence_for.parquet`: `1,326,240` rows.

Evidence design:

- `evidence_nodes` stores existing extracted text with stable `evidence_id`.
- `evidence_for` links evidence to contract nodes by `ocid`; when `lot_id` is
  present it first tries lot-level matching against `contract_nodes.related_lots`
  and falls back to OCID-level matching if needed.
- This is still a pointer layer, not a document parser and not an offline
  document crawl.

Validation result after region-set stabilization:

- `23 PASS`
- `3 WARN`
- `0 FAIL`

Additional region checks:

- `address_regions` is valid JSON for every org node.
- non-empty primary `address_region` is always included in `address_regions`.

The three warnings are coverage signals:

- buyer edge coverage: `215,218 / 215,221` contract nodes.
- supplier award coverage: `204,186 / 215,221` contract nodes.
- evidence pointer coverage: `215,202 / 215,221` contract nodes.
