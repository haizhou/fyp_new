# Rebuild Plan: Data → Entity Resolution → KG

**Scope:** New data/KG layer only. Existing reasoning pipeline (planner/executor/reflector/Step3)
plugs in later via an adapter; the old pipeline directory is not inside this workspace.  
**Status:** Implementation planning pass — structure and decisions finalised, no code written yet.

**Decisions already fixed (not open for re-discussion):**
- Latest-release-per-OCID is the default KG snapshot.
- GB-FTS is never used as a canonical entity ID; always stored as alias only.
- GB-COH / GB-NHS / GB-UKPRN / GB-CHC / GB-MPR → deterministic canonical merge.
- Exact normalised-name + address/region → conservative second-phase merge only.
- Fuzzy matching produces a candidate report (`er_candidates.csv`) for human review; never
  produces automatic merges.
- Framework-like contracts are flagged (`is_framework` column), not excluded.
- Old pipeline compatibility is handled later through `kg_adapter.py`; the KG schema itself
  carries no backward-compat hacks.

See `docs/project_structure_proposal.md` for the full three-option comparison.

---

## 1. Adopted Project Structure (Option C — Hybrid)

The recommended and adopted structure is **Option C** from the structure proposal: flat `src/`
package + numbered `pipelines/` scripts + strict four-layer data directory.

```
fyp_new/
│
├── data/
│   ├── raw/                     # Immutable downloaded .jsonl.gz — never modified by code
│   │   ├── 2022.jsonl.gz
│   │   ├── 2023.jsonl.gz
│   │   ├── 2024.jsonl.gz
│   │   ├── 2025.jsonl.gz
│   │   └── 2026.jsonl.gz
│   ├── interim/                 # Flattened releases (all years merged, deduped by OCID)
│   │   └── releases.parquet
│   ├── entities/                # Entity resolution outputs
│   │   ├── canonical_orgs.parquet    # One row per canonical entity
│   │   ├── alias_map.parquet         # raw_id → canonical_id (many-to-one)
│   │   ├── er_candidates.csv         # Fuzzy-match candidates for human review
│   │   └── er_audit.csv              # Per-entity merge decision provenance
│   └── kg/
│       ├── nodes/
│       │   ├── org_nodes.parquet
│       │   ├── contract_nodes.parquet
│       │   └── cpv_nodes.parquet
│       └── edges/
│           ├── buyer_of.parquet
│           ├── supplier_of.parquet
│           └── categorized_by.parquet
│
├── src/                         # Flat importable library (no subpackages)
│   ├── __init__.py
│   ├── ingest.py                # OCDS decompression, streaming, flattening
│   ├── normalise.py             # Name/ID normalisation utilities (shared)
│   ├── er_phase1.py             # Deterministic ER: official-ID priority merge
│   ├── er_phase2.py             # Heuristic ER: exact norm-name + region merge
│   ├── er_candidates.py         # Fuzzy candidate report (review only, no auto-merge)
│   ├── kg_nodes.py              # Build org/contract/CPV node parquets
│   ├── kg_edges.py              # Build buyer_of/supplier_of/categorized_by parquets
│   ├── kg_query.py              # Stable aggregation query API
│   └── kg_adapter.py            # Old-pipeline compatibility shim (stub for now)
│
├── pipelines/                   # Numbered orchestration scripts — no business logic
│   ├── 01_ingest.py             # Calls src/ingest.py → writes data/interim/
│   ├── 02_er_phase1.py          # Calls src/er_phase1.py → writes data/entities/
│   ├── 03_er_phase2.py          # Calls src/er_phase2.py → extends data/entities/
│   ├── 04_build_kg.py           # Calls src/kg_nodes.py + kg_edges.py → writes data/kg/
│   └── 05_validate_kg.py        # Calls src/kg_query.py → prints sanity checks
│
├── configs/
│   ├── settings.yaml            # Paths, thresholds, flags
│   └── gov_lookup.json          # Curated: normalised govt name → canonical_id
│
├── notebooks/
│   └── 01_explore_ocds.ipynb    # EDA only — never imported by src/
│
├── tests/
│   ├── test_normalise.py
│   ├── test_er_phase1.py
│   ├── test_er_phase2.py
│   ├── test_kg_nodes.py
│   └── test_kg_query.py
│
└── docs/
    ├── reference_study.md
    ├── ocds_data_analysis.md
    ├── rebuild_plan.md               # ← this document
    └── project_structure_proposal.md
```

**Import rules (enforced by convention, not tooling):**
- `kg_*` modules never import from `er_*` or `ingest`.
- `er_*` modules never import from `kg_*`.
- `ingest.py` never imports from `er_*` or `kg_*`.
- `pipelines/` scripts never import from each other.
- `data/raw/` is never written to by any script.

---

## 2. Stage-by-Stage Pipeline Plan

### Stage 1 — Ingest (`pipelines/01_ingest.py` → `src/ingest.py`)

**Input:** `data/raw/*.jsonl.gz`  
**Output:** `data/interim/releases.parquet`

#### What it does

1. Decompress each year file line by line (streaming, never loads full year into memory).
2. For each OCDS release, extract a flat row covering all fields needed downstream.
   A release becomes **two sub-tables** merged into a single wide row:
   - **Release-level fields** (one per OCID): `ocid`, `release_id`, `date`, `year`, `tag`,
     `buyer_raw_id`, `buyer_name`,
     `tender_title`, `tender_value_amount`, `tender_cpv_id`, `tender_cpv_description`,
     `tender_method`, `tender_category`, `tender_period_end`,
     `contract_period_start`, `contract_period_end`.
   - **Party fields** stored as JSON strings (lists of dicts) in the same row:
     `parties_json` — serialised array of all party records with all fields from OCDS
     `parties[]`: `{id, name, roles, identifier.scheme, identifier.id, identifier.legalName,
     address.region, address.postalCode, details.url, details.classifications}`.
   - **Award/contract fields**: `contracts_json` — serialised array of contract records:
     `{contract_id, value_amount, value_currency, date_signed, period_start, period_end,
     award_id_ref, supplier_raw_ids (list)}`.
3. Collect all rows from all 5 years into one DataFrame.
4. **Deduplication:** group by `ocid`, keep the row with the latest `date`. This implements
   the "latest-release-per-OCID" decision. A `source_years` column records which year files
   contained this OCID (for audit).
5. Write to `data/interim/releases.parquet`.

#### Schema of `releases.parquet` (key columns)

| Column | Type | Source |
|--------|------|--------|
| `ocid` | str | release.ocid |
| `release_id` | str | release.id |
| `date` | datetime | release.date |
| `year` | int | derived from date |
| `source_years` | str | pipe-delimited list of years that had this OCID |
| `buyer_raw_id` | str | release.buyer.id |
| `buyer_name` | str | release.buyer.name |
| `tender_title` | str | release.tender.title |
| `tender_value_amount` | float | release.tender.value.amount (nullable) |
| `tender_cpv_id` | str | release.tender.classification.id (nullable) |
| `tender_cpv_description` | str | release.tender.classification.description (nullable) |
| `tender_method` | str | release.tender.procurementMethod |
| `tender_category` | str | release.tender.mainProcurementCategory |
| `tender_period_end` | datetime | release.tender.tenderPeriod.endDate (nullable) |
| `contract_period_start` | datetime | release.tender.contractPeriod.startDate (nullable) |
| `contract_period_end` | datetime | release.tender.contractPeriod.endDate (nullable) |
| `parties_json` | str | JSON-serialised list of party dicts |
| `contracts_json` | str | JSON-serialised list of contract dicts |

**Why JSON columns for nested data?** Parquet supports list/struct columns natively, but
pandas + polars handle them inconsistently. Serialising as JSON strings keeps the intermediate
file universally readable and avoids version-specific schema pain. The ER and KG stages
deserialise on read.

#### `src/ingest.py` — Functions to implement

```
load_year(path: Path) -> Iterator[dict]
    Streams decompressed JSONL; yields one parsed dict per line.

extract_release_row(release: dict) -> dict
    Extracts flat row from a single OCDS release dict.
    Serialises parties and contracts to JSON strings.
    Returns {} for malformed records (caller skips).

flatten_all_years(raw_dir: Path) -> pd.DataFrame
    Calls load_year for each .jsonl.gz in raw_dir.
    Collects rows, deduplicates by OCID (keep latest date).
    Returns the full releases DataFrame.

write_interim(df: pd.DataFrame, out_path: Path) -> None
    Writes parquet with snappy compression.
```

---

### Stage 2a — Deterministic ER (`pipelines/02_er_phase1.py` → `src/er_phase1.py`)

**Input:** `data/interim/releases.parquet`  
**Output:** `data/entities/canonical_orgs.parquet` (initial), `data/entities/alias_map.parquet` (initial)

#### Goal

Assign a `canonical_id` to every entity that has an official non-FTS identifier.
These assignments are certain and never revisited.

#### Identifier Priority Order

| Priority | Scheme | Canonical ID format | Registry |
|----------|--------|--------------------|---------| 
| 1 | GB-COH | `GB-COH-<id>` | UK Companies House |
| 2 | GB-NHS | `GB-NHS-<code>` | NHS Organisation Data Service |
| 3 | GB-UKPRN | `GB-UKPRN-<number>` | UKRI Provider Reference |
| 4 | GB-CHC | `GB-CHC-<number>` | Charity Commission England & Wales |
| 5 | GB-SC | `GB-SC-<number>` | Office of the Scottish Charity Regulator |
| 6 | GB-NIC | `GB-NIC-<number>` | Charity Commission Northern Ireland |
| 7 | GB-MPR | `GB-MPR-<number>` | Mutuals Public Register |

#### Logic

1. Read `parties_json` from releases; deserialise each party.
2. Collect all unique `(raw_id, party_scheme, party_official_id, party_legal_name, party_name,
   party_roles, party_address_region, party_org_category)` tuples across all records.
3. For each unique `raw_id`:
   - If `party_scheme` is in the priority list above → `canonical_id = "<scheme>-<official_id>"`.
   - Normalise the official_id (strip whitespace, uppercase for NHS codes).
4. Cross-reference: for each `ocid`, if a party appears with two different raw_ids but
   one has an official scheme and the other is GB-FTS, link them as aliases:
   - Check: same `ocid`, same `role`, `normalise(name_A) == normalise(name_B)` → alias.
5. Build `alias_map`: all raw_ids (including official-scheme raw_ids) → canonical_id.
6. Build initial `canonical_orgs`: one row per canonical_id with aggregated metadata.
7. Mark all GB-FTS-only entities as `er_status = "unresolved"` → deferred to Phase 2.
8. Mark officially resolved entities as `er_status = "deterministic"`.

#### `src/er_phase1.py` — Functions to implement

```
collect_parties(releases_df: pd.DataFrame) -> pd.DataFrame
    Deserialises parties_json; returns flat party table with one row per (ocid, raw_id, role).

is_official_scheme(scheme: str) -> bool
    Returns True for GB-COH, GB-NHS, GB-UKPRN, GB-CHC, GB-SC, GB-NIC, GB-MPR.

scheme_priority(scheme: str) -> int
    Returns priority rank (1 = highest). GB-FTS returns 99.

resolve_official(parties_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]
    Assigns canonical_id to all parties with official scheme.
    Cross-links GB-FTS aliases within same ocid by name match.
    Returns (canonical_orgs_df, alias_map_df).
```

Depends on: `src/normalise.py`

---

### Stage 2b — Heuristic ER (`pipelines/03_er_phase2.py` → `src/er_phase2.py`, `src/er_candidates.py`)

**Input:** `data/entities/canonical_orgs.parquet`, `data/entities/alias_map.parquet`,
          `configs/gov_lookup.json`, `data/interim/releases.parquet`  
**Output:** Updated `data/entities/canonical_orgs.parquet`, updated `data/entities/alias_map.parquet`,
            `data/entities/er_candidates.csv`, `data/entities/er_audit.csv`

#### Goal

Assign `canonical_id` to the remaining GB-FTS-only entities using conservative heuristics.
Produce a fuzzy-match candidate report for human review (no automatic merges from fuzzy matching).

#### Step 2a — Government Entity Lookup (buyers, deterministic)

Config file `configs/gov_lookup.json` maps normalised canonical names to stable synthetic IDs:
```json
{
  "MINISTRY OF DEFENCE": "GOV-MOD",
  "UK RESEARCH AND INNOVATION": "GOV-UKRI",
  "NHS ENGLAND": "GOV-NHSE",
  "SCOTTISH GOVERNMENT": "GOV-SCOTGOV",
  ...
}
```

For each unresolved entity: `normalise(name)` → look up in gov_lookup → if found,
assign `canonical_id = GOV-<SLUG>`, `er_status = "gov_lookup"`.

This handles the Ministry of Defence (77 GB-FTS IDs) and similar large-org fragmentation
cases deterministically.

#### Step 2b — Exact Normalised Name + Region Merge

For remaining GB-FTS-only entities:
1. Group by `(normalise(name), address_region)`.
2. If a group has >1 distinct raw_id → all are aliases of each other.
3. Assign `canonical_id = "MERGED-<sha256(norm_name + region)[:12]>"`, `er_status = "name_region_merge"`.
4. If region is null/blank for all members, group by `normalise(name)` only → `er_status = "name_only_merge"`.
5. Singletons (no match) retain their GB-FTS raw_id as canonical_id, `er_status = "singleton"`.

#### Step 2c — Fuzzy Candidate Report (no auto-merge)

`src/er_candidates.py` generates `data/entities/er_candidates.csv` for human review:
- Uses Jaro-Winkler similarity (threshold > 0.92) between unresolved entity normalised names.
- Outputs: `(entity_a_id, entity_a_name, entity_b_id, entity_b_name, similarity, shared_region)`.
- **This file is never read by any pipeline stage.** It is for human inspection only.
- After review, confirmed matches can be added manually to `configs/gov_lookup.json`.

#### Audit Log

`data/entities/er_audit.csv` — one row per canonical entity:

| Column | Content |
|--------|---------|
| `canonical_id` | The assigned canonical ID |
| `er_status` | One of: `deterministic`, `gov_lookup`, `name_region_merge`, `name_only_merge`, `singleton` |
| `evidence` | Pipe-delimited list of raw_ids that were merged into this canonical_id |
| `canonical_name` | Display name used |
| `n_aliases` | Total number of raw_ids pointing to this entity |

#### `src/er_phase2.py` — Functions to implement

```
apply_gov_lookup(unresolved_df: pd.DataFrame, lookup: dict) -> pd.DataFrame
    Applies gov_lookup.json to unresolved entities. Returns updated entity rows.

merge_by_name_region(unresolved_df: pd.DataFrame) -> pd.DataFrame
    Groups by (norm_name, region) and assigns MERGED-* canonical_id.

build_audit_log(canonical_orgs: pd.DataFrame, alias_map: pd.DataFrame) -> pd.DataFrame
    Produces per-entity audit rows.
```

`src/er_candidates.py` — Functions to implement

```
generate_candidates(unresolved_df: pd.DataFrame, threshold: float = 0.92) -> pd.DataFrame
    Computes pairwise Jaro-Winkler on normalised names for remaining singletons.
    Returns candidate pair rows above threshold. Does not write anything.

write_candidates(candidates_df: pd.DataFrame, out_path: Path) -> None
```

Depends on: `src/normalise.py`

---

### Stage 3 — KG Construction (`pipelines/04_build_kg.py` → `src/kg_nodes.py`, `src/kg_edges.py`)

**Input:** `data/entities/canonical_orgs.parquet`, `data/entities/alias_map.parquet`,
          `data/interim/releases.parquet`  
**Output:** `data/kg/nodes/*.parquet`, `data/kg/edges/*.parquet`

#### Node Schemas

**`org_nodes.parquet`** — one row per canonical entity:

| Column | Type | Notes |
|--------|------|-------|
| `canonical_id` | str | Primary key |
| `canonical_name` | str | Best display name (most recent / most common) |
| `org_type` | str | `buyer`, `supplier`, or `both` |
| `er_status` | str | From audit: `deterministic`, `gov_lookup`, etc. |
| `official_scheme` | str | Non-FTS scheme if available, else null |
| `official_id` | str | ID within that scheme |
| `alias_ids` | str | JSON list of all raw_ids (GB-FTS + official) |
| `alias_names` | str | JSON list of all name variants seen |
| `address_region` | str | Most common NUTS region code |
| `org_category` | str | TED_CA_TYPE (MINISTRY, BODY_PUBLIC, etc.) |
| `cofog_code` | str | COFOG sector code (nullable) |
| `first_seen` | date | Earliest `date` of any release involving this entity |
| `last_seen` | date | Latest `date` |
| `n_contracts_buyer` | int | Number of contracts as buyer |
| `n_contracts_supplier` | int | Number of contracts as supplier |

**`contract_nodes.parquet`** — one row per contract (from `contracts_json`):

| Column | Type | Notes |
|--------|------|-------|
| `contract_id` | str | `<ocid>/<contract.id>` — unique key |
| `ocid` | str | Parent contracting process |
| `date_signed` | date | `contracts[].dateSigned` (nullable) |
| `release_date` | date | Date of the OCDS release |
| `year` | int | Year of release_date |
| `value_amount` | float | Signed contract value (GBP) — most reliable |
| `value_currency` | str | Always GBP here |
| `tender_value` | float | Pre-award estimated value (nullable, ~31% coverage) |
| `is_framework` | bool | True if tender has multiple lots OR value > 500M GBP |
| `tender_title` | str | From parent release tender.title |
| `tender_method` | str | `open`, `limited`, `negotiated`, etc. |
| `tender_category` | str | `goods`, `services`, `works` |
| `cpv_id` | str | Primary CPV code (nullable) |
| `cpv_description` | str | CPV label (nullable) |
| `buyer_canonical_id` | str | Denormalised for fast buyer-centric queries |

**`cpv_nodes.parquet`** — one row per unique CPV code seen:

| Column | Type | Notes |
|--------|------|-------|
| `cpv_id` | str | e.g. `42636000` |
| `cpv_description` | str | e.g. `Presses` |
| `cpv_division` | str | First 2 digits (e.g. `42`) |
| `cpv_division_label` | str | Human label for the division (from a static CPV lookup) |

#### Edge Schemas

**`buyer_of.parquet`**:

| Column | Type |
|--------|------|
| `buyer_canonical_id` | str |
| `contract_id` | str |
| `raw_buyer_id` | str |

**`supplier_of.parquet`**:

| Column | Type |
|--------|------|
| `supplier_canonical_id` | str |
| `contract_id` | str |
| `raw_supplier_id` | str |
| `award_id` | str |

**`categorized_by.parquet`**:

| Column | Type |
|--------|------|
| `contract_id` | str |
| `cpv_id` | str |

#### `src/kg_nodes.py` — Functions to implement

```
build_org_nodes(canonical_orgs: pd.DataFrame, alias_map: pd.DataFrame,
                releases: pd.DataFrame) -> pd.DataFrame
    Joins entity table with release statistics. Returns org_nodes_df.

build_contract_nodes(releases: pd.DataFrame, alias_map: pd.DataFrame) -> pd.DataFrame
    Deserialises contracts_json; creates one row per contract.
    Applies is_framework detection logic.
    Denormalises buyer_canonical_id by joining alias_map on buyer_raw_id.

build_cpv_nodes(contract_nodes: pd.DataFrame) -> pd.DataFrame
    Extracts unique CPV codes from contract_nodes.
    Adds division from first 2 digits.
```

#### `src/kg_edges.py` — Functions to implement

```
build_buyer_of(contract_nodes: pd.DataFrame) -> pd.DataFrame
    Reads buyer_canonical_id from contract_nodes. Returns buyer_of_df.

build_supplier_of(releases: pd.DataFrame, alias_map: pd.DataFrame) -> pd.DataFrame
    Deserialises contracts_json to get (contract_id, award_id, supplier_raw_ids).
    Joins alias_map to get supplier_canonical_ids.
    Returns supplier_of_df.

build_categorized_by(contract_nodes: pd.DataFrame) -> pd.DataFrame
    Extracts (contract_id, cpv_id) pairs from contract_nodes.
    Returns categorized_by_df.
```

Neither `kg_nodes.py` nor `kg_edges.py` imports from `er_phase1`, `er_phase2`, or `ingest`.
They read only from parquet files via pandas.

---

### Stage 4 — Query Interface (`src/kg_query.py`)

**Input:** `data/kg/nodes/*.parquet`, `data/kg/edges/*.parquet` (loaded once at import time)  
**Output:** Python return values — not files

#### Design Principles

- Loads all parquet files into DataFrames at module import (or lazily on first call).
  At ~166K contracts + ~50K entities the in-memory footprint is manageable (<1 GB RAM).
- All public functions accept `canonical_id` as the entity key.
- All functions that accept `year` accept `None` to mean "all years".
- Return types are typed dataclasses (`OrgEntity`, `Contract`) — not raw dicts.
  This gives the old pipeline a stable typed contract to code against.

#### Public API

```python
# Entity lookup
def get_org(canonical_id: str) -> OrgEntity | None
def search_org_by_name(name: str, top_k: int = 5) -> list[OrgEntity]
    # exact normalised match first; falls back to alias_names scan

# Contract sets
def contracts_of(entity_id: str, role: str = "any",
                 year: int | None = None,
                 exclude_frameworks: bool = False) -> list[Contract]

# Aggregations (all return None if no data found)
def count_contracts(entity_id: str, role: str = "any",
                    year: int | None = None) -> int
def sum_value(entity_id: str, role: str = "any",
              year: int | None = None,
              exclude_frameworks: bool = False) -> float | None
def max_value_contract(entity_id: str, role: str = "any",
                       year: int | None = None) -> Contract | None
def first_contract(entity_id: str, role: str = "any") -> Contract | None
def last_contract(entity_id: str, role: str = "any") -> Contract | None

# Relational
def suppliers_to_buyer(buyer_id: str, year: int | None = None) -> list[OrgEntity]
def buyers_of_supplier(supplier_id: str, year: int | None = None) -> list[OrgEntity]
def top_suppliers_by_value(buyer_id: str, n: int = 10,
                           year: int | None = None) -> list[tuple[OrgEntity, float]]
def top_buyers_by_value(supplier_id: str, n: int = 10,
                        year: int | None = None) -> list[tuple[OrgEntity, float]]

# Category
def contracts_by_cpv(cpv_id: str, year: int | None = None) -> list[Contract]
def orgs_by_cpv(cpv_id: str, role: str = "supplier") -> list[OrgEntity]
```

#### `src/kg_adapter.py` (stub)

```python
# Placeholder — will be implemented when old pipeline is reconnected.
# Purpose: translate between old pipeline's entity key format and canonical_id.
# Does NOT modify kg_query.py or any KG schema.

from kg_query import get_org, contracts_of  # etc.

def legacy_get_entity(old_key: str):
    # old_key may be a GB-FTS id, a raw name string, or a canonical_id
    # Step 1: try canonical_id directly
    # Step 2: look up in alias_map
    # Step 3: name search fallback
    raise NotImplementedError("Adapter not yet implemented")
```

---

## 3. Shared Utility: `src/normalise.py`

Used by `er_phase1.py`, `er_phase2.py`, `er_candidates.py`, and `kg_query.py`'s name search.
Must be self-contained (no imports from other `src/` modules).

```python
# Functions to implement:

def normalise_name(name: str) -> str
    # uppercase → strip whitespace → expand & → AND
    # strip common legal suffixes (LIMITED, LTD, PLC, CIC, LLP, INCORPORATED, INC)
    # strip trailing punctuation → collapse multiple spaces
    # Examples:
    #   "NHS England " → "NHS ENGLAND"
    #   "Ingeus UK Limited" → "INGEUS UK"
    #   "Dept. for Energy Security & Net Zero" → "DEPT FOR ENERGY SECURITY AND NET ZERO"

def normalise_id(scheme: str, raw_id: str) -> str
    # Strip whitespace from raw_id; uppercase for NHS codes; keep as-is for COH numbers
    # Return "<scheme>-<normalised_id>"

def canonical_id_from_party(scheme: str, official_id: str) -> str
    # Returns "<scheme>-<normalise_id(scheme, official_id)>"

SUFFIX_PATTERNS: list[re.Pattern]  # compiled once at module load
```

---

## 4. Configuration: `configs/settings.yaml`

```yaml
data:
  raw_dir: data/raw
  interim_dir: data/interim
  entities_dir: data/entities
  kg_dir: data/kg

ingest:
  deduplicate_by: latest_date   # "latest_date" | "all" (future option)

entity_resolution:
  official_schemes:             # in priority order
    - GB-COH
    - GB-NHS
    - GB-UKPRN
    - GB-CHC
    - GB-SC
    - GB-NIC
    - GB-MPR
  gov_lookup_path: configs/gov_lookup.json
  fuzzy_threshold: 0.92         # Jaro-Winkler; candidates only, no auto-merge

kg:
  framework_value_threshold: 500_000_000   # GBP; contracts above this get is_framework=True
  framework_multi_lot_threshold: 5         # >=5 lots → is_framework=True
```

---

## 5. Tests to Write (Before or Alongside Implementation)

All tests use small fixture data — never the full 166K record dataset.

### `tests/test_normalise.py`

```
test_normalise_name_uppercase()
test_normalise_name_ampersand_expansion()
test_normalise_name_suffix_strip()
test_normalise_name_whitespace()
test_normalise_name_edge_cases()   # empty string, None, unicode
test_normalise_id_nhs_uppercase()
test_normalise_id_coh_whitespace()
```

### `tests/test_er_phase1.py`

```
test_official_scheme_gets_canonical_id()
test_gb_fts_marked_unresolved()
test_cross_ref_alias_within_same_ocid()
test_priority_order_coh_wins_over_nhs()
test_same_coh_id_two_names_merged()
```

Fixture: a small DataFrame with ~20 synthetic party rows covering all scheme types
and the cross-reference case.

### `tests/test_er_phase2.py`

```
test_gov_lookup_applied_to_ministry()
test_name_region_merge_creates_merged_id()
test_name_only_merge_without_region()
test_singleton_keeps_fts_id()
test_audit_log_has_row_per_canonical()
test_fuzzy_candidates_not_auto_merged()
```

### `tests/test_kg_nodes.py`

```
test_contract_node_has_signed_value()
test_is_framework_flag_high_value()
test_is_framework_flag_multi_lot()
test_buyer_canonical_id_denormalised()
test_cpv_node_division_derived()
test_org_node_type_both_when_buyer_and_supplier()
```

### `tests/test_kg_query.py`

```
test_get_org_returns_none_for_unknown()
test_search_org_by_name_exact_match()
test_search_org_by_alias_name()
test_count_contracts_buyer()
test_count_contracts_supplier()
test_count_contracts_filtered_by_year()
test_sum_value_excludes_frameworks_when_flagged()
test_max_value_contract_correct()
test_first_last_contract_by_date()
test_top_suppliers_sorted_by_value()
```

All tests use a ~50-row fixture KG (10 orgs, 30 contracts, 50 edges) built in a `conftest.py` fixture.

---

## 6. Risk Register (Updated)

| Risk | Severity | Mitigation | Status |
|------|----------|------------|--------|
| GB-FTS IDs not stable cross-year | HIGH | Run ER across all 5 years combined; name+region merge catches cross-year same-name entities | Mitigated by design |
| Framework agreements inflate value sums | MEDIUM | `is_framework` flag on ContractNode; `exclude_frameworks` param in `sum_value()` | Decision fixed: flag, don't exclude |
| OCID duplication across year files | MEDIUM | Latest-date-per-OCID deduplication in ingest | Decision fixed: latest-only |
| Gov lookup table incomplete (MoD has 77 IDs) | MEDIUM | `er_candidates.csv` for human review; `gov_lookup.json` is incrementally extensible | Design handles it |
| Missing contract values (only 31% have tender value; 0% have award value) | LOW | Store both `value_amount` (signed) and `tender_value` (estimated) separately; signed is primary | Mitigated by design |
| Old pipeline uses GB-FTS keys as entity references | MEDIUM | `kg_adapter.py` provides alias-map lookup; old pipeline connects only through adapter | Deferred; adapter is a stub |
| Normalised name merge creates false positives (e.g. "NHS" matching unrelated entities) | LOW | Region secondary check; gov_lookup has priority; singletons preferred over false merges | Mitigated by two-factor check |

---

## 7. Implementation Order

When implementation is approved, work in this sequence. Each item is one working session.

| Step | File(s) | Depends on | Output |
|------|---------|-----------|--------|
| 1 | `src/normalise.py` + `tests/test_normalise.py` | nothing | Utility functions, all tests green |
| 2 | `src/ingest.py` + `tests/` fixture data | normalise.py | `data/interim/releases.parquet` |
| 3 | `pipelines/01_ingest.py` | ingest.py, settings.yaml | Verify parquet row count = expected |
| 4 | `src/er_phase1.py` + `tests/test_er_phase1.py` | normalise.py | canonical_orgs + alias_map (partial) |
| 5 | `configs/gov_lookup.json` (initial ~50 entries) | data analysis | Lookup table ready |
| 6 | `src/er_phase2.py` + `src/er_candidates.py` + tests | normalise.py, gov_lookup.json | canonical_orgs + alias_map (complete) + er_audit.csv |
| 7 | `pipelines/02_er_phase1.py` + `pipelines/03_er_phase2.py` | er_phase1, er_phase2 | Full entity resolution run |
| 8 | `src/kg_nodes.py` + `tests/test_kg_nodes.py` | alias_map, releases | org_nodes, contract_nodes, cpv_nodes |
| 9 | `src/kg_edges.py` | alias_map, releases | buyer_of, supplier_of, categorized_by |
| 10 | `pipelines/04_build_kg.py` | kg_nodes, kg_edges | Full KG parquet files written |
| 11 | `pipelines/05_validate_kg.py` | kg_query.py | Sanity check: MoD = 1 node, NHS England = 1 node, value sums reasonable |
| 12 | `src/kg_query.py` + `tests/test_kg_query.py` | kg parquets | Stable query API green |
| 13 | `src/kg_adapter.py` (stub → real) | kg_query.py + old pipeline inspection | Old pipeline reconnected |

Total: ~13 implementation sessions, ~10 source files + 5 test files + 2 config files.
