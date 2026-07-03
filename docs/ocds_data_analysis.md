# OCDS Data Analysis: UK Public Procurement (2022–2026)

**Data source:** https://data.open-contracting.org/en/publication/41  
**Format:** JSONL.gz (one JSON object per line, each line = one compiled OCDS release)  
**Analysis date:** 2026-06-28  
**Tooling:** PowerShell + .NET GZipStream, direct inspection of decompressed JSONL

---

## 1. Data Scale

| Year | Records | File size (compressed) | Notes |
|------|---------|----------------------|-------|
| 2022 | 24,431  | 26 MB  | Pre-Procurement Act regime |
| 2023 | 24,009  | 26 MB  | Pre-Procurement Act regime |
| 2024 | 25,950  | 31 MB  | Pre-Procurement Act regime |
| 2025 | 52,398  | 51 MB  | Procurement Act 2023 took effect Feb 2025 — volume roughly doubled |
| 2026 | 39,489  | 41 MB  | Partial year (data through mid-2026) |
| **Total** | **166,277** | **175 MB** | ~5× decompressed ≈ 875 MB |

The sharp jump in 2025 reflects the UK Procurement Act 2023 introducing new mandatory notice
types (Preliminary Market Engagement, Pipeline, Transparency). The record count is not purely
"contracts awarded" — each OCDS `compiled` record is a snapshot of one contracting process
(procurement notice → award → contract). Multiple releases for the same OCID may exist across
years; the `compiled` tag means this is the latest merged state.

---

## 2. OCDS Record Structure

Each line is a flat compiled release (not a record package). Top-level keys:

```json
{
  "id":            "ocds-h6vhtk-0419af-2024-02-28T09:24:18Z",  // release ID (ocid + timestamp)
  "ocid":          "ocds-h6vhtk-0419af",                         // stable contract identifier
  "date":          "2024-02-28T09:24:18Z",
  "tag":           "compiled",
  "initiationType":"tender",
  "language":      "en",
  "buyer":         { "id": "GB-COH-RC000667", "name": "UNIVERSITY OF SHEFFIELD" },
  "parties":       [...],   // all orgs involved with roles + full identifier + address
  "tender":        {...},   // CPV, value, lots, dates, procedure type
  "awards":        [...],   // suppliers + award value (often null value here)
  "contracts":     [...],   // signed contract value + dateSigned
  "bids":          { "statistics": [...] }  // bid count stats
}
```

### Key structural points

- **`buyer`** at record level contains only `{id, name}` — a reference, not the full org record.
- **`parties`** contains the full org record for every entity involved, with `roles` array
  (possible values: `buyer`, `supplier`, `reviewBody`, `centralPurchasingBody`, `tenderer`).
  This is where `identifier.scheme`, `identifier.id`, `address`, `contactPoint`, etc. live.
- **`awards[].suppliers`** contains `{id, name}` references (same format as `buyer`).
- **`contracts[].value.amount`** is where the signed contract value lives — **not** `awards[].value`
  (award value was consistently null in the 2024 full population).
- **`tender.value.amount`** is the estimated value at time of notice — present in ~31% of 2024 records.
- **`tender.classification`** holds the primary CPV code (scheme + id + description).
- **`contracts[].dateSigned`** provides the contract execution date.
- **`tender.contractPeriod`** provides intended start/end dates.

---

## 3. Entity Identification: How Buyers and Suppliers Are Identified

### 3.1 The Two-Level ID System

Every party has two IDs:
1. **`parties[].id`** — the `buyer.id` reference key used in `buyer` and `awards[].suppliers`.
   Format is always `<SCHEME>-<VALUE>` (e.g., `GB-FTS-18165`, `GB-COH-06884292`).
2. **`parties[].identifier`** — the OCDS official identifier object: `{scheme, id, legalName}`.
   This is only present for entities that have a properly registered OCDS identifier.
   For GB-FTS entities, `identifier.scheme` is often absent (only `legalName` is set).

In practice, `parties[].id` and `<scheme>-<id>` from `parties[].identifier` are *usually* the same
but can differ (e.g., `buyer.id = GB-FTS-91445` while `parties[].identifier` has only `legalName`).

### 3.2 Identifier Schemes Observed (2024, full 25,950 records)

**Buyers** (scheme derived from `buyer.id` prefix):

| Scheme | Count | % of records | Registry |
|--------|-------|-------------|---------|
| GB-FTS | 23,091 | 89.0% | Contracts Finder internal platform ID |
| GB-NHS | 1,767  | 6.8%  | NHS organisation codes (official) |
| GB-COH | 901    | 3.5%  | UK Companies House registration number (official) |
| GB-UKPRN | 95   | 0.4%  | UK Provider Reference Number (universities, official) |
| GB-CHC | 62     | 0.2%  | Charity Commission for England & Wales (official) |
| GB-MPR | 31     | 0.1%  | Mutuals Public Register |
| GB-SC, GB-NIC | 3 | 0%  | Scottish Charities / NI Charities |
| Name-only | 0  | 0%  | None — all records have at least a platform ID |

**Suppliers** (scheme from `awards[].suppliers[].id` prefix, 2024 full population, 71,541 refs):

| Scheme | Count | % of refs | Registry |
|--------|-------|----------|---------|
| GB-FTS | 61,327 | 85.7% | Contracts Finder internal platform ID |
| GB-COH | 8,883  | 12.4% | Companies House (official) |
| GB-NHS | 1,115  | 1.6%  | NHS codes |
| GB-CHC | 172    | 0.2%  | Charity Commission |
| GB-UKPRN | 34   | 0.05% | Universities |
| Others | 10     | 0.01% | Various |
| Name-only | 0   | 0%    | None |

### 3.3 Critical Finding: GB-FTS Is NOT a Canonical Entity Identifier

`GB-FTS` (Find a Tender Service) is Contracts Finder's *per-profile* ID assigned to an
organisation's account on the platform. Key problem: **the same real-world entity creates
multiple profiles** (different procurement teams, regional offices, different eTendering portals),
each receiving a unique GB-FTS ID.

Evidence from 2024 data (5,000 record sample):

| Entity (canonical name) | Number of distinct GB-FTS IDs |
|------------------------|------------------------------|
| Ministry of Defence | **77** |
| UK Research & Innovation | 18 |
| NHS Wales Shared Services Partnership | 13 |
| NHS England | 12 |
| London Luton Airport Operations Ltd | 11 |
| Scottish Government | 8 |
| Southampton City Council | 7 |
| Staffordshire County Council | 6 |
| National Highways | 6 |

This means 89% of buyer records and 85% of supplier records use IDs that are **not** canonical
entity identifiers. The entity resolution problem is primarily about consolidating GB-FTS IDs
into canonical entities, not about handling name-only records.

### 3.4 Official ID Coverage (Non-FTS)

For buyers: ~11% of records use an official non-FTS scheme.
For suppliers: ~14% of references use an official non-FTS scheme (dominated by GB-COH at 12.4%).

These ~14% of supplier references are anchored to genuine canonical IDs. The remaining ~86%
(GB-FTS) require disambiguation.

### 3.5 Name Variation for Same Official ID

Even entities with official IDs show name variation (2024, 5,000 record sample):

| Entity ID | Name variants observed |
|-----------|----------------------|
| GB-NHS-QHM | "NORTH EAST & NORTH CUMBRIA INTEGRATED CARE BOARD" / "...AND NORTH CUMBRIA..." / "NHS NORTH EAST AND NORTH CUMBRIA..." / "NORTH OF ENGLAND COMMISSIONING SUPPORT" |
| GB-NHS-13T | 4 variants including old (pre-ICB merger) name |
| GB-COH-12664966 | "CGTC MANUFACTURING INNOVATION CENTRE LIMITED" / "CELL & GENE THERAPY CATAPULT..." / "CELL AND GENE THERAPY CATAPULT - BRAINTREE" |
| GB-COH-04320853 | "INGEUS UK LIMITED" / "INGEUS" |

This confirms: **canonical ID should drive merging, not name**. Name is stored as a property/alias,
not as the entity key.

### 3.6 Buyer ↔ Supplier Overlap

In the 500-record sample, zero entity IDs appeared as both buyer and supplier — consistent with
the domain logic (government bodies buy; private companies supply). However, universities and
NHS Trusts sometimes appear on both sides. This should be revisited at full-population scale
after entity consolidation (a GB-COH entity might appear as buyer in one record and supplier
in another, only discoverable after merging GB-FTS aliases).

---

## 4. Contract Values and Aggregation Fields

### 4.1 Value Fields

| Field | Location | Coverage (2024) | Notes |
|-------|----------|----------------|-------|
| Signed contract value | `contracts[].value.amount` | 34,678 / ~34K contract entries (~100%) | Most reliable; post-award |
| Tender estimated value | `tender.value.amount` | ~31% of records | Pre-award estimate |
| Award value | `awards[].value.amount` | 0% (null in all checked records) | Not used in this dataset |

2024 signed contract value statistics:
- **Total:** GBP 1,418 billion (includes multi-year framework values — extreme outliers present)
- **Average:** GBP 40.9M (heavily skewed by large frameworks)
- **Max:** GBP 51 billion (likely a framework agreement, not a single contract)
- **Min:** GBP 0.01

The large max values suggest framework agreements are included alongside individual contracts.
Aggregation queries must be aware of this — SUM of contract values is not "total spend."

### 4.2 Fields Sufficient for Core Aggregation Queries

| Query type | Fields needed |
|-----------|---------------|
| COUNT contracts per buyer/supplier | `buyer.id` + canonical entity resolution |
| SUM / MAX contract value | `contracts[].value.{amount,currency}` |
| Temporal first/last contract | `contracts[].dateSigned` or `date` (release date) |
| Highest-value contract | `contracts[].value.amount` + link to `tender.title` via `awardID` |
| CPV category aggregation | `tender.classification.{id,description}` (CPV code) |
| Procurement method | `tender.procurementMethod` + `tender.procurementMethodDetails` |
| Geographic | `parties[].address.{region,postalCode}` |
| Lot-level detail | `tender.lots[].{id,status,value}` + `awards[].relatedLots` |

### 4.3 Date Fields

The `date` field (release date) covers the full year evenly (2024: 1,800–2,700 records/month).
`contracts[].dateSigned` is more precise for "when was this contract executed."
`tender.tenderPeriod.endDate` = bid deadline.
`contracts[].period.{startDate,endDate}` = contract duration (when present).

---

## 5. Entity Disambiguation Difficulty Assessment

### Summary

| Entity type | Total refs (2024) | With canonical non-FTS ID | GB-FTS only | Difficulty |
|-------------|-------------------|--------------------------|-------------|------------|
| Buyers | 25,950 records | ~3,000 (11%) | ~23,000 (89%) | **High** — large govt orgs highly fragmented |
| Suppliers | 71,541 refs | ~10,300 (14%) | ~61,300 (86%) | **High** — most private companies only have FTS ID |

### Difficulty by Entity Type

**Buyers (easier overall):**
- Central government departments: known names, can build a canonical lookup table (~50 departments).
  The problem is GB-FTS fragmentation (77 IDs for MoD), but the name is consistent.
- NHS bodies: GB-NHS codes are the canonical ID — already reliable. Name variation is the issue
  (ICB mergers caused name changes while the code was updated inconsistently).
- Local authorities: no canonical non-FTS ID in most cases; name matching is the only option
  (but names are relatively stable and unique within the UK).
- Universities: GB-UKPRN available for ~0.4% of buyer refs — highly reliable when present.

**Suppliers (harder):**
- Private companies: GB-COH is the gold standard — 12.4% coverage. The remaining 87.6% are
  GB-FTS only. A private company with GB-FTS-XXXX and no GB-COH listed cannot be deterministically
  linked to a Companies House record without fuzzy matching or external API lookup.
- NHS trusts as suppliers: GB-NHS available — small subset but reliable.
- Charities: GB-CHC available — small subset.
- International companies: sometimes use non-GB schemes (e.g., `GB-COH-HRB 304054` for a German
  company — an OCDS misuse). Cannot link to Companies House reliably.

### Name Variation Sources

1. **Abbreviation vs. full name:** "MoD" vs "Ministry of Defence"
2. **Punctuation/spacing:** "NHS England" vs "NHS England " (trailing space)
3. **Ampersand vs. "and":** "Department for Energy Security & Net Zero" vs "...and Net Zero"
4. **Organisational restructuring:** "NHS South, Central and West CSU" (absorbed into ICBs)
5. **ICB mergers (2022-2023):** NHS trusts merged into Integrated Care Boards — same GB-NHS code,
   different legal name
6. **Casing:** "UNIVERSITY OF SHEFFIELD" vs "University of Sheffield"

---

## 6. Relation to Old Pipeline

The old system's entity/KG issues (noted by user) likely stem from:
1. Treating GB-FTS IDs as canonical entity keys — this fragments large orgs into 77+ nodes
2. No disambiguation pass: Ministry of Defence appears as 77 separate buyer entities
3. Aggregation queries (SUM, COUNT per buyer) return fragmented, incorrect results

The new system must produce a single canonical entity node per real-world organization,
with all GB-FTS aliases stored as properties — before any graph is built.
