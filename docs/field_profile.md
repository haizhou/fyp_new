# OCDS Field Profile and Layer Assignment

Profiled from: `data/raw/2025.jsonl.gz` (300 records) and `data/raw/2024_sample.jsonl` (5 records).
Full data: 166,277 unique OCIDs across 2022–2026.

Coverage tiers: FREQUENT >80%, MODERATE 20–80%, SPARSE <20%.

---

## Summary Decision Table

| Layer | Purpose | Output file |
|---|---|---|
| **Core structured** | KG nodes/edges, ER input, filterable attributes | `data/interim/releases.parquet` (already), `data/extracted/tender_core.parquet`, `data/extracted/lots.parquet`, `data/extracted/awards.parquet`, `data/extracted/contracts.parquet`, `data/extracted/award_criteria.parquet`, `data/extracted/bid_stats.parquet` |
| **Text evidence** | QA evidence layer, not KG attributes | `data/extracted/text_evidence.parquet` |
| **Document/URL metadata** | External links for later retrieval | `data/extracted/documents.parquet` |
| **Raw provenance** | Full fidelity, recovery of any field | `data/interim/releases.parquet` → `parties_json`, `contracts_json` (existing); raw files preserved |

---

## Field-by-Field Assignments

### TOP-LEVEL

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `ocid` | 100% | string | CORE KEY | Stable contract identifier |
| `id` (release_id) | 100% | string | CORE | Already in interim |
| `date` | 100% | datetime | CORE | Already in interim |
| `tag[]` | 100% | array | CORE | e.g. `["compiled"]`; already in interim |
| `initiationType` | 100% | string | IGNORE | Always `"tender"` in this dataset |
| `language` | 100% | string | IGNORE | Always `"en"` |

---

### BUYER

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `buyer.id` | 100% | string | CORE (ER) | raw_id for buyer entity; already in interim |
| `buyer.name` | ~100% | string | CORE (ER) | Already in interim |

---

### PARTIES

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `parties[].id` | 100% | string | CORE (ER) | raw_id; already in parties_json |
| `parties[].name` | 100% | string | CORE (ER) | Already in parties_json |
| `parties[].roles[]` | 100% | array | CORE (ER) | Already in parties_json |
| `parties[].identifier.scheme` | ~100% | string | CORE (ER) | GB-COH/GB-FTS/GB-NHS/GB-PPON etc; in parties_json |
| `parties[].identifier.id` | ~100% | string | CORE (ER) | Already in parties_json |
| `parties[].identifier.legalName` | ~80% | string | CORE (ER) | Best canonical name source; in parties_json |
| `parties[].address.region` | ~100% | string | CORE (ER) | NUTS/ITL region code; in parties_json |
| `parties[].address.postalCode` | ~99% | string | CORE | Already captured |
| `parties[].address.streetAddress` | ~98% | string | TEXT EVIDENCE | Too noisy for KG attribute; useful for dedup |
| `parties[].address.locality` | ~100% | string | CORE | Town/city — useful for ER confirmation |
| `parties[].address.countryName` | 100% | string | CORE | Nearly always "United Kingdom" |
| `parties[].details.url` | ~99% | string | DOCUMENT/URL | Platform profile URL |
| `parties[].details.buyerProfile` | ~20% | string | DOCUMENT/URL | External buyer profile page |
| `parties[].details.classifications[].id` | ~96% | string | CORE | Org category code e.g. `publicAuthoritySubCentralGovernment` |
| `parties[].details.classifications[].scheme` | ~99% | string | CORE | `UK_CA_TYPE` or `TED_CA_TYPE` — used for org_category |
| `parties[].details.classifications[].description` | ~99% | string | IGNORE | Human label for .id; redundant with code |
| `parties[].details.scale` | ~20% | string | CORE | `"sme"` flag — useful KG attribute |
| `parties[].contactPoint.email` | 100% | string | DOCUMENT/URL | Contact detail; not a KG node attribute |
| `parties[].contactPoint.url` | SPARSE | string | DOCUMENT/URL | |
| `parties[].contactPoint.name` | SPARSE | string | IGNORE | Too sparse and too personal |
| `parties[].contactPoint.telephone` | SPARSE | string | IGNORE | Too sparse |
| `parties[].additionalIdentifiers[]` | ~8% | array | CORE (ER) | Additional official IDs; extract scheme+id pairs |
| `parties[].identifier.noIdentifierRationale` | ~7% | string | IGNORE | `"notOnAnyRegister"` — informational |

**New extractions needed in parties_json:** `address.locality`, `address.streetAddress`, `details.scale`, `details.buyerProfile`, `additionalIdentifiers[]`, `contactPoint.email`.

---

### TENDER (core structured)

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `tender.id` | 100% | string | CORE | Local tender reference; already in interim |
| `tender.title` | 100% | string | CORE | Short title already in interim; also TEXT EVIDENCE (full) |
| `tender.status` | ~99% | string | CORE | `complete`/`active`/`cancelled` |
| `tender.value.amount` | ~36% | number | CORE | 36% fill — missing = framework/no-value tenders |
| `tender.value.currency` | 100% | string | CORE | |
| `tender.procurementMethod` | 100% | string | CORE | `open`/`selective`/`limited`/`direct` |
| `tender.procurementMethodDetails` | ~100% | string | CORE | e.g. `"Open procedure"` — human label |
| `tender.mainProcurementCategory` | 100% | string | CORE | `goods`/`services`/`works` |
| `tender.classification.id` | 100% | string | CORE | Primary CPV code |
| `tender.classification.scheme` | 100% | string | CORE | Always `"CPV"` |
| `tender.classification.description` | 100% | string | CORE | CPV description |
| `tender.legalBasis.id` | 100% | string | CORE | e.g. `"32014L0024"` or `"2023/54"` (PCR reference) |
| `tender.legalBasis.scheme` | 100% | string | CORE | `"CELEX"` or `"UKPGA"` |
| `tender.coveredBy[]` | ~80% | array | CORE | e.g. `["GPA"]` — WTO GPA coverage flag |
| `tender.hasRecurrence` | ~100% | bool | CORE | Recurring contract flag |
| `tender.tenderPeriod.endDate` | ~100% | string | CORE | Submission deadline; already in interim |
| `tender.tenderPeriod.startDate` | SPARSE | string | CORE | When present |
| `tender.awardPeriod.startDate` | ~60% | string | CORE | Expected award date |
| `tender.bidOpening.date` | ~60% | string | CORE | |
| `tender.submissionMethod[]` | ~100% | array | CORE | e.g. `["electronicSubmission"]` |
| `tender.submissionTerms.variantPolicy` | ~100% | string | CORE | `"allowed"`/`"notAllowed"` |
| `tender.submissionTerms.languages[]` | ~100% | array | CORE | e.g. `["en"]` |
| `tender.secondStage.minimumCandidates` | ~40% | number | CORE | Restricted procedure shortlist size |
| `tender.secondStage.maximumCandidates` | ~40% | number | CORE | |
| `tender.techniques.hasFrameworkAgreement` | ~20% | bool | CORE | Framework agreement flag |
| `tender.techniques.frameworkAgreement.maximumParticipants` | SPARSE | number | CORE | |
| `tender.techniques.hasDynamicPurchasingSystem` | SPARSE | bool | CORE | DPS flag |
| `tender.lotDetails.maximumLotsBidPerSupplier` | SPARSE | number | CORE | |
| `tender.enquiryPeriod.endDate` | ~8% | string | CORE | Clarification deadline |
| `tender.communication.futureNoticeDate` | ~40% | string | CORE | Planned notice date |

**Text evidence (long free-text, not KG attributes):**

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `tender.description` | ~100% | string LONG | TEXT EVIDENCE | Often 200–1000+ chars; core QA evidence |
| `tender.reviewDetails` | SPARSE | string LONG | TEXT EVIDENCE | Review body / challenge procedure |
| `tender.contractTerms.performanceTerms` | SPARSE | string LONG | TEXT EVIDENCE | Performance requirements |
| `tender.selectionCriteria.criteria[].description` | ~7% | string LONG | TEXT EVIDENCE | Selection criteria text |

**Document/URL metadata:**

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `tender.submissionMethodDetails` | ~60% | string URL | DOCUMENT/URL | Submission portal URL |
| `tender.documents[]` | ~80% | array | DOCUMENT/URL | See documents section below |

**Ignore:**

| Field | Reason |
|---|---|
| `tender.submissionTerms.bidValidityPeriod.*` | SPARSE, niche |
| `tender.submissionTerms.electronicCataloguePolicy` | SPARSE |
| `tender.contractTerms.hasElectronicPayment` | SPARSE boolean, low signal |
| `tender.contractTerms.hasElectronicOrdering` | SPARSE boolean |
| `tender.participationFees[]` | Very sparse, niche |
| `tender.amendments[]` | SPARSE; change-tracking not needed now |
| `planning.*` | ~8% fill; milestones are long-text anyway |

---

### TENDER LOTS

| Field | Coverage (within records with lots) | Type | Layer | Notes |
|---|---|---|---|---|
| `tender.lots[].id` | ~99% | string | CORE | Lot identifier |
| `tender.lots[].status` | ~98% | string | CORE | `active`/`cancelled`/`complete` |
| `tender.lots[].title` | ~60% | string | CORE | Short lot title |
| `tender.lots[].hasOptions` | ~100% | bool | CORE | Option to extend |
| `tender.lots[].hasRenewal` | ~100% | bool | CORE | Renewal flag |
| `tender.lots[].value.amount` | ~40% | number | CORE | Lot-level value |
| `tender.lots[].value.currency` | ~40% | string | CORE | |
| `tender.lots[].contractPeriod.startDate` | ~80% | string | CORE | |
| `tender.lots[].contractPeriod.endDate` | ~100% | string | CORE | |
| `tender.lots[].contractPeriod.durationInDays` | ~40% | number | CORE | |
| `tender.lots[].submissionTerms.variantPolicy` | ~100% | string | CORE | |
| `tender.lots[].suitability.vcse` | ~7% | bool | CORE | Voluntary/charity sector flag |
| `tender.lots[].awardCriteria.criteria[].type` | ~80% | string | CORE | `"price"`/`"quality"` — goes to award_criteria table |
| `tender.lots[].awardCriteria.criteria[].name` | ~60% | string | CORE | |
| `tender.lots[].awardCriteria.criteria[].weight` | ~60% | number | CORE | Weighting % |
| `tender.lots[].description` | ~100% | string LONG | TEXT EVIDENCE | Per-lot scope description |
| `tender.lots[].renewal.description` | ~60% | string LONG | TEXT EVIDENCE | Renewal terms |
| `tender.lots[].options.description` | SPARSE | string LONG | TEXT EVIDENCE | Options terms |
| `tender.lots[].awardCriteria.criteria[].description` | ~80% | string LONG | TEXT EVIDENCE | Criteria text (often just a number like "75" but sometimes long) |

---

### TENDER ITEMS

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `tender.items[].id` | ~100% | string | CORE | Item id |
| `tender.items[].relatedLot` | ~100% | string | CORE | Links item to lot |
| `tender.items[].deliveryAddresses[].region` | ~100% | string | CORE | Delivery region (NUTS code) |
| `tender.items[].additionalClassifications[].id` | ~60% | string | CORE | Additional CPV codes |
| `tender.items[].additionalClassifications[].scheme` | ~60% | string | CORE | Always `"CPV"` |
| `tender.items[].additionalClassifications[].description` | ~60% | string | IGNORE | Redundant with CPV id |

---

### TENDER DOCUMENTS

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `tender.documents[].id` | ~80% | string | DOCUMENT/URL | Document reference |
| `tender.documents[].documentType` | ~80% | string | DOCUMENT/URL | e.g. `"economicSelectionCriteria"`, `"tenderNotice"` |
| `tender.documents[].url` | ~60% | string URL | DOCUMENT/URL | External document link |
| `tender.documents[].title` | MODERATE | string | DOCUMENT/URL | |
| `tender.documents[].description` | SPARSE | string | DOCUMENT/URL | Brief description |
| `tender.documents[].format` | ~8% | string | DOCUMENT/URL | MIME type e.g. `"text/html"` |
| `tender.documents[].datePublished` | SPARSE | string | DOCUMENT/URL | |

---

### AWARDS

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `awards[].id` | ~60% | string | CORE | Award identifier |
| `awards[].status` | ~60% | string | CORE | `"active"`/`"unsuccessful"` |
| `awards[].suppliers[].id` | ~60% | string | CORE (ER) | Supplier raw_id; already via contracts_json |
| `awards[].suppliers[].name` | ~60% | string | CORE (ER) | |
| `awards[].relatedLots[]` | ~60% | string | CORE | Which lot(s) this award covers |
| `awards[].value.amount` | ~7% | number | CORE | Award-level value (use contracts[].value when absent) |
| `awards[].value.amountGross` | ~7% | number | CORE | Gross including VAT |
| `awards[].value.currency` | ~7% | string | CORE | |
| `awards[].contractPeriod.startDate` | ~7% | string | CORE | |
| `awards[].contractPeriod.endDate` | ~7% | string | CORE | |
| `awards[].aboveThreshold` | ~7% | bool | CORE | Above-threshold notice flag |
| `awards[].title` | ~40% | string | CORE | Short award title |
| `awards[].documents[].url` | ~7% | string URL | DOCUMENT/URL | Award notice URL (Find a Tender link) |
| `awards[].documents[].documentType` | ~7% | string | DOCUMENT/URL | e.g. `"awardNotice"` |
| `awards[].documents[].noticeType` | ~7% | string | DOCUMENT/URL | e.g. `"UK6"` |
| `awards[].milestones[]` | ~7% | array | IGNORE | Future signature dates; very sparse |

---

### CONTRACTS

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `contracts[].id` | ~80% | string | CORE | Contract identifier |
| `contracts[].status` | ~80% | string | CORE | `"active"`/`"terminated"` |
| `contracts[].awardID` | ~80% | string | CORE | Links to award |
| `contracts[].value.amount` | ~80% | number | CORE | Contract value; already in contracts_json |
| `contracts[].value.currency` | ~80% | string | CORE | |
| `contracts[].dateSigned` | ~80% | string | CORE | Already in contracts_json |
| `contracts[].title` | ~20% | string | CORE | |
| `contracts[].period.startDate` | ~60% | string | CORE | |
| `contracts[].period.endDate` | ~60% | string | CORE | |

---

### BIDS STATISTICS

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `bids.statistics[].measure` | ~40% | string | CORE | `"bids"`, `"smeBids"`, `"electronicBids"`, `"lowestValidBidValue"` |
| `bids.statistics[].value` | ~40% | number | CORE | Count or value |
| `bids.statistics[].relatedLot` | ~40% | string | CORE | Per-lot stat |
| `bids.statistics[].id` | ~40% | string | IGNORE | Just an index number |

---

### PLANNING (sparse — deferred)

| Field | Coverage | Type | Layer | Notes |
|---|---|---|---|---|
| `planning.milestones[].type` | ~8% | string | IGNORE NOW | e.g. `"engagement"` — too sparse to warrant a table |
| `planning.milestones[].description` | ~8% | string LONG | IGNORE NOW | Engagement milestone descriptions |
| `planning.milestones[].dueDate` | ~7% | string | IGNORE NOW | |
| `planning.budget.*` | very sparse | various | IGNORE NOW | Budget planning; present in <5% |

---

## New Extraction Tables (what `src/extract.py` produces)

All tables share `ocid` as the join key to `releases.parquet`.

### 1. `tender_core.parquet` — one row per OCID
Fields not already in `releases.parquet`:
`tender_status`, `tender_legal_basis_id`, `tender_legal_basis_scheme`,
`tender_covered_by`, `tender_submission_method`, `tender_variant_policy`,
`tender_has_recurrence`, `tender_has_framework`, `tender_framework_max_participants`,
`tender_has_dps`, `tender_second_stage_min`, `tender_second_stage_max`,
`tender_award_period_start`, `tender_bid_opening_date`,
`tender_enquiry_period_end`, `tender_future_notice_date`,
`tender_submission_url`.

### 2. `lots.parquet` — one row per (ocid, lot_id)
`ocid`, `lot_id`, `lot_status`, `lot_title`, `has_options`, `has_renewal`,
`lot_value_amount`, `lot_value_currency`,
`contract_start`, `contract_end`, `contract_duration_days`,
`variant_policy`, `is_vcse`,
`delivery_regions` (JSON list from items with this relatedLot),
`additional_cpv_ids` (JSON list from items with this relatedLot).

### 3. `award_criteria.parquet` — one row per (ocid, lot_id, criterion_index)
`ocid`, `lot_id`, `criterion_index`, `criterion_type`, `criterion_name`, `criterion_weight`.

### 4. `awards.parquet` — one row per (ocid, award_id)
`ocid`, `award_id`, `award_status`, `award_title`,
`award_value_amount`, `award_value_currency`, `award_value_gross`,
`award_period_start`, `award_period_end`,
`related_lots` (JSON list), `supplier_raw_ids` (JSON list),
`above_threshold`.

### 5. `bid_stats.parquet` — one row per (ocid, measure, related_lot)
`ocid`, `measure`, `stat_value`, `related_lot`.

### 6. `text_evidence.parquet` — one row per (ocid, field_path)
`ocid`, `field_path`, `lot_id` (nullable), `text`.
Covers: `tender.description`, `tender.reviewDetails`, `tender.contractTerms.performanceTerms`,
`tender.selectionCriteria.criteria[].description`,
`tender.lots[].description`, `tender.lots[].renewal.description`, `tender.lots[].options.description`.

### 7. `documents.parquet` — one row per (ocid, source, doc_id)
`ocid`, `source` (e.g. `"tender"`, `"award"`), `doc_id`, `document_type`,
`url`, `title`, `description`, `format`, `date_published`, `notice_type`.
Also captures: `tender.submissionMethodDetails`, `parties[].details.url`,
`parties[].details.buyerProfile`, `parties[].contactPoint.email`.

---

## Fields Excluded and Why

| Field | Reason |
|---|---|
| `initiationType` | Always `"tender"` |
| `language` | Always `"en"` |
| `parties[].contactPoint.name` | Personal name, too sparse, PII-adjacent |
| `parties[].contactPoint.telephone` | Too sparse, PII |
| `parties[].identifier.noIdentifierRationale` | Informational, not queryable |
| `parties[].details.classifications[].description` | Redundant with `.id` code |
| `tender.submissionTerms.bidValidityPeriod.*` | Sparse, niche procurement detail |
| `tender.contractTerms.hasElectronic*` | Sparse booleans, minimal signal |
| `tender.participationFees[]` | Very sparse |
| `tender.amendments[]` | Sparse; change-tracking not in scope |
| `planning.*` | <8% fill; deferred until needed |
| `awards[].milestones[]` | Very sparse, future-state only |
| `tender.items[].additionalClassifications[].description` | Redundant with CPV id |
