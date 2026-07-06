# Chapter 3 — Data Engineering and Knowledge-Graph Construction

## Chapter thesis (what this chapter must convince the reader of)

Hard verification is only possible over a substrate whose semantics are explicit: this chapter
shows how five years of OCDS releases become a typed, convention-explicit, audit-first
knowledge graph of 215,221 contract nodes and 131,502 canonical organisations. Two design
stances carry the argument: precision-first entity resolution (false merges corrupt every
downstream count and sum, so merging is tiered, conservative, and audited), and conventions-
as-data (additivity of money values, the flat first-party record universe, latest-release
snapshots are checkable flags, not folklore). The reader must accept that the KG's residual
ambiguity is documented and bounded — because chapter 4's oracle audit and chapter 5's
verifier both lean on exactly these properties.

## Section outline

### 3.1 Source data and ingestion
OCDS releases 2022–2026 from `.jsonl.gz` dumps; deduplication to the latest release per OCID
(166,277 releases); award-level extraction. Why latest-release: amendment chains multiply
schema complexity for questions the benchmark never asks; first-release discards corrections.

### 3.2 Graph schema and coverage
Typed Parquet node/edge tables: 215,221 contract nodes, 131,502 organisation nodes, 3,870 CPV
nodes; buyer/supplier/classification/evidence edges. Coverage as measured facts: buyer edges
215,218/215,221; supplier awards 204,186/215,221; evidence pointers 215,202/215,221; 176,002
records carry additive contract values; 193,544 carry signed award dates.

### 3.3 Entity resolution: conservative, tiered, audited
204,711 raw aliases → 131,502 canonical organisations via ordered tiers: registry-ID
deterministic merges (26,704), safe deterministic variants (6,332), normalised name+region
(14,875), name-only (1,286), LLM-adjudicated borderline cases (255, human-reviewable),
government lookup (37); 82,013 singletons left unmerged; fuzzy matching demoted to producing
human-review candidate files only. Notice-scoped identifiers (GB-FTS) are never canonical.

### 3.4 Conventions as checkable data
`value_is_additive` makes "additive-only money aggregation" a guard the executor and verifier
can enforce; the flat first-party record universe and the latest-release snapshot are stated
conventions that the chapter-4 oracle audit later stress-tests. Residual ER ambiguity is
handled at query time: variant equivalence classes ("The X" / "X (BEIS)" → IN-filter) and
evidence-driven abbreviation expansion (ICB → Integrated Care Board), both added only after
the oracle audit showed these were the ambiguity classes with practical impact.

### 3.5 Columnar tables, not a graph database
Vectorised joins outperform query-engine round-trips at this scale, keep the executor
deterministic and unit-testable, and make the independent-oracle audit possible with plain
dataframes. Operational surface argument, not expressiveness.

### 3.6 What the KG hands to the rest of the thesis
One-page contract: typed slots and closed enums (grounding targets), explicit conventions
(verifier guards), coverage bounds (abstain-no-results semantics). Forward pointers to ch. 4–5.

## Evidence manifest

All KG numbers below are currently [DOC-SOURCED] — none are in `results_master_table.md`,
which is eval-focused. Per writing rule 1, promote each cited number into the master table
(with artifact path) before the writing pass, or cite the artifact-bearing doc section.

| Number | Where used | Source / artifact |
|---|---|---|
| 215,221 contract nodes | §3.2 | [DOC-SOURCED: kg_enrichment_plan.md] data/kg/nodes/contract_nodes.parquet |
| 131,502 organisation nodes | §3.2 | [DOC-SOURCED: kg_enrichment_plan.md] data/kg/nodes/org_nodes.parquet |
| 3,870 CPV nodes | §3.2 | [DOC-SOURCED: PROJECT_STRUCTURE.md] data/kg/nodes/cpv_nodes.parquet |
| 166,277 deduplicated releases | §3.1 | [DOC-SOURCED: thesis_draft.md §3.1] |
| buyer 215,218 / supplier 204,186 / evidence 215,202 (of 215,221) | §3.2 | [DOC-SOURCED: kg_enrichment_plan.md] data/kg/edges/*.parquet |
| 176,002 additive values; 193,544 signed dates | §3.2/§3.4 | [DOC-SOURCED: thesis_draft.md §3.1] |
| 204,711 aliases → 131,502 canonical; tier counts 26,704/6,332/14,875/1,286/255/37; 82,013 singletons | §3.3 | [DOC-SOURCED: thesis_draft.md §3.3, PROJECT_STRUCTURE.md] data/entities/ |
| Figure slot: KG build + ER tier funnel diagram | §3.3 | [PENDING: render; no F-number assigned yet] |

Note: the task brief's shorthand "137k orgs" does not match the artifacts; the frozen doc
number is 131,502 canonical organisations. Use 131,502 everywhere.

## Claims discipline (this chapter must NOT)

- MUST NOT claim ER recall or completeness — the design is precision-first; the documented
  cost (residual case-variant splits of one organisation) must be stated alongside the tiers.
- MUST NOT call the KG "ground truth" — correctness downstream is convention-relative
  (flat first-party universe, additive-only money, latest-release snapshot); those conventions
  are choices this chapter documents, and ch. 8 lists convention-relativity as a limitation.
- MUST NOT claim complete coverage of UK procurement — the KG covers the ingested OCDS dumps;
  coverage numbers are within-corpus, not within-reality.
- MUST NOT smuggle eval-flavoured claims ("the KG enables 85.65%") into this chapter — system
  numbers live in ch. 6–7; this chapter's numbers are counts and coverage only.
- MUST NOT describe deterministic checks enabled by the KG as "detecting all errors" — the
  projection framing (ch. 5) owns that wording; this chapter only establishes that projectable
  properties exist because the substrate is typed.
