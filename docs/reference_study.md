# Reference Study: Data → Entity Disambiguation → KG Construction

Surveyed 4 projects spanning procurement KG, enterprise entity resolution, and biomedical KG.
The goal was to extract patterns applicable to our case: structured OCDS data where most entities
carry official identifiers (GB-COH, GB-NHS, etc.) but the dominant scheme (GB-FTS) is
a platform-internal ID rather than a canonical entity identifier.

---

## Project 1: DerwenAI/ERKG — Entity Resolved Knowledge Graph

**Source:** https://github.com/DerwenAI/ERKG  
**Domain:** Business entity deduplication (Las Vegas business registry, ~85K records, ~2% duplicates)

### Pipeline

```
Raw structured data (CSV/JSON)
    ↓
Senzing SDK — probabilistic entity resolution
    ↓  (merges duplicate business names + addresses)
JSONL output (resolved entities + relations)
    ↓
Neo4j ingestion — graph construction
    ↓
GDS / Cypher queries + PyVis visualization
```

### Entity Disambiguation Approach

- Delegates entirely to **Senzing**: a commercial/open-source ER engine that uses feature-based
  heuristic matching (name tokens, address tokens, phone numbers) with configurable thresholds.
- No custom NLP or embedding logic in the project itself.
- Senzing outputs a resolved entity ID that replaces the raw source IDs.
- The graph nodes use Senzing's resolved entity IDs; original source IDs become aliases.

### What Is Applicable to Our Case

| Aspect | Applicability |
|--------|---------------|
| Senzing for fuzzy org name matching | Potentially useful for name-only entities, but heavyweight dependency; overkill when we have official scheme+id |
| Resolved entity ID as KG node key | ✅ Directly applicable: we need a stable canonical ID layer above raw GB-FTS IDs |
| Aliases / source IDs stored as properties | ✅ We should store all GB-FTS variants as aliases on a canonical entity node |
| Post-resolution graph construction (Neo4j) | ✅ Same pattern we need |

### What Is Not Applicable

- Probabilistic ER over unstructured business descriptions — we have structured OCDS fields.
- Senzing as a hard dependency is not needed when GB-COH/GB-NHS IDs already deterministically identify entities.

---

## Project 2: DerwenAI/strwythura — Ontology-Driven ERKG

**Source:** https://github.com/DerwenAI/strwythura  
**Domain:** Corporate directories + researcher profiles (ORCID, Scopus) mixed with unstructured documents

### Pipeline

```
Multiple structured datasets (corporate dirs, ORCID, Scopus)
    ↓  Stage 1
Senzing SDK — heuristic feature matching → merged entities + relations (JSONL)
    ↓  Stage 2
RDFlib + domain taxonomy (RDF Turtle) → semantic layer → NetworkX property graph (ERKG backbone)
    ↓  Stage 3 (parallel)
Document crawl → chunking → GLiNER NER → entity linking to domain taxonomy IRIs
LanceDB vector store for chunk embeddings
    ↓  Stage 4
Human-in-the-loop curation of entity definitions
    ↓  Stage 5
Textgraph algorithms → entity ranking → Word2Vec embeddings
    ↓  Stage 6
DSPy + LLM → GraphRAG QA
```

### Entity Disambiguation Approach

Three layers:
1. **Deterministic (Senzing features):** Name tokens + structured attributes matched across datasets.
2. **Semantic (IRI mapping):** Extracted entities are linked to domain taxonomy IRIs (SKOS-based thesaurus).
   This is the "canonical ID" layer — IRI acts as the stable entity identifier.
3. **Co-occurrence (Textgraph):** Synonyms identified through lexical graph co-occurrence patterns.

Key insight from their docs: *"The quality of entity definitions poses a limiting factor for how well
relation resolution performs downstream."* — entity resolution quality gates everything else.

### What Is Applicable to Our Case

| Aspect | Applicability |
|--------|---------------|
| Canonical IRI / stable ID as the graph node key | ✅ We need exactly this: canonical `org:<scheme>-<id>` URI above raw IDs |
| Domain taxonomy for entity types (buyer types, CPV codes) | ✅ We have CPV classification + org type (TED_CA_TYPE, COFOG) — can enrich KG nodes |
| Separation: ER backbone first, then relations | ✅ Exact same principle: build clean entity layer before building award/contract edges |
| Stage-gated pipeline with separate modules | ✅ Informs our file structure design |

### What Is Not Applicable

- GLiNER NER for unstructured text — we have structured OCDS; no free-text NER needed.
- LanceDB vector store / embedding layer — we're building a structured aggregation KG, not RAG.
- Human-in-the-loop curation at scale — impractical for 166K records; automate instead.

---

## Project 3: Open Contracting Partnership — OCDS + OpenCorporates Reconciliation

**Source:**  
- https://www.open-contracting.org/2016/05/09/linking-matching-contracting-company-data/  
- https://standard.open-contracting.org/latest/en/schema/identifiers/

**Domain:** UK/global public procurement data (directly our domain)

### Pipeline

```
OCDS JSON data (supplier names + optional company numbers)
    ↓
OCDS Validator → spreadsheet conversion (suppliers tab)
    ↓
OpenRefine + OpenCorporates Reconciliation API
    ↓  (fuzzy name match → candidate company records)
Manual review / threshold accept
    ↓
Download matched company data from OpenCorporates API (JSON)
    ↓
Linked dataset with canonical company IDs
```

### Entity Disambiguation Approach

- **Primary:** Official organization identifiers (OCDS `scheme` + `id` fields). OCDS documentation
  explicitly defines a hierarchy: primary registers (GB-COH = Companies House) > secondary
  (VAT/tax IDs) > third-party databases > local lists.
- **Secondary:** OpenCorporates Reconciliation API — fuzzy name matching to link suppliers
  without official IDs to canonical company records.
- The `scheme:id` pair is designed to be globally unique and stable:
  `GB-COH:06368740` → one specific legal entity.

### Key Data Quality Finding (Directly Relevant to Our Data)

From the OCP analysis of Contracts Finder specifically:
- **Buyers:** Almost 100% have an organizational identifier in the data.
  BUT: the dominant scheme is **GB-FTS** (internal Contracts Finder platform ID), not GB-COH.
  "Ministry of Defence" alone maps to 77 different GB-FTS IDs in 2024 data.
- **Suppliers:** ~84% have a GB-FTS ID; ~12% have GB-COH; ~2% have NHS/other official ID;
  ~0% are truly name-only (at least a platform ID exists for all).
- The GB-FTS scheme is a *per-organisation-profile* ID in the Contracts Finder platform,
  not a canonical legal entity identifier. The same government department creates multiple
  profiles (different procurement teams/portals), each getting a different GB-FTS ID.

### What Is Applicable to Our Case

| Aspect | Applicability |
|--------|---------------|
| OCDS `scheme+id` as the canonical identifier | ✅ Core strategy: use GB-COH/GB-NHS as ground truth; treat GB-FTS as alias |
| OpenCorporates reconciliation for name-only suppliers | Partially applicable — we have no truly name-only records, but useful for GB-FTS → GB-COH enrichment |
| Identifier hierarchy (primary > secondary > local) | ✅ Directly informs our disambiguation priority order |

### What Is Not Applicable

- Manual OpenRefine workflow — we need an automated pipeline for 166K records.
- Fuzzy name matching as primary strategy — we have platform IDs (GB-FTS) for everything,
  so the problem is ID consolidation, not name matching.

---

## Project 4: alisonmitchell/Biomedical-Knowledge-Graph

**Source:** https://github.com/alisonmitchell/Biomedical-Knowledge-Graph  
**Domain:** Biomedical literature (PubMed / arXiv papers)

### Pipeline

```
Raw text (Europe PMC / arXiv) → GROBID PDF parsing
    ↓
spaCy normalization + fastcoref coreference resolution
    ↓
scispaCy NER + EntityLinker (queries UMLS knowledge base)
  OR KAZU end-to-end NER+linking
    ↓
REBEL seq2seq relation extraction (200+ relation types)
    ↓
NetworkX + PyVis graph construction and visualization
```

### Entity Disambiguation Approach

- **Entity Linking to a knowledge base (UMLS):** After NER extracts entity mentions,
  EntityLinker maps them to canonical UMLS concept IDs (CUIs). This is the disambiguation step —
  two mentions that map to the same CUI become one node.
- **Coreference resolution** first, then linking — handles pronoun references before entity lookup.
- KAZU performs NER + linking in one step, using a pre-built entity dictionary.

### What Is Applicable to Our Case

| Aspect | Applicability |
|--------|---------------|
| Canonical KB identifier (UMLS CUI) as node key | ✅ Same principle: our official scheme+id is our "CUI" |
| Entity linking → one canonical node per entity | ✅ Many GB-FTS aliases → one canonical node |
| Staged pipeline with numbered folders | ✅ Clean separation of NER, linking, graph construction |

### What Is Not Applicable

- NLP/NER pipeline — we have structured data, not free text.
- UMLS knowledge base — irrelevant domain.
- REBEL relation extraction — our relations are explicit in OCDS awards/contracts.
- Coreference resolution — no pronouns in structured procurement records.

---

## Synthesis: What to Borrow for Our System

### Core Strategy (from Projects 1 + 3 + 4)

**Use official scheme+id as the canonical entity identifier.** This is analogous to UMLS CUI
(biomedical) or Senzing resolved entity ID (enterprise ER). In our case:
- `GB-COH-<number>` → canonical for private companies
- `GB-NHS-<code>` → canonical for NHS trusts/ICBs
- `GB-UKPRN-<number>` → canonical for universities
- `GB-CHC-<number>` → canonical for charities
- `GB-FTS-<number>` → treat as **alias only**, requires consolidation

### Two-Phase Disambiguation (from Projects 1 + 3)

**Phase 1 — Deterministic (high confidence):**
- If any appearance of an entity has a non-FTS official ID (GB-COH, GB-NHS, etc.),
  that becomes the canonical ID.
- All GB-FTS IDs for the same entity become aliases.
- Cross-year: same GB-COH number = same entity, regardless of name variation.

**Phase 2 — Heuristic (for GB-FTS-only entities):**
- Normalized name matching (strip Ltd/Limited/PLC suffixes, uppercase, trim whitespace).
- Secondary signals: address, email domain, `details.url`.
- For buyers: UK government departments have known canonical names — build a lookup table.

### Project Structure Pattern (from Projects 2 + 4)

Both strwythura and Biomedical-KG use **numbered stage folders** with clear separation:
```
01_ingest/
02_entity_resolution/
03_kg_construction/
04_qa_interface/   ← existing pipeline plugs in here
```
This separation is the key lesson: **never mix entity resolution logic with graph query logic.**

### What NOT to Do (lessons from all projects)

- Don't use probabilistic ER (Senzing, embedding similarity) as the primary strategy when
  deterministic official IDs are available — it introduces false merges.
- Don't try to reconcile against OpenCorporates at scale without caching — API rate limits
  make it impractical for 166K records without significant infrastructure.
- Don't build the graph before the entity layer is clean — "garbage in, garbage out" is
  amplified in a graph because incorrect merges propagate through all relationships.
