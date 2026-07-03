# Project Structure Proposal

**Purpose:** Evaluate three structural options for the clean rebuild and recommend one.  
**Constraints already fixed:**  
- GB-FTS is never canonical; stored only as alias  
- Latest-release-per-OCID is the default KG snapshot  
- Fuzzy matching produces candidate reports only, no automatic merges  
- Framework contracts flagged, not excluded  
- Old pipeline compatibility handled via an adapter layer, not baked into KG schema  

---

## Engineering Lessons from Reference Projects

Before proposing options, here is what the surveyed projects teach about structure:

| Project | Key structural lesson |
|---------|----------------------|
| **strwythura** | Numbered top-level scripts (`1_er.py`, `2_sem.py`, …) make the pipeline order obvious, but everything lands in one flat `data/` directory — raw inputs sit next to generated artefacts, making it hard to know what is computed vs downloaded |
| **ERKG** | Pure notebooks (`datasets.ipynb`, `graph.ipynb`); clean pipeline order, zero reusability — no importable modules, no tests. Acceptable for a demo, unusable for a project that evolves |
| **FareedKhan/KG-Pipeline** | Single `code.ipynb` monolith — negative example; everything mixed |
| **docling-graph** | Best engineering discipline: `src/<package>/`, `tests/`, `docs/`, `pyproject.toml`. Pipeline stages are subpackages inside the main package (`pipeline/`, `core/`, `db_clients/`). Config is a YAML file, not scattered inline. CLI is separate from library logic |
| **Cookiecutter Data Science** | Industry-standard layout for data pipelines: `data/raw/`, `data/interim/`, `data/processed/` for data lineage; `src/<module>/` for importable library code; `notebooks/` for exploration only; `reports/` for outputs |
| **Biomedical-KG** | Numbered Jupyter notebooks (`01_Data_Collection/`, `02_EDA/`, …) — readable for an academic audience but hard to test and import from |

**Anti-patterns to avoid:**
1. Mixing computed artefacts with source files (strwythura's flat `data/` problem).
2. Mixing pipeline orchestration scripts with library code (makes unit testing impossible).
3. Mixing KG construction logic with query/interface logic (the old project's root bug).
4. Numbered scripts at the project root (strwythura): works for demos, breaks for anything with >1 contributor or re-run cadence.
5. Hardcoding pipeline compatibility concerns inside KG schema (couples two independent problems).

---

## Option A — Simple Research Prototype

**Inspiration:** strwythura + Biomedical-KG (sequential numbered scripts, data beside code)

### Directory Tree

```
fyp_new/
│
├── data/
│   ├── raw/                    # Downloaded .jsonl.gz — never modified
│   ├── interim/                # Decompressed + flattened releases (parquet)
│   ├── entities/               # ER outputs (canonical_orgs, alias_map, er_log)
│   └── kg/                     # Final KG artefacts (node/edge parquets)
│
├── scripts/
│   ├── 01_parse_releases.py
│   ├── 02_er_phase1.py
│   ├── 03_er_phase2.py
│   ├── 04_build_kg.py
│   └── 05_validate_kg.py
│
├── notebooks/
│   └── explore_data.ipynb
│
├── configs/
│   └── gov_entity_lookup.json   # Curated govt body → canonical_id map
│
├── docs/
│   └── *.md
│
├── tests/
│   └── test_er.py
│
└── requirements.txt
```

### Responsibility of Each Folder

| Folder | Responsibility |
|--------|---------------|
| `data/raw/` | Immutable downloaded files — never overwritten by pipeline |
| `data/interim/` | Single large parquet after flattening; regenerable from raw |
| `data/entities/` | ER outputs: one canonical entity per row, alias map, audit log |
| `data/kg/` | Final node/edge parquet files consumed by query layer |
| `scripts/` | Numbered orchestration scripts — run top to bottom to rebuild KG |
| `notebooks/` | Exploration and one-off analysis only |
| `configs/` | Static lookup tables and thresholds — not code |
| `tests/` | Unit tests for logic inside scripts |

### Pros

- Dead simple to understand: `run 01, 02, 03, 04` in order.
- No Python packaging overhead.
- Fast to write the first working version.
- Academic readers can follow the numbered sequence easily.

### Cons

- **No importable library:** Each script re-implements helpers. When `02` and `03` both need name
  normalisation, you either duplicate it or do a fragile `import` from a sibling script.
- **No clear module boundary:** The query layer (`05`) imports from `04`, which imports from `03` —
  tight coupling re-emerges even though the folders look separate.
- **Hard to unit-test:** Scripts are not importable as modules (they execute on import).
- **Does not support adapter pattern:** Adding an old-pipeline adapter means either a new numbered
  script or a hacky import chain.
- **Data folder conflates different artefact types:** `data/entities/` and `data/kg/` are distinct
  concerns but both live under `data/`.

### Fit for Old Pipeline Connection

Poor. The only integration point is "run `05_validate_kg.py` and then hand the parquet files to the
old pipeline". There is no clean programmatic API to call.

---

## Option B — Modular Production-Like Structure

**Inspiration:** docling-graph (subpackages per concern) + Cookiecutter Data Science (data layers)

### Directory Tree

```
fyp_new/
│
├── data/
│   ├── raw/                    # Immutable downloads — never touched by code
│   ├── interim/                # Pipeline-generated intermediate artefacts
│   │   └── releases.parquet
│   ├── entities/               # Entity resolution outputs
│   │   ├── canonical_orgs.parquet
│   │   ├── alias_map.parquet
│   │   ├── er_candidates.csv   # Fuzzy-match candidates for human review
│   │   └── er_audit.csv        # Provenance: how each entity was resolved
│   └── kg/                     # Final KG artefacts (nodes + edges)
│       ├── nodes/
│       │   ├── org_nodes.parquet
│       │   ├── contract_nodes.parquet
│       │   └── cpv_nodes.parquet
│       └── edges/
│           ├── buyer_of.parquet
│           ├── supplier_of.parquet
│           └── categorized_by.parquet
│
├── src/
│   └── procurement_kg/         # Installable Python package
│       ├── __init__.py
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── loader.py        # Decompress + stream JSONL
│       │   └── flattener.py     # OCDS record → flat row schema
│       ├── entity_resolution/
│       │   ├── __init__.py
│       │   ├── normalise.py     # Name normalisation utilities
│       │   ├── phase1.py        # Deterministic: official ID priority merge
│       │   ├── phase2.py        # Heuristic: exact-norm-name + region merge
│       │   └── candidates.py   # Fuzzy candidate report (no auto-merge)
│       ├── kg/
│       │   ├── __init__.py
│       │   ├── nodes.py         # Build org/contract/CPV node parquets
│       │   └── edges.py         # Build buyer_of/supplier_of/categorized_by
│       └── interface/
│           ├── __init__.py
│           ├── query.py         # Core aggregation query functions
│           └── adapter.py       # Old-pipeline compatibility shim (later)
│
├── pipelines/                   # Thin orchestration scripts (not library code)
│   ├── run_ingest.py
│   ├── run_entity_resolution.py
│   ├── run_kg_build.py
│   └── run_validate.py
│
├── configs/
│   ├── settings.yaml            # Paths, thresholds, flags
│   └── gov_lookup.json          # Curated govt canonical names
│
├── notebooks/
│   └── 01_explore_ocds.ipynb
│
├── tests/
│   ├── unit/
│   │   ├── test_normalise.py
│   │   ├── test_phase1.py
│   │   ├── test_phase2.py
│   │   └── test_query.py
│   └── integration/
│       └── test_full_pipeline.py
│
├── docs/
│   └── *.md
│
├── pyproject.toml               # Editable install: pip install -e .
└── .gitignore
```

### Responsibility of Each Folder

| Folder | Responsibility |
|--------|---------------|
| `data/raw/` | Immutable source files — gitignored, never modified |
| `data/interim/` | First stable intermediate: flattened releases (single parquet) |
| `data/entities/` | All entity resolution outputs including audit trail and fuzzy candidates |
| `data/kg/nodes/` | Final canonical node tables |
| `data/kg/edges/` | Final edge tables; together with nodes form the complete KG |
| `src/procurement_kg/ingest/` | Loading and flattening logic; knows about OCDS schema |
| `src/procurement_kg/entity_resolution/` | All disambiguation logic; knows nothing about graph |
| `src/procurement_kg/kg/` | Graph construction; reads alias map + flat releases; knows nothing about ER logic |
| `src/procurement_kg/interface/` | Query API; reads only kg/ parquets; knows nothing about ER or ingest |
| `pipelines/` | Thin scripts that call library code in order; no business logic here |
| `configs/` | External configuration: paths, thresholds, lookup tables |
| `tests/unit/` | Pure unit tests for individual functions (no file I/O) |
| `tests/integration/` | End-to-end tests running the full pipeline on a small fixture dataset |
| `notebooks/` | Exploration only — never imported by `src/` |

### Pros

- **Clean import boundaries:** `entity_resolution` never imports from `kg`; `kg` never imports
  from `interface`. Dependency graph is a strict DAG.
- **Fully testable:** Each module is importable; unit tests exercise functions in isolation.
- **Adapter pattern is natural:** `interface/adapter.py` is just another module — it wraps
  `query.py` in whatever format the old pipeline expects, without touching the KG schema.
- **Data lineage is explicit:** `raw/` → `interim/` → `entities/` → `kg/` is a one-way flow.
  Any stage can be rerun independently by deleting its output folder.
- **Config externalised:** Changing thresholds or lookup tables does not require editing code.

### Cons

- **More upfront scaffolding:** `pyproject.toml`, `__init__.py` files, package install step.
- **Steeper curve for readers** unfamiliar with Python packaging conventions.
- **Slight over-engineering** for a research project that will not be distributed as a library.
- The `pipelines/` layer adds one extra file-per-stage that does nothing but call `src/`.

### Fit for Old Pipeline Connection

Excellent. `interface/query.py` provides a stable, documented API. `interface/adapter.py` wraps
it without touching anything upstream. The old pipeline imports only from `interface/`.

---

## Option C — Hybrid (Recommended for MSc/FYP)

**Inspiration:** Cookiecutter Data Science data layers + strwythura numbered pipeline clarity +
docling-graph's separation of library code from scripts.

The core idea: use a **flat `src/` package** (no subpackages) instead of Option B's nested
subpackages, while keeping strict data-layer separation and the adapter pattern.

### Directory Tree

```
fyp_new/
│
├── data/
│   ├── raw/                     # Immutable downloads
│   ├── interim/                 # Intermediate pipeline outputs
│   │   └── releases.parquet     # Flattened OCDS (all years, deduped)
│   ├── entities/                # Entity resolution outputs
│   │   ├── canonical_orgs.parquet
│   │   ├── alias_map.parquet
│   │   ├── er_candidates.csv    # Fuzzy candidates for review (never auto-merged)
│   │   └── er_audit.csv         # Per-entity resolution decision log
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
├── src/                         # All importable library code (flat package)
│   ├── __init__.py
│   ├── ingest.py                # OCDS loading, decompression, flattening
│   ├── normalise.py             # Name/ID normalisation utilities (shared)
│   ├── er_phase1.py             # Deterministic entity resolution
│   ├── er_phase2.py             # Heuristic entity resolution
│   ├── er_candidates.py         # Fuzzy candidate report generator
│   ├── kg_nodes.py              # Node table construction
│   ├── kg_edges.py              # Edge table construction
│   ├── kg_query.py              # Aggregation query functions (the stable interface)
│   └── kg_adapter.py            # Old-pipeline compatibility shim (stub for now)
│
├── pipelines/                   # Numbered orchestration scripts — no logic inside
│   ├── 01_ingest.py
│   ├── 02_er_phase1.py
│   ├── 03_er_phase2.py
│   ├── 04_build_kg.py
│   └── 05_validate_kg.py
│
├── configs/
│   ├── settings.yaml            # Paths, thresholds
│   └── gov_lookup.json          # Curated govt body canonical name → canonical_id
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
├── docs/
│   ├── reference_study.md
│   ├── ocds_data_analysis.md
│   ├── rebuild_plan.md
│   └── project_structure_proposal.md
│
├── requirements.txt             # Simple pip requirements (no pyproject.toml needed)
└── .gitignore
```

### Responsibility of Each Folder

| Folder | Responsibility |
|--------|---------------|
| `data/raw/` | Immutable downloaded `.jsonl.gz` files — gitignored; not modified by any script |
| `data/interim/` | First transform output: all 5 years merged into a single flat parquet; regenerable |
| `data/entities/` | All entity resolution outputs. Four files: canonical table, alias map, fuzzy candidates (review only), audit log (provenance) |
| `data/kg/nodes/` | Canonical node tables. Built entirely from `entities/` + `interim/`; never modified after build |
| `data/kg/edges/` | Edge tables. Link canonical entity IDs to contract IDs; no raw IDs here |
| `src/` | All importable Python logic. Each file = one cohesive concern. Files are importable modules — no side effects on import |
| `pipelines/` | Numbered orchestration scripts. Each does: load config → call `src/` functions → write output. Zero business logic inside |
| `configs/` | External configuration and lookup tables. Changing these requires no code edit |
| `notebooks/` | Exploration only. Must not import from `pipelines/` |
| `tests/` | One test file per `src/` module. Uses small fixture data, not full 166K records |
| `docs/` | All planning and reference documents |

### Pros

- **Simpler than Option B** — flat `src/` means no package install, no `__init__.py` nesting,
  no subpackage confusion. A reader or examiner can open `src/` and immediately see all modules.
- **Cleaner than Option A** — library code is separated from orchestration scripts; every
  function is importable and testable.
- **Numbered pipelines** preserve the readable top-down order from strwythura/Biomedical-KG.
- **Strict data-layer separation** from Cookiecutter Data Science makes data lineage auditable:
  delete `data/kg/` to rebuild the graph without rerunning ER; delete `data/entities/` to rerun ER.
- **Adapter pattern works cleanly:** `kg_adapter.py` is just another importable module.
- **Right size for an FYP:** Testable without being over-engineered. Could be shipped as-is
  to an examiner without explaining Python packaging or pyproject.toml.

### Cons

- **Flat `src/` grows unwieldy** if the project eventually adds 15+ modules. For now (8–10 files)
  it is fine. This is an acceptable trade-off for FYP scope.
- **No package namespace** means name collisions are possible if the old pipeline uses the same
  module names. Mitigation: the old pipeline is in a separate folder; `sys.path` is controlled
  by the adapter.
- **`requirements.txt` instead of `pyproject.toml`** means the package is not installable
  as a library — acceptable for FYP, slightly less professional.

### Fit for Old Pipeline Connection

Good. `kg_adapter.py` is a stub now, implemented later. The old pipeline adds
`sys.path.insert(0, "path/to/fyp_new/src")` and calls `from kg_adapter import ...`.
No changes to KG schema needed.

---

## Comparison Table

| Criterion | Option A (Prototype) | Option B (Production) | Option C (Hybrid) ✅ |
|-----------|---------------------|-----------------------|---------------------|
| Time to first working pipeline | Fast | Slowest | Fast |
| Testability | Poor | Excellent | Good |
| Data lineage clarity | Medium | Excellent | Excellent |
| ER / KG separation | Weak | Strong | Strong |
| Old pipeline adapter | Awkward | Excellent | Good |
| FYP examiner readability | Good | Medium (packaging knowledge assumed) | Excellent |
| Future extension | Poor | Good | Medium |
| Overengineering risk | None | High | Low |

---

## Recommendation: Option C — Hybrid

**Justification:**

1. **The most important constraint is clean separation of ER from KG from query.** All three
   options can achieve this in principle, but Option A's import-from-sibling-script pattern
   constantly fights against it. Option C enforces it structurally: pipeline scripts call `src/`
   functions, never each other.

2. **The FYP context requires an examiner/reader to follow the code.** Option B's nested
   subpackages require understanding Python packaging conventions that not all examiners share.
   Option C's flat `src/` is immediately legible: open folder, see modules, understand pipeline.

3. **The numbered `pipelines/` scripts solve the reproducibility requirement** that examiners
   and future pipeline users care about: "how do I regenerate the KG from scratch?" → run
   `01_ingest.py` through `04_build_kg.py` in order.

4. **The data-layer separation (`raw/interim/entities/kg/`) is non-negotiable** given what
   we know about the data: entity resolution decisions are expensive to rerun, and the audit
   log in `data/entities/er_audit.csv` is essential for justifying merge decisions in the
   dissertation.

5. **`kg_adapter.py` as a stub** costs nothing now and provides a clean future seam for
   connecting the old pipeline later — without polluting the KG schema with backward-compat hacks.

---

## What the Structure Enforces (Dependency Rules)

The following import rules must be respected. They are not enforced by tooling but must be
followed during implementation:

```
pipelines/01  →  src/ingest.py                only
pipelines/02  →  src/er_phase1.py, src/normalise.py
pipelines/03  →  src/er_phase2.py, src/er_candidates.py, src/normalise.py
pipelines/04  →  src/kg_nodes.py, src/kg_edges.py
pipelines/05  →  src/kg_query.py  (validation queries)

src/er_phase1.py    →  src/normalise.py         only
src/er_phase2.py    →  src/normalise.py         only
src/er_candidates.py → src/normalise.py         only
src/kg_nodes.py     →  (no src/ imports)        reads parquet files
src/kg_edges.py     →  (no src/ imports)        reads parquet files
src/kg_query.py     →  (no src/ imports)        reads kg/ parquet files
src/kg_adapter.py   →  src/kg_query.py          only

notebooks/          →  src/*                    allowed (read-only exploration)
tests/              →  src/*                    allowed
```

**Hard rules:**
- `kg_*` modules MUST NOT import from `er_*` or `ingest` modules.
- `er_*` modules MUST NOT import from `kg_*` modules.
- `ingest.py` MUST NOT import from `er_*` or `kg_*`.
- No `pipelines/` script imports from another `pipelines/` script.
- No `src/` module modifies `data/raw/` ever.
