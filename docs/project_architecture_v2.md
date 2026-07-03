# Project Architecture v2

Purpose: define the next stable project layout before adding KG construction,
QA construction, evaluation, and a fuller reference-data workflow.

This supersedes the earlier flat-`src` hybrid plan in
`docs/project_structure_proposal.md` for future work. Existing files do not need
to be moved immediately; this document is the migration target.

## Design Inputs

The layout follows a few patterns from mature data and graph/RAG projects:

| Reference project | Useful pattern |
| --- | --- |
| Microsoft GraphRAG | Treat graph construction as a pipeline suite, not a single script. Keep docs, configs, tests, and package code separated. |
| deepset Haystack | Keep pipeline components modular so retrieval, QA, and evaluation can evolve independently. |
| Neo4j GraphRAG Python | Keep the importable package under `src/`, with examples/docs/tests outside the package. |
| Cookiecutter Data Science | Keep data lineage explicit: raw inputs, interim data, derived outputs, reports, and experiments do not mix. |

## Target Tree

```text
fyp_new/
  configs/
    paths.yaml
    reference.yaml
    er.yaml
    extract.yaml
    kg_schema.yaml
    qa.yaml
    eval.yaml
    lookups/
      gov_lookup.json

  data/
    raw/
    interim/
    reference/
    entities/
    extracted/
    kg/
      nodes/
      edges/
      exports/
    qa/
      corpora/
      questions/
      gold/
      generated/
    eval/
      runs/
      metrics/
    ablation/
      reference/
      er/
      kg/
      qa/

  src/
    procurement_graph/
      __init__.py
      common/
        __init__.py
        paths.py
        io.py
        logging.py
        schemas.py
        normalise.py
      reference/
        __init__.py
        fetchers.py
        store.py
        lookup.py
        ablation.py
      ingest/
        __init__.py
        ocds_loader.py
        flatten.py
        dedupe.py
      er/
        __init__.py
        phase1.py
        phase2.py
        candidates.py
        audit.py
      extract/
        __init__.py
        tenders.py
        awards.py
        lots.py
        documents.py
        evidence.py
        coverage.py
      kg/
        __init__.py
        schema.py
        nodes.py
        edges.py
        build.py
        validate.py
        export.py
        query.py
      qa/
        __init__.py
        corpus.py
        question_generation.py
        retrieval.py
        answer.py
        evaluation.py
      experiments/
        __init__.py
        reference_ablation.py
        er_ablation.py
        kg_ablation.py
        qa_ablation.py

  pipelines/
    00_reference_refresh.py
    10_ingest.py
    20_er_phase1.py
    21_er_phase2.py
    22_er_candidates.py
    30_extract.py
    40_build_kg.py
    41_validate_kg.py
    50_build_qa.py
    60_evaluate.py
    run_full_pipeline.ps1

  scripts/
    diagnostics/
    maintenance/
    one_off/

  reports/
    reference/
    er/
    extraction/
    kg/
    qa/
    ablation/

  docs/
    architecture.md
    data_contracts.md
    reference_flow.md
    er_method.md
    extraction_method.md
    kg_schema.md
    qa_method.md
    experiments.md
    handoff/

  tests/
    unit/
      common/
      reference/
      ingest/
      er/
      extract/
      kg/
      qa/
    integration/
      test_full_small_pipeline.py
    fixtures/

  notebooks/
    exploratory/
```

## Directory Responsibilities

| Directory | Responsibility |
| --- | --- |
| `configs/` | Human-editable settings and lookups. No generated files. |
| `data/raw/` | Immutable downloaded source files. Pipeline code never overwrites this directory. |
| `data/interim/` | Stable intermediate tables, especially flattened/deduped OCDS releases. |
| `data/reference/` | API/cache snapshots plus cache metadata. Reproducible runs read from here, not live APIs. |
| `data/entities/` | Entity-resolution outputs: canonical orgs, alias maps, audit logs, ER candidates. |
| `data/extracted/` | Structured extraction tables from the OCDS releases. |
| `data/kg/` | Built graph artifacts: node tables, edge tables, and optional export formats. |
| `data/qa/` | QA corpora, generated questions, gold labels, and generated answers. |
| `data/eval/` | Evaluation runs and machine-readable metrics. |
| `data/ablation/` | Experiment outputs, split by experiment family. |
| `src/procurement_graph/` | Importable library code. This is where logic lives. |
| `pipelines/` | Thin orchestration entrypoints. They call library code and write official artifacts. |
| `scripts/` | Diagnostics, maintenance, and one-off utilities that are not core pipeline stages. |
| `reports/` | Human-readable reports, figures, summaries, and reviewed candidate files. |
| `docs/` | Architecture, method notes, schemas, experiment plans, and handoff notes. |
| `tests/` | Unit and integration tests. Fixtures live under `tests/fixtures/`. |
| `notebooks/` | Exploration only. Not imported by library or pipeline code. |

## Pipeline Numbering

Use two-digit stage families so that new stages can be inserted without
renumbering the whole project.

| Family | Stage | Main output |
| --- | --- | --- |
| `00` | Reference refresh/cache | `data/reference/` |
| `10` | Ingest and dedupe | `data/interim/releases.parquet` |
| `20` | Entity resolution | `data/entities/` |
| `30` | Structured extraction | `data/extracted/` |
| `40` | KG build and validation | `data/kg/`, `reports/kg/` |
| `50` | QA corpus/question/answer build | `data/qa/` |
| `60` | Evaluation | `data/eval/`, `reports/qa/`, `reports/ablation/` |

## Import Boundaries

The dependency direction should remain one-way:

```text
common
  -> reference
  -> ingest
  -> er
  -> extract
  -> kg
  -> qa
  -> experiments/eval
```

Rules:

- `pipelines/` scripts may import from `src/procurement_graph/*`, but not from
  each other.
- `common/` must not import project domains such as `er`, `kg`, or `qa`.
- `reference/` can be used by `er` and experiments, but should not import `er`.
- `er/` should not import from `kg` or `qa`.
- `kg/` reads entity/extraction artifacts, but should not call ER implementation
  code directly.
- `qa/` may read KG outputs and extracted evidence, but should not modify KG
  schema or entity-resolution outputs.
- `experiments/` may call stable public functions from each domain, but
  experiment-only logic stays out of production pipeline modules.

## Data Contracts

Each major artifact should have a short schema contract in `docs/data_contracts.md`
or a domain-specific doc:

| Artifact | Contract owner |
| --- | --- |
| `data/interim/releases.parquet` | `ingest/` |
| `data/reference/*.json` | `reference/` |
| `data/entities/canonical_orgs.parquet` | `er/` |
| `data/entities/alias_map.parquet` | `er/` |
| `data/extracted/*.parquet` | `extract/` |
| `data/kg/nodes/*.parquet` | `kg/` |
| `data/kg/edges/*.parquet` | `kg/` |
| `data/qa/*` | `qa/` |
| `data/eval/*` | `qa/evaluation` and `experiments/` |

## Reference Flow

Reference data is a first-class workflow, not a small ER helper.

```text
pipelines/00_reference_refresh.py
  -> procurement_graph.reference.fetchers
  -> data/reference/*.json
  -> procurement_graph.reference.store
  -> procurement_graph.reference.lookup
  -> ER candidates or ablation reports
```

Important conventions:

- Live APIs are only called by the reference refresh pipeline.
- Normal pipeline runs use cached snapshots.
- Cache metadata records record count, timestamp, source URL, and checksum.
- Reference matches start as evidence/candidates; auto-merge policy must be
  explicit and test-covered before being used in ER.

## KG Flow

```text
data/interim/releases.parquet
data/entities/canonical_orgs.parquet
data/entities/alias_map.parquet
data/extracted/*.parquet
  -> procurement_graph.kg.nodes
  -> procurement_graph.kg.edges
  -> data/kg/nodes/*.parquet
  -> data/kg/edges/*.parquet
  -> procurement_graph.kg.validate
```

KG construction should be deterministic and file-based. It should not call live
APIs and should not rerun ER internally.

## QA Flow

```text
data/kg/*
data/extracted/text_evidence.parquet
data/extracted/documents.parquet
  -> procurement_graph.qa.corpus
  -> procurement_graph.qa.question_generation
  -> procurement_graph.qa.retrieval
  -> procurement_graph.qa.answer
  -> procurement_graph.qa.evaluation
```

QA artifacts should be stored separately from KG artifacts. The KG remains the
factual substrate; QA stores task-specific corpora, prompts, generated questions,
answers, and evaluation outputs.

## Experiment and Ablation Layout

Machine-readable experiment outputs go under `data/ablation/<family>/`.
Human-readable summaries go under `reports/ablation/<family>/`.

Examples:

```text
data/ablation/reference/reference_entity_candidates.parquet
reports/ablation/reference/summary.md

data/ablation/qa/retrieval_variants.parquet
reports/ablation/qa/retrieval_metrics.csv
```

This avoids mixing reference ablation, ER ablation, KG ablation, and QA ablation
in the same flat folder.

## Migration Plan

Do not move everything at once. Use compatibility wrappers until the new package
layout is stable.

### Phase 1: Add the Package Skeleton

Create `src/procurement_graph/` and move shared utilities first:

- `src/normalise.py` -> `src/procurement_graph/common/normalise.py`
- `src/reference_lookup.py` -> `src/procurement_graph/reference/lookup.py`
- `pipelines/00_fetch_reference.py` logic -> `src/procurement_graph/reference/fetchers.py`
- `scripts/run_entity_ablation.py` logic -> `src/procurement_graph/experiments/reference_ablation.py`

Keep old files as thin wrappers temporarily.

### Phase 2: Move Existing Domains

Move current importable modules into domain packages:

- `src/ingest.py` -> `src/procurement_graph/ingest/`
- `src/er_phase1.py` -> `src/procurement_graph/er/phase1.py`
- `src/er_phase2.py` -> `src/procurement_graph/er/phase2.py`
- `src/er_candidates.py` -> `src/procurement_graph/er/candidates.py`
- `src/extract.py` -> `src/procurement_graph/extract/`

Update pipelines after each move and run import checks.

### Phase 3: Add KG in the New Layout

Add KG modules only under `src/procurement_graph/kg/`.
Do not add new flat files such as `src/kg_nodes.py`.

### Phase 4: Add QA in the New Layout

Add QA modules only under `src/procurement_graph/qa/`.
Keep QA corpora and generated data under `data/qa/`, not `data/kg/`.

### Phase 5: Move Reports and Ablations

Move current reference ablation outputs from:

```text
data/ablation/
reports/ablation/
```

to:

```text
data/ablation/reference/
reports/ablation/reference/
```

Only do this once all scripts that read/write those files are updated.

## Immediate Next Steps

1. Keep current working pipeline stable.
2. Create the new package skeleton.
3. Move reference code first, because it is already becoming its own workflow.
4. Move ER/extract after reference imports are stable.
5. Build KG and QA directly in the new package structure.

## Things Not To Do

- Do not put business logic in `pipelines/`.
- Do not let QA code write into `data/kg/`.
- Do not let KG construction call live APIs.
- Do not let experiments mutate canonical production artifacts.
- Do not keep adding flat files to `src/` as new domains appear.
- Do not mix generated artifacts with configs or docs.
