# Project Structure Map (2026-07-05)

One page to navigate the repository. Status legend: **ACTIVE** (current path), *legacy* (working
historical stage, do not delete), ~~archived~~ (moved under `archive/` or `data/qa/_archive/`).

## Pipeline stages → code → data → docs

| Stage | Code | Data artefacts | Design doc |
|---|---|---|---|
| 0. Reference data | `pipelines/00_fetch_reference.py`, `src/reference_lookup.py` *(legacy flat)* | `data/reference/` | `docs/reference_study.md` |
| 1. Ingest OCDS | `pipelines/01_ingest.py`, `src/ingest.py` *(legacy flat)* | `data/raw/*.jsonl.gz` → `data/interim/releases.parquet` (166,277 releases, 2022-2026) | `docs/ocds_data_analysis.md` |
| 2. Entity resolution | `pipelines/02/03_er_*.py`, `src/er_*.py` *(legacy flat)*, `src/procurement_graph/er/` | `data/entities/` (204,711 aliases → 131,502 canonical orgs; tiers: singleton 82k / deterministic 27k / name+region 15k / safe 6.3k / name-only 1.3k / LLM-adjudicated 255 / gov-lookup 37) | `docs/rebuild_plan.md` |
| 3. Extraction | `pipelines/04_extract.py`, `src/extract.py` *(legacy flat)* | `data/extracted/` | `docs/field_profile.md` |
| 4. KG build | `pipelines/40/41_*.py`, `src/procurement_graph/kg/` | `data/kg/nodes/` (215,221 contracts / 131,502 orgs / 3,870 CPV) + `edges/` | `docs/kg_enrichment_plan.md` |
| 5. QA generation (L1) | `scripts/build_targeted_v2.py`, `src/procurement_graph/qa/targeted_v2/` | `data/qa/generated/` → `data/qa/generated_clean_l1/` (9,044 → 6,772 after cleaning) | `docs/targeted_v2_benchmark_design.md` |
| 6. QA naturalisation (L2) | `scripts/build_multilevel_qa.py`, `src/procurement_graph/qa/multilevel.py` | `data/qa/multilevel_v2_full2k/` | `docs/multilevel_qa_design.md` |
| 7. QA quality line | **ACTIVE** `scripts/qa_build_v2.py` → `qa_curate_v3.py` → `qa_rebalance_v31.py` → `qa_fill_scarce.py` → `qa_surface_diversify.py`; audit: `qa_independent_eval.py` | v1 `cicada_merged_l1_l2_trainbalanced_v1` → v2 (`_v2`, gold backfill + 99.88% dual verification) → v3 (`cicada_core_v3` + surplus) → **v4 `cicada_core_v4` (12,828 rows, the current benchmark)**; audits in `data/qa/audit_dual_eval_*` | worklog 2026-07-04/05 entries |
| 8. Reasoning runtime | **ACTIVE** `src/procurement_graph/reasoning/` (typed_planning, graph_planning, pipeline, executor, grounding, schema_grounding, entity_resolution, reflector, verifier…) | probe outputs `data/qa/plan_probe/`, `data/qa/understanding_probe/` | `docs/reasoning_pipeline_design.md`, `docs/reasoning_architecture_review.md` |
| 9. Probes / eval harness | **ACTIVE** `scripts/probe_plan_step2.py`, `probe_understanding_step1.py`, `eval_targeted_v2.py` | `data/qa/plan_probe/step2_*` (kept: all runs cited in worklog) | worklog |
| 10. Teacher data engine | **ACTIVE** `scripts/run_teacher.py` | `data/qa/teacher_strat50/`, `teacher_b500/` | `docs/trace_first_teacher_pipeline.md`, `docs/cicada_planner_training_plan.md` |

## Conventions that everything depends on

- **Flat record universe**: one row per contract node, FIRST buyer/supplier name per contract
  (`kg_interface.ParquetKGQueryBackend`). Edge-level any-party matching is a different universe;
  bridge-family oracles are defined on the flat one (dual-eval 2026-07-04).
- **Money aggregation is additive-only** (`value_is_additive=true`), explicit in every sum gold plan.
- **Split integrity**: no plan_id may appear in both train and any eval split; enforced by hard
  gates in every QA build script.
- **Model-specific Step-2 config**: grok → lean prompt + optional schema; nano → capability card +
  all-required schema (`resolve_planner_variants`).

## Archived / removed (2026-07-05 cleanup)

- deleted: `NUL`, `.tmp/` (700 scratch files), `__pycache__`, `.pytest_cache`
- `archive/legacy_tmp/`: old `tmp/` smoke outputs
- `data/qa/_archive/superseded_runs/`: multilevel pilots & smokes (11 dirs), teacher smokes,
  the failed grok v7 run
- `src/*.py` flat modules (`ingest/extract/er_*/normalise`) are **not** archived: still imported
  by `pipelines/00-04` (the KG build lineage).


## System lineage: three planner generations (audit 2026-07-05)

Knowing which results came from which generation is essential when reading old run artefacts.

| Generation | Modules | Consumers / artefacts | Status |
|---|---|---|---|
| Gen-1 rule dry-run | `reasoning/planner.py` (RuleBasedDryRunPlanner) | earliest smokes | superseded (protocol class still defines the planner interface) |
| Gen-2 rule-decomposition + first LLM planner | `planner_decomposition.py`, `llm_planner.py`, `decomposition.py` (exec path), `trace_reflector.py`, `memory.py`, `verbalize.py` | `run_compare.py --system ours` (**the 91.8% RAG comparison**), `run_hard20_nano.py`, hard100 evals, `eval_multilevel.py` | legacy — keep for artefact reproducibility; **not** the current system |
| **Gen-3 CICADA (current)** | `typed_planning.py` (two-step TypedLLMPlanner), `graph_planning.py` (T1-T12 + gates), `pipeline.py` (gated reflector, flagged-answer repair), `schema_grounding.py`, `entity_resolution.py`, `executor/grounding/verifier/answer_*` | `probe_plan_step2.py`, `run_teacher.py`, `run_compare.py --system cicada` (added 2026-07-05) | **ACTIVE** |

Dormant-by-configuration in Gen-3: `diagnostics.py` (advisory hooks, off), `trace_reflector.py`
(off), `retrieval.semantic_repair` (no candidate retriever wired), `decomposition.py` execution
branch (no Gen-3 planner emits decompositions), `documents/*` (need_documents=False).

**Result-provenance warnings**: the 220-question RAG comparison (91.8% "ours") and the
hard20/hard100 evals are Gen-2 numbers; the dev_smoke trajectory 56→84% and all teacher-harness
numbers are Gen-3. The comparison is being rerun with Gen-3 (`--system cicada`) on the same
220 questions; RAG-side numbers are unaffected.
