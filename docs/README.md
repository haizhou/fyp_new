# Documentation Index

Start here:

- `project_architecture_v2.md` - current target architecture for the next phase:
  reference workflow, ER, extraction, KG, QA, evaluation, and ablations.
- `rebuild_plan.md` - earlier rebuild plan for the current flat-`src` implementation.
- `project_structure_proposal.md` - historical comparison of structure options.

Domain notes:

- `reference_study.md` - reference-data research and entity-resolution lessons.
- `ocds_data_analysis.md` - OCDS source-data analysis.
- `field_profile.md` - field coverage and extraction notes.
- `kg_enrichment_plan.md` - KG-ready enrichment layers, value/date semantics,
  and recommended KG build order.
- `document_evidence_pipeline.md` - verdict-time document evidence boundary.
- `qa_benchmark_pipeline.md` - QA benchmark generation and validation design.
- `qa_benchmark_design.md` - frozen KG v0.1 QA field semantics and known
  coverage limitations for samplers.
- `reasoning_pipeline_design.md` - runtime KGQA reasoning design: planner,
  deterministic executor, evidence verdict, answer card, and reflector.

Use `project_architecture_v2.md` as the source of truth for new files and future
module placement.
