# Appendix A — Artifact Manifest

Every quantitative claim in this dissertation is verifiable against a build
artifact in the project repository. This appendix lists the artifacts behind
the figures quoted in the text. All counts are enforced by validation gates
that fail the build on violation (Chapter 3, §3.5).

## A.1 Knowledge graph (Chapter 3)

| Artifact | Rows | Figures it verifies |
|---|---|---|
| `data/kg/nodes/contract_nodes.parquet` | 215,221 | contract-award count; value-source split (160,362 contract / 15,640 award / 19,623 tender / 19,596 none); 176,002 additive; 118,088 distinct OCIDs |
| `data/kg/nodes/org_nodes.parquet` | 131,502 | canonical organisations; role counts (7,125 buyers, 114,281 suppliers, 1,269 both) |
| `data/kg/nodes/cpv_nodes.parquet` | 3,870 | CPV code inventory |
| `data/kg/nodes/evidence_nodes.parquet` | 535,731 | evidence-text counts by field |
| `data/kg/edges/buyer_of.parquet` | 215,218 | buyer coverage |
| `data/kg/edges/supplier_of.parquet` | 334,063 | supplier coverage; 22,075 multi-supplier awards |
| `data/kg/edges/categorized_by.parquet` | 164,691 | CPV coverage |
| `data/kg/edges/evidence_for.parquet` | 1,326,240 | provenance coverage (215,202 of 215,221 nodes) |
| `data/entities/alias_map.parquet` | 204,711 | alias universe; `alias_source` provenance |
| `data/entities/canonical_orgs.parquet` | 131,502 | ER tier sizes via `er_status` (26,704 / 6,332 / 14,875 / 1,286 / 255 / 37 / 82,013) |
| `data/entities/er_audit.csv` | 131,502 | one audit row per canonical entity |
| `ocds_data_analysis.md` | — | the source-data analysis cited as "(source-data analysis; Appendix A)" |

## A.2 Benchmark (Chapter 4)

| Artifact | Content |
|---|---|
| frozen v4.1 split files | 12,828 rows = 9,267 / 556 / 671 / 49 / 2,285; zero train↔test plan overlap |
| `scripts/qa_independent_eval.py` | the independent second oracle (99.88% agreement, 14,752/14,770) |
| dual-report table | 19 cue-matched rows; two disclosed verbatim duplicates |

## A.3 Compose track and PACS (Chapters 9–10)

| Artifact | Content |
|---|---|
| `data/qa/pacs_v1/pacs_dev.jsonl` | 231 rows / 173 clusters, sha256 61efabc9… |
| `data/qa/pacs_v1/pacs_test.jsonl` | 922 rows / 694 clusters, sha256 be20efcf…, sealed one-run |
| `data/qa/pacs_v1/intent_pool.jsonl` | intent clusters with `compose_tree`, logical signatures, isolation identifiers |
| `data/qa/pacs_v1/necessity_rows.jsonl` | per-row predicate-necessity audit records |
| `data/qa/compose_probe_v1/eval_*/summary_*` | all evaluation arms incl. `pacstest_v3_a` (78.31/82.43), `pacstest_teacher_a` (50.33/56.62), reflect arms |

## A.4 WTQ portability (Chapter 11)

| Artifact | Content |
|---|---|
| `data/qa/wtq/squall_split/QUARANTINE.md` | leak audit (zero test overlap), fold manifests, protocol amendment, consumed-status record |
| `data/qa/wtq/differential_audit.jsonl` + `da_{baseline,translator_only,loader_only,both}.jsonl` | three-metric separation and the four-way attribution audit |
| `data/qa/wtq/eval_pristine_{base,v3,A,C}.jsonl` | the one-shot sealed-test run (22.51 / 27.33 / 44.43 / 51.80, official evaluator) |
| `data/qa/wtq/pristine_manifest.txt` | commit hash + file checksums recorded at launch |
| `data/qa/wtq/harvest_A_*.jsonl`, `harvest_dev_*.jsonl` | denotation-only bootstrap pools (26.5% / 27.2% yield) |

## A.5 Process record

`docs/cicada_worklog.md` is the append-only project worklog; every result in
this dissertation appears there with its date, configuration, and any
correction history (e.g. the PACS v1.1 relabelling, the zero-shot claim
revision). Citations of the form "(project worklog, YYYY-MM-DD)" resolve to
its dated entries.
