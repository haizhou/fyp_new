# VERIFICATION REPORT — draft v0.1 → v0.2 (2026-07-07)

Verification officer run was twice interrupted (API error, then session limit) after completing
its CHECK phase; the apply phase was executed in the main session. Audit trail below.

## 1. Marker resolutions (8/8 verified in primary sources, zero corrections needed)

| Figure | Chapter | Source verified | Disposition |
|---|---|---|---|
| 215,221 contract records | ch1 | parquet (pandas) + kg_enrichment_plan.md | PROMOTED |
| 131,502 canonical orgs | ch1/ch3 | parquet (pandas) | PROMOTED |
| 9,762/9,762 LLM-checker approvals | ch2 | thesis_draft.md(legacy) §2 | PROMOTED (motivation-only label) |
| 99.88% (14,752/14,770) dual-oracle | ch2/ch4 | worklog 2026-07-04 + legacy §4.4 | PROMOTED |
| 86.7% vs 52.3% bridge stratum (n=1,351) | ch2 | worklog 2026-07-05, computed from rsft_qwen_r1 traces | PROMOTED |
| 0.429 trigram-Jaccard | ch2 | legacy §4.5 | PROMOTED (distributional-evidence label) |
| 166,277 releases | ch3 | ocds_data_analysis.md (interim parquet deleted) | kept DOC-SOURCED citation |
| GB-FTS 77-IDs / money-semantics survey | ch3 | ocds_data_analysis.md | kept DOC-SOURCED citation |

## 2. Automated cross-check (all % / pt / large-count tokens in 9 files vs master table)

Headline class: five-pairing scoreboard, DEV ladders, teacher floor, decay figures, schema
diagnostic, cue-split, benchmark arithmetic — ALL MATCH, consistent across the 6 files that cite
them (same precision, same set labels).
Residual flags manually classified (no corrections required):
- Corpus/composition statistics with in-text artifact paths (ch3 ER tier counts, ch4 plan_id
  counts 6,674/2,184, census 12,779) — [ARTIFACT-VERIFIED] class, table rows not required.
- Derived deltas of table rows (ch7 "+2.0" = 83.5−81.5 v2.2; "−5.4" = 82.7−77.3) — derivable.
- Process counts from committed summaries (export sizes 3,084/3,169; DPO pool 1,079; step1 5,091;
  harvest internals 4,712/2,889/1,051) — worklog/export_report-sourced with citations.
- ch7 "66.5" cites the ARCHIVED v2.0 teacher run in the noise-chasing narrative — process fact,
  correctly not presented as citable.
- PROMOTED this pass: cross-pairing +5.30 CI[+3.62,+6.97] p=8.7e-10 (post-hoc label) into the
  paired-significance table.

## 3. Discipline sweeps

- "OOD" as claim: 0 occurrences (all matches were 'goods'). Reflector-as-verifier: 0. DEV numbers
  as headline: 0 found (final_test numbers carry confirmatory labels; DEV carries dev labels).
- Figure placeholders pending render: ch5 Fig 5.1 (architecture diagram — to draw), Fig 5.2
  (dumbbell — exists as F3, reference to fix in style pass).

## 4. Unresolved / deferred

- ch2 dev_smoke motivation figures: motivation-only label kept, no promotion (per marker note).
- Full per-sentence audit by a fresh agent deferred (session limits); the CHECK phase of the
  interrupted officer reported no discrepancies beyond the above before termination.
