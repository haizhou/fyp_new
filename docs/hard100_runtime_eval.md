# Hard-100 Runtime Reasoning Evaluation

Date: 2026-07-01

Scale-up of the runtime reasoning eval to 100 "hard but diagnosable" questions extracted from the
verified v1 benchmark (`scripts/extract_hard100.py`, read-only). Companion to
`docs/hard20_runtime_eval.md`; same runner (`scripts/run_hard20_nano.py`) and same shared-executor
property (a runtime miss on a well-posed item is a *query-understanding* error, not an execution
mismatch).

- Set: `data/qa/eval/hard100_answer_key.jsonl` (+ `hard100_questions_only.jsonl`), 100 questions —
  sum 20 · boundary 8 · conjunction 18 · count 16 · cpv_count 8 · temporal 12 · factoid 18; 74 hard,
  26 compositional, 8 value-sanity WARN; `value_source` factoids excluded (unsupported by design).

## Result (gpt-5.4-nano planner)

| metric | value |
| --- | --- |
| oracle-match (raw) | **81 / 100** |
| by category | sum 20/20 · boundary 8/8 · count 16/16 · cpv_count 8/8 · temporal 12/12 · factoid **16/18** · conjunction **1/18** |
| planner source | `llm` ×100 (no fallback) |

All reduction families are perfect. The LLM planner lifts factoids from 0 (rule planner) to 16/18.

### The misses decompose cleanly

- **17 conjunction mismatches = `golden_over_constrained`** (benchmark artifact). Each is a faithful
  count that runs higher than the golden because the golden spec carries hidden `supplier_count>=1`
  / `buyer_count>=1` coverage guards absent from the question. `0` are `planning_gap`. See
  `docs/hard20_runtime_eval.md` §2 and the "Coverage Guards Are Answer-Changing For Counts" known
  limitation. **No change to `answer_key` / benchmark** — recorded and deferred to the v2
  `coverage_fixed` subset.
- **2 factoid misses**, both entity/value *matching* brittleness, not reasoning:
  - **factoid_0021 (date) — fix verified.** KG stores `award_date_signed =
    '2021-09-28T00:00:00+01:00'`; the planner correctly emitted `eq '2021-09-28'`, but a date-only
    value cannot `eq`-match an ISO timestamp. Fixed at the KG-translation boundary
    (`kg_backend._translate`: a date-only `eq` on a timestamp column matches by day via `contains`).
    Verified against the real KG — the planner's existing constraints now return exactly one row →
    `CSG Usher's Ltd` (= oracle). This **recovers factoid_0021 → 17/18 factoids, 82/100** overall.
    (Tested: `tests/test_qa_real_kg_backend.py::TestConstraintTranslation`.)
  - **factoid_0017 (casing) — KG entity-resolution known issue, not fixed here.** The same
    organisation exists as two distinct `buyer_name` strings — `'University of Sheffield'` and
    `'UNIVERSITY OF SHEFFIELD'` — and exact-`eq` splits them. This is a KG entity-resolution gap;
    left to the semantic-repair track / a canonical org resolver (see `docs/qa_benchmark_design.md`
    Known Limitations). **No runtime casing patch applied.**

## Scorecard

| | value |
| --- | --- |
| nano raw | 81 / 100 |
| + date fix (verified) | **82 / 100** |
| conjunction mismatches = benchmark artifact | 17 (`golden_over_constrained`) |
| remaining non-artifact miss | **1** (factoid_0017 casing / entity-resolution) |
| change to `answer_key` / benchmark | **none** |

Reading: the planner's query understanding is essentially solid on the answerable set. The residue is
(1) the conjunction coverage-guard artifact (→ v2 `coverage_fixed`) and (2) one KG entity-resolution
gap (→ canonical org resolver / semantic repair). Neither is a reasoning failure.

## Reproduction

```powershell
$env:AZURE_OPENAI_API_KEY = "<key>"
python -B scripts\run_hard20_nano.py --planner llm --model gpt-5.4-nano `
  --questions data\qa\eval\hard100_questions_only.jsonl `
  --answer-key data\qa\eval\hard100_answer_key.jsonl `
  --out data\qa\eval\hard100_runtime_nano.jsonl
```
