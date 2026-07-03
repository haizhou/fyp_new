# Hard-20 Runtime Reasoning Evaluation

Date: 2026-07-01

Experiment analysis for the 20 "hard but answerable" eval questions run through the runtime
reasoning pipeline (`procurement_graph.reasoning`). This is a **smoke / diagnostic** eval, not the
full benchmark: it isolates *query understanding* (planning) quality, because the runtime and the
benchmark oracle share one deterministic executor.

- Questions: `data/qa/eval/hard20_questions_only.jsonl` (id / question / difficulty_reason only).
- Answer key: `data/qa/eval/hard20_answer_key.jsonl` (oracle + full constraints; **read-only, never
  modified**).
- Runner: `scripts/run_hard20_nano.py` (`--planner rule` offline, or `--planner llm --model
  gpt-5.4-nano`).
- Stratification: 6 sum · 5 count(conjunction) · 4 factoid · 3 exhaustive · 2 boundary.

## 1. Headline results

| planner | answered | oracle-match | sum | count | factoid | exhaustive | boundary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rule (offline) | 16/20 | **12/20** | 6/6 | 1/5 | 0/4 | 3/3 | 2/2 |
| gpt-5.4-nano   | 19/20 | **15/20** | 6/6 | 1/5 | 3/4 | 3/3 | 2/2 |

Two code fixes (details in `docs/reasoning_architecture_review.md` §6) lifted nano from 12 → 15:
grounding was made authoritative over *every* preflight-referenced field (an LLM-invented
`dedupe_key` no longer causes a `schema_error`), and the planner is now shown the exact answerable
column names (schema linking), so it stops inventing `award_signed_date` for `award_date_signed`.

The remaining nano non-matches are **not query-understanding errors** — see the gap diagnosis.

## 2. Gap diagnosis: benchmark artifact vs planning error

Raw oracle-match accuracy conflates two very different failures. `scripts/run_hard20_nano.py`
classifies every numeric-reduction mismatch (`_gap_diagnosis`) and records it per question
(`gap_diagnosis`) and in the summary (`gap_diagnoses`, `benchmark_artifact_ids`):

- **`golden_over_constrained`** — the golden spec applies a hidden coverage guard
  (`supplier_count>=1` / `buyer_count>=1` / `value_is_additive`) that is **not in the
  natural-language question**. The runtime answers the question faithfully; the oracle silently
  over-constrains. Always an *over-count* (the guard only removes rows). This is a **benchmark
  defect**, not a planner error.
- **`planning_gap`** — a genuine query-understanding error (wrong/missing/extra filter).

Result on the hard-20: **all 4 conjunction mismatches are `golden_over_constrained`; 0 are
`planning_gap`.**

### 2.1 Ablation confirmation (applying the hidden guards)

Re-running each conjunction query with the golden's hidden `supplier_count>=1` / `buyer_count>=1`
guards appended reconciles the runtime count to the oracle **exactly**:

| question | runtime (question-only) | oracle (golden) | runtime + hidden guards |
| --- | ---: | ---: | ---: |
| conjunction_count_1958 | 102 | 99 | **99** |
| conjunction_count_0456 | 97 | 93 | **93** |
| conjunction_count_1099 | 94 | 93 | **93** |
| conjunction_count_0249 | 98 | 91 | **91** |

The gap is 100% the coverage guards; the plans are correct. (For 1958 the 3-count difference is
exactly the 3 matching notices with `supplier_count = 0`, i.e. no resolved supplier edge.)

### 2.2 Artifact-adjusted reading

Excluding the 4 items whose golden answers carry answer-changing hidden guards (confirmed above):

- **nano** made **0 genuine planning errors** on the answerable set. Its one remaining non-match is
  `factoid_0003_value_source`, which asks for the value's *provenance* field — deliberately treated
  as answer-only / non-queryable in KG v0.1, so the runtime **abstains by design** (the answer key
  still marks it answerable; this is a design-boundary disagreement, not a planning failure).
- **rule** also mismatches the same 4 conjunction artifacts, but additionally fails all 4 factoids
  for a *real* reason (it cannot resolve a multi-attribute natural-language anchor to a contract).
  This is the concrete value of the LLM planner: factoids 0 → 3.

So the honest planning-quality picture is **nano ≈ perfect on well-posed answerable items**, with the
headline 15/20 depressed only by a benchmark artifact (×4) and a deliberate design boundary (×1).

## 3. Post-execution answer verification (why counts now self-explain)

The audit that produced §2 also found that verification only ever checked the *plan*
(`preflight_checks`), never the *answer*: a faithfully-planned, cleanly-executed count could pass at
`high` confidence while silently including notices with no recorded supplier/buyer. Added
`verifier.postflight_checks` (wired in `pipeline.py`): for a passed count/sum it reports **population
coverage** (matched rows lacking a supplier/buyer), for a `select_unique` it reports **answer
multiplicity**. A non-trivial gap is disclosed on the answer card, e.g. for conjunction_count_1958:

```
population coverage: of 102 matched contracts, 3 with no recorded supplier
  (all counted; a supplier/buyer-complete figure would be lower)
```

This makes a faithful count *explain* its difference from a coverage-filtered figure instead of
surprising the reader — the same lens the gap diagnosis uses, surfaced to the end user.

## 4. Decisions and follow-ups

- **Chosen now (Plan B):** do **not** regenerate Stage-2, do **not** modify `answer_key`. The
  conjunction mismatches are recorded as `golden_over_constrained` benchmark artifacts in this
  analysis and in the smoke summary, and excluded from planning-error accounting.
- **Deferred (Plan A), next Stage-1 rebuild:** drop **answer-changing** coverage guards
  (`supplier_count>=1`, `buyer_count>=1`) from conjunction golden specs — or verbalise them in the
  question. Tracked as a known issue in `docs/qa_benchmark_design.md`.
- **`--coverage-mode` is intentionally NOT added as a default.** Appending the hidden guards at
  runtime would cater to the hidden oracle rather than answer the natural-language question. It is
  admissible only as an *ablation* (the §2.1 table is that ablation, computed as a one-off); it must
  never be the default answering path.

## 5. Reproduction

```powershell
# offline, deterministic (no API)
python -B scripts\run_hard20_nano.py

# live nano planner
$env:AZURE_OPENAI_API_KEY = "<key>"
python -B scripts\run_hard20_nano.py --planner llm --model gpt-5.4-nano --out data\qa\eval\hard20_runtime_nano.jsonl
```

Outputs: `<out>.jsonl` (per question: predicted, oracle, match, `gap_diagnosis`, `limitations`,
plan/grounding/preflight trace) and `<out>.summary.json` (accuracy, per-category, `gap_diagnoses`,
`benchmark_artifact_ids`).
