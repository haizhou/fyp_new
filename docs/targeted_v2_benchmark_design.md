# Targeted v2 Benchmark — Design Spec

Date: 2026-07-01
Status: design + 50-per-subset pilot (human review gate before scaling to 2k)

v2 is a **targeted, capability-probing** benchmark that sits *beside* v1 (the 9,044-row verified
set). It does not regenerate or overwrite v1. Where v1 measured single-table query understanding,
v2 adds four capability axes each v1 under-tested: phrasing robustness, calibrated abstention, a
coverage-clean count oracle, and genuine multi-hop.

## 0. Non-negotiables

1. **v1 is frozen.** v2 writes only under `data/qa/targeted_v2/`; nothing in `data/qa/generated/`
   or `data/qa/eval/hard*` is touched.
2. **Verification-first.** Every row's `oracle_answer` / `expected_status` is confirmed against the
   real KG at build time. An `unanswerable` row must be *provably* unanswerable (0 rows / non-KG
   field / >1 match), never assumed.
3. **Derive from v1 where the oracle must be preserved** (`naturalized`, `coverage_fixed`) rather
   than regenerate — this inherits Gate A/B correctness and isolates exactly the new variable.

## 1. Row schema (every subset, every row)

```
id                 str   "<subset>_<seq>" e.g. naturalized_0007
subset             str   naturalized | unanswerable | coverage_fixed | bridge_join
question           str   the natural-language question
answer_type        enum  count|sum|factoid|boolean|comparison|min_max|set_list|top_k|unsupported|ambiguous
answer_operation   enum  select_unique|count|sum|exists|argmax|argmin|distinct_set|rank_top_k|compare|none
expected_status    enum  answerable|unsupported|ambiguous|no_results
constraints        list  [{field, op, value}, ...]  (question-visible predicates only)
oracle_answer      any   the verified answer, or null when expected_status != answerable
evidence_ids       list  supporting contract_node_ids (capped at 50). REQUIRED for every answerable
                         row with a concrete match set (naturalized/coverage_fixed/factoid/bridge).
evidence_count     int   exact size of the match set (auditable even when evidence_ids is sampled/
                         capped, or empty for pure aggregates like top_k)
difficulty_reason  str   why this item is hard / what it probes
requires_decomposition bool  true if answering needs >1 sub-query (bridge/compare)
executor_support   enum  supported | needs_op:<name> | needs_decomposition | n/a
generation_notes   str   provenance + how oracle/status was verified
```

Additional bookkeeping carried through when derived from v1: `source_spec_id`, `evidence_count`,
`difficulty`, `generalization_class`, `domain_slice`.

## 2. answer_type × answer_operation × executor_support

`executor_support` is the key new field: it makes v2 double as a **work-list for the shared
executor** (extend the executor once; benchmark oracle and runtime both inherit it — the
architecture-review "one executor, two consumers" principle).

| answer_type | answer_operation | executor_support (today) | needs |
| --- | --- | --- | --- |
| count | count | supported | — |
| sum | sum | supported | — |
| factoid | select_unique | supported | — |
| boolean | exists (count>0) | supported | — |
| set_list | distinct_set | `needs_op:distinct_set` | return the distinct set of a field over the match set |
| min_max | argmax / argmin | `needs_op:argmax` | argmax/argmin over a numeric field |
| top_k | rank_top_k | `needs_op:top_k` | group + rank + head-k |
| comparison | compare | `needs_decomposition` | two sub-counts + comparator |
| (bridge) any | * over a semijoin | `needs_decomposition` + `needs_op:in_subquery` | bind a sub-answer set as an `in` filter |
| unsupported | none | n/a | (abstain) |
| ambiguous | none | n/a | (abstain) |

`exists`/boolean is deliberately **supported today** (it is `count>0`), so v2 broadens answer-type
coverage without waiting on executor work.

## 3. expected_status semantics

- **answerable** — a unique verified answer exists; `oracle_answer` set. Runtime must produce it.
- **unsupported** — asks for a field/operation outside KG v0.1 (e.g. number of bidders, social-value
  score, average). `oracle_answer=null`. Correct runtime behaviour = abstain (`mark_unsupported`).
- **ambiguous** — under-specified: the anchor matches >1 contract with differing answers.
  `oracle_answer=null`. Correct behaviour = ask-clarifying (never guess).
- **no_results** — a well-formed query with 0 matches (a factoid over a non-existent
  entity/combination). `oracle_answer=null`. Correct behaviour = report no matching evidence.
  (Note: a *count* with 0 matches is `answerable` with `oracle=0`, NOT `no_results`.)

## 4. Subsets

### naturalized (oracle-preserving paraphrase — planner robustness)
- **Source:** sampled v1 verified specs (count/sum/factoid).
- **Method:** re-verbalise the question into diverse surface forms; **constraints + oracle
  unchanged by construction.** Pilot uses deterministic paraphrase templates (guaranteed faithful,
  API-free). 2k plan: nano paraphrase + a Gate-B faithfulness check that rejects any paraphrase
  whose decoded {operation, filters} differ from the source spec.
- answer_type ∈ {count, sum, factoid}; expected_status=answerable; executor_support=supported;
  requires_decomposition=false.

### unanswerable (calibrated abstention)
- **Composition (verified against KG):** unsupported (non-KG field/op), ambiguous (>1 matching
  contract with differing answers), no_results (0-match factoid over a real-but-empty combination).
- **Verification:** unsupported = field ∉ schema; ambiguous = executor returns ≥2 distinct answers;
  no_results = executor returns 0 rows. Each confirmed at build time.
- answer_type ∈ {unsupported, ambiguous, factoid(→no_results)}; oracle_answer=null;
  executor_support=n/a.

### coverage_fixed (clean count oracle)
- **Source:** v1 `conjunction` specs (the family whose golden carried hidden
  `supplier_count>=1`/`buyer_count>=1` guards).
- **Method:** remove the answer-changing coverage guards; **recompute the oracle** as the faithful
  count over the visible predicates (via the shared executor against the KG). This is the corrected
  count the hard-100 runtime already produced. Alternative (documented, not default): verbalise the
  guard into the question ("... awarded to a supplier on record"); the pilot uses removal.
- answer_type=count; expected_status=answerable; executor_support=supported. `generation_notes`
  records the removed guards and both oracles (v1 vs recomputed) for audit.

### extended_ops (single-hop, needs a new reduction op — NOT decomposition)
- **Purpose:** answer types the current executor lacks but that are computed from **one** query +
  one new reduction op: `min_max` (argmax/argmin), `top_k` (group+rank), `set_list` (distinct set),
  `comparison` (two **independent** counts + comparator).
- **`requires_decomposition = false`** for all of them — a comparison of two independent counts is
  *not* a bridge (no sub-answer binds the next query). `executor_support = needs_op:{argmax|top_k|
  distinct_set|compare}`.
- **Oracle:** computed offline (pandas) so each is well-posed ground truth; evidence_ids populated
  where a concrete match set exists (argmax → the winning contract; set_list → the match set),
  evidence_count otherwise (top_k over a whole category; comparison's two sub-counts live in the
  oracle object).

### bridge_join (PURE multi-hop — sub-answer set binds the next query)
- **Purpose:** only genuine semijoins — hop-1 resolves an entity **set** (suppliers of a buyer /
  buyers of a supplier), hop-2 filters on membership in that set (`in_subquery`). This is the sole
  pattern that actually needs the decomposition planner.
- **`requires_decomposition = true`**, `executor_support = needs_op:in_subquery`. Oracle computed
  offline (2-step); evidence_ids = the hop-2 match set (sampled). **Not fed to the current runtime
  as answerable** — it specifies the `in_subquery` executor op + the decomposition planner.
- Quality over quantity: only well-posed joins are emitted (no padding to a fixed count).

## 5. Stage-1 generation rule change (coverage guard)

For families whose golden **is** the count (`count`, `conjunction`, `cpv`, `temporal`), Stage-1 must
not apply answer-changing coverage guards (`supplier_count>=1`, `buyer_count>=1`). Either drop them
or verbalise them into the question. `coverage_fixed` operationalises the "drop + recompute" path on
existing v1 specs without regenerating v1. (Tracked in `qa_benchmark_design.md` Known Limitations.)

## 6. Layout & review gate

```
data/qa/targeted_v2/
  naturalized_50.jsonl      unanswerable_50.jsonl
  coverage_fixed_50.jsonl   bridge_join_50.jsonl
  pilot_summary.json
scripts/build_targeted_v2_pilot.py     # read-only over v1 + KG; deterministic
```

Gate: build 50 per subset → print/inspect `pilot_summary.json` → human review → only then scale to
2,000. No LLM calls in the pilot (deterministic paraphrase + KG verification), so it is reproducible.

## 7. Design changes I am proposing (for your review)

1. **`executor_support` as an actionable enum** (`supported` / `needs_op:<name>` /
   `needs_decomposition`), not a bool — so v2 is simultaneously the executor extension backlog.
2. **Split `unanswerable` into three verified reasons** (`unsupported` / `ambiguous` / `no_results`)
   rather than one bucket — abstention quality differs by reason and the runtime routes them
   differently (`mark_unsupported` vs `ask_clarifying` vs `report_no_results`).
3. **A count with 0 matches is `answerable` (oracle=0), not `unanswerable`.** Only 0-match *factoids*
   are `no_results`. This avoids poisoning the abstention set with legitimate zero answers.
4. **ADOPTED — 5th subset `extended_ops`.** Single-hop advanced answer types (min_max, top_k,
   set_list, comparison) live in `extended_ops` (`requires_decomposition=false`); `bridge_join` is
   now **pure multi-hop** (`in_subquery` semijoins only, `requires_decomposition=true`). A comparison
   of two independent counts is explicitly NOT a bridge.
5. **`boolean`/`exists` is executor-supported today** (count>0), so include it immediately rather
   than deferring — cheap answer-type coverage.

## 8. Locked-in decisions & pre-2k fixes (2026-07-01 review)

- **Five subsets:** naturalized · unanswerable · coverage_fixed · extended_ops · bridge_join.
- **Auditable evidence on every answerable row.** Concrete-match-set rows carry `evidence_ids`
  (capped 50) + exact `evidence_count`: naturalized 50/50, coverage_fixed 50/50, bridge_join hop-2
  set. Aggregate rows carry a *recomputable* evidence recipe instead of an id list:
  - `top_k` → `group_by_field`, `metric`, `k`, `evidence_count` (population size),
    `evidence_kind="aggregate_recomputable"`;
  - `comparison` → `comparison_breakdown` `{buyerA:{count}, buyerB:{count}}`, `evidence_count`
    (sum of the two sub-counts), `evidence_kind="two_subcounts_recomputable"`.
  So a top_k/comparison row is not "no evidence" — its evidence is the stated group-by, re-derivable.
- **naturalized at 2k switches to nano rewrite + a Gate-B faithfulness check** (reject any paraphrase
  whose decoded {operation, filters} differ from the source spec) — HARD REQUIREMENT. The template
  paraphrase is a pilot-only device: rule-planner-friendly and NOT representative of 2k diversity.
- **Quality-first, not padded.** A subset that falls short of N is written as `<subset>_pilot.jsonl`
  (never a misleading `_50`); exactly-N subsets are `<subset>_50.jsonl`.
- Pilot outputs under `data/qa/targeted_v2/`: `naturalized_50`, `unanswerable_50`,
  `coverage_fixed_50` (exactly 50) + `extended_ops_pilot` (33), `bridge_join_pilot` (32) +
  `pilot_summary.json`. Human review gate before any 2k scale-up (do NOT generate 2k yet).

## 9. QAv2 v0.2 freeze: targeted full2k dataset

Date frozen: 2026-07-01

Frozen artifact root:

```
data/qa/targeted_v2/full2k/
```

Dataset label: `qav2_v0.2_full2k`

Generation command:

```
python -B scripts\build_targeted_v2.py --subset all --limit 2000 --llm off --out-dir data\qa\targeted_v2\full2k --run-tag full2k --seed 42
```

Important scope note: this is a deterministic, template-controlled QAv2 dataset.
It does not use nano rewrite or live Gate-B calls. It is intended as the frozen
targeted evaluation set for the next reasoning-pipeline pass, not as a replacement
for the separate Stage 2 generated benchmark in `data/qa/generated/`.

Final accepted row counts:

| subset | target | accepted | rejected | validation_failed | duplicate_ids | duplicate_questions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| naturalized | 2,000 | 2,000 | 1,124 | 0 | 0 | 0 |
| coverage_fixed | 2,000 | 2,000 | 0 | 0 | 0 | 0 |
| unanswerable | 2,000 | 1,990 | 10 | 0 | 0 | 0 |
| extended_ops | 2,000 | 2,000 | 39 | 0 | 0 | 0 |
| bridge_join | 2,000 | 2,000 | 21 | 0 | 0 | 0 |
| **total** | **10,000** | **9,990** | **1,194** | **0** | **0** | **0** |

Known deviations and thesis/report notes:

- The frozen set has `9,990` accepted rows, not `10,000`.
- `unanswerable` accepted `1,990/2,000`; the 10 missing rows were rejected by the
  duplicate id/question guard. This is kept as-is rather than padded.
- `bridge_join` is structurally valid and all rows have `requires_decomposition=true`,
  but family balance is uneven at full scale: the large supplier/buyer/CPV-set
  families dominate, while `cpv_suppliers_other_cpv`, `year_suppliers_next_year`,
  `category_buyers_count`, and `supplier_set_compare` are small.
- All accepted rows have `validation_failed=0`.
- All accepted rows have `duplicate_ids=0` and `duplicate_questions=0`.

Freeze metadata:

- Summary: `data/qa/targeted_v2/full2k/validation_summary.full2k.json`
- Manifest: `data/qa/targeted_v2/full2k/manifest.full2k.json`
- Run tag: `full2k`
- Seed: `42`
- QAv1 / Stage 2 generated files under `data/qa/generated/` remain untouched by this
  targeted-v2 freeze.
