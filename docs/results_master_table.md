# Master results table (single source of truth for every citable number)


> **IRON RULE (2026-07-07, after 3 gate bugs):** every RATE-type / gate-decided number here
> must have had 10 deterministic-random raws (passes AND failures) human-scanned before entry,
> via `scripts/dump_raws.py`. End-to-end EM numbers (five-pairing matrix) are oracle-double-
> verified and exempt; gate-judged numbers (schema diagnostic, ood_probe, gain decomposition)
> are NOT exempt. A suspiciously clean/terrible number is treated as an instrument fault until
> its raws are opened.

Rule: the thesis cites ONLY numbers from this table; each carries dataset, metric, pipeline version, artifact path.

- Benchmark arithmetic: **12,828** = train 9,267 + dev_tune 556 + dev_select 671 + dev_smoke 49 + final_test 2,285. compare_set_v4 (260) ⊂ final_test.
- compare_set_v4 = DEV set (model selection; pipeline v2 fixes were derived from its v1 errors). final_test = confirmatory, report incl./excl. 19 cue-matched rows.
- 'Data-engine yield' (harvest) and 'system accuracy' (eval) are DIFFERENT METRICS; never compare across.

| System | Pipeline | Dataset | n | Metric | Value | Artifact | Runs | Note |
|---|---|---|---|---|---|---|---|---|
| RAG naive | v1 | compare_v4(dev) | 260 | EM acc | 31.5% (82/260) | outputs/eval/baselines/rag_naive/compare_rag_naive.summary.json | 1 |  |
| RAG strong | v1 | compare_v4(dev) | 260 | EM acc | 28.8% (75/260) | outputs/eval/baselines/rag_strong/compare_rag_strong.summary.json | 1 |  |
| Teacher (nano+grok) | v1 | compare_v4(dev) | 260 | EM acc | 70.0% (182/260) | outputs/eval/baselines/teacher/compare_cicada.summary.json | 1 | single run |
| Qwen zeroshot | v1 | compare_v4(dev) | 260 | EM acc | 61.5% (160/260) | outputs/eval/matrix/cicada-qwen3-zeroshot/compare_cicada.summary.json | 1 |  |
| Qwen sft | v1 | compare_v4(dev) | 260 | EM acc | 76.1% (198/260) | outputs/eval/matrix/cicada-qwen3-sft/compare_cicada.summary.json | 1 |  |
| Qwen rsft | v1 | compare_v4(dev) | 260 | EM acc | 76.1% (198/260) | outputs/eval/matrix/cicada-qwen3-rsft/compare_cicada.summary.json | 1 |  |
| Qwen dpo | v1 | compare_v4(dev) | 260 | EM acc | 78.8% (205/260) | outputs/eval/matrix/cicada-qwen3-dpo/compare_cicada.summary.json | 1 |  |
| Qwen zeroshot | v2.2 | compare_v4(dev) | 260 | EM acc | 70.4% (183/260) | outputs/eval/matrix_v2/cicada-qwen3-zeroshot/compare_cicada.summary.json | 1 |  |
| Qwen SFT | v2.2 | compare_v4(dev) | 260 | EM acc | 81.2% (211/260) | outputs/eval/matrix_v2/cicada-qwen3-sft/compare_cicada.summary.json | 1 |  |
| Qwen RSFT | v2.2 | compare_v4(dev) | 260 | EM acc | 81.5% (212/260) | outputs/eval/matrix_v2/cicada-qwen3-rsft/compare_cicada.summary.json | 1 |  |
| Qwen DPO-v1 | v2.2 | compare_v4(dev) | 260 | EM acc | 83.5% (217/260) | outputs/eval/matrix_v2/cicada-qwen3-dpo/compare_cicada.summary.json | 1 |  |
| Qwen DPO-v2a | v2.2 | compare_v4(dev) | 260 | EM acc | 80.0% (208/260) | outputs/eval/matrix_v2/cicada-qwen3-dpo-v2a/compare_cicada.summary.json | 1 |  |
| Qwen DPO-v2b(IPO) | v2.2 | compare_v4(dev) | 260 | EM acc | 82.3% (214/260) | outputs/eval/matrix_v2/cicada-qwen3-dpo-v2b/compare_cicada.summary.json | 1 |  |
| Llama zeroshot | v2.2 | compare_v4(dev) | 260 | EM acc | 60.0% (156/260) | outputs/eval/matrix_v2/cicada-llama31-zeroshot/compare_cicada.summary.json | 1 |  |
| Llama SFT | v2.2 | compare_v4(dev) | 260 | EM acc | 83.1% (216/260) | outputs/eval/matrix_v2/cicada-llama31-sft/compare_cicada.summary.json | 1 |  |
| Llama RSFT | v2.2 | compare_v4(dev) | 260 | EM acc | 82.7% (215/260) | outputs/eval/matrix_v2/cicada-llama31-rsft/compare_cicada.summary.json | 1 |  |
| Llama DPO-v1 | v2.2 | compare_v4(dev) | 260 | EM acc | 77.3% (201/260) | outputs/eval/matrix_v2/cicada-llama31-dpo/compare_cicada.summary.json | 1 |  |
| FULLY-LOCAL Qwen (step1+dpo) | v2.2 | compare_v4(dev) | 260 | EM acc | 86.2% (224/260) | outputs/eval/matrix_v2/fully-local-qwen/compare_cicada.summary.json | 1 |  |
| FULLY-LOCAL Llama (step1+sft) | v2.2 | compare_v4(dev) | 260 | EM acc | 84.2% (219/260) | outputs/eval/matrix_v2/fully-local-llama/compare_cicada.summary.json | 1 |  |
| Teacher (nano+grok) | v2.2 | compare_v4(dev) | 260 | EM acc, 3 runs | mean 72.6% ± 1.1% (71.9%, 71.9%, 73.9%) | outputs/eval/matrix_v2/teacher* | 3 | noise floor |
| Teacher harvest | harvest-cfg | train pool | 9267 | oracle-correct on answerable (DATA-ENGINE YIELD, not system acc) | 65.5% (5605/8555) [RECOMPUTED from traces 2026-07-07; summary.json had been clobbered by a 3-question salvage run] | data/qa/teacher_full_v1/traces.jsonl | 1 | repair budget 2 |
| Qwen-SFT self-harvest r1 | harvest-cfg | train pool | 9267 | oracle-correct on answerable (DATA-ENGINE YIELD, not system acc) | 76.6% (6556/8555) | data/qa/rsft_qwen_r1/summary.json | 1 | repair budget 2, temp per config |
| Llama-SFT self-harvest r1 | harvest-cfg | train pool | 9267 | oracle-correct on answerable (DATA-ENGINE YIELD, not system acc) | 78.1% (6685/8555) | data/qa/rsft_llama_r1/summary.json | 1 | repair budget 2, temp per config |


## Paired significance & slice analyses (2026-07-06, compare_v4 DEV)

| Comparison | Discordant | McNemar p | Verdict |
|---|---|---|---|
| Qwen DPO-v1 vs teacher r1/r2/r3 | +35/-5, +34/-4, +30/-5 | 3e-7, 5e-7, 2e-5 | student >> teacher, robust to provider noise |
| Qwen fully-local vs hybrid(dpo-v1) | +13/-6 | 0.167 | direction +, NOT individually significant |
| Llama fully-local vs hybrid(sft) | +14/-11 | 0.690 | direction +, NOT individually significant |
| CROSS (post-hoc): Llama-FL vs Qwen-hybrid, final_test | +252/-131, Δ=+5.30 CI[+3.62,+6.97] | 8.7e-10 | briefing quality dominates base capability; non-preregistered |

Fully-local-vs-hybrid at scale: decided by final_test pairing (n=2,285), pending.

### iid -> ood_candidate decay (absolute values; ood_candidate = HARD-CATEGORY + L2 composite slice, NOT a strict holdout — see merge_stratify_l1_l2.py:is_ood_candidate; both train and test contain this class; plans disjoint, surfaces novel)

| System | iid (n=103) | ood_candidate (n=136) | slope |
|---|---|---|---|
| fully-local qwen | 84.5% | 85.3% | +0.8 |
| qwen dpo-v1 | 86.4% | 78.7% | -7.7 |
| teacher r1/r2/r3 | 75.7/73.8/78.6% | 65.4/67.6/66.2% | -10.3/-6.1/-12.5 |

Strict-holdout probe available: 1 test-only template family (small n) — report in limitations.


## FINAL_TEST headline block (2026-07-07, n=2,285, v4.1, eval protocol repair-1) — CONFIRMATORY SET

| System | Acc | vs teacher (paired) | vs hybrid (paired) | Artifact |
|---|---|---|---|---|
| Teacher (nano+grok), 1 replicate | 69.76% (1594/2285) | — | — | outputs/eval/final_test/teacher_r1 |
| Hybrid student (nano step1 + dpo_v1) | 78.03% (1783/2285) | +8.27pt, +263/-74, p<1e-15 | — | outputs/eval/final_test/hybrid_qwen_dpo |
| FULLY-LOCAL student (step1_v1 + dpo_v1) | **85.65% (1957/2285)** | **+15.89pt, CI[+14.07,+17.70], +406/-43, p<1e-15** | +7.61pt, CI[+6.10,+9.13], +242/-68, p<1e-15 | outputs/eval/final_test/fully_local_qwen |

Asymmetric dev->test decay: hybrid 83.5->78.0 (-5.5pt), fully-local 86.2->85.65 (-0.5pt) — denoising at scale.
Pending fills: incl./excl. 19 cue-matched rows split; llama double-arm; r2 rung.


### FINAL scoreboard incl. Llama arms (2026-07-07, all pairings McNemar p<1e-14)

| System | final_test | vs teacher | vs own hybrid | dev->test decay |
|---|---|---|---|---|
| Teacher (1 replicate) | 69.76% | -- | -- | -- |
| Llama hybrid (nano+sft) | 75.97% | +6.21 [+4.68,+7.74] | -- | -7.1pt |
| Qwen hybrid (nano+dpo_v1) | 78.03% | +8.27 [+6.70,+9.85] | -- | -5.5pt |
| Llama FULLY-LOCAL (step1+sft) | 83.33% | +13.57 [+11.81,+15.32] | +7.35 [+5.73,+8.97] | -0.9pt |
| Qwen FULLY-LOCAL (step1+dpo_v1) | **85.65%** | **+15.89 [+14.07,+17.70]** | +7.61 [+6.10,+9.13] | **-0.5pt** |

Four-point decay pattern: fully-local flat on both bases, hybrid steep on both bases -> denoising mechanism cross-base at n=2,285.


### Cue-split dual report (disclosure for the dev-derived _UNSUPPORTED_CUES; 19/2,285 rows string-matched)

| System | all 2,285 | excl. 19 cue rows | delta |
|---|---|---|---|
| teacher | 69.76% | 69.55% | -0.21pt |
| qwen hybrid | 78.03% | 77.89% | -0.14pt |
| qwen fully-local | 85.65% | 85.57% | -0.08pt |
| llama hybrid | 75.97% | 75.82% | -0.16pt |
| llama fully-local | 83.33% | 83.23% | -0.10pt |

Bound: every headline moves <=0.21pt; rankings and significance unchanged. (Deltas negative: the cue rows were mostly answered correctly by ALL systems.)


### Schema-valid diagnostic (no guided decoding, n=100 DEV, DUAL-METRIC, raws scanned 2026-07-07)

| Model | JSON-parse | grounded-any-schema | target-contract-conformant |
|---|---|---|---|
| Qwen3-8B (untrained base) | 100 | **98** | **3** |
| cicada-qwen3-sft | 100 | 98 | 98 |
| cicada-qwen3-dpo | 100 | 99 | 99 |

Verified by opening 10 random base raws (not trusting the gate): the base emits question-grounded
plans with REAL entities (Wessex Water, NHS England, real CPV/years) but in a self-invented schema
(op:filter_records, args.filters, id/inputs) — NOT the compiler's contract. Conclusion: fine-tuning
does NOT teach planning ability (the base already extracts entities and builds filter structures) —
it teaches CONFORMANCE TO THE SPECIFIC EXECUTABLE CONTRACT (target 3 -> 98/99). Guided decoding at
eval time enforces exactly this contract, so the accuracy ladder isolates planning QUALITY within
the contract, not the ability to emit it. Honesty caveats: (1) diagnostic stored only failures, so
trained-arm PASSES were not individually scanned — the claim rests on the base arm's verified
grounded-but-off-schema pattern; (2) 1 base abstain via return.operation (not question_type) was
under-counted as "empty" — the 3-vs-98 direction is insensitive. This diagnostic had THREE prior
opposite-signed gate flaws (extractor over-count / shape over-count / content under-count), each
caught by spot-check — the audit trail is itself an eval-methodology contribution.
artifact: docs/artifacts/schema_valid_diagnostic.json.

### r2 (bootstrap round 3) — SUPPLEMENTARY ROW per pre-committed gate (Δ<+3pt, n.s.)

| Metric | r2 | fully-local DPO champion | verdict |
|---|---|---|---|
| DEV (compare_v4) | 86.5% (225/260) | 86.2% (224/260) | +0.4pt, McNemar +3/-2 p=1.0 |
| bridge_join bucket | 15/20 | 14/20 | +1, p=1.0 (n.s.) |
| harvest answerable yield | 86.3% | (r1 76.6%) | yield keeps rising... |
| harvest bridge yield | 75% | (r1 52%) | ...but training gain saturates |

**Bootstrap convergence**: r1 gave +11pt over teacher at harvest AND translated to the ladder;
r2 harvest yield still climbs (answerable 86.3%, bridge 75%) but the eval gain is +0.4pt (n.s.).
The verifier-gated self-improvement loop has a natural ceiling on this task. Headline UNCHANGED
(fully-local DPO, final_test 85.65%). r2 reported as a convergence data-point, not a new headline.
Llama-r2: literal gate (bridge>=15/20) met on a technicality but the underlying gain is null;
skipped: NOT because it would be null (Qwen convergence does not imply Llama convergence — DPO behaved oppositely across bases), but because the convergence claim is ALREADY established on the main base, single-base evidence suffices for it, and near deadline the 4 GPU-h has higher marginal value on ood_probe — pending user confirmation.


## Promoted rows (2026-07-07 verification pass — draft v0.1 marker resolution)

| Figure | Value | Primary source | Verification |
|---|---|---|---|
| Canonical organisations in KG | 131,502 | data/kg/nodes/org_nodes.parquet | [ARTIFACT-VERIFIED pandas count 2026-07-07] |
| Contract-award records in KG | 215,221 | data/kg/nodes/contract_nodes.parquet | [ARTIFACT-VERIFIED pandas count; also kg_enrichment_plan.md] |
| Dual-oracle agreement | 99.88% (14,752/14,770) | worklog 2026-07-04 + thesis_draft.md (legacy) §4.4 | [DOC-VERIFIED both sources] |
| LLM-checker approves all rewrites (motivates mechanical checks) | 9,762/9,762 | thesis_draft.md (legacy) §2 | [DOC-VERIFIED; process fact, motivation-only] |
| Bridge self-harvest verifier-blind stratum | verified 86.7% vs oracle-correct 52.3% (n=1,351) | worklog 2026-07-05 (computed from rsft_qwen_r1/traces.jsonl in-session) | [ARTIFACT-DERIVED] |
| Median pairwise trigram-Jaccard of accepted L2 surfaces | 0.429 | thesis_draft.md (legacy) §4.5 | [DOC-VERIFIED; distributional evidence, not a diversity claim] |
| OCDS release corpus size | 166,277 releases (2022-2026 five year-files) | ocds_data_analysis.md | [DOC-SOURCED; interim releases.parquet deleted] |
| MoD under 77 distinct GB-FTS IDs (2024 sample) + money-semantics field survey | qualitative context | ocds_data_analysis.md | [DOC-SOURCED analysis document] |

## Pending (filled as artifacts land)
- final_test fully-local qwen (RUNNING)
- r2 rung
- schema-valid diagnostic
- OOD-slice table (computed 2026-07-06: student advantage LARGEST on ood_candidate: +18-20pt vs teacher)
## E13g — C-v5 (v4b pool, 5,532 ex, retrained on turin after save corruption) 2026-07-23

| Arm | n | Metric | Value | Paired vs C-v3 | Artifact |
|---|---|---|---|---|---|
| C-v5 (gold-program, v4b pool) | 300 clean-eval-devfold | internal denotation match | **55.67% (167/300)** | +21/-7, exact McNemar p=0.0125 | data/qa/wtq/eval_wtq_clean_C5.jsonl |

Ladder: C-v1 40.0 -> C-v2 47.67 -> C-v3 51.00 -> C-v5 55.67; slice ceiling ~64.6 (coverage-bounded).
Outcomes: answered 288, eval_failed 10, truncated 1, invalid_tree 1, abstain 0 (expected for C arm).
Provenance note: first C-v5 save was quota-corrupted (443/504 zero tensors, 23.33% garbage number DISCARDED, never ledgered);
retrained clean on turin GPU3 (3h08m shared card), integrity assertion 0 zero tensors, eval against curated 300 slice.

## E13h — grok-4-20-reasoning trials on zero-hit pool (2026-07-23, three-arm control on same 20 fast-failed questions)

| Arm | Rescue | Cost/q notes |
|---|---|---|
| fast alone k=2 (pilot) | 0/20 by construction | $0.0007 |
| fast + reasoning-brief k=2 | 3/20 | brief burns ~2,950 thinking tk/q |
| reasoning direct k=1 | **7/20 (35%)** | ~2,658 thinking tk/q |

Verdict: understanding-arm (brief) handoff loses most of reasoning's advantage; planning in the
algebra, not comprehension, is the bottleneck. Reasoning-DIRECT is the pool-completion tool.
Artifacts: pilot_grok_reasoning.jsonl, pilot_grok_wtq_brief.json, pilot_fast_with_brief.json.
Fast pilot economics (E13c artifact recomputed): 155/1000 = 15.5% incremental, zero-hit pool 8,049/11,332.
Pricing tier of grok-4-20-reasoning UNRESOLVED (dashboard shows GBP 0 across 66 req / 45.1k tk —
free preview or billing lag; deployment blade check pending).

## E13i — two-stage v2 (faithful main-experiment card port) + template census (2026-07-23)

| Arm (same 20 fast-failed) | Rescue |
|---|---|
| two-stage v2: 8-section brief + template shells | **1/20** (v1 prose brief: 3/20; reasoning direct: 7/20) |

Template distribution on the 20: ordered_navigation 6, filter_aggregate 7, comparison 4, grouped_extreme 2, row_lookup 1.
Mechanism: brief classified 6 as out-of-language -> shell steers Step-2 to abstain -> dead by construction,
while reasoning-direct SOLVES some of the same questions by recasting in-flight. Frontier questions need
JOINT understanding+planning search; any Step-1->Step-2 freeze kills the recast. Replicates the main
pipeline's own lean-beats-card finding for strong planners. Understanding-arm CLOSED for zero-hit pool.
Template vocabulary validated: 9,663 verified trees classify 99.9% into 8 templates
(filter_aggregate 40.2, row_lookup 29.3, superlative_row 20.0, grouped_extreme 7.0, comparison 2.8, rest 0.5).
Artifacts: pilot_brief_v2.jsonl, scripts/wtq/pilot_brief_v2.py.
E13i amendment: v3 (free-composition + advisory brief + explicit recast rules) = 3/20, ties v1;
recast guidance did rescue 1/6 ordered_navigation. Three handoff protocols (tight/loose/advisory)
all plateau at 1-3/20 vs reasoning-direct 7/20 -> bottleneck is fast's in-algebra planning, not
brief quality or handoff format. Understanding-arm verdict FINAL. Artifact: pilot_brief_v3.json.

## E13j — local B-arm paired trial: trained planner vs briefs (2026-07-23)

| Arm (same 20 zero-hit fast-failed, k=2 t=1.0, strict schema) | Rescue |
|---|---|
| LOCAL C-v5 bare | **6/20** |
| LOCAL C-v5 + reasoning brief | 6/20 (identical question set: brief-only 0, bare-only 0) |

In-grammar training (~1000 steps) nearly matches cloud reasoning-direct (7/20) at zero marginal cost;
the brief is fully redundant for a trained planner (+0, perfect overlap). Understanding-arm closed from
both ends: weak planner can't exploit briefs (3/20 ceiling), strong planner doesn't need them.
Harvest chain implication: LOCAL C-v5 resample of zero-hit pool FIRST (free), grok-fast on residue,
reasoning-direct on the hard core. Correction to language-boundary claim: v4 views already made row
order DATA; the true wall is Predicate<Expr,Expr> (literal-only filters), and anchor-visible navigation
questions are winnable via plan-time literal inlining. Artifact: pilot_barm_local.json.

## E13k — two-step coupling engineering, five-protocol convergence (2026-07-23)

| Protocol (same 20 fast-failed) | Rescue | Mechanism added |
|---|---|---|
| v1 prose brief | 3/20 | brief |
| v2 template shells | 1/20 | structured protocol (backfired) |
| v3 advisory + recast rules | 3/20 | free composition |
| v4 hard rules + mech repair | 3/20 | Step-2.5 deterministic adapter |
| v5 + typed-feedback loop | 4/20 | full reflector (1 via repair, 1 via feedback) |

Linkage machinery WORKS (repairs and feedback each converted trees); the ceiling is grok-fast's
multi-fault tree assembly (~20% on this residue). Two-step is not dead but FUSED: reasoning-direct
(7/20) performs understanding->planning jointly inside one forward pass, immune to handoff freezing.
Teacher config: fused reasoning for cloud, in-grammar-trained local planner (C-v5 6/20) for free tier.
Infrastructure committed: tree_repair.py (5 deterministic variant classes), typed-diagnosis feedback loop.
Artifacts: pilot_twostep_v4.json, pilot_twostep_v5.json, linkage_debug.json (six-case fault traces).
E13k amendment 2 (v6, user-designed rewrite protocol): table-grounded question REWRITE by reasoning
(sees catalog, eliminates ordinals via sortable columns) -> fast standard prompt = 4/20 + 4 honest
REWRITE_IMPOSSIBLE flags. Decisive exhibit nt-1: rewrite is a word-perfect English rendering of the
winning tree ("the venue__part_1 in the row with the largest year among rows where position is 1st")
and fast STILL fails to assemble it -> assembly, not question indirection, is the binding constraint.
Six protocols now: 3/1/3/3/4/4 vs reasoning-direct 7, local C-v5 6. Byproduct adopted: the rewriter's
REWRITE_IMPOSSIBLE flag becomes a free out-of-language detector to prune the paid teacher pool.
Artifact: pilot_twostep_v6.json.
E13k CORRECTION (user-driven ground-truth check): the nt-1 "decisive exhibit" was MISATTRIBUTED.
Table inspection shows position cells are exactly "1st" and the hand-built tree with field=venue
matches gold perfectly; the v6 rewrite's only error was targeting view column venue__part_1
("Bangkok") instead of original venue ("Bangkok, Thailand"). fast plausibly assembled the tree AS
INSTRUCTED. Assembly-is-the-constraint claim is falsified for this exhibit; v6 ceiling unknown
pending v6.1 (rewrite prompt: report values from ORIGINAL columns only; repair: new de-base variant).

## E13l — teacher ensemble union analysis (2026-07-23, closes the two-step investigation)

v6.1 (rewrite prompt: answers from ORIGINAL columns only + de-base repair variant) = 5/20, new
two-step best; nt-1 rescued mechanically. Three-lane overlap on the same 20 hardest residue:
local C-v5 6 (0 unique), reasoning-direct 7 (2 unique: nt-14 nt-25), two-step v6.1 5 (1 unique:
nt-12 via operand-swap repair). UNION 9/20 = 45% of the both-samplers-zero-hit tail.
Final cascade (evidence-based): C-v5 free sweep -> reasoning-direct -> two-step v6.1 mop-up;
rewrite_impossible flags prune the paid pool. The user's two-step strategy earns its ensemble seat
with a unique contribution; protocol v6.1 + tree_repair de-base are the production versions.
Artifacts: pilot_twostep_v61.json; union computed from pilot_barm_local/grok_reasoning artifacts.

## E13m — v4c normalization views: the grounding port (2026-07-23, user-driven)

User challenge "主实验不就有grounding吗" was correct: the transfer ported the data loader but NOT the
main pipeline's normalisation discipline (procurement side: 131,502 canonical orgs, ER phases,
normalise.py; WTQ side: raw-cell exact matching). Exhibit nt-9: the PERFECT in_expr semijoin plan
returns only 1 of 2 gold players because position cells split as "Middle blocker"/"Middle Blocker";
runtime eq/in_expr is exact (casefold only in sort keys). No prompt protocol can fix a representation
gap. v4c: __norm casefolded/whitespace-collapsed twin views, created ONLY where a text column's
values actually merge under folding (conservative trigger). nt-9 tree via position__norm: exact gold
match. Census: 26/703 zero-hit tables (3.7%) gain views; 37/1000 zero-hit questions on affected
tables. Misattribution corrected: nt-9 was representation debt, not planning failure — same taxonomy
as the paper's one-seventh finding. Gated reharvest runs under v4b+v4c views from here on.

## E13n — hand-audit of the 10 unrescued: taxonomy flips from wall to search failure (2026-07-23)

| Case | Verdict | Class |
|---|---|---|
| nt-5 | WON by hand: gcombine(diff, groupby-sum, groupby-sum) -> keys_where ge 3 -> size | search failure |
| nt-23 | WON: keys_where le 1 (tied minimum, literal-inlined) | search failure |
| nt-24 | WON: row_index literal inline (right-after navigation) | search failure |
| nt-31 | WON: row_index lte inline (before-first-full) | search failure |
| nt-9 | WON: v4c __norm (E13m) | representation debt, fixed |
| nt-6 | one view away: paren-suffix "(D3)" needs __noparen | representation debt, open |
| nt-30 | true mean 3.87 vs gold "4" | gold noise |
| nt-10, nt-18 | nationality / hyperlinks absent from table | info-absent |
| nt-38, nt-27 | run-length; cross-column majority comparison | true language wall |

14/20 of the hardest tail now DEMONSTRATED winnable (union 9 + 4 hand + nt-9), 15 with __noparen.
Language wall shrank 6 -> 2. Dominant failure = SEARCH: winning idioms (gcombine rowwise arithmetic,
keys_where tied extremum, literal-inline navigation) are <0.5% of training pools; no sampler proposes
what no pool taught (circular). Break: seed harvest prompts with these idiom exemplars (zero cost).
Note: comparison-op vocabulary split (pred gte/lte vs keys_where ge/le) tripped the auditor too —
prompt should state both vocabularies explicitly.

## E13o — idiom seeding breaks the search circularity; infrastructure suite complete (2026-07-23)

Idiom-seeded harvest prompt (3 exemplars: gcombine rowwise arithmetic, keys_where tied extremum,
literal-inline navigation): reasoning AUTONOMOUSLY wins nt-5 and nt-23 (taught idioms transfer to
unseen tables). Round 2 with linker row_index annotation: nt-24 autonomous. Twin-view repair
(base->__noparen/__norm): nt-6 repaired-win offline. Final tally of the 20 hardest-tail questions:
15/20 proven winnable (13 machine-autonomous incl. repairs), residual = nt-9 (needs self-exclusion
idiom #4), nt-31 (semantic anchor), nt-30 (gold noise), nt-10/18 (info absent), nt-27/38 (true wall).
Shipped infrastructure: v4c __norm + __noparen loader views, linker render_with_rows (anchor
row_index), tree_repair 8 variant classes incl. twin-view upgrades, idiom-seeded prompt block.
All harvest stages (local cascade + harvest_teacher) inherit these by default.
Artifacts: pilot_idiom_seed.json, pilot_idiom_seed_r2.json.

## E13p — user-authored Step-1 closes the audit: 18/20 (2026-07-23)

User-written briefs (design principles: Step-1 must know the operator inventory; Step-1 supplies
world knowledge explicitly) + repair v3 (inner select->values under in_expr; last-resort
eq->contains, oracle-guarded):
| nt-9 | user brief (self-exclusion) + inner-select repair | WIN |
| nt-31 | user brief ("beta-related") + eq->contains repair | WIN ("Beta-pre" row 8) |
| nt-30 | user's recheck order: gold was RIGHT, Totals row was the culprit; ratio over Totals-excluded = 4.0 | WIN (my gold-noise verdict retracted) |
| nt-27 | literal diff 1996-1874 from brief author's table read | WIN, faithfulness caveat (plan does not derive 1874) |
| nt-10 | world-knowledge in-list (Falappa, Pirovano) per user principle | WIN |

FINAL: 18/20 of the hardest tail winnable (was 0/20 for fast at session start). Survivors: nt-18
(hyperlinks destroyed in CSV export — dead), nt-38 (run-length — the sole true language wall).
Production adoption: Step-1 contract gains the two user principles; repair suite now 10 variant
classes. Artifacts: pilot_user_step1.json.

## E13q — holdout-20 replication: toolkit gain does NOT transfer as-is; method does (2026-07-23)

Production config (reasoning + 6 idioms + v4c views + row-index hints + 10-variant repair) on a
FRESH 20 from the same fast-failed pool: 7/20, all direct, repair fired zero times — exactly the
bare-reasoning baseline. Pre-registered read: the original-20's 13/20 was substantially fitted
(idioms/repairs were distilled FROM those failures). Fresh failures are NEW species: pred-layer
ge/le vocab error (mechanical variant added), argext key-vs-value ("most per year" wants the count),
does-A-or-B wants a NAME not a boolean, Totals-row-as-answer (inverse of nt-30), unit-carrying gold
("1 year"), multi-column union counts, date arithmetic. Conclusion: the hard tail is a LONG TAIL;
the asset is not a static idiom list but the audit->distill->bank LOOP, growing coverage batch by
batch. Stable planning number: bare reasoning-direct ~35% replicated across both 20s (7/20, 7/20) —
use for Job A+ yield forecasts. Artifacts: pilot_holdout20.json.
