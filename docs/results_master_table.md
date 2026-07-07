# Master results table (single source of truth for every citable number)

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


### Schema-valid diagnostic (no guided decoding, n=100 stratified DEV, dual-gate)

| Model | JSON-parse | well-formed shell | CONTENT-valid |
|---|---|---|---|
| Qwen3-8B (untrained base) | 100% | 100% | **0%** |
| cicada-qwen3-sft | 100% | 99% | 94% |
| cicada-qwen3-dpo | 100% | 100% | 99% |

Three-layer separation: JSON SYNTAX is free (all 100%), structural SHELL is near-free (99-100%, but degenerate/empty for the base — 2-var no-filter shells), CONTENT-valid planning (question_type + >=1 grounded filter) is ENTIRELY acquired by fine-tuning (0 -> 94 -> 99). Supports the control-variable claim: guided decoding at eval time enforces the near-free shell, so ladder accuracy deltas isolate the learned content. [artifact outputs/eval/schema_valid_diagnostic.json; instrument twice-verified: JSON extractor fence/brace-fair, shape gate spot-checked and split into wellformed+content after base emitted empty shells.]


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
skipped (would cost ~4 GPU-h to replicate a non-significant result) — pending user confirmation.

## Pending (filled as artifacts land)
- final_test fully-local qwen (RUNNING)
- r2 rung
- schema-valid diagnostic
- OOD-slice table (computed 2026-07-06: student advantage LARGEST on ood_candidate: +18-20pt vs teacher)