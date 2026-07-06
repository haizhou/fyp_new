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
| Teacher harvest | harvest-cfg | train pool | 9267 | oracle-correct on answerable (DATA-ENGINE YIELD, not system acc) | 0.0% (2/8555) | data/qa/teacher_full_v1/summary.json | 1 | repair budget 2, temp per config |
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

## Pending (filled as artifacts land)
- final_test fully-local qwen (RUNNING)
- r2 rung
- schema-valid diagnostic
- OOD-slice table (computed 2026-07-06: student advantage LARGEST on ood_candidate: +18-20pt vs teacher)