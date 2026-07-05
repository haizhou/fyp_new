# CICADA Core v4 — Question-Type Distribution Across Splits

Source of truth: `data/qa/cicada_core_v4/*.jsonl` (12,828 rows total), tabulated 2026-07-05
from the frozen artifacts. Canonical benchmark axis is `train_bucket` (13 buckets — the same
axis used to build `compare_set_v4` as 13 x 20). `question_type` below is the finer
generator-level typology (15 types).

Split sizes: train 9,267 · dev_tune 556 · dev_select 671 · dev_smoke 49 · final_test 2,285
(9,267 + 556 + 671 + 49 + 2,285 = 12,828).

## Table 1 — Bucket distribution by split (n, % of split)

| Bucket | train | dev_tune | dev_select | dev_smoke | final_test |
|---|---:|---:|---:|---:|---:|
| abstain_ambiguous | 244 (2.6%) | 20 (3.6%) | 42 (6.3%) | 4 (8.2%) | 120 (5.3%) |
| abstain_no_results | 235 (2.5%) | 20 (3.6%) | 41 (6.1%) | 4 (8.2%) | 120 (5.3%) |
| abstain_unsupported | 233 (2.5%) | 20 (3.6%) | 41 (6.1%) | 2 (4.1%) | 120 (5.3%) |
| boolean | 382 (4.1%) | 18 (3.2%) | 26 (3.9%) | 2 (4.1%) | 121 (5.3%) |
| bridge_join | 1,351 (14.6%) | 30 (5.4%) | 64 (9.5%) | 8 (16.3%) | 350 (15.3%) |
| categorical | 403 (4.3%) | 20 (3.6%) | 14 (2.1%) | 1 (2.0%) | 120 (5.3%) |
| comparison | 970 (10.5%) | 29 (5.2%) | 63 (9.4%) | 5 (10.2%) | 306 (13.4%) |
| count | 1,790 (19.3%) | 106 (19.1%) | 231 (34.4%) | 15 (30.6%) | 250 (10.9%) |
| factoid | 1,230 (13.3%) | 20 (3.6%) | 31 (4.6%) | 2 (4.1%) | 250 (10.9%) |
| min_max | 632 (6.8%) | 82 (14.7%) | 21 (3.1%) | 1 (2.0%) | 107 (4.7%) |
| set | 631 (6.8%) | 77 (13.8%) | 21 (3.1%) | 1 (2.0%) | 93 (4.1%) |
| sum | 802 (8.7%) | 29 (5.2%) | 63 (9.4%) | 4 (8.2%) | 200 (8.8%) |
| top_k | 364 (3.9%) | 85 (15.3%) | 13 (1.9%) | 0 (0.0%) | 128 (5.6%) |
| **Total** | **9,267** | **556** | **671** | **49** | **2,285** |

Design notes for the text:
- The three abstain buckets are deliberately over-sampled in final_test (120 each, 15.8%
  combined, vs 7.7% in train): abstention quality is a headline claim, so the test set
  buys statistical power there.
- final_test flattens the bucket skew relative to train (count drops 19.3% -> 10.9%;
  every bucket has >= 93 items), so per-bucket test accuracies have comparable error bars.
- dev_tune oversamples min_max/set/top_k (development-era hardening targets); dev_smoke is
  a 49-item sanity slice — neither is claimed as an unbiased estimate.

## Table 2 — Fine-grained question_type (train vs final_test)

| question_type | train | final_test |
|---|---:|---:|
| abstain_ambiguous | 244 (2.6%) | 120 (5.3%) |
| abstain_no_results | 235 (2.5%) | 120 (5.3%) |
| abstain_unsupported | 233 (2.5%) | 120 (5.3%) |
| aggregation_count | 272 (2.9%) | 31 (1.4%) |
| aggregation_sum | 518 (5.6%) | 100 (4.4%) |
| boolean | 382 (4.1%) | 121 (5.3%) |
| categorical_cpv | 271 (2.9%) | 31 (1.4%) |
| comparison | 970 (10.5%) | 306 (13.4%) |
| count | 1,809 (19.5%) | 390 (17.1%) |
| factoid | 1,633 (17.6%) | 370 (16.2%) |
| min_max | 632 (6.8%) | 107 (4.7%) |
| set | 631 (6.8%) | 93 (4.1%) |
| sum | 800 (8.6%) | 217 (9.5%) |
| temporal | 273 (2.9%) | 31 (1.4%) |
| top_k | 364 (3.9%) | 128 (5.6%) |

(`train_bucket` collapses this typology into the 13 evaluation buckets; both fields are
carried on every row, so either granularity can be reported.)

## Table 3 — Difficulty axes (train vs final_test)

**Generalization class** (final_test is deliberately harder):

| class | train | final_test |
|---|---:|---:|
| iid | 4,937 (53.3%) | 843 (36.9%) |
| compositional | 157 (1.7%) | 193 (8.4%) |
| ood_candidate | 4,173 (45.0%) | 1,249 (54.7%) |

**Surface level** (L1 = cleaned canonical surfaces, L2 = rewritten/diversified surfaces):

| level | train | final_test |
|---|---:|---:|
| L1 | 3,156 (34.1%) | 609 (26.7%) |
| L2 | 6,111 (65.9%) | 1,676 (73.3%) |

**Expected status**:

| status | train | final_test |
|---|---:|---:|
| answerable | 8,555 (92.3%) | 1,925 (84.2%) |
| ambiguous | 244 (2.6%) | 120 (5.3%) |
| no_results | 235 (2.5%) | 120 (5.3%) |
| unsupported | 233 (2.5%) | 120 (5.3%) |

## Split integrity (leakage guarantees)

- **Plan-level disjointness**: train covers 6,674 distinct `plan_id`s, final_test 2,184;
  overlap = **0**. No test question shares its underlying logical plan with any training
  question — the split separates plans, not just surface strings.
- **Template families**: 37 in train, 35 in final_test; 34 shared, 1 test-only family
  (held-out family probe).
- The teacher harvest (`teacher_full_v1`) reads **only** `train.jsonl`; students therefore
  never see final_test or compare_set_v4 questions (or their plans) in any training pool.
- `compare_set_v4` (260 = 13 buckets x 20) is drawn from final_test only.
