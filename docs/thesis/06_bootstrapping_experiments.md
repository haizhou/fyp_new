# Chapter 6 — Bootstrapping Experiments: Ladder, Yields, and the Five-Pairing Confirmation

## Chapter thesis (what this chapter must convince the reader of)

The learning layer works, three times over. First instantiation: teacher-filtered distillation
— an 8B student trained only on verifier-passed, oracle-gated teacher outputs climbs a
four-rung ladder on DEV. Second: the student replaces the teacher inside the same harness and
its temp-0.7×4 rejection-sampled self-harvest EXCEEDS the teacher's data-engine yield
(76.6% vs 65.5% oracle-correct on answerable). Third: Step-1 denoising distillation — briefings
filtered by whole-pipeline oracle-correct outcome produce a fully-local stack. The confirmatory
evidence is one held-out scoreboard: teacher 69.76 < llama-hybrid 75.97 < qwen-hybrid 78.03 <
llama-fully-local 83.33 < qwen-fully-local 85.65 (n=2,285, every pairing McNemar p<1e-14).
Signal came only from the verifiable region; the resulting systems beat the signal's source.

## Section outline

### 6.1 Protocols: two budgets, one metric discipline
Harvest config (repair budget 2, temperature per config — data engine, maximise verified
yield) vs eval protocol (repair budget 1, plan_samples 2, guided-json on every rung, matrix-
internally comparable). Data-engine YIELD and system ACCURACY are different metrics, never
compared across. EM scoring is type-aware; abstention credit defined in ch. 4.

### 6.2 The ladder on DEV (v2.2, n=260)
Qwen: zero-shot 70.4 → SFT 81.2 → RSFT 81.5 → DPO 83.5. Llama: 60.0 → 83.1 → 82.7 → 77.3
(DPO regression analysed in ch. 7). Scaffolding floor vs training gain decomposition (pillar
2): the zero-shot rung with full deterministic scaffolding already doubles RAG (v1: 61.5 vs
31.5/28.8); training adds on top of the raised floor. DEV robustness: DPO beats each of three
teacher replicates (+35/−5, +34/−4, +30/−5; p ≤ 2e-5).

### 6.3 Bootstrap yields: the student out-harvests its teacher (F5)
Train-pool data-engine yield, oracle-correct on answerable (n=8,555): teacher 65.5%
(5,605/8,555) vs Qwen-SFT self-harvest 76.6% (6,556/8,555, +11.1pt) vs Llama-SFT 78.1%
(6,685/8,555). Correct abstention 88.1% vs teacher 86.8%. Rejection sampling + verifier gate +
repair loop finds correct plans beyond both the teacher's and the student's own greedy
decoding (repair gain 2,889: verified@1 4,712 → @k 7,601). Honesty flags stay in the text:
bridge verified 86.7% but oracle-correct 52.3% — the 34.4pt gap is verifier-passing-but-wrong,
i.e. check-3 leakage, routed to hard negatives (1,051) and on-policy DPO pairs (689).

### 6.4 Step-1 denoising distillation → the fully-local stack
5,091 briefings filtered by WHOLE-PIPELINE oracle-correct outcome (partial-verifiability
filter) train local Step-1 adapters. DEV: fully-local Qwen 86.2 (+2.7 over hybrid), fully-
local Llama 84.2 (+1.1) — cross-base replication of the recipe-level effect; third independent
validation of the partial-verifiability claim.

### 6.5 The confirmatory five-pairing scoreboard (final_test, n=2,285, v4.1, repair-1)
Full table with per-pairing discordants, deltas, CIs; all p<1e-14. Qwen FL vs teacher +15.89
[+14.07,+17.70] (+406/−43); vs own hybrid +7.61 [+6.10,+9.13] (+242/−68); Llama FL vs teacher
+13.57 [+11.81,+15.32]; vs own hybrid +7.35 [+5.73,+8.97]; hybrids vs teacher +8.27 / +6.21.
Dev→test decay column: FL −0.5/−0.9 vs hybrid −5.5/−7.1 (analysis deferred to ch. 7).
Dual-report incl./excl. the 19 cue rows per ch. 4.

### 6.6 Round-2 self-harvest (r2)
[PENDING: r2 rung — harvest was running at ~5,000/9,267 on 2026-07-07.] Pre-committed gates
(frozen in abstract_variants.md BEFORE any r2 DEV number existed): headline swap only if
r2 ≥ champion +3pt on DEV with significant bridge improvement; matrix-row addition if ≥ +0pt;
below 0 reported honestly as diminishing second-round returns against r1's +11.1pt yield gain.

## Evidence manifest

| Number | Where used | Source / artifact |
|---|---|---|
| Scoreboard 69.76 / 75.97 / 78.03 / 83.33 / 85.65; all pairings p<1e-14; CIs as listed | §6.5 | [TABLE-SOURCED] FINAL scoreboard; outputs/eval/final_test/* |
| Decay row −7.1 / −5.5 / −0.9 / −0.5 | §6.5 | [TABLE-SOURCED] FINAL scoreboard |
| Qwen v2.2 ladder 70.4/81.2/81.5/83.5 (183/211/212/217 of 260) | §6.2 | [TABLE-SOURCED] outputs/eval/matrix_v2/cicada-qwen3-*/ |
| Llama v2.2 ladder 60.0/83.1/82.7/77.3 (156/216/215/201 of 260) | §6.2 | [TABLE-SOURCED] outputs/eval/matrix_v2/cicada-llama31-*/ |
| DEV robustness +35/−5, +34/−4, +30/−5; p=3e-7/5e-7/2e-5 | §6.2 | [TABLE-SOURCED] paired-significance table |
| RAG naive 31.5% / strong 28.8% (v1, DEV) vs zero-shot 61.5% (v1) | §6.2 | [TABLE-SOURCED] outputs/eval/baselines/rag_*/; matrix v1 zeroshot |
| Teacher yield 65.5% (5,605/8,555); abstention 86.8% (618/712) | §6.3 | [WORKLOG-SOURCED: 2026-07-05 — master-table row is mangled ("0.0% (2/8555)"), FIX before citing; artifact data/qa/teacher_full_v1/ traces] |
| Qwen self-harvest 76.6% (6,556/8,555); Llama 78.1% (6,685/8,555) | §6.3 | [TABLE-SOURCED] data/qa/rsft_{qwen,llama}_r1/summary.json |
| repair gain 2,889 (4,712→7,601); hard negatives 1,051; on-policy pairs 689 (+ teacher 390 = 1,079); bridge verified 86.7 vs oracle 52.3 | §6.3 | [WORKLOG-SOURCED: 2026-07-05 self-harvest entry — promote] |
| Step-1: 5,091 filtered briefings; FL DEV 86.2 (224/260) / 84.2 (219/260); FL-vs-hybrid DEV pairings +13/−6 p=0.167, +14/−11 p=0.690 (individually n.s.) | §6.4 | [TABLE-SOURCED] outputs/eval/matrix_v2/fully-local-*/; 5,091 [WORKLOG-SOURCED — promote] |
| Figure F2 ladder; F5 bootstrap yield; F7 training curves | §6.2–6.4 | outputs/figures/F2_ladder_main.pdf, F5_bootstrap_yield.pdf, F7_training_curves.pdf (DEV-era renders; final_test re-render [PENDING]) |

## Claims discipline (this chapter must NOT)

- MUST NOT compare yield numbers with accuracy numbers in any sentence (65.5/76.6/78.1 are
  DATA-ENGINE YIELD on the train pool under repair-2; 69.76–85.65 are system accuracy under
  repair-1 on final_test). The bare 67.2% teacher overall yield is never reported without its
  stratification (L1 77.1 vs L2 62.0; the pool is deliberately hard).
- MUST NOT cite DEV ladder numbers as headline results — headline lives on final_test only;
  DEV carries model selection and the three-replicate robustness role, stated as such.
- MUST NOT claim fully-local beats hybrid FROM the DEV pairings (+13/−6, +14/−11 are
  individually non-significant); that claim is licensed only by the final_test pairings.
- MUST NOT use smoke numbers (35/50, 38/50) anywhere.
- MUST NOT describe the self-harvest surpassing the teacher as "student is more accurate than
  teacher" — it is a yield statement about the data engine (sampling + verifier gate + repair),
  not an eval-accuracy statement; the eval statement has its own numbers and set.
- MUST NOT let the r2 section improvise: outcomes map mechanically onto the pre-committed
  gates; a negative r2 is reported as a convergence-curve observation, verbatim per the rule.
- MUST NOT drop "oracle filters, never authors" when describing verified_sft acceptance.
