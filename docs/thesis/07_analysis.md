# Chapter 7 — Analysis: Mechanisms, Noise Floors, Failure Modes, and Ceilings

## Chapter thesis (what this chapter must convince the reader of)

The headline numbers of chapter 6 are explained, not just reported. Four analyses carry the
chapter: (i) the four-point decay pattern — fully-local flat (−0.5/−0.9) versus hybrid steep
(−5.5/−7.1) on both bases — identifies distillation-as-denoising as the mechanism, with a
three-strand causal account of WHY cloud briefings collapse on the harder natural mix;
(ii) DPO's cross-base dose curve plus a two-arm recipe ablation shows near-miss hard negatives
are the poison — an honest, mechanism-explained negative result; (iii) the teacher noise floor
(72.6% ± 1.1) disciplines every teacher-delta claim in the dissertation; (iv) error-mode
decomposition and two ceilings (the boolean structural ceiling shared with the teacher; the
schema-format diagnostic) locate what remains. The reader should finish believing the numbers
because their failure structure is accounted for.

## Section outline

### 7.1 The four-point decay pattern and the denoising mechanism
Dev→test decay: qwen-FL −0.5, llama-FL −0.9, qwen-hybrid −5.5, llama-hybrid −7.1 — flat on
both bases where Step-1 is distilled, steep on both bases where Step-1 is nano. Three-strand
mechanism paragraph (one paragraph in the body, each strand with its experiment footnote,
none in the appendix): (i) final_test's natural mix upweights bridge+comparison to 28.7% (vs
uniform 15.4%) — exactly the types where briefing structure drives planning; (ii) nano
briefings on hard questions are irregular, while the student Step-2 trained on regular
briefing distribution (distilled Step-1 output lands inside it, nano's free-form output lands
outside); (iii) boundary questions abstain bistably-conservatively (null autopsy: 4/6
deterministic abstain, 2/6 nano churn). Also footnote the file-order audit: the @850 rolling-
window dip for hybrid AND teacher co-locates with the 200/200 bridge_join cluster at file
positions 800–1000 — a file-order effect, not a temporal incident.

### 7.2 Cross pairing: briefing quality dominates base capability (post-hoc)
Weak base + local briefing > strong base + cloud briefing: llama-FL 83.33% vs qwen-hybrid
78.03% (+252/−131, Δ=+5.30pt, CI [+3.62,+6.97], p=8.7e-10) — although Qwen sits 2–3pt above
Llama in every same-briefing configuration. One sentence in the body + footnote: post-hoc,
non-preregistered pairing.

### 7.3 The hard-composite slice: where the student's advantage is largest
iid (n=103) → hard-composite (n=136) absolute values: teacher replicates 75.7/73.8/78.6 →
65.4/67.6/66.2 (slopes −10.3/−6.1/−12.5); qwen dpo-v1 86.4 → 78.7 (−7.7); fully-local qwen
84.5 → 85.3 (+0.8, "no measurable decay" mode wording). Student advantage on the slice:
+18–20pt vs teacher [PENDING: promote the computed OOD-slice table rows into the master table
body — currently in its Pending list].

### 7.4 DPO near-miss toxicity: two-arm ablation and the cross-base dose curve
Diagnosis: pair pool 71% bridge+set, 766/1,079 oracle-gated-repair source, no chosen anchor.
Recipe arms (on-policy-first rebuilt pools; qwen 389 pairs 94% on-policy): arm A
sigmoid+pref_ftx 80.0% (bridge 5), arm B IPO 82.3% (bridge 9) — BOTH lose to champion DPO-v1
83.5% (bridge 11); RSFT keeps best bridge (16). Cross-base dose curve: qwen RSFT→DPO +2.0pt;
llama RSFT→DPO −5.4pt (same pathology, higher dose; saturated displacement: chosen logp
memorised, margins grown only by pushing rejected down). Mechanism: on-policy rejected bridge
plans are NEAR-MISS negatives — string-adjacent to correct plans — so suppression drags the
correct modes down (consistent with Razin et al.); balanced pools CONCENTRATED the poison.
Consequence adopted: the bridge lever moves to ADDITIVE self-distillation (r2), not stronger
suppression.

### 7.5 The teacher noise floor
Three identical v2.2 DEV runs: 71.9 / 71.9 / 73.9 → 72.6% ± 1.1 (SD); pairwise discordance
13–22 questions. Discipline derived: single-run teacher deltas below ~3pt are not claimable;
the v2.0/v2.1 "teacher regression" chase was partly noise-chasing (recorded as a lesson).
final_test teacher 69.76% is difficulty (harder natural mix), not API decay — integrity-audit
evidence: teacher's same-260 cross-run comparison was net POSITIVE (+6).

### 7.6 Error-mode decomposition (F6)
v1 matrix decomposition: 21 abstained-on-answerable / 12 answered-on-unsupported / 22
wrong-value → four root-caused scaffolding fixes (ch. 5's worked example); v2.2 re-decomposition
per rung; what training fixes vs what scaffolding fixes (pillar 2 closure).

### 7.7 The boolean structural ceiling
On DEV boolean (n=20), nearly every system lands 14/20 (llama-SFT and teacher-r3 13/20): the
same 6 questions are missed by qwen-FL, llama-FL, AND the hybrid champion — 6/6/6 — and all 6
are also missed by every teacher replicate. A shared structural ceiling, not a training gap:
distillation cannot exceed what neither teacher nor scaffolding can currently express.
[ARTIFACT-DERIVED: computed from outputs/eval/matrix_v2/*/compare_cicada.results.jsonl —
promote to master table before citing.]

### 7.8 Schema-format diagnostic: is the output contract internalised?
[PENDING: schema-valid diagnostic results.] Success criteria PRE-COMMITTED before reading any
output: trained ≥90% no-guidance plan-shape rate with base significantly lower → "contract
internalised"; both-high → pre-committed flip: "format is carried by the prompt; SFT gains are
semantic" (strengthens the control-variable claim). Read against the 3-cell parse/shape grid;
extractor patched (fence-aware) before conclusions; the base arm is true base weights.

## Evidence manifest

| Number | Where used | Source / artifact |
|---|---|---|
| Decay −0.5/−0.9 vs −5.5/−7.1 (both bases) | §7.1 | [TABLE-SOURCED] FINAL scoreboard decay column |
| bridge+comparison 28.7% vs uniform 15.4%; null autopsy 4/6 deterministic; @850 = 200/200 bridge cluster | §7.1 | [SOURCED: thesis_narrative_core.md hybrid-decay section; worklog 2026-07-07 audit entries — promote process numbers] |
| llama-FL vs qwen-hybrid +5.30pt, CI [+3.62,+6.97], +252/−131, p=8.7e-10 | §7.2 | [SOURCED: thesis_narrative_core.md cross-observation — promote to master table] |
| iid→hard-composite: 75.7/73.8/78.6→65.4/67.6/66.2; 86.4→78.7 (−7.7); 84.5→85.3 (+0.8); n=103/136 | §7.3 | [TABLE-SOURCED] iid→ood_candidate table |
| Student advantage +18–20pt on the slice | §7.3 | [TABLE-SOURCED (Pending list) — promote to table body before citing] |
| DPO arms: v2a 80.0 (208/260), v2b 82.3 (214/260), DPO-v1 83.5 (217/260); llama DPO-v1 77.3 vs RSFT 82.7 (−5.4) | §7.4 | [TABLE-SOURCED] outputs/eval/matrix_v2/cicada-qwen3-dpo{,-v2a,-v2b}/, cicada-llama31-{rsft,dpo}/ |
| Pool anatomy 71% bridge+set; 766/1,079; 389 pairs 94% on-policy; bridge sub-scores 5/9/11/16 | §7.4 | [WORKLOG-SOURCED: 2026-07-06 DPO deep-dive — promote] |
| Teacher floor 72.6% ± 1.1 (71.9/71.9/73.9); discordance 13–22; same-260 teacher net +6 | §7.5 | [TABLE-SOURCED] outputs/eval/matrix_v2/teacher*; audit numbers [WORKLOG-SOURCED: 2026-07-07 integrity audit — promote] |
| Error modes 21 / 12 / 22 (v1 matrix) | §7.6 | [WORKLOG-SOURCED: 2026-07-06 pipeline-v2 entry — promote] |
| Boolean ceiling: 14/20 typical; same 6 missed by FL-qwen/FL-llama/hybrid champion + all 3 teacher replicates | §7.7 | [ARTIFACT-DERIVED: outputs/eval/matrix_v2/*/compare_cicada.results.jsonl — promote] |
| Figures: four-point decay figure [PENDING: render]; F4 bucket heatmap; F6 error modes | §7.1/7.6 | outputs/figures/F4_bucket_heatmap.pdf, F6_error_modes.pdf (DEV-era; final_test re-render [PENDING]) |
| Schema diagnostic | §7.8 | [PENDING: patched rerun + 10-failure manual review per pre-commit] |

## Claims discipline (this chapter must NOT)

- MUST NOT headline the cross pairing (§7.2) — post-hoc, non-preregistered: one mechanism
  sentence plus footnote is its entire allowance; "briefing quality is the dominant variable"
  is the licensed phrasing.
- MUST NOT call the hard-composite slice "OOD" (§7.3) — and must state it is a difficulty
  composite present in both train and test (plans disjoint, surfaces novel); compositional
  generalisation belongs to ood_probe_v1 (pre-registered, pending).
- MUST NOT report the hybrid steep decay as a single-cause claim — three footnoted strands,
  merged into one body paragraph, per the frozen narrative-core instruction.
- MUST NOT claim any single-run teacher delta below ~3pt (noise floor); never present
  69.76% as API degradation (the audit refutes it).
- MUST NOT spin the DPO arms as tuning failures — they are mechanism-explained negative
  results; equally MUST NOT generalise to "DPO is toxic" beyond near-miss-negative regimes.
- MUST NOT interpret the boolean ceiling as benchmark defect without evidence — it is a
  shared expressiveness ceiling (teacher-inclusive), stated as such.
- MUST NOT read the schema diagnostic beyond its pre-committed 3-cell grid; first-run numbers
  are provisional until the patched rerun.
