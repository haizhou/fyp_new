# Results Ledger — post-freeze track (compose / PACS / WTQ)

Role of this file: experiment record, not prose. Each entry: research
question, frozen config + commit, scale, STATUS (FORMAL sealed one-shot |
DEV diagnostic | POST-HOC), numbers with significance, supported conclusion,
limits, artifacts. Writing decisions (what goes to Abstract/Intro/Discussion/
Appendix/nowhere) are made later from this ledger. Main-line (ch.1–8) results
live in their chapters and are not re-recorded here.

---

## E1. PACS-test confirmatory run — STATUS: FORMAL (sealed, one-shot)
Q: Does the frozen compositional planner survive a benchmark it did not author?
Config: frozen compose-v3, guided decoding, single call, channel a; seal sha256 be20efcf…; v1.1 relabel (46 rows, boolean-empty defect; stored trees re-executed offline, zero model calls).
Scale: 922 rows / 694 clusters (test); dev 231/173 for diagnosis only.
Numbers: strict 78.31; safe 82.43; answerable 80.00 (584/730), cluster-bootstrap 95% CI [77.06, 82.92]; status axis 71.9 SEM / 91.7 Safe (n=192); unseen-exposure gap 11.5pt [5.6, 17.2]; channel b 77.11 (paired naturalization cost −2.37 on n=465); cost 1 call / 1.15 s / ≈1.3k tok.
Families: F1 99, F2 96, F3 41, F4 94, F5 97, F6 70, F7 39 (approx per frozen table).
Supports: primary-benchmark claim; conditional-extrapolation boundary.
Artifacts: data/qa/pacs_v1/*, data/qa/compose_probe_v1/eval_pacstest_v3_a*, worklog entries.

## E2. PACS same-seal references — STATUS: FORMAL (same one-run discipline)
Numbers: base 36.01 strict / 42.41 safe; lexical RAG (8B reader) 21.90 answerable-subset; teacher grok-4-1-fast (free decode, API cannot constrain) 50.33 / 56.62 → student +27.98; teacher format failures invalid 156 + unparseable 57 = 23.1%.
Incident record: first teacher run stalled 7 h (Azure trial exhausted, 401), rerun after account upgrade with streaming driver (concurrency 4, ~13 min, ~$0.5); 18 api_error rows resume-retried. Commit 1f8342a/fbc92c7.
Supports: student-surpasses-teacher on primary benchmark; format-channel deployment argument (asymmetry disclosed).

## E3. Legacy paired comparison (Set A) — STATUS: FORMAL on frozen legacy test
Q: Does the single-call compositional planner replace the two-stage legacy system?
Scale: n=2,285 paired per-item; answerable 1,925; abstention 360.
Numbers: answerable 87.38 vs 83.01 (+237/−153, McNemar p=2.5e-05); abstention strict 13.9% (safe recovers 181, denies 58 unfaithful); overall 75.80 strict / 83.72 safe vs 85.65 (safe diff +237/−281, p=0.059 n.s.). Compose mix carried only 37 abstention demonstrations.
Verdict: NO retirement claim (pre-registered key-bucket bar); answerable competence transfers at lower cost, refusal does not come free.

## E4. Constraint-necessity audit — STATUS: POST-HOC dataset audit (gold side only)
Q: Is the PACS headline inflated by logically reducible questions?
Method: delete each oracle-tree predicate, re-execute, compare. Partial sample n=441 of ~903 (one batch lost to machine OOM, disclosed).
Numbers: test predicate necessity 58.5% (622/1,064); irreducible rows 42.7% (141); v3 acc irreducible 78.7% vs reducible 81.5%. Dev: 63.4% / 50.5%; acc 73.2 vs 87.3. Earlier full-dev pass: 62.1% necessity, 50.3% irreducible (consistent).
Supports: headline not shortcut-inflated (−1..3pt on irreducible subset).
Artifacts: data/qa/pacs_v1/necessity_rows.jsonl. Commits 9cda92a, 69d48ff.

## E5. Typed-feedback reflect on PACS-dev — STATUS: DEV diagnostic
Config: oracle-blind; triggers only unparseable/no_tree/type-check/malformed-runtime; abstain/answered/multiple_answers/no_results FINAL.
Numbers: strict 71.86→73.59 (+1.73), safe 82.25→84.42 (+2.17); fixed 4 / broke 0 (p=0.125); triggers 9/231; abstention families unchanged (unsupported 33→33; no_results safe 24→25).
Supports: repair value small-and-safe on mature planner; design encodes ch.6's 84→66 lesson. Commit 96d4f90.
Open decision (user): whether a reflect config spends PACS-test budget.

## E6. WTQ quarantine + baselines — STATUS: DEV diagnostic / infrastructure
Leak audit: Squall 11,276 ∩ WTQ-test = 0 (tables 0/1,617; questions 0); folds train 9,030 / dev 2,246 table-disjoint; wtq-test.json never opened. QUARANTINE.md; commit 8288bfb.
Loader v2: official TSV, 2,108/2,108 tables, 0 silently dropped rows (naive CSV path corrupted 81 tables), dual raw/typed view; loader_audit.py = standing regression. Commit 5dc155a.
Zero-shot floors (clean-300): base 15.3 vs v3 15.7, p≈0.56; original-300: 17.33 vs 18.67, McNemar p=0.557; behavior: abstain 57 vs 21-25, truncation 10 vs 28-31.

## E7. Differential oracle audit (v2 algebra) — STATUS: DEV diagnostic (train/dev folds)
Three-metric separation: syntactic coverage 54.66 train / 52.72 dev; translation fidelity 92.40 / 92.08; executor|A 94.11 / 94.17; reference ceiling 88.99 / 88.78.
Supports: coverage-limited (not execution-limited) verdict; instrument for all later closure claims.
Artifacts: data/qa/wtq/differential_audit.jsonl; da_baseline.jsonl.

## E8. Value-linker four-way ablation — STATUS: DEV diagnostic (pre-adoption gate)
Numbers (base, clean-300): none 15.0 | +columns 16.3 | +cells 19.0 | +random 13.3 (control NEGATIVE). Cells vs none: fixed 15 / broke 3, p=0.0075.
Decision: linker enters frozen recipe. Locked rules: current-table only, fixed top-k 4×5, never gold-conditioned, no row id, column-aware rendering. Commit eee9e96.

## E9. Supervision ladder — STATUS: DEV diagnostic (clean-300, tables disjoint from all pools)
A (denotation-only, 3,739 ex): 31.67; vs floor +54/−5, p=1.9e-11. Reaches its 32.7% gold-translation ceiling.
B (shape-filtered, 1,244 ex = 1/3 pool): 30.00; vs A 14/19, p=0.487 → spurious programs NOT the bottleneck; harvest coverage is.
C (gold-program, 3,956 ex): 40.00; vs A +41/−16, p=0.0013; 7.3pt ABOVE own ceiling; 34 corrects on translator-inexpressible questions.
C decomposition: oracle-solvable 85.3 (81/95); expressible A∪B 73.5 (86/117); valid-tree 98.7; exec|valid 94.6; normalization-recoverable 3/160. Wrong-attribution: 82 gold-audit-uncovered (EXCLUDED from fixable denominator, per correction), 35 empty-grounding, 20 same-shape-grounding, 17 intent-flagged (manual audit: 5 detector FPs, 4 answer-form, 4 true intent ≈2.5%, 4 grounding), 6 structure.
Reflect after SFT: C+reflect = C (neutral) — repair value decreasing in planner maturity (cross-domain replication).
Commits 2d3c88a, 6416781, e132710, a9ef58f.

## E10. Learning curve × init control — STATUS: DEV diagnostic
base-init: 100st 37.3 | 200 38.0 | 500 39.3 | 1000 40.7 (peak) | 2000 39.7.
v3init: 100 39.3 | 200 37.7 | 500 38.3; marks 1000/2000 LOST (quota crash at step ~800, loss 0.0017 converged; not rerun, disclosed).
Reading: curves overlap ±2pt; 100 steps ≈ 1.6k examples lifts 15→37; plateau = data-coverage property; frozen step choice 1000. Commit a7fa817.

## E11. WTQ reflect arms — STATUS: DEV diagnostic
v3 18.67→20.67 (+2.0; 6/0, p=0.0312); base 17.33→20.67 (+3.33; 10/0, p=0.0020); truncation immune to "write shorter" feedback (28→28). Commit fc386bb.

## E12. PRISTINE one-shot — STATUS: FORMAL (sealed test consumed 2026-07-21)
Config frozen: 1,000 steps, extended(v2) algebra, linker hints, guided, single call; pools train+dev per PRE-DECLARED amendment; manifest = commit + 7 sha256 (data/qa/wtq/pristine_manifest.txt); official evaluator (py3 port dev-validated: official 0.49 vs internal 47.67 on C-v2).
Numbers (n=4,344, official): base 22.51 | v3 27.33 | A-final 44.43 | C-final 51.80.
Significance: v3>base +380/−177 p=4.9e-18; A>v3 +810/−89 p=2.7e-146; C>A +604/−299 p=1.7e-24.
Revision recorded: weight prior +4.8pt IS significant at test power (dev n=300 could not detect) → claim refined: weights small prior, recipe delivers bulk (+17 / +24.5 beyond prior).
Positioning (user-approved wording): A 44.43 matches classical weakly-supervised parsers (≈37–46); C 51.80 exceeds early table-pretrained baselines such as TAPAS; NOT competitive with recent specialised TableQA; coverage is the bottleneck.
Consumed: any future run on this set is post-hoc; formal numbers require a new holdout. Commit 9182e16.

## E13. Phase-1 representation/compilation closure — STATUS: POST-PRISTINE DIAGNOSTIC
(does not replace the sealed WTQ test result)
Finding:
  Frozen v2 algebra coverage: train 54.66%, dev 52.72%.
  After translator + loader closure: train 61.03%, dev 60.06%.
  Attribution over 5,196 initially unsupported cases:
    translator only 562 · loader only 154 · interaction 14 · still 4,466.
  Regressions 0/6,080; module overlap 0; fidelity held 92.4–92.7.
Interpretation: 14.0% of the apparent gap was representation/compiler debt; 86.0% remains outside the frozen execution language (nested subqueries 1,482 → Predicate⟨Expr,Expr⟩; row navigation 525 → OrderedRelation; rejected ambiguous parsers 1,870; tails ~380).
Process: audit gates caught 3 implementation bugs (colmap pollution; NaN display map, 549 tables; stale suffix gate) invisible to merged accuracy.
Internal effect: C-v2 (v2 pool 4,378 + hints + 1000 steps) 47.67 on clean-300 vs C-v1 40.0 (+32/−9, p=4.3e-04), invalid 0, truncated 0; converges 2pt under new ceiling.
Commits 4efe7e1, 485ba7e, ea9b9b0, 9660c29, fa3a78d.

## E14. Probe E (hand-authored, authorship disclosed) — STATUS: frozen probe, run once
40 questions; author = co-developing agent at user request (disclosed; NOT independent human evidence; upgrade path: user replaces ≥10). Battery: v3 5/9 vs base 4/9; shared failures deixis/ambiguity/grammar-hole (#39); v3 false-abstain on avg (#12); record-id-vs-title extremum gap (#40) — motivated extreme_rows on WTQ side. First run VOIDED (harness defect, disclosed); fixed run frozen. Commit f7d2fe2.

## E15. Environment/process records (for reproducibility appendix)
Machine root cause: vm.overcommit_memory=2 + other users' commit reservations (513/517 GB) → all "OOM" symptom families; OPENBLAS_NUM_THREADS≤4 mandatory; 50 GB home quota (NFS lag); geneva-env-traps memory note.
Key commands: scripts/wtq/{loader_audit,differential_audit,audit_4way.sh,run_pristine.sh,zero_shot,harvest_a,build_sft_data}.py|sh; scripts/run_compose_probe_eval.py (streaming + --reflect).

## Open items (decisions pending, no work in flight beyond C-v3 gated run)
- v3 type system (Predicate⟨Expr,Expr⟩, OrderedRelation): designed, NOT built — next paper.
- Teacher-harvest grok pilot: $1 gate, awaiting A/C-v3 gap after local re-harvest.
- PACS-test reflect config; E-probe replacement rows; H (RLVR): user decisions.

## E13b. C-v3 ceiling check — STATUS: POST-PRISTINE DIAGNOSTIC (clean-300)
Config: phase-1 grammar (translator+loader closure), train-fold-only pool
4,810 (clean-300 stays table-disjoint), 1000 steps, linker hints.
Numbers: 51.00 (153/300) vs C-v2 47.67 (+22/-12, p=0.121, direction positive, underpowered at n=300); behavior answered 280 / eval_failed
16 / invalid 4 / truncated 0. Sits ~5pt under the widened ceiling
(60.06% dev coverage x ~93% executor). Chain: C-v1 40.0 -> C-v2 47.67 ->
C-v3 51.00 as coverage 52.7 -> 60.1: capability tracks the grammar boundary
(third consecutive confirmation).
Incident: gated runner retrained needlessly after an eval-phase server
failure (train+eval now split into separately retryable stages); attempt-10
adapter preserved at outputs/wtq_C_v3_final.

## E13c. A-v3 (answer-only, v3 grammar + hints + 1000 steps) — STATUS: POST-PRISTINE DIAGNOSTIC (clean-300)
Pool: re-harvest under v3 grammar, yield 26.5%->28.7% train / 27.8% dev-fold;
train-fold pool 4,215 ex (3,283 q). Numbers: **40.00** (120/300) vs A-v1
31.67; answered 243 / eval_failed 47 / invalid 10. Chain A-v1 31.67 -> A-v3
40.00 mirrors C-v1 40.0 -> C-v3 51.0 (both +8~11 from the same phase-1
closure). A-v3 to C-v3 gap = 11.0pt > 3pt pre-declared threshold ->
teacher-harvest $1 pilot WARRANTED per rule; AWAITING USER SPEND APPROVAL.
Pilot design (frozen): 1,000 questions with zero local hits, grok k=2,
metrics = unique-new-verified-questions vs local, cost per new trace,
estimated $0.8-1.0.
