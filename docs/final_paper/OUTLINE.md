# ELEC0054 paper. Outline before prose.

Deliverable confirmed. TMLR format, 8 pages main text + 4 pages appendix,
single column. TMLR's acceptance criteria are exactly two questions. Are the
claims supported by clear and convincing evidence. Would some of TMLR's
audience be interested. No novelty bar. Claim-evidence alignment is the
whole game, which is the ledger discipline by another name. This file fixes the claim, maps every
claim to its evidence, pre-runs the referee, and locks the style contract.
Prose starts only after this file is agreed.

## 0. Style contract (binding for every sentence)
- One idea per paragraph. Active voice. Simple words at the same information.
- State facts with numbers. No judgement words (well, promising, impressive).
- No em-dashes. Colons only where a list construction forces one. Short
  sentences over subordinate chains.
- Every acronym defined at first use. Every claim carries evidence in this
  paper or a citation.
- Numbers come from RESULTS_LEDGER.md entries only. No number outside it.
- Register per the lecture. "In this paper we show that" appears once, in
  the introduction, and the shown thing is exactly the claim below.

## 1. The claim (specific, falsifiable, scoped)
An 8-billion-parameter local model, trained only on supervision that passed
a deterministic compile, ground, execute and verify chain, answers UK
procurement questions better than the cloud model that taught it. On the
held-out test set the local system reaches 85.65 percent against the
teacher's 69.76 (n=2,285, McNemar p<1e-15). The recipe is the portable
object. Moved to WikiTableQuestions it lifts a 22.5 percent zero-shot floor
to 44.4 with answer-only supervision and 51.8 with gold programs on the
official sealed test, while the checkpoint alone moves the floor 4.8 points.

Scope statement for the paper. Claims are about this recipe on these two
benchmarks. No state-of-the-art claim anywhere.

## 2. Section plan (8 main pages, TMLR single column)

### §1 Introduction (1.2 p)
Problem. Public spending questions need answers a reader can audit, and a
refusal when the data cannot support one. Gap, three sentences. Retrieval
cannot aggregate exhaustively (our RAG controls 31.5/28.8 vs 61.5 zero-shot
scaffolding, E-ledger ch.1 numbers). Distillation assumes a trusted teacher
and ours answers 7 in 10. Self-training filters accept on proxies that pass
wrong outputs silently. Claim paragraph. Contribution list, four bullets,
each forward-referencing its evidence section (SPJ rule).
C1 verified-bootstrap recipe, evidence §5.1.
C2 typed compositional algebra with dual evaluators, evidence §4, §5.2.
C3 sealed necessity-audited benchmark PACS, evidence §5.2.
C4 cross-domain transfer with migration accounting, evidence §5.3.

### §2 Related work (1.0 p)
Three clusters, each with how it works, why relevant, why not enough
(lecture rule, one sentence each).
Cluster A filtered self-training (STaR, RFT, ReST, ExeSQL, SCD). Not enough
because pass criteria are proxies and passing-but-wrong enters the pool
silently. Our filter is deterministic and multi-stage and the leak is
measured (86.7 vs 52.3 on bridge harvest).
Cluster B LLM-over-KG agents (Pangu, ChatKBQA, RoG, StructGPT). Not enough
because none treats verifier rejects as labelled supervision and none scores
abstention.
Cluster C executable table QA (TAPAS, TAPEX, Binder, classical parsers).
Not enough because verifiability of the program language is not the design
object. Positions our WTQ numbers honestly (44.4 matches the classical
weakly-supervised band, 51.8 above TAPAS-large, below modern specialised
systems).

### §3 System (1.5 p) + Figure 1
Three layers in one diagram. Typed algebra (17+1 node types, closed grammar,
open composition). Deterministic chain. Guided decoding as format authority.
Abstention with provenance-gated empty results. Verified-bootstrap loop
(teacher or self samples, dual-execution filter, train on survivors).
Notation kept to one paragraph (lecture: terminology section is for symbols
used later, nothing else).

### §4 Benchmarks and evaluation protocol (0.8 p)
Procurement benchmark (12,828 questions, dual-oracle agreement 99.88, plan
level split isolation). PACS (sealed 922, abstention axis, one confirmatory
run). WTQ official test consumed once under a recorded manifest. Metrics,
repair budget, significance tests all named here once.

### §5 Experiments (2.5 p)
5.1 Main result. Five-system scoreboard table. Controls: RAG baseline,
zero-shot scaffolding floor. Rival: cloud teacher, plus noise floor 72.6±1.1
discipline. Ledger E-main.
5.2 PACS. 78.31 strict, teacher 50.33, base 36.01. Anti-shortcut control:
necessity audit, irreducible subset 78.7 vs 80.0 headline. Unseen-exposure
gap 11.5 [5.6,17.2]. Ledger E1/E2/E4.
5.3 Transfer. Floors, supervision ladder A/B/C on dev, one-shot official
test 22.51/27.33/44.43/51.80 with per-rung McNemar. Migration cost table.
Four-way attribution audit as the closing decomposition (14 percent of the
gap was compiler debt). Ledger E6-E13.

### §6 Discussion and limitations (0.7 p)
What the evidence does and does not support. Abstention does not transfer
free (legacy comparison, no-retirement verdict). Preference optimisation
hazard in near-miss regimes, two sentences. Coverage-limited verdict and
the two designed-not-built extensions. Benchmark surfaces are LLM-generated.
WTQ test consumed. Single KG for the main claim.

### §7 Conclusion (0.3 p)
Restate claim, one paragraph, no new material.

## 3. Referee pre-attack (write answers into the draft before submission)
R1 "You built your own benchmark." Answer with measured integrity. Dual
implementation 99.88, split isolation zero overlap, necessity audit, PACS
surfaces independent of the training renderer.
R2 "Teacher comparison is unfair, students are fine-tuned." Scoped claim
wording. The claim is about the recipe versus the deployed teacher config,
stated in §6. PACS gives the symmetric zero-shot-vs-frozen view.
R3 "LLM-generated questions." Disclosed limitation. PACS channel-b paired
naturalization cost quantified at -2.37.
R4 "Is 51.8 good?" Positioning sentence with the classical band and the
explicit non-competitiveness statement. No SOTA language anywhere.
R5 "Cherry-picked significance." Every paired comparison pre-declared,
discordant pairs printed, ledger commits cited in the repro appendix.
R6 "Test set reuse." WTQ manifest, one shot, consumed status; PACS one
confirmatory run per frozen config, dev/test governance stated.
R7 "Guided decoding does the work, not learning." Schema diagnostic 3/98/99
and the format-channel finding. One paragraph in §5.1.
R8 "Why not RL." One sentence, demonstrations dominated exploration in our
boundary experiment, deferred with the r2 convergence evidence.
R9 "Reproducibility." Seeds fixed, configs frozen, commit+sha manifest,
artifact appendix table.
R10 "Novelty of the loop." The loop is standard and we say so. The filter
strength and blind-region accounting are the contribution (ch.2 divide).

## 2b. Appendix plan (4 pages)
A. Reproducibility. Commit+sha manifests, frozen configs, seeds, serving
   command, cost table.
B. Benchmark integrity detail. Dual-oracle residual characterisation, PACS
   seal and audit record, necessity-audit tables (dev+test).
C. Full result tables. PACS family breakdown, WTQ supervision ladder with
   discordant pairs, learning curves both inits, four-way attribution and
   migration matrix.
D. Algebra reference. Node inventory, typed grammar, one worked tree.

## 4. Open items for the user
- Whether the paper's headline is the main-line scoreboard (recommended,
  rubric wants focused) with PACS+WTQ as generality evidence, or the
  compose/PACS line as headline.
- Figure budget. F1 architecture, F2 scoreboard or ladder, F3 attribution
  matrix. Three figures fit five pages.

## 5. Writing references consulted
Peyton Jones, How to write a great research paper (structure, contributions
drive the paper, claim-evidence forward referencing). Module lecture
transcript 2026-06-23 (claim properties, baselines control and rival,
honesty about failure, introduction-first narrative, one idea per
paragraph, no judgements, define acronyms, no overclaiming).
