# Top-reviewer pass over the full experimental program (2026-07-19)

Verdict: the program's discipline (frozen specs, dual verification, holdout
rotation, perturbation battery, audit) is above the venue bar; the remaining
weaknesses are concentrated in BASELINE COVERAGE and INTERVAL REPORTING on the
primary benchmark, plus a few disclosed construction defects.

## Major (fix before submission)
1. **Primary table lacks a same-set baseline.** Base model was run on PACS-dev
   only; the 78.31% headline floats without a PACS-test reference.
   -> FIX IN PROGRESS: base (its own frozen config, one-run rule) on test/a.
2. **No confidence intervals on primary numbers.** Spec mandates intent-level
   cluster bootstrap; not yet computed. -> compute offline from per-item files
   (overall, per-family, and the seen-unseen gap).
3. **Family cells are unequal n** (F7 n=49 vs F3 112); per-family CIs required
   before any cross-family narrative hardens.
4. **Channel-c promise unmet.** §3.5 promises paired a−c/b−c deltas; c was
   (correctly) never run on test, and the generic c-renderer is under-specified
   for complex templates. -> either fix wording (deltas on dev only, c
   diagnostic) or repair the c renderer and report dev-only deltas.
5. **Sequencing deviation, disclose plainly**: the confirmatory test run was
   EXECUTED before the audit finished (acceptance of numbers did follow the
   audit + relabel + offline re-scoring). One limitation sentence.

## Moderate (adds that materially strengthen)
6. **Cloud-model reference on PACS** absent: one single-run API arm (teacher
   class) on test/a would position local-vs-cloud on the primary benchmark.
7. **Human-authored mini-probe** (30–50 questions written by a person, oracled
   through the algebra): kills the "all surfaces are synthetic" objection.
8. **RAG baseline on PACS answerable subset**: intro's RAG claim currently
   leans on historical dev numbers only.
9. **Cost table for suite A** (calls/tokens/latency): promised in the protocol,
   currently qualitative ("1 call vs 2–4").
10. **Channel-b fallback rate 35%**: report pass-only-subset accuracy as a
    sensitivity check (fallback rows duplicate channel a).
11. **F7 anchor-constrained quotas** (30/45, 12/45) + smallest family n:
    widen anchor space in v1.2 and top up.

## Minor
12. Reproducibility manifest per artifact (seed/commit/hash) — consolidate the
    worklog entries into one table in the appendix.
13. v1.2 backlog (scope placeholder fill; protected-word list incl. "additive";
    decoration cosmetics) — already recorded.

## Additions ranked by value/cost
A. base on PACS-test (running) — closes the primary table. ~30 min.
B. Cluster-bootstrap CIs (offline) — closes #2/#3. ~30 min.
C. Channel-b pass-only sensitivity + dev-only channel deltas — closes #4/#10.
D. Teacher single run on PACS-test/a — closes #6. One API pass.
E. Human mini-probe — closes #7. Needs the user's questions (30–50).
F. RAG-on-PACS baseline — closes #8. Half day.
G. Cost table — closes #9. Instrumentation pass over stored runs + one timed replay.
H. RLVR exploration-vs-demonstration study — the open methodological question;
   optional for this paper, natural follow-up paper.
I. Second procurement corpus (another OCDS country) — future work, cheapest
   external-validity extension.
