# Multi-level QA generation: separating executor validation from language generalization

**Date:** 2026-07-02 · **Status:** implemented (plan bank + L1 offline; L2/L3 need one nano run)

## The problem with the current benchmark

v1/v2 questions are template-generated, and the rule planner was (inevitably) written against the
same surface conventions. Offline rule-only accuracy of 92–96% therefore measures *template
recovery*, not language understanding — a benchmark the system can pattern-match is a control, not
a result. Worse, one number conflates two different claims:

1. the deterministic executor computes correct, exhaustive, auditable answers **given the right
   plan** (the FYP's core reliability claim);
2. the planner maps **natural language** to the right plan (the generalization claim, where the
   LLM should visibly matter).

## Design: plan-first, one oracle, many surfaces

Each item is an executor-validated **plan** (constraints + operation + oracle, reused from the
accepted targeted-v2 rows — v1/v2 artifacts are not modified). Every plan then carries multiple
**surface realizations** that all share that one oracle:

| Level | Surface | What it measures | Expected profile |
|---|---|---|---|
| L0 | none (plan bank) | executor ceiling, language removed (`--mode executor`) | ~100% by construction |
| L1 | source template question | rule-planner control (it should saturate this) | rule ≈ hybrid |
| L2 | LLM paraphrase (syntax/register/order varied, atoms verbatim) | language-to-plan generalization | rule drops, hybrid holds |
| L3 | LLM adversarial (distractor preamble, indirection, no new facts) | robustness of the plan layer | rule drops further; hybrid degrades gracefully |

Because oracles are shared, `accuracy(L1) − accuracy(L2)` is attributable to language alone —
the **generalization gap**, per planner. This is the experiment that makes the LLM contribution
legible: the hybrid cascade's value is the area between the rule curve and the hybrid curve
across L2/L3.

## Generation is generate-then-validate (like everything else in the pipeline)

LLM rewrites are accepted only through the deterministic gate `check_surface`
(`src/procurement_graph/qa/multilevel.py`), with rejection sampling (`--retries`):

- **atom preservation** — every visible constraint value (org names, years, CPV codes, dates,
  thresholds, category) must appear verbatim; hidden guards (`value_is_additive`) are exempt;
- **no new numbers** — any digit string not in the plan atoms or source question rejects the
  candidate (blocks accidental new filters/answer leakage);
- **no new KG organisations** — capitalized org-shaped spans are resolved against the KG; an
  exact-resolving org outside the plan rejects (blocks alternative readings); containment
  whitelisting only applies to org-length atoms, so short atoms like `services` cannot whitelist
  foreign org names;
- **unanswerable trigger survives** — an `unsupported` item must keep its trigger phrase
  ("social value", …), so abstention items stay abstention items after paraphrase;
- **actually rewritten** — token-Jaccard vs the source > 0.9 rejects (no lazy copies at L2/L3);
- **exactly one question**, sane length.

The gate can reject a good rewrite (conservative) but cannot accept a semantics-changing one
under the modeled failure modes; residual risk (e.g. a paraphrase inverting a comparison
direction while keeping all atoms) is documented and sampled manually.

## Files

- `src/procurement_graph/qa/multilevel.py` — atoms, gate, prompts, row assembly
- `scripts/build_multilevel_qa.py` — builds `data/qa/multilevel/{plan_bank,surfaces.L*}.jsonl`
- `scripts/eval_multilevel.py` — level × planner matrix (shares scoring with `eval_targeted_v2`)
- `tests/test_qa_multilevel.py` — gate/assembly/rejection-sampling tests

## Commands

```powershell
# offline (done): 500 plans (100/subset stride) + L1 surfaces
python -B scripts\build_multilevel_qa.py

# with API key: generate + validate L2/L3 surfaces (~1000 nano calls, well under rate limits)
$env:AZURE_OPENAI_API_KEY="<key>"
python -B scripts\build_multilevel_qa.py --llm on --model gpt-5.4-nano

# the generalization-gap matrix
python -B scripts\eval_multilevel.py --mode executor                 # L0 ceiling
python -B scripts\eval_multilevel.py --planner rule_decomp           # offline rule curve
python -B scripts\eval_multilevel.py --planner hybrid                # hybrid curve (API)
python -B scripts\eval_multilevel.py --planner llm                   # LLM-only curve (API)
```

## Round 2 (2026-07-02, after the first live L2/L3 run)

Live results (user's nano run, 500 plans): rule 99→80→88, hybrid 99→83→91 (L1→L2→L3).
`scripts/analyze_generalization_gap.py` decomposes the gap per item:

| class | L2 | L3 | meaning |
|---|---|---|---|
| llm_recovered − hybrid_regressed | +12 | +10 | the cascade's net measured contribution |
| rule_misfire_unescalated | 25 | 19 | rule answered WRONG, LLM never consulted (blind spot) |
| escalated_llm_failed | 44 | 19 | LLM consulted, its plan was wrong (→ preference-log data) |

Two architecture responses (no new rule coverage — per the research principle):

- **`VerifyingHybridPlanner` (verify-then-escalate)** — *revised to a conservative policy after
  the first live run regressed L1 99→98*. Post-mortem (`scripts/analyze_escalation.py`,
  `escalation_report.json`): every L1 regression came from the aggressive `degenerate_zero`
  trigger escalating predicates whose CORRECT answer is False; `multiple_answers` escalations
  (64 calls) hit underdetermined items the LLM cannot fix. Conservative policy: accept a rule
  plan that passes deterministic verification; escalate ONLY on hard failure (grounding failure
  or `no_results` probe) plus the classic rule-`ambiguous` path; a failed LLM probe falls back to
  the rule plan. Offline recombination predicts L1 99% / L2 83% at ~half the LLM calls
  (88→41, 123→75). Honest finding: hard-verification escalation restores the control and buys
  efficiency + a per-question ledger, but does NOT beat classic hybrid on accuracy — the damaging
  rule misfires execute "successfully" with wrong semantics, so catching them needs semantic
  (not execution-level) verification. Run: `eval_multilevel.py --planner verified_hybrid`.
- **Level 4 — plan-verbalized surfaces**: questions are generated from the PLAN ALONE
  (`verbalize_messages`; the template question is withheld from the LLM), so L4 carries no
  template DNA at all. The similarity gate is deliberately off at L4 (independent generation may
  legitimately resemble a template). Unanswerable items are excluded (their unanswerability lives
  in the question concept, which a plan cannot carry).
  Build: `build_multilevel_qa.py --llm on --levels 4`.

Expected reading: rule accuracy L1 ≥ L3 ≥ L2 ≥ L4; the verified-hybrid curve should stay flat
across levels, and the flatness differential IS the thesis result.

## Dissertation framing

Report L1 as the control ("the rule layer saturates its own templates — by design"), and the
L2/L3 matrix as the result: executor reliability (L0) is decoupled from language generalization,
and the hybrid cascade's contribution is measured as the rule→hybrid gap on non-template
language. This also preempts the "your planner is overfit to your benchmark" examiner question —
the answer is "yes, on L1, deliberately; here is what happens off-template."
## Pilot Decision (2026-07-03): L2 Generator / Checker Split

Latest decision: **use `gpt-5.4-nano` as the L2 generator and
`grok-4-1-fast-non-reasoning` as the semantic checker**.

Rationale from the 100-plan pilots:

- `data/qa/multilevel_l2_pilot_v4` (`nano` generation + `grok` checking) accepted **97/100**
  plans, preserved all 20 unanswerable/ambiguous/no-results items, and produced usable L2
  coverage across all subsets. Remaining quality issues are prompt-level (too many phrases like
  "matching procurement records", "recorded in the KG", and some weak role phrasing), not a
  broken model split.
- `data/qa/multilevel_l2_pilot_grokgen_nanocheck` (`grok` generation + `nano` checking)
  accepted **82/100** and only **3/20** unanswerable items. The nano checker repeatedly treated
  `unsupported` / `no_results` rows as failures because no executable derivation exists, even
  though abstention is the reference answer/status. This makes nano unsuitable as the current
  semantic checker for abstention cases.
- Grok generation sometimes gives livelier bridge wording, but it also overuses phrases such as
  "matching procurement records" / "recorded in the KG" and can drift the answer target ("which
  buyers..." instead of count). Those gains do not offset the checker-side loss of abstention
  coverage.

Current L2 contract:

- deterministic hard gate checks numeric invariants only: years, CPV codes, dates, counts,
  thresholds, money amounts, new numbers, and single-question format;
- Grok checker handles semantic logic: same answer target, role direction, organisation reference
  (including acceptable abbreviations), comparison direction, bridge relationship, and whether the
  route can derive the reference answer/status;
- for abstention rows, `unsupported`, `ambiguous`, or `no_results` is itself the reference
  answer/status, so the checker must preserve the same reason to abstain rather than require an
  entity-valued answer.

Command for the next accepted pilot / full run:

```powershell
python -B scripts\build_multilevel_qa.py `
  --per-subset 20 `
  --llm on `
  --levels 2 `
  --model gpt-5.4-nano `
  --checker-model grok-4-1-fast-non-reasoning `
  --org-gate off `
  --progress-every 5 `
  --out-dir data\qa\multilevel_l2_pilot_v4
```

Prompt follow-up before scaling to full: tighten the generator notes to discourage "matching
procurement records", "recorded in the KG", "What number of...", and any bridge wording that adds
temporal relations such as "later", "after", or "subsequently" unless present in L1.
