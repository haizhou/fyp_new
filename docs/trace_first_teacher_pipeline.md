# Trace-First Closed-Loop Teacher Pipeline

This document defines the offline teacher/data-generation pipeline used before any Qwen fine-tuning.
The goal is not to treat nano as the final model or as a source of gold labels. The goal is to use an
LLM planner inside a deterministic KG-verification loop, then keep only verifier-accepted artifacts
for baseline analysis, SFT data, DPO pairs, and error analysis.

## Core Definition

The system is a **trace-first closed-loop teacher pipeline**:

```text
Planner -> Grounding -> Schema Check -> Executor -> Verifier
if rejected:
  Reflector sees structured trace summary -> repaired plan
  -> Grounding -> Schema Check -> Executor -> Verifier again
```

The reflector is not the whole loop. It is the repair component inside a verifier-guided closed loop.

Recommended paper wording:

> The verifier judges each attempt, while the reflector proposes the next attempt. The reflector never
> has access to the hidden reference answer, and every repaired plan must be re-executed and re-verified
> before being accepted.

## Roles

### Planner

The planner proposes a structured plan from a QA question.

It may be `gpt-5.4-nano` for the first teacher run. The planner does not answer the question directly.
Its output is a candidate plan, not a gold plan.

### Grounding / Schema Check / Executor

These deterministic components convert a candidate plan into an executable runtime spec, validate fields
and constraints, execute against the KG, and produce a submitted answer or a structured failure.

### Verifier

The verifier is the judge. It does not repair plans.

Allowed verifier outputs:

```json
{
  "accepted": false,
  "final_verdict": "rejected",
  "failure_stage": "schema|grounding|executor|verifier|answer_sanity|postflight",
  "failure_reason": "wrong_answer|schema_error|no_results|multiple_answers|unsupported_operation|unsafe_sum|...",
  "answer_correct": false,
  "submitted_answer": 42,
  "repair_hints": ["missing_year_constraint", "possible_role_flip"]
}
```

`repair_hints` must be rule-like labels only. The verifier must not synthesize a new plan.

In offline data-construction mode the verifier may use hidden reference answers internally. However, the
reflector must not receive the reference answer. Public/LLM-facing fields should use `answer_correct`,
`accepted`, or `final_verdict`; `oracle_match` should remain an internal implementation detail if used.

### Reflector

The reflector is the repair module. It sees a compact structured trace summary and proposes the next
attempt.

Allowed reflector outputs:

```json
{
  "diagnosis": "The plan constrained supplier_name, but the question asks who published the notices.",
  "repair_action": "repair_plan|swap_buyer_supplier|change_operation|relax_constraint|abstain|no_repair",
  "repaired_plan": {},
  "abstain": false,
  "unchanged_fields": ["release_year"],
  "changed_fields": ["supplier_name -> buyer_name"]
}
```

A repaired plan is never accepted because the reflector says it is fixed. It must go back through grounding,
schema checks, execution, and verification.

## Trace Separation

Save both complete logs and compact reflector inputs.

### Full Trace

`full_traces.jsonl` is for audit, debugging, data construction, and case studies. It may contain:

- question id and question text
- planner model and raw LLM output
- parsed plan
- grounding input/output, changes, and issues
- schema/preflight checks
- executor status, submitted answer, evidence counts, failed checks
- verifier result, including hidden-reference correctness when running offline
- answer sanity and postflight checks
- deterministic reflector action
- LLM reflector raw output and parsed repair
- repair execution and verification result

### Reflector Trace Summary

`reflector_inputs.jsonl` is the only trace format sent to the LLM reflector. It should be compact and
structured:

```json
{
  "question_id": "...",
  "question": "...",
  "failed_plan": {},
  "failure_stage": "verifier",
  "grounding_issues": [],
  "schema_errors": [],
  "executor_status": "passed",
  "submitted_answer": 42,
  "verifier_feedback": {
    "accepted": false,
    "final_verdict": "rejected",
    "failure_reason": "wrong_answer",
    "answer_correct": false,
    "repair_hints": ["operation_mismatch"]
  },
  "answer_sanity": {
    "ok": true,
    "flags": []
  },
  "allowed_repair_actions": [
    "repair_plan",
    "swap_buyer_supplier",
    "change_operation",
    "relax_constraint",
    "abstain"
  ]
}
```

Do not include the hidden reference answer in this summary.

Do not send long raw evidence dumps, raw KG rows, or noisy debug logs to the LLM reflector. Those belong in
`full_traces.jsonl`.

## Attempt Protocol (budget)

```text
Attempt 0: initial plan
Attempt 1: repair after feedback 0
Attempt 2: repair after feedback 1
Attempt 3: repair after feedback 2   (teacher default: up to 3 repairs; cost-saving: 2)
```

Rules:

1. **Every repair re-runs the full grounding -> schema check -> executor -> verifier path.** The
   reflector never certifies its own fix.
2. Each repair is driven by the feedback of the LATEST failed attempt (not the initial one).
3. Stop at the first verifier-accepted attempt.
4. Runtime knob: `ReasoningPipeline.max_feedback_replans` (runtime default 1; teacher runs 2-3).
   Per-attempt records land in `metadata.feedback_replan.attempts` with `first_verified_attempt`,
   which is exactly what Repair@k metrics and DPO pairing read.

Paper wording:

> We allow up to three verifier-guided repair attempts. After each repair, the revised plan is
> re-grounded, re-executed, and re-verified. Successful repairs produce SFT targets and DPO chosen
> responses, while failed attempts serve as rejected responses when paired with a later verified plan.

## Data Collection by Outcome

| Outcome | Artifacts written |
|---|---|
| Attempt 0 accepted | `verified_plans`: SFT `question -> verified plan` |
| Repair k accepted | `verified_plans` + `repair_sft`: `question + failed_plan(k-1) + feedback(k-1) -> plan(k)` + `dpo_pairs` |
| Some accepted, some failed | `dpo_pairs` (see pairing rules) |
| All attempts failed | `failures` only — **never a DPO pair on its own** |

**DPO pairing rules** (a pair is always same-question):

- **Primary pair**: `chosen = first verifier-accepted attempt (k)`, `rejected = attempt (k-1)` —
  the nearest failure. This trains exactly the "repair according to verifier feedback" behaviour.
- Optional weaker pair: `chosen = attempt k`, `rejected = attempt 0` (use with lower weight).
- An all-failed question enters `dpo_pairs` only if a later verified plan for the SAME question
  exists (stronger teacher, oracle plan, or another prompt round supplies the chosen side).
- Chosen must satisfy the full accept bar (schema + faithful + executor correct); normalize
  serialization and record `length_delta` per the training plan.

## Loop Metrics

```text
First-pass verified accuracy      attempt-0 acceptance rate
Repair@1 / Repair@2 / Repair@3    share first accepted at attempt k
Final verified accuracy           acceptance within the full budget
Average attempts per solved       cost/efficiency
Failure-after-budget taxonomy     which question types stay unsolved (by failure_stage/reason)
```

## Teacher Mix (optional)

Step-1 understanding and Step-2 structuring may use DIFFERENT models
(`TypedLLMPlanner.understanding_client/understanding_model` vs `client/model`), e.g. nano reads
the question while a stricter-JSON model fills the type shell. Two invariants keep this safe:

1. Label authority is the executor/verifier — the teacher mix never defines correctness.
2. Every artifact records provenance: `teacher: {"understanding": ..., "plan": ..., "repair": ...}`
   so mixes can be ablated or filtered later.

Pick the mix empirically on `dev_smoke` by comparing shape-failure rates (nesting echo,
placeholder slots, operation alternation) — the L2 probe showed output SHAPE, not extraction,
is the small-model bottleneck.

## Data Products

Recommended output files:

```text
full_traces.jsonl
reflector_inputs.jsonl
reflector_outputs.jsonl
verified_plans.jsonl
repair_sft.jsonl
dpo_pairs.jsonl
failures.jsonl
summary.json
```

Definitions:

- `verified_plans.jsonl`: accepted initial or repaired plans. These are verifier-accepted plans, not gold plans.
- `repair_sft.jsonl`: examples of `question + failed_plan + structured_feedback -> repaired verified plan`.
- `dpo_pairs.jsonl`: `chosen = verifier-accepted plan`, `rejected = failed/non-executable/wrong-answer plan`.
- `failures.jsonl`: attempts still rejected after the bounded repair loop, for error analysis.

## First Experiment

Use the current split:

```text
data/qa/cicada_merged_l1_l2_trainbalanced_v1/dev_smoke.jsonl
```

Initial goal:

1. Run nano as planner on `dev_smoke`.
2. Execute and verify every candidate plan.
3. For rejected attempts, create compact reflector inputs.
4. Run nano as reflector once.
5. Re-run deterministic grounding/executor/verifier on repaired plans.
6. Write all trace/data-product JSONL files.
7. Report:
   - initial accepted rate
   - repaired accepted rate
   - final accepted rate
   - rejection taxonomy by failure stage/reason
   - counts of SFT initial examples, repair SFT examples, DPO pairs, unrepaired failures

Use `dev_smoke` only to validate the loop and logging format. Do not tune against final test.
