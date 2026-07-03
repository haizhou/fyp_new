# LLM integration architecture + ranked innovation points (2026-07-02)

## 1. How the LLM cooperates with the deterministic KG executor

Design rule everywhere: **generate-then-validate**. The LLM proposes structure (plans, entity
choices, repair routes); a deterministic layer validates each proposal against a closed schema /
candidate list; the KG executor is the only component that produces answers. The LLM can therefore
be wrong but never hallucinate an answer — a bad proposal fails validation or fails execution.

Three touchpoints, one per pipeline phase:

| Phase | Component | LLM role | Validation boundary |
|---|---|---|---|
| Plan time | `HybridPlanner` (rule → LLM cascade) | NLU for phrasings the rule recognisers cannot map (status `ambiguous`); emits a `RuntimeQuerySpec` + optional N-hop `decomposition` | `ground_spec` schema grounding; org mentions snapped to canonical KG entities via `OrgResolver`; executor runs the spec |
| In-loop repair | `semantic_repair_spec` + `LLMCandidateSelector` | pick the intended entity among retrieval candidates when an exact filter matched nothing | choice must come from the retrieved candidate list; re-executed deterministically |
| Trace time | `TraceReflector` + `LLMTraceAdvisor` (new) | judge an *uncertain finished trace*; recommend one repair action; optionally pick an entity candidate | action must be in the closed `REPAIR_ACTIONS` set; entity choice must be in the resolver candidate list; any repair re-runs ground → execute (bounded to 1 round) |

This answers the five requested LLM roles:
1. **difficult entity matching** — plan-time resolver snapping + in-loop `LLMCandidateSelector` +
   trace-time `entity_choice` (validated against candidates).
2. **ambiguous question-type detection** — `HybridPlanner` escalates only on rule `ambiguous`;
   the trace reflector independently cross-checks question-type cues vs the executed operation.
3. **plan repair** on `no_results` / `multiple_answers` / `incomplete_evidence` — deterministic
   policy first (`re_link_entity`, `re_plan_question_type`, `relax_constraints`), LLM advisor only
   when the deterministic policy lands on abstain/ask.
4. **trace-aware reflection** — `TraceReflector.reflect_trace` verifies evidence-faithfulness
   (gold-free) and plan validity over the full trace.
5. **preference logging** — `log_preference` writes good / bad / safe_abstain plan records
   (chosen vs rejected plans) as JSONL for future fine-tuning.

## 2. Trace-aware reflector (implemented)

`src/procurement_graph/reasoning/trace_reflector.py`, wired as `ReasoningPipeline(trace_reflector=…)`.
Additive: results attach under `trace.metadata["trace_reflection"]` / `["trace_repair"]`; no
existing JSON contract changed; without the reflector the pipeline behaves exactly as before.

- **Faithfulness (answer vs its own evidence, no gold needed):** count == evidence_count; exists
  consistent with match count; sum has contributors and is ≥ its max contributor; select_unique
  uniqueness held; every decomposition hop passed. Verdict: `faithful | abstained | suspicious`.
  A `suspicious` answer is abstained on — the hallucination guard of last resort.
- **Plan validity:** question-type cues vs executed operation; entity links with confidence < 0.6;
  org constraints not grounded in the question text.
- **Targeted repair (closed set):** `re_link_entity` (next resolver candidate; for a *passed zero*
  answer only a case/punctuation **sibling variant** of the same org may be swapped in),
  `re_plan_question_type` (re-issue under the cued operation), `relax_constraints`,
  `ask_clarifying_question`, `abstain`, `mark_ambiguous` (underdetermined factoid; distinct values
  recorded). One bounded ground→execute round; the executor stays the answer authority.
- **Preference log:** JSONL rows `{label: good|bad|safe_abstain, chosen_plan, rejected_plans,
  faithfulness, action, execution_status, …}` — DPO-style pairs for planner fine-tuning.

Ablation experiment: `run_compare.py --system ours --planner rule_decomp --reflect on` and
`eval_targeted_v2.py --mode pipeline --reflect on [--reflect-advisor on]`.
Metrics: accuracy delta, hallucination count (answers with `suspicious` faithfulness), repair
success rate (`trace_repair.status == passed`), abstention precision by reason.

## 3. Ranked innovation points

Ranking = impact × feasibility ÷ risk, for this codebase specifically.

**R1. Trace-aware reflect–repair with faithfulness verification — IMPLEMENTED (above).**
Problem: a verified-by-construction executor can still be fed a wrong plan (wrong entity variant,
wrong operation) and return a confidently wrong 0 / wrong subset; nothing audited the finished
trace. Metric: ablation on/off (accuracy, hallucinations, repair success). Contract risk: none
(metadata-only).

**R2. Preference-pair logging for planner fine-tuning — IMPLEMENTED (part of R1).**
Problem: LLM planner quality is static; every eval run discards supervision. Now each trace yields
(question, chosen plan, rejected plans, outcome label). Short-term: log + curate; medium-term:
DPO/SFT a small planner model on pairs; metric: LLM-planner plan-validity rate before/after
fine-tuning on a held-out hard set. Contract risk: none (new JSONL file).

**R3. Organisation canonicalisation layer (variant clusters).**
Problem (measured): the KG stores "The Newcastle Upon/upon Tyne…" (177 vs 13 rows) and
"EDS Ltd" vs "EDS Ltd." as different entities; every eq filter silently under-counts. Current
mitigation is resolve-time (exact-surface preference, sibling re-link). Proper fix: offline
clustering pass (casefold + punctuation-normalised key, optionally company-number joins) emitting
`org_canonical_id`; runtime constraints become `in {variants}`. Feasibility: short-term for the
clustering + a *disclosed* variant-union mode; medium-term as default because **golden answers
shift** (bridge oracle 29 247 → 29 417) — needs oracle regeneration under union semantics.
Metric: entity-completeness (rows covered per mention) + bridge accuracy under union oracles.
Contract risk: HIGH if default (changes v1/v2 oracles); LOW as an opt-in disclosed mode.

**R4. Benchmark ill-posedness auditor.**
Problem (found by hand): conjunction oracles embed hidden filters (`supplier_count>=1`) absent from
the question; some factoid questions underdetermine their contract (4 NHS buyers match). Automate:
for each benchmark row, re-derive the constraint set from the question text alone (the runtime
linker), execute both, and flag rows where oracle-spec ≠ question-derived results (hidden-filter
class) or where the answer key is non-unique (underdetermined class). Feasibility: short-term
(one script over the shared executor). Metric: % benchmark items flagged, with manual audit of a
sample; cleaner benchmark = more credible headline numbers. Contract risk: none (report only).

**R5. Calibrated selective escalation (rule → LLM cost model).**
Problem: hybrid escalates on `ambiguous` only; rule misfires with high confidence never reach the
LLM (the count-101 case), and every escalation costs tokens. Learn a small confidence calibrator
(features: recogniser id, link confidences, constraint count, evidence_count) to decide
escalate/trust; the trace reflector's plan-validity signal supplies training labels for free.
Feasibility: medium-term. Metric: accuracy at fixed LLM-call budget (cost–accuracy curve).
Contract risk: none.

**R6. Answer cards with faithfulness-scored natural-language explanations.**
Problem: answers are scalars + limitations; the dissertation demo would benefit from grounded
explanations. Use the existing `LLMVerbalizer` + `answer_preserves_atoms` to generate an
explanation whose atoms (numbers, names) must all appear in the evidence bundle; reject otherwise.
Feasibility: short-term (components exist). Metric: atom-preservation rate; human readability
rating. Contract risk: none (new field).

**Deliberately not pursued:** more regex recognisers (diminishing returns — remaining offline gaps
are LLM-shaped phrasings like "the contract between X and Y"), unbounded graph traversal (rejected
by design), and letting any LLM output reach the user unvalidated.
