"""Deterministic repair routing (design Stage 7).

The reflector reads a failed or low-confidence artifact and chooses a repair route. It never
invents an answer and never mutates prior stages; it only proposes the next action, optionally
with a concrete `suggested_constraints` set the orchestrator can apply for a bounded retry.

Diagnosis is deterministic first (this module). An LLM reflection step can be layered on top
later for genuinely open-ended failures, but the common procurement failures — over-specific
filters, non-additive sums, contradictory constraints — are repaired by rule here.
"""

from __future__ import annotations

from .models import CandidatePlan, ExecutionResult, QueryConstraint, ReflectionAction

# Higher rank = more specific = dropped first when broadening an over-constrained query.
_SPECIFICITY: dict[str, int] = {
    "tender_cpv_id": 5,
    "has_award_signed_date": 4,
    "value_source": 3,
    "tender_category": 2,
    "release_year": 1,
}
_ANSWER_ANCHORS = frozenset({"contract_node_id", "ocid", "award_id"})


def reflect_plan(plan: CandidatePlan | None) -> ReflectionAction:
    """Reflect on a planner outcome before execution (unsupported / ambiguous)."""
    if plan is None:
        return ReflectionAction("ask_clarifying_question", "the planner returned no candidate",
                                message_to_user="Could you rephrase or add more detail to your question?")
    if plan.status == "unsupported" or plan.query_spec.intent == "unsupported":
        return ReflectionAction("mark_unsupported", plan.rationale or "the question is not answerable from KG v0.1",
                                message_to_user=plan.rationale or "This question needs data outside the current knowledge graph.")
    if plan.status == "ambiguous" or plan.query_spec.intent == "ambiguous":
        return ReflectionAction("ask_clarifying_question", plan.rationale or "the question is ambiguous",
                                message_to_user="Your question is ambiguous. Which contracts, year, or category did you mean?")
    return ReflectionAction("no_repair_needed", "plan is executable")


def reflect(result: ExecutionResult, plan: CandidatePlan, *, rounds_used: int = 0, max_rounds: int = 3) -> ReflectionAction:
    """Reflect on a deterministic execution result and choose the next action."""
    spec = result.query_spec
    if result.passed:
        return ReflectionAction("no_repair_needed", "execution passed verification")

    budget_left = rounds_used < max_rounds - 1

    if result.status == "incomplete_evidence":
        # Non-additive rows in a sum: retry with the additive-value guard if not already present.
        if spec.answer_operation == "sum" and not _has_additive_guard(spec):
            return ReflectionAction(
                "replan_query",
                "sum matched non-additive value rows; restrict to additive award/contract values",
                rollback_to="query_spec",
                suggested_constraints=spec.constraints + (
                    QueryConstraint("value_is_additive", "eq", True, visible_to_user=False,
                                    source_text="additive value safety"),
                ),
            )
        return ReflectionAction("report_insufficient_evidence",
                                "the matching evidence cannot be safely aggregated as requested")

    if result.status == "no_results":
        relaxed = _relaxed_constraints(spec.constraints)
        if relaxed is not None and budget_left:
            dropped = _dropped_field(spec.constraints, relaxed)
            return ReflectionAction(
                "relax_non_answer_constraints",
                f"no matches; relaxing the most specific non-answer filter ({dropped})",
                rollback_to="query_spec",
                suggested_constraints=relaxed,
                metadata={"dropped_field": dropped},
            )
        return ReflectionAction("report_insufficient_evidence", "no matching contracts in the KG for this question")

    if result.status == "multiple_answers":
        return ReflectionAction(
            "ask_clarifying_question",
            "the question does not uniquely identify one contract",
            message_to_user="This matches several contracts. Can you specify the buyer, supplier, year, or CPV?",
        )

    if result.status == "constraint_conflict":
        deduped = _dedupe_eq_conflicts(spec.constraints)
        if deduped is not None and budget_left:
            return ReflectionAction("replan_query", "contradictory constraints; keeping the first value per field",
                                    rollback_to="query_spec", suggested_constraints=deduped)
        return ReflectionAction("ask_clarifying_question", "the question contains contradictory constraints",
                                message_to_user="Your question seems to ask for two conflicting values. Which did you mean?")

    if result.status in {"schema_error", "unsupported_operation"}:
        return ReflectionAction("mark_unsupported",
                                f"execution status {result.status} is outside KG v0.1 runtime support")

    if result.status == "error":
        return ReflectionAction("escalate_for_manual_review", result.error or "unexpected execution error")

    return ReflectionAction("report_insufficient_evidence", f"unhandled execution status: {result.status}")


def _has_additive_guard(spec) -> bool:
    return any(c.field == "value_is_additive" and c.op == "eq" and bool(c.value) for c in spec.constraints)


def _relaxed_constraints(constraints: tuple[QueryConstraint, ...]) -> tuple[QueryConstraint, ...] | None:
    """Drop the single most specific non-answer, user-visible constraint. None if nothing droppable."""
    droppable = [c for c in constraints if c.visible_to_user and c.field not in _ANSWER_ANCHORS]
    if len(droppable) <= 1:
        return None
    target = max(droppable, key=lambda c: _SPECIFICITY.get(c.field, 0))
    return tuple(c for c in constraints if c is not target)


def _dropped_field(before: tuple[QueryConstraint, ...], after: tuple[QueryConstraint, ...]) -> str:
    after_ids = {id(c) for c in after}
    for c in before:
        if id(c) not in after_ids:
            return c.field
    return ""


def _dedupe_eq_conflicts(constraints: tuple[QueryConstraint, ...]) -> tuple[QueryConstraint, ...] | None:
    seen_eq: set[str] = set()
    kept: list[QueryConstraint] = []
    changed = False
    for c in constraints:
        if c.op == "eq" and c.field in seen_eq:
            changed = True
            continue
        if c.op == "eq":
            seen_eq.add(c.field)
        kept.append(c)
    return tuple(kept) if changed else None


__all__ = ["reflect", "reflect_plan"]
