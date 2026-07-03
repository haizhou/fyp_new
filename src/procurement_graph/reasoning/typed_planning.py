"""Typed planning: the LLM fills a question-type slot schema; deterministic layers verify it.

Architecture shift (2026-07-03): the rule recognisers stop being the primary understander. The
supported question types become a reasoning DSL — the LLM (1) classifies the question into one of
the types below and (2) fills that type's slots. Two deterministic layers then take over:

  Plan Consistency Check (this module): is the typed plan FAITHFUL TO THE QUESTION TEXT?
    years/CPVs/thresholds in the plan must appear in the question; none may be invented;
    comparison direction words must match; buyer/supplier role phrasing must match the role the
    plan constrains; the operation must belong to the declared type. This catches the class the
    execution probe provably cannot: plans that run "successfully" with wrong semantics.

  Compile + grounding: slots become a RuntimeQuerySpec / DecompositionPlan (org surfaces resolved
  against the KG; structured errors like multiple_entity_candidates are surfaced, not guessed).
  The shared executor remains the sole answer authority.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from .decomposition import Binding, DecompositionPlan, SubQuery
from .models import CandidatePlan, QueryConstraint, RuntimeQuerySpec

QUESTION_TYPES: dict[str, dict[str, Any]] = {
    "factoid":      {"operations": ("select_unique",), "slots": ("answer_field", "constraints")},
    "count":        {"operations": ("count",), "slots": ("constraints",)},
    "sum":          {"operations": ("sum",), "slots": ("constraints",)},
    "boolean":      {"operations": ("exists", "predicate"), "slots": ("constraints", "comparison")},
    "comparison":   {"operations": ("predicate",), "slots": ("constraints", "comparison")},
    "min_max":      {"operations": ("argmax", "argmin"), "slots": ("constraints",)},
    "top_k":        {"operations": ("top_k",), "slots": ("constraints", "group_by", "k", "metric")},
    "set":          {"operations": ("distinct_set",), "slots": ("answer_field", "constraints")},
    "compare_two":  {"operations": ("compare",), "slots": ("left", "right", "constraints")},
    "bridge_join":  {"operations": ("count", "sum"), "slots": ("steps",)},
    "unanswerable": {"operations": (), "slots": ("reason",)},
}
_SLOT_FIELDS = {"buyer": "buyer_name", "supplier": "supplier_name", "year": "release_year",
                "cpv": "tender_cpv_id", "category": "tender_category", "title": "tender_title",
                "date": "award_date_signed"}
_ROLE_CUES = {"buyer_name": ("awarded by", "published by", "buyer", "contracting authority", "by "),
              "supplier_name": ("awarded to", "supplier", "won by", "contract with", "vendor")}
_GT_WORDS = ("more than", "above", "over", "greater", "exceed", "at least")
_LT_WORDS = ("less than", "below", "under", "fewer", "at most", "no more than")
_NUM = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])")


def question_understanding_messages(question: str) -> tuple[str, str]:
    """Step 1 prompt: understand the question before filling compiler slots."""
    system = (
        "You are helping a UK public-procurement QA system understand a user's question. "
        "Do not answer the question and do not compile it into database fields yet. First explain "
        "what kind of question it is, what the final answer should return, what information is "
        "explicitly known from the wording, and the reasoning chain needed to get the answer. "
        "Return a compact JSON object so the next planning step can read it."
    )
    user = json.dumps({
        "task": "Step 1: understand the procurement question.",
        "question": question,
        "question_types": list(QUESTION_TYPES),
        "return_fields": {
            "question_type": "best high-level type, e.g. count, sum, factoid, set, compare_two, bridge_join, unanswerable",
            "needs_to_return": "plain-English description of what the final answer should be",
            "known_information": [
                "entities, roles, years, CPV codes, categories, titles, dates, amounts, thresholds explicitly stated"
            ],
            "reasoning_chain": [
                "ordered natural-language steps needed to answer, including joins/comparisons if any"
            ],
            "missing_or_unsupported_information": "why the system should abstain, if applicable; otherwise null",
            "notes_for_planner": "pitfalls such as buyer/supplier role direction or comparison direction"
        },
        "notes": [
            "Copy entity names, years, CPV codes, dates, amounts, and titles from the question when mentioning them.",
            "Do not invent filters or entities.",
            "If the question asks for an unsupported concept such as fairness, payment terms, social value, delivery performance, or bidder counts, mark it as unanswerable and explain why.",
            "If a bridge is needed, describe the intermediate set in words, e.g. suppliers who also worked with buyer X, then count/sum over their contracts.",
            "If two organisations are compared, state the left side, right side, metric, and comparison direction."
        ],
    }, ensure_ascii=False)
    return system, user


def typed_plan_messages(question: str, understanding: dict[str, Any] | None = None) -> tuple[str, str]:
    system = ("You map UK public-procurement questions onto a fixed reasoning DSL. First decide "
              "question_type, then fill ONLY that type's slots with surfaces copied verbatim from "
              "the question. Never invent filters, never answer. Return strict JSON.")
    user = json.dumps({
        "question": question,
        "step1_understanding": understanding,
        "question_types": {t: spec["slots"] for t, spec in QUESTION_TYPES.items()},
        "output_schema": {
            "question_type": "one of question_types",
            "operation": "the concrete operation for that type",
            "answer_field": "KG-ish field name or role word (buyer/supplier/category/value/date)",
            "constraints": [{"slot": "buyer|supplier|year|cpv|category|title|date",
                             "surface": "verbatim question text", "value": "normalised value"}],
            "comparison": {"operator": "> | < | after | before | =", "threshold": "number or ISO date"},
            "steps": "bridge only: [{step, action, constraints, from_step}]",
            "left/right": "compare_two only: the two organisation surfaces",
        },
    }, ensure_ascii=False)
    return system, user


@dataclass(frozen=True)
class ConsistencyVerdict:
    ok: bool
    issues: tuple[str, ...] = ()


def plan_consistency_check(question: str, payload: dict[str, Any]) -> ConsistencyVerdict:
    """Deterministic surface-fidelity check: question -> typed plan, BEFORE any KG access."""
    issues: list[str] = []
    low = " ".join(question.casefold().split())
    qtype = str(payload.get("question_type", ""))
    spec = QUESTION_TYPES.get(qtype)
    if spec is None:
        return ConsistencyVerdict(False, (f"unknown_question_type:{qtype}",))
    operation = str(payload.get("operation", ""))
    if spec["operations"] and operation not in spec["operations"]:
        issues.append(f"operation_outside_type:{operation}!in:{qtype}")

    question_numbers = set(_NUM.findall(low.replace(",", "")))
    for constraint in payload.get("constraints") or ():
        surface = str(constraint.get("surface", "") or "")
        value = constraint.get("value")
        if surface and surface.casefold() not in low:
            issues.append(f"surface_not_in_question:{surface[:40]}")
        for number in _NUM.findall(str(value)):
            if number not in question_numbers and number.lstrip("0") not in question_numbers:
                issues.append(f"invented_number:{number}")
        slot = str(constraint.get("slot", ""))
        field = _SLOT_FIELDS.get(slot, slot)
        if field in _ROLE_CUES and surface:
            span = low.find(surface.casefold())
            window = low[max(0, span - 40): span]
            other = "supplier_name" if field == "buyer_name" else "buyer_name"
            if any(cue in window for cue in _ROLE_CUES[other]) \
                    and not any(cue in window for cue in _ROLE_CUES[field]):
                issues.append(f"role_flipped:{slot}:{surface[:30]}")

    comparison = payload.get("comparison") or {}
    if comparison:
        op = str(comparison.get("operator", ""))
        if op in {">", "after", ">="} and not any(w in low for w in _GT_WORDS + ("after",)):
            issues.append("comparison_direction_gt_unsupported_by_text")
        if op in {"<", "before", "<="} and not any(w in low for w in _LT_WORDS + ("before",)):
            issues.append("comparison_direction_lt_unsupported_by_text")
        for number in _NUM.findall(str(comparison.get("threshold", "")).replace(",", "")):
            scaled = {number, number.rstrip("0") or number}  # "1000000" may appear as "1" + "million"/"m"
            if not (scaled & question_numbers):
                issues.append(f"invented_threshold:{number}")
    return ConsistencyVerdict(not issues, tuple(issues))


def compile_typed_plan(question: str, payload: dict[str, Any], *,
                       org_resolver: Any = None) -> CandidatePlan:
    """Typed plan -> executable candidate. Entity surfaces are grounded via the resolver;
    unresolvable or ambiguous entities produce a structured non-planned candidate, never a guess."""
    qtype = str(payload.get("question_type", ""))
    operation = str(payload.get("operation", "")) or (QUESTION_TYPES.get(qtype, {}).get("operations") or ("",))[0]
    spec_id = "typed_" + re.sub(r"\W+", "", qtype)[:24]
    if qtype == "unanswerable":
        spec = _spec(question, spec_id, "unsupported", (), intent="unsupported")
        return CandidatePlan(plan_id=f"{spec_id}:p0", query_spec=spec, status="unsupported",
                             rationale=str(payload.get("reason", "typed as unanswerable")),
                             planner_source="typed_llm")

    constraints: list[QueryConstraint] = []
    for item in payload.get("constraints") or ():
        field = _SLOT_FIELDS.get(str(item.get("slot", "")), str(item.get("slot", "")))
        value = item.get("value", item.get("surface"))
        if field in ("buyer_name", "supplier_name") and org_resolver is not None:
            hits = org_resolver.resolve(str(value))
            if not hits:
                return _not_planned(question, spec_id, f"entity_not_found:{value}")
            exact = [h for h in hits if h.source.startswith("records_")and h.score >= 0.85]
            if not exact and len(hits) > 1:
                return _not_planned(question, spec_id, "multiple_entity_candidates:"
                                    + ";".join(h.linked_id for h in hits[:4]))
            value = (exact or hits)[0].linked_id
        if field == "release_year":
            value = int(str(value)[:4]) if str(value)[:4].isdigit() else value
        constraints.append(QueryConstraint(field, "eq", value, source_text=str(item.get("surface", ""))))

    metadata: dict[str, Any] = {}
    answer_field, value_type = "", "string"
    comparison = payload.get("comparison") or {}
    if qtype in ("boolean", "comparison") and comparison and operation == "predicate":
        op_map = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte", "=": "eq",
                  "after": "after", "before": "before"}
        metadata = {"predicate_subject": "sum" if "value" in str(payload.get("answer_field", "")) or comparison.get("of") == "sum" else "field",
                    "comparator": op_map.get(str(comparison.get("operator")), "gt"),
                    "threshold": comparison.get("threshold"),
                    "threshold_type": "date" if str(comparison.get("operator")) in ("after", "before") else "number"}
        if metadata["predicate_subject"] == "sum":
            answer_field = "value_amount"
        else:
            metadata["predicate_field"] = _SLOT_FIELDS.get(str(payload.get("answer_field", "")),
                                                           str(payload.get("answer_field", "")) or "award_date_signed")
        value_type = "boolean"
    elif operation in ("select_unique", "distinct_set"):
        answer_field = _SLOT_FIELDS.get(str(payload.get("answer_field", "")),
                                        str(payload.get("answer_field", "")) or "supplier_name")
    elif operation in ("argmax", "argmin"):
        answer_field = "contract_node_id"
    elif operation == "top_k":
        metadata = {"group_by_field": _SLOT_FIELDS.get(str(payload.get("group_by", "")), "buyer_name"),
                    "metric": str(payload.get("metric", "count")), "k": int(payload.get("k", 3)),
                    "metric_field": "value_amount"}
    elif operation == "sum":
        answer_field, value_type = "value_amount", "currency"

    decomposition = None
    if qtype == "compare_two":
        left, right = str(payload.get("left", "")), str(payload.get("right", ""))
        year = tuple(c for c in constraints if c.field == "release_year")
        steps = tuple(SubQuery(sid, _spec(question, spec_id, "count",
                                          year + (QueryConstraint("buyer_name", "eq", name),)))
                      for sid, name in (("a", left), ("b", right)))
        decomposition = DecompositionPlan(spec_id, question, steps, combine="compare_gt",
                                          final_steps=("a", "b"))
        operation = "compare"
    elif qtype == "bridge_join" and payload.get("steps"):
        decomposition = _bridge_from_steps(question, spec_id, payload["steps"], constraints, operation)
        if decomposition is None:
            return _not_planned(question, spec_id, "bridge_steps_uncompilable")

    spec = _spec(question, spec_id, operation, tuple(constraints), answer_field=answer_field,
                 value_type=value_type, intent=qtype, metadata=metadata,
                 sort_field="value_amount" if operation in ("argmax", "argmin") else "")
    return CandidatePlan(plan_id=f"{spec_id}:p0", query_spec=spec, status="planned", confidence=0.7,
                         rationale=f"typed plan: {qtype}/{operation}", planner_source="typed_llm",
                         decomposition=decomposition)


def _bridge_from_steps(question, spec_id, steps, extra, operation):
    """Compile a bridge skeleton (find_X -> find_Y(from_step) -> aggregate) into a DecompositionPlan."""
    try:
        entity_steps = [s for s in steps if not any(c.get("from_step") for c in s.get("constraints", ()))]
        hop2 = next(s for s in steps if any(c.get("from_step") for c in s.get("constraints", ())))
    except StopIteration:
        return None
    if not entity_steps:
        return None
    anchor = entity_steps[0]
    anchor_cons = tuple(QueryConstraint(_SLOT_FIELDS.get(str(c.get("slot", "")), str(c.get("slot", ""))),
                                        "eq", c.get("value", c.get("surface")))
                        for c in anchor.get("constraints", ()) if not c.get("from_step"))
    emit = _SLOT_FIELDS.get(str(hop2.get("bind_slot", "buyer")),
                            "buyer_name" if any(c.field == "supplier_name" for c in anchor_cons) else "supplier_name")
    h1 = SubQuery("h1", _spec(question, spec_id, "count", anchor_cons), kind="entity_set", emit_field=emit)
    hop2_cons = tuple(QueryConstraint(_SLOT_FIELDS.get(str(c.get("slot", "")), str(c.get("slot", ""))),
                                      "eq", c.get("value", c.get("surface")))
                      for c in hop2.get("constraints", ()) if not c.get("from_step")) + tuple(extra)
    op_spec = _spec(question, spec_id, operation, hop2_cons,
                    answer_field="value_amount" if operation == "sum" else "",
                    value_type="currency" if operation == "sum" else "string")
    ans = SubQuery("ans", op_spec, binds=(Binding("h1", emit),))
    return DecompositionPlan(spec_id, question, (h1, ans), final_steps=("ans",))


def _spec(question, spec_id, op, constraints, *, answer_field="", value_type="string",
          intent="typed", metadata=None, sort_field=""):
    return RuntimeQuerySpec(spec_id=spec_id, question=question, intent=intent,
                            constraints=tuple(constraints), answer_operation=op,
                            answer_field=answer_field, answer_value_type=value_type,
                            sort_field=sort_field, dedupe_key="contract_node_id",
                            requires_exhaustive_retrieval=True, metadata=metadata or {},
                            planner_version="typed_planner_v1")


def _not_planned(question, spec_id, reason):
    spec = _spec(question, spec_id, "count", (), intent="ambiguous")
    return CandidatePlan(plan_id=f"{spec_id}:p0", query_spec=spec, status="ambiguous",
                         rationale=reason, planner_source="typed_llm",
                         warnings=(reason,))


@dataclass
class TypedLLMPlanner:
    """LLM fills the DSL; deterministic consistency check + compile gate the result.

    A consistency failure or grounding ambiguity yields a structured non-planned candidate whose
    rationale carries the mismatch reason — the reflector's repair signal, never a silent guess.
    """

    client: Any
    model: str
    org_resolver: Any = None

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        u_system, u_user = question_understanding_messages(question)
        try:
            understanding = getattr(self.client.complete_json(model=self.model, system=u_system, user=u_user),
                                    "parsed", None)
        except Exception as exc:  # pragma: no cover - live boundary
            return (_not_planned(question, "typed_error", f"llm_understanding_error:{exc!r}"),)
        if not isinstance(understanding, dict):
            return (_not_planned(question, "typed_error", "unparseable_question_understanding"),)

        system, user = typed_plan_messages(question, understanding=understanding)
        try:
            payload = getattr(self.client.complete_json(model=self.model, system=system, user=user),
                              "parsed", None)
        except Exception as exc:  # pragma: no cover - live boundary
            return (_not_planned(question, "typed_error", f"llm_error:{exc!r}"),)
        if not isinstance(payload, dict):
            return (_not_planned(question, "typed_error", "unparseable_typed_plan"),)
        verdict = plan_consistency_check(question, payload)
        if not verdict.ok:
            return (_not_planned(question, "typed_inconsistent",
                                 "plan_semantic_mismatch:" + ";".join(verdict.issues[:4])),)
        candidate = compile_typed_plan(question, payload, org_resolver=self.org_resolver)
        return (replace(candidate, raw_response={"understanding": understanding, "typed_plan": payload}),)


__all__ = ["QUESTION_TYPES", "ConsistencyVerdict", "TypedLLMPlanner", "compile_typed_plan",
           "plan_consistency_check", "question_understanding_messages", "typed_plan_messages"]
