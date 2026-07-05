"""LLM planner adapter (design Stage 1).

Wraps any chat client exposing ``complete_json(model, system, user) -> result.parsed`` (the
Stage 2 ``ChatClient`` fits) to turn a question into schema-validated `CandidatePlan`s. The LLM
is a planner only — it emits an executable query spec; the deterministic executor remains the
answer authority. On any parse/schema failure it falls back to the rule-based planner so the
runtime always returns a candidate. Kept out of the package ``__init__`` so importing the core
reasoning logic needs no LLM/KG dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .linking import UNSUPPORTED_TERMS
from .entity_resolution import resolve_confident_org
from .models import CandidatePlan
from .planner import (
    PLANNER_SCHEMA_VERSION,
    PlanningError,
    RuleBasedDryRunPlanner,
    candidate_from_payload,
    planner_output_schema,
)

# Fields the planner is allowed to constrain on (frozen KG v0.1, question-facing only).
PLANNER_FIELD_VOCABULARY: dict[str, str] = {
    "contract_node_id": "exact internal contract id (only when the user quotes one)",
    "buyer_name": "contracting authority / buyer name",
    "supplier_name": "awarded supplier name",
    "release_year": "publication year (2022-2026, integer)",
    "tender_title": "exact tender/notice title when the user quotes or clearly names a specific notice",
    "award_title": "exact award or lot title when the user quotes or clearly names a specific award/lot",
    "tender_category": "goods | services | works",
    "tender_cpv_id": "8-digit CPV classification code",
    "tender_cpv_description": "CPV label/description, only when the user gives the exact CPV wording",
    "has_award_signed_date": "boolean: the award has a signed date on record",
    "value_is_additive": "internal guard for sums (award/contract values only); set by the planner, not the user",
}

# Fields that may appear as `answer_field` (what a factoid/sum returns). Naming the exact column
# here is "schema linking" (text-to-SQL DIN-SQL/RESDSQL): it stops a small model inventing a
# variant like `award_signed_date` or `total_contract_value` for a real column.
PLANNER_ANSWER_FIELDS: dict[str, str] = {
    "buyer_name": "the buyer / contracting authority",
    "supplier_name": "the awarded supplier",
    "tender_category": "goods | services | works",
    "tender_cpv_id": "the 8-digit CPV code",
    "tender_cpv_description": "the CPV label/description",
    "release_year": "the publication year",
    "award_date_signed": "the date the award/contract was signed (USE THIS EXACT NAME for a 'signed date')",
    "value_amount": "the contract/award money value in GBP (also the answer_field for every sum)",
    "value_source": "provenance of the value figure (answer only; NEVER a filter)",
    "has_award_signed_date": "whether a signed date exists",
}


class ChatLike(Protocol):
    def complete_json(self, *, model: str, system: str, user: str) -> Any:
        ...


# --- LLM decomposition -> DecompositionPlan (LLM proposes shape+mentions; engine grounds+executes) --

_ORG_FIELDS = frozenset({"buyer_name", "supplier_name"})


def _resolve_constraint(item: dict[str, Any], org_resolver: Any):
    from .models import QueryConstraint

    field = str(item["field"])
    op = item["op"]
    value = item.get("value")
    # snap an org MENTION to the canonical KG name; the LLM need not know exact strings.
    if org_resolver is not None and field in _ORG_FIELDS and op == "eq" and isinstance(value, str):
        resolved = resolve_confident_org(org_resolver, value)
        if resolved.ok and resolved.hit is not None:
            value = resolved.hit.linked_id
    return QueryConstraint(field=field, op=op, value=value)


def decomposition_from_payload(question: str, plan_payload: dict[str, Any], org_resolver: Any = None):
    """Build a DecompositionPlan from an LLM plan's optional `decomposition` block. None if absent."""
    from .decomposition import Binding, DecompositionPlan, SubQuery
    from .models import RuntimeQuerySpec

    block = plan_payload.get("decomposition")
    if not isinstance(block, dict) or not isinstance(block.get("steps"), list) or not block["steps"]:
        return None

    def _spec(op: str, constraints, answer_field: str = "", value_type: str = "string") -> RuntimeQuerySpec:
        return RuntimeQuerySpec(spec_id=_sid(question, op), question=question, intent="decomposition",
                                constraints=tuple(constraints), answer_operation=op, answer_field=answer_field,
                                answer_value_type=value_type, dedupe_key="contract_node_id",
                                requires_exhaustive_retrieval=True)

    steps = []
    for raw in block["steps"]:
        constraints = [_resolve_constraint(c, org_resolver) for c in raw.get("constraints", [])]
        binds = tuple(Binding(str(b["from_step"]), str(b["into_field"]), str(b.get("op", "in")))
                      for b in raw.get("binds", []))
        if raw.get("kind") == "entity_set":
            steps.append(SubQuery(str(raw["step_id"]), _spec("count", constraints), binds=binds,
                                  kind="entity_set", emit_field=str(raw.get("emit_field", ""))))
        else:
            steps.append(SubQuery(str(raw["step_id"]),
                                  _spec(str(raw.get("answer_operation", "count")), constraints,
                                        str(raw.get("answer_field", "")), str(raw.get("answer_value_type", "string"))),
                                  binds=binds, kind="answer"))
    return DecompositionPlan(plan_id=_sid(question, "llm_decomp"), question=question, steps=tuple(steps),
                             combine=str(block.get("combine", "identity")),
                             final_steps=tuple(str(x) for x in block.get("final_steps", [])))


def _sid(question: str, tag: str) -> str:
    import hashlib

    return "llm_" + hashlib.sha1(f"{tag}|{question}".encode("utf-8")).hexdigest()[:12]


@dataclass
class LLMReasoningPlanner:
    client: ChatLike
    model: str
    max_plans: int = 2
    fallback: Any = field(default_factory=RuleBasedDryRunPlanner)
    org_resolver: Any = None  # snaps LLM-proposed org MENTIONS to canonical KG entities (deterministic)

    @classmethod
    def from_env(cls, model: str, **overrides: Any) -> "LLMReasoningPlanner":
        from ..qa.benchmark.chat import ChatClient

        return cls(client=ChatClient.from_env(), model=model, **overrides)

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        system, user = build_planner_messages(question)
        try:
            result = self.client.complete_json(model=self.model, system=system, user=user)
        except Exception as exc:  # pragma: no cover - live only
            return self._fallback(
                question,
                reason="parse_error",
                raw_response={"fallback_detail": "client_exception", "error": repr(exc)},
            )
        payload = getattr(result, "parsed", result)
        if not isinstance(payload, dict):
            return self._fallback(question, reason="parse_error", raw_response={"parsed": payload})
        payload.setdefault("schema_version", PLANNER_SCHEMA_VERSION)
        raw_plans = payload.get("plans")
        if not isinstance(raw_plans, list) or not raw_plans:
            return self._fallback(
                question,
                reason="schema_error",
                raw_response={"fallback_detail": "empty_plans", "payload": payload},
            )
        plans: list[CandidatePlan] = []
        failures: list[str] = []
        for index in range(min(self.max_plans, len(raw_plans))):
            try:
                candidate = candidate_from_payload(question, payload, index=index, planner_version=self.model)
                decomposition = decomposition_from_payload(question, raw_plans[index], self.org_resolver)
                if decomposition is not None:
                    candidate = replace(candidate, decomposition=decomposition)
                plans.append(candidate)
            except (PlanningError, KeyError, TypeError, ValueError) as exc:
                failures.append(f"plan_{index}: {type(exc).__name__}: {exc}")
                continue
        return tuple(plans) or self._fallback(
            question,
            reason="schema_error",
            raw_response={"payload": payload, "plan_errors": failures},
        )

    def _fallback(self, question: str, *, reason: str, raw_response: Any = None) -> tuple[CandidatePlan, ...]:
        if self.fallback is None:
            return ()
        fallback_name = getattr(self.fallback, "planner_version", type(self.fallback).__name__)
        return tuple(
            replace(
                plan,
                planner_source="llm",
                fallback_used=True,
                fallback_reason=reason,
                fallback_planner=str(fallback_name),
                raw_response=raw_response,
            )
            for plan in self.fallback.plan(question)
        )


def build_planner_messages(question: str) -> tuple[str, str]:
    system = (
        "You are a query planner for a UK public-procurement knowledge-graph QA system. Convert the "
        "user's question into one or more EXECUTABLE structured query plans. You do NOT answer the "
        "question; a deterministic engine executes your plan.\n"
        "Rules:\n"
        "1. Use ONLY the allowed fields. If the question needs a field that is not listed, return a "
        "single plan with intent='unsupported'.\n"
        "2. Classify the operation: select_unique (one contract's field), count, or sum (money total).\n"
        "3. For a sum, add a hidden constraint {field:'value_is_additive', op:'eq', value:true, "
        "visible_to_user:false}; the user never states it. The sum's answer_field is always 'value_amount'.\n"
        "4. Extract every visible filter (year, category, CPV, buyer/supplier, signed-date) as a constraint.\n"
        "5. answer_field MUST be one of answer_fields below, spelled EXACTLY. For a contract's signed "
        "date use 'award_date_signed'; for a money value use 'value_amount'. Do not invent variants.\n"
        "6. Leave dedupe_key empty and omit target_node_type; the engine deduplicates by contract.\n"
        "7. If the question is vague or under-specified, return intent='ambiguous'.\n"
        "8. Never invent entities, years, amounts, or CPV codes not present in the question.\n"
        "9. MULTI-HOP / COMPARISON: if answering needs an entity SET from one sub-query to feed the "
        "next (e.g. 'contracts awarded to suppliers who also worked with BUYER' -> resolve BUYER's "
        "suppliers, then aggregate over their contracts), or a comparison of two independent groups, "
        "emit a 'decomposition' (see output_schema.decomposition) instead of flat constraints. Use "
        "'entity_set' steps (emit_field = the linking entity) then a final 'answer' step whose 'binds' "
        "reference the prior step; combine='identity' for a bridge, 'compare_gt' for 'X more than Y'. "
        "Put the org NAME as the constraint value; the engine resolves it to the KG entity.\n"
        "Respond with strict JSON only, matching output_schema."
    )
    payload = {
        "question": question,
        "allowed_fields": PLANNER_FIELD_VOCABULARY,
        "answer_fields": PLANNER_ANSWER_FIELDS,
        "unsupported_concepts": sorted(set(UNSUPPORTED_TERMS.values())),
        "output_schema": planner_output_schema(),
        "decomposition_example": {
            "intent": "bridge", "answer_operation": "sum", "answer_field": "value_amount",
            "answer_value_type": "currency", "constraints": [],
            "decomposition": {
                "steps": [
                    {"step_id": "h1", "kind": "entity_set", "emit_field": "supplier_name",
                     "constraints": [{"field": "buyer_name", "op": "eq", "value": "<BUYER NAME>"}]},
                    {"step_id": "ans", "kind": "answer", "answer_operation": "sum",
                     "answer_field": "value_amount", "constraints": [],
                     "binds": [{"from_step": "h1", "into_field": "supplier_name"}]},
                ],
                "combine": "identity", "final_steps": ["ans"],
            },
        },
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "LLMReasoningPlanner",
    "ChatLike",
    "build_planner_messages",
    "PLANNER_FIELD_VOCABULARY",
    "PLANNER_ANSWER_FIELDS",
]
