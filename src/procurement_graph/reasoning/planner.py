"""Planner contracts and a deterministic dry-run planner.

The production planner will usually be an open-source model that emits strict
JSON. This module defines the JSON shape and a small rule-based planner so the
runtime reasoning loop can be tested without any model call.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from .linking import link_question
from .models import CandidatePlan, QueryConstraint, RuntimeQuerySpec


PLANNER_SCHEMA_VERSION = "runtime_planner_schema_v1"
RULE_BASED_PLANNER_VERSION = "rule_based_planner_v1"


class ReasoningPlanner(Protocol):
    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        ...


@dataclass(frozen=True)
class PlanningError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


class RuleBasedDryRunPlanner:
    """Small deterministic planner for tests and smoke runs.

    It intentionally covers only the benchmark-like operation families that the
    first runtime executor supports: count, sum, and a narrow factoid shape.
    """

    planner_version = RULE_BASED_PLANNER_VERSION

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        text = " ".join(str(question or "").split())
        link = link_question(text)
        if link.unsupported_reason:
            return (self._unsupported(text, link.unsupported_reason),)

        constraints = link.constraints
        if link.asks_sum:
            constraints = constraints + (
                QueryConstraint(
                    "value_is_additive",
                    "eq",
                    True,
                    visible_to_user=False,
                    source_text="additive value safety",
                    metadata={"reason": "sum uses only additive award/contract value rows"},
                ),
            )
            return (
                self._planned(
                    text,
                    intent="aggregation_sum",
                    constraints=constraints,
                    answer_operation="sum",
                    answer_field="value_amount",
                    answer_value_type="currency",
                    rationale="Detected a total/sum value question.",
                ),
            )

        if link.asks_count:
            fields = {constraint.field for constraint in constraints}
            intent = "aggregation_count"
            if "tender_cpv_id" in fields:
                intent = "categorical_cpv"
            elif "has_award_signed_date" in fields:
                intent = "temporal"
            elif len([constraint for constraint in constraints if constraint.visible_to_user]) >= 2:
                intent = "conjunction_constraint"
            return (
                self._planned(
                    text,
                    intent=intent,
                    constraints=constraints,
                    answer_operation="count",
                    answer_field="contract_node_id",
                    answer_value_type="integer",
                    rationale="Detected a count question.",
                ),
            )

        if link.factoid_field and link.explicit_contract_id:
            return (
                self._planned(
                    text,
                    intent="factoid",
                    constraints=(
                        QueryConstraint(
                            "contract_node_id", "eq", link.explicit_contract_id,
                            source_text=link.explicit_contract_id,
                        ),
                    ),
                    answer_operation="select_unique",
                    answer_field=link.factoid_field,
                    answer_value_type="string",
                    rationale="Detected a narrow factoid over an explicit contract identifier.",
                    warnings=("explicit internal identifiers should not be used in final user-facing benchmarks",),
                ),
            )

        return (
            self._ambiguous(
                text,
                "Could not map the question to a supported count, sum, or narrow factoid query.",
            ),
        )

    def _planned(
        self,
        question: str,
        *,
        intent: Any,
        constraints: tuple[QueryConstraint, ...],
        answer_operation: Any,
        answer_field: str,
        answer_value_type: str,
        rationale: str,
        warnings: tuple[str, ...] = (),
    ) -> CandidatePlan:
        spec = RuntimeQuerySpec(
            spec_id=_spec_id(question, answer_operation, answer_field, constraints),
            question=question,
            intent=intent,
            constraints=constraints,
            answer_operation=answer_operation,
            answer_field=answer_field,
            answer_value_type=answer_value_type,
            dedupe_key="contract_node_id",
            target_node_type="contract",
            requires_exhaustive_retrieval=answer_operation in {"count", "sum"},
            planner_version=self.planner_version,
            metadata={"schema_version": PLANNER_SCHEMA_VERSION},
        )
        return CandidatePlan(
            plan_id=f"{spec.spec_id}:plan0",
            query_spec=spec,
            status="planned",
            confidence=0.8,
            rationale=rationale,
            warnings=warnings,
            planner_source="rule",
        )

    def _ambiguous(self, question: str, reason: str) -> CandidatePlan:
        spec = RuntimeQuerySpec(
            spec_id=_spec_id(question, "unsupported", "", tuple()),
            question=question,
            intent="ambiguous",
            constraints=tuple(),
            answer_operation="unsupported",
            answer_field="",
            answer_value_type="",
            requires_exhaustive_retrieval=False,
            planner_version=self.planner_version,
            metadata={"schema_version": PLANNER_SCHEMA_VERSION},
        )
        return CandidatePlan(
            plan_id=f"{spec.spec_id}:ambiguous",
            query_spec=spec,
            status="ambiguous",
            confidence=0.0,
            rationale=reason,
            planner_source="rule",
        )

    def _unsupported(self, question: str, reason: str) -> CandidatePlan:
        spec = RuntimeQuerySpec(
            spec_id=_spec_id(question, "unsupported", "", tuple()),
            question=question,
            intent="unsupported",
            constraints=tuple(),
            answer_operation="unsupported",
            answer_field="",
            answer_value_type="",
            requires_exhaustive_retrieval=False,
            planner_version=self.planner_version,
            metadata={"schema_version": PLANNER_SCHEMA_VERSION, "unsupported_reason": reason},
        )
        return CandidatePlan(
            plan_id=f"{spec.spec_id}:unsupported",
            query_spec=spec,
            status="unsupported",
            confidence=1.0,
            rationale=reason,
            planner_source="rule",
        )


def planner_output_schema() -> dict[str, Any]:
    """Schema-like contract expected from an LLM planner adapter."""
    return {
        "schema_version": PLANNER_SCHEMA_VERSION,
        "plans": [
            {
                "intent": "factoid | aggregation_count | aggregation_sum | conjunction_constraint | comparison | superlative_top_k | temporal | categorical_cpv | role_path | unsupported | ambiguous",
                "answer_operation": "select_unique | count | sum | set | compare | top_k | duration | unsupported",
                "answer_field": "KG field name",
                "answer_value_type": "string | integer | number | currency | date | boolean",
                "constraints": [
                    {
                        "field": "KG field name",
                        "op": "eq | in | contains | exists | gte | lte | between",
                        "value": "JSON value",
                        "visible_to_user": "boolean",
                        "source_text": "question span or planner note",
                    }
                ],
                "dedupe_key": "leave empty; the engine deduplicates by contract",
                "target_node_type": "leave empty; set by the engine",
                "requires_exhaustive_retrieval": "boolean",
                "rationale": "short reason",
                "confidence": "0.0-1.0",
            }
        ],
    }


def candidate_from_payload(
    question: str,
    payload: dict[str, Any],
    *,
    index: int = 0,
    planner_version: str = "llm_planner",
    planner_source: str = "llm",
) -> CandidatePlan:
    """Build a CandidatePlan from schema-validated planner JSON."""
    if payload.get("schema_version") != PLANNER_SCHEMA_VERSION:
        raise PlanningError(f"unexpected planner schema_version: {payload.get('schema_version')!r}")
    plans = payload.get("plans")
    if not isinstance(plans, list) or index >= len(plans):
        raise PlanningError("planner payload contains no plan at requested index")
    plan = plans[index]
    constraints = tuple(
        QueryConstraint(
            field=str(item["field"]),
            op=item["op"],
            value=item.get("value"),
            visible_to_user=bool(item.get("visible_to_user", True)),
            source_text=str(item.get("source_text", "")),
            confidence=float(item.get("confidence", 1.0)),
            metadata={key: value for key, value in item.items() if key not in {"field", "op", "value", "visible_to_user", "source_text", "confidence"}},
        )
        for item in plan.get("constraints", [])
    )
    spec = RuntimeQuerySpec(
        spec_id=_spec_id(question, str(plan.get("answer_operation", "")), str(plan.get("answer_field", "")), constraints),
        question=question,
        intent=plan["intent"],
        constraints=constraints,
        answer_operation=plan["answer_operation"],
        answer_field=str(plan.get("answer_field", "")),
        answer_value_type=str(plan.get("answer_value_type", "")),
        dedupe_key=str(plan.get("dedupe_key", "")),
        target_node_type=str(plan.get("target_node_type", "")),
        requires_exhaustive_retrieval=bool(plan.get("requires_exhaustive_retrieval", True)),
        planner_version=planner_version,
        metadata={"schema_version": PLANNER_SCHEMA_VERSION},
    )
    status = "planned"
    if spec.intent == "unsupported":
        status = "unsupported"
    elif spec.intent == "ambiguous":
        status = "ambiguous"
    return CandidatePlan(
        plan_id=f"{spec.spec_id}:plan{index}",
        query_spec=spec,
        status=status,
        confidence=float(plan.get("confidence", 0.0)),
        rationale=str(plan.get("rationale", "")),
        raw_response=payload,
        planner_source=planner_source,
    )


def _spec_id(question: str, operation: str, field: str, constraints: tuple[QueryConstraint, ...]) -> str:
    payload = "|".join(
        [
            question.casefold(),
            operation,
            field,
            ";".join(f"{c.field}:{c.op}:{c.value}" for c in constraints),
        ]
    )
    return f"runtime_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


__all__ = [
    "PLANNER_SCHEMA_VERSION",
    "RULE_BASED_PLANNER_VERSION",
    "PlanningError",
    "ReasoningPlanner",
    "RuleBasedDryRunPlanner",
    "candidate_from_payload",
    "planner_output_schema",
]
