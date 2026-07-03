"""General bounded multi-hop decomposition (Phase 2).

A `DecompositionPlan` is an ordered list of `SubQuery` steps. A step may **bind** a prior step's
emitted *entity set* as an `in` (set-membership) filter on one of its own fields — that is one hop.
The design is deliberately N-hop, not 2-hop: a 2-hop bridge is two steps, a 3-hop chain is three, and
`compare` is two INDEPENDENT steps plus a `combine`. Depth is bounded (`max_hops`) because the KG is
tabular and genuine chains are short; unbounded expansion is the graph-era machinery we rejected.

Each step runs through the SAME deterministic core: entity-set steps collect the distinct `emit_field`
over an exhaustive match set; answer steps run `execute_query_spec` (so the additive-sum guard,
exhaustiveness, and schema grounding all apply per hop). The executor stays the sole answer authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .executor import execute_query_spec
from .grounding import ground_spec
from .models import QueryConstraint, RuntimeQuerySpec
from .verifier import RuntimeQueryBackend, backend_fields


@dataclass(frozen=True)
class Binding:
    """Bind the emitted set of `from_step` as an `in` filter on `into_field` of this step."""
    from_step: str
    into_field: str
    op: str = "in"


@dataclass(frozen=True)
class SubQuery:
    step_id: str
    spec: RuntimeQuerySpec
    binds: tuple[Binding, ...] = ()
    kind: str = "answer"          # "entity_set" (emit a set) | "answer" (terminal reduction)
    emit_field: str = ""          # entity_set: which field's distinct values to expose


@dataclass(frozen=True)
class DecompositionPlan:
    plan_id: str
    question: str
    steps: tuple[SubQuery, ...]
    combine: str = "identity"     # "identity" | "compare_gt"
    final_steps: tuple[str, ...] = ()
    max_hops: int = 4


@dataclass(frozen=True)
class HopTrace:
    step_id: str
    kind: str
    status: str
    output_size: int
    bound_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecompositionResult:
    plan_id: str
    status: str                   # passed | no_results | unsupported | error | <exec status>
    answer: Any = None
    hops: tuple[HopTrace, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def execute_decomposition(backend: RuntimeQueryBackend, plan: DecompositionPlan, *,
                          allowed_fields: frozenset[str] | set[str] | None = None) -> DecompositionResult:
    allowed = frozenset(allowed_fields) if allowed_fields else frozenset(backend_fields(backend))
    if len(plan.steps) > plan.max_hops:
        return DecompositionResult(plan.plan_id, "unsupported",
                                   reason=f"plan depth {len(plan.steps)} exceeds max_hops={plan.max_hops}")
    outputs: dict[str, Any] = {}          # step_id -> frozenset (entity_set) | scalar (answer)
    hops: list[HopTrace] = []
    evidence_ids: tuple[str, ...] = ()

    for step in plan.steps:
        constraints = list(step.spec.constraints)
        bound_from = tuple(b.from_step for b in step.binds)
        for bind in step.binds:
            members = outputs.get(bind.from_step)
            if members is None:
                return DecompositionResult(plan.plan_id, "error", hops=tuple(hops),
                                           reason=f"step {step.step_id!r} binds unknown/out-of-order step {bind.from_step!r}")
            if not members:
                hops.append(HopTrace(step.step_id, step.kind, "no_results", 0, bound_from))
                return DecompositionResult(plan.plan_id, "no_results", hops=tuple(hops),
                                           reason=f"empty bind set from {bind.from_step!r}")
            constraints.append(QueryConstraint(bind.into_field, bind.op, tuple(sorted(members, key=str)),
                                               visible_to_user=False, source_text=f"bound from {bind.from_step}"))
        spec = replace(step.spec, constraints=tuple(constraints))

        if step.kind == "entity_set":
            rows = backend.query(spec.constraints)
            members = frozenset(row.get(step.emit_field) for row in rows if row.get(step.emit_field) not in (None, ""))
            outputs[step.step_id] = members
            hops.append(HopTrace(step.step_id, "entity_set", "passed" if members else "no_results", len(members), bound_from))
            if not members:
                return DecompositionResult(plan.plan_id, "no_results", hops=tuple(hops),
                                           reason=f"step {step.step_id!r} emitted an empty set")
        else:
            grounded = ground_spec(replace(spec, requires_exhaustive_retrieval=True), allowed_fields=allowed)
            if not grounded.ok:
                hops.append(HopTrace(step.step_id, "answer", "unsupported", 0, bound_from))
                return DecompositionResult(plan.plan_id, "unsupported", hops=tuple(hops), reason=grounded.reason)
            result = execute_query_spec(backend, grounded.spec)
            outputs[step.step_id] = result.answer
            if result.evidence:
                evidence_ids = tuple(result.evidence.evidence_ids)
            hops.append(HopTrace(step.step_id, "answer", result.status, 1, bound_from))
            if not result.passed:
                return DecompositionResult(plan.plan_id, result.status, hops=tuple(hops),
                                           reason=f"step {step.step_id!r}: {result.status}")

    answer, reason = _combine(plan, outputs)
    if reason:
        return DecompositionResult(plan.plan_id, "error", hops=tuple(hops), reason=reason)
    return DecompositionResult(plan.plan_id, "passed", answer=answer, hops=tuple(hops),
                               evidence_ids=evidence_ids, metrics={"hops": len(plan.steps)})


def _combine(plan: DecompositionPlan, outputs: dict[str, Any]) -> tuple[Any, str]:
    finals = plan.final_steps or (plan.steps[-1].step_id,)
    if plan.combine == "identity":
        return outputs.get(finals[0]), ""
    if plan.combine == "compare_gt":
        if len(finals) != 2:
            return None, "compare_gt needs exactly two final_steps"
        a, b = outputs.get(finals[0]), outputs.get(finals[1])
        return {"answer": bool(_num(a) > _num(b)), finals[0]: a, finals[1]: b}, ""
    return None, f"unknown combine {plan.combine!r}"


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


__all__ = ["Binding", "SubQuery", "DecompositionPlan", "HopTrace", "DecompositionResult",
           "execute_decomposition"]
