"""Generate-then-ground boundary (adapted from the prior pipeline's step1 grounding).

An LLM planner may *propose* a query spec, but its fields, operation, and safety guards are
rebuilt deterministically here before execution. This is the single most important safety layer
around a learned planner: it prevents the model from

- naming a non-schema field ("winner", "year") -> aliased to a real field or rejected;
- dropping the additive-sum guard -> re-added, so a sum never silently includes tender-fallback
  (non-additive) values;
- requesting an operation the deterministic executor cannot run -> marked unsupported cleanly;
- forgetting that reductions must be exhaustive -> forced on.

The prior project scattered this across ~300 lines of `ground_llm_interpretations`; here it is one
small, tested function. It is deliberately schema-light (no beam search, no path templates) because
the KG is tabular.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .models import QueryConstraint, RuntimeQuerySpec

# Canonical question-facing KG fields (frozen KG v0.1). value_is_additive is internal-only.
CANONICAL_FIELDS: frozenset[str] = frozenset(
    {
        "contract_node_id",
        "buyer_name",
        "supplier_name",
        "release_year",
        "tender_title",
        "award_title",
        "tender_category",
        "tender_cpv_id",
        "tender_cpv_description",
        "has_award_signed_date",
        "value_amount",
        "value_source",
        "award_date_signed",
        "value_is_additive",
        "supplier_count",
        "buyer_count",
    }
)

# Natural field names an LLM tends to emit -> canonical KG fields. This is the "schema linking"
# layer (cf. text-to-SQL DIN-SQL / RESDSQL): a learned planner names columns in natural language,
# and every mention is snapped to a real schema column before execution rather than trusted.
FIELD_ALIASES: dict[str, str] = {
    "year": "release_year",
    "published_year": "release_year",
    "publication_year": "release_year",
    "category": "tender_category",
    "procurement_category": "tender_category",
    "cpv": "tender_cpv_id",
    "cpv_code": "tender_cpv_id",
    "cpv_id": "tender_cpv_id",
    "cpv_description": "tender_cpv_description",
    "buyer": "buyer_name",
    "contracting_authority": "buyer_name",
    "authority": "buyer_name",
    "supplier": "supplier_name",
    "winner": "supplier_name",
    "value": "value_amount",
    "amount": "value_amount",
    "contract_value": "value_amount",
    "total": "value_amount",
    "total_value": "value_amount",
    "total_amount": "value_amount",
    "total_contract_value": "value_amount",
    "contract_amount": "value_amount",
    # signed-date variants a small model reaches for when it is not shown the exact column name.
    "signed_date": "award_date_signed",
    "award_signed_date": "award_date_signed",
    "date_signed": "award_date_signed",
    "contract_signed_date": "award_date_signed",
    "signing_date": "award_date_signed",
    "award_date": "award_date_signed",
    "has_signed_date": "has_award_signed_date",
}

_SUPPORTED_OPERATIONS = frozenset(
    {"select_unique", "count", "sum", "exists", "argmax", "argmin", "distinct_set", "set", "top_k", "predicate"}
)
_REDUCTIONS = frozenset({"count", "sum", "exists", "argmax", "argmin", "distinct_set", "set", "top_k", "predicate"})
_COUNT_LIKE = frozenset({"count", "exists", "top_k", "predicate"})  # answer over contract_node_id; no field to select
# compare (two independent counts) and bridge in_subquery need the decomposition planner, not a
# single-query executor op -> still routed as unsupported here.
_UNSUPPORTED_OP_ALIASES = {
    "average": "aggregate 'average' is not supported by the deterministic runtime yet",
    "avg": "aggregate 'average' is not supported by the deterministic runtime yet",
    "mean": "aggregate 'mean' is not supported by the deterministic runtime yet",
    "median": "aggregate 'median' is not supported by the deterministic runtime yet",
    "compare": "comparison needs the decomposition planner (two independent sub-counts), not yet wired",
    "in_subquery": "bridge/semijoin needs the decomposition planner + set-membership, not yet wired",
}


@dataclass(frozen=True)
class GroundingResult:
    spec: RuntimeQuerySpec
    ok: bool
    changes: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    reason: str = ""


def ground_spec(spec: RuntimeQuerySpec, *, allowed_fields: frozenset[str] | set[str] | None = None) -> GroundingResult:
    """Validate and repair a planned spec against the schema. ok=False means un-executable."""
    allowed = set(allowed_fields) if allowed_fields else set(CANONICAL_FIELDS)
    changes: list[str] = []
    issues: list[str] = []

    # 0. operation support
    op = spec.answer_operation
    if op in _UNSUPPORTED_OP_ALIASES:
        return GroundingResult(spec=spec, ok=False, reason=_UNSUPPORTED_OP_ALIASES[op])
    if op not in _SUPPORTED_OPERATIONS:
        return GroundingResult(spec=spec, ok=False, reason=f"answer_operation '{op}' is not executable")

    # 1. alias + validate constraint fields
    new_constraints: list[QueryConstraint] = []
    for constraint in spec.constraints:
        field = FIELD_ALIASES.get(constraint.field, constraint.field)
        if field != constraint.field:
            changes.append(f"aliased constraint field {constraint.field!r}->{field!r}")
            constraint = replace(constraint, field=field)
        if field not in allowed:
            return GroundingResult(spec=spec, ok=False, reason=f"constraint field {field!r} is not in the KG schema")
        new_constraints.append(constraint)

    # 2. alias + validate answer field (count answers over contract_node_id, so any is fine)
    answer_field = FIELD_ALIASES.get(spec.answer_field, spec.answer_field)
    if answer_field != spec.answer_field:
        changes.append(f"aliased answer_field {spec.answer_field!r}->{answer_field!r}")
    if op in _COUNT_LIKE:
        answer_field = answer_field or "contract_node_id"
    elif op == "sum":
        answer_field = "value_amount"  # the only additive numeric field
        if answer_field != spec.answer_field:
            changes.append("forced sum answer_field to value_amount")
    elif op in {"distinct_set", "set"} and not answer_field:
        return GroundingResult(spec=spec, ok=False, reason=f"{op} needs an answer_field to collect")
    elif answer_field and answer_field not in allowed:
        return GroundingResult(spec=spec, ok=False, reason=f"answer_field {answer_field!r} is not in the KG schema")

    # 3. additive-sum guard (safety): a sum must only aggregate additive rows
    if op == "sum" and not any(
        c.field == "value_is_additive" and c.op == "eq" and bool(c.value) for c in new_constraints
    ):
        new_constraints.append(
            QueryConstraint("value_is_additive", "eq", True, visible_to_user=False,
                            source_text="additive value guard (grounded)")
        )
        changes.append("added value_is_additive guard for sum")

    # 4. reductions must be exhaustive; factoids need a dedupe key
    requires_exhaustive = spec.requires_exhaustive_retrieval
    if op in _REDUCTIONS and not requires_exhaustive:
        requires_exhaustive = True
        changes.append("forced exhaustive retrieval for reduction")

    # 5. bookkeeping fields (dedupe_key, sort_field) are also referenced by preflight's schema
    # check, so an LLM-invented value here would wrongly fail an otherwise-valid plan with a
    # `schema_error`. Repair, don't reject (cf. Pangu / ChatKBQA grounded generation): alias then
    # default an unknown dedupe_key to contract_node_id and drop an unknown sort_field. After this,
    # grounding is authoritative over EVERY preflight-referenced field.
    dedupe_key = _ground_bookkeeping_field(spec.dedupe_key, allowed, changes, name="dedupe_key",
                                           fallback="contract_node_id") or "contract_node_id"
    sort_field = _ground_bookkeeping_field(spec.sort_field, allowed, changes, name="sort_field",
                                           fallback="")

    # 6. top_k groups by a field carried in metadata; alias + validate it (reject if unknown, since
    # the answer is grouped BY it).
    metadata = {**spec.metadata, "grounded": True, "grounding_changes": changes}
    if op == "top_k":
        group_by = metadata.get("group_by_field", "")
        aliased = FIELD_ALIASES.get(group_by, group_by)
        if not aliased or aliased not in allowed:
            return GroundingResult(spec=spec, ok=False, reason=f"top_k group_by_field {group_by!r} is not in the KG schema")
        if aliased != group_by:
            changes.append(f"aliased group_by_field {group_by!r}->{aliased!r}")
        metadata["group_by_field"] = aliased

    # a sum is inherently numeric; force currency so a sloppy value_type (e.g. an LLM leaving it
    # 'string') can't break the Decimal aggregation.
    answer_value_type = "currency" if op == "sum" else spec.answer_value_type

    grounded = replace(
        spec,
        constraints=tuple(new_constraints),
        answer_field=answer_field,
        answer_value_type=answer_value_type,
        dedupe_key=dedupe_key,
        sort_field=sort_field,
        requires_exhaustive_retrieval=requires_exhaustive,
        metadata=metadata,
    )
    return GroundingResult(spec=grounded, ok=True, changes=tuple(changes), issues=tuple(issues))


def _ground_bookkeeping_field(
    value: str, allowed: set[str], changes: list[str], *, name: str, fallback: str
) -> str:
    """Alias/validate a non-answer bookkeeping field (dedupe_key/sort_field) that preflight
    references. An unknown value is reset to `fallback` (never a `schema_error`)."""
    if not value:
        return fallback
    aliased = FIELD_ALIASES.get(value, value)
    if aliased != value:
        changes.append(f"aliased {name} {value!r}->{aliased!r}")
    if aliased not in allowed:
        changes.append(f"reset unknown {name} {aliased!r} to {fallback!r}")
        return fallback
    return aliased


__all__ = ["GroundingResult", "ground_spec", "CANONICAL_FIELDS", "FIELD_ALIASES"]
