"""Deterministic verification helpers for runtime query specs."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from .models import QueryConstraint, RuntimeQuerySpec


class RuntimeQueryBackend(Protocol):
    """Minimal backend contract used by runtime reasoning."""

    def query(self, constraints: tuple[QueryConstraint, ...]) -> list[dict[str, Any]]:
        ...

    def record_id(self, record: dict[str, Any]) -> str:
        ...


def backend_fields(backend: Any) -> set[str]:
    """Best-effort field discovery for schema checks."""
    fields_attr = getattr(backend, "fields", None)
    if callable(fields_attr):
        return {str(field) for field in fields_attr()}
    if fields_attr is not None:
        return {str(field) for field in fields_attr}
    records = getattr(backend, "records", None)
    if records:
        return {str(key) for record in records for key in record.keys()}
    frame = getattr(backend, "records_df", None)
    columns = getattr(frame, "columns", None)
    if columns is not None:
        return {str(column) for column in columns}
    return set()


def preflight_checks(backend: RuntimeQueryBackend, spec: RuntimeQuerySpec) -> tuple[dict[str, Any], ...]:
    """Validate schema and obvious constraint conflicts before execution."""
    checks: list[dict[str, Any]] = []
    fields = backend_fields(backend)
    referenced = {constraint.field for constraint in spec.constraints}
    if spec.answer_field:
        referenced.add(spec.answer_field)
    if spec.dedupe_key:
        referenced.add(spec.dedupe_key)
    if spec.sort_field:
        referenced.add(spec.sort_field)

    if fields:
        missing = sorted(field for field in referenced if field and field not in fields)
        checks.append(_check("schema_fields", not missing, missing_fields=missing))
    else:
        checks.append(_check("schema_fields", True, reason="backend did not expose fields; skipped"))

    conflicts = constraint_conflicts(spec.constraints)
    checks.append(_check("constraint_conflicts", not conflicts, conflicts=conflicts))

    if spec.answer_operation in {"count", "sum", "exists", "argmax", "argmin", "distinct_set", "set", "compare", "top_k"}:
        checks.append(
            _check(
                "exhaustive_required",
                bool(spec.requires_exhaustive_retrieval),
                operation=spec.answer_operation,
            )
        )
    return tuple(checks)


def constraint_conflicts(constraints: tuple[QueryConstraint, ...]) -> list[dict[str, Any]]:
    """Detect simple contradictions without querying the KG."""
    conflicts: list[dict[str, Any]] = []
    eq_values: dict[str, set[str]] = {}
    in_values: dict[str, set[str]] = {}
    lower_bounds: dict[str, Decimal] = {}
    upper_bounds: dict[str, Decimal] = {}

    for constraint in constraints:
        field = constraint.field
        if constraint.op == "eq":
            eq_values.setdefault(field, set()).add(_stable_value(constraint.value))
        elif constraint.op == "in":
            in_values.setdefault(field, set()).update(_stable_value(value) for value in (constraint.value or []))
        elif constraint.op == "gte":
            value = _to_decimal_or_none(constraint.value)
            if value is not None:
                lower_bounds[field] = max(value, lower_bounds.get(field, value))
        elif constraint.op == "lte":
            value = _to_decimal_or_none(constraint.value)
            if value is not None:
                upper_bounds[field] = min(value, upper_bounds.get(field, value))
        elif constraint.op == "between":
            values = list(constraint.value or [])
            if len(values) == 2:
                low = _to_decimal_or_none(values[0])
                high = _to_decimal_or_none(values[1])
                if low is not None and high is not None:
                    if low > high:
                        conflicts.append({"field": field, "reason": "between lower bound exceeds upper bound"})
                    lower_bounds[field] = max(low, lower_bounds.get(field, low))
                    upper_bounds[field] = min(high, upper_bounds.get(field, high))

    for field, values in eq_values.items():
        if len(values) > 1:
            conflicts.append({"field": field, "reason": "multiple eq values", "values": sorted(values)})
        allowed = in_values.get(field)
        if allowed is not None and not values <= allowed:
            conflicts.append(
                {
                    "field": field,
                    "reason": "eq value outside allowed in-set",
                    "eq_values": sorted(values),
                    "allowed_values": sorted(allowed),
                }
            )

    for field, low in lower_bounds.items():
        high = upper_bounds.get(field)
        if high is not None and low > high:
            conflicts.append({"field": field, "reason": "lower bound exceeds upper bound", "gte": str(low), "lte": str(high)})
    return conflicts


def postflight_checks(spec: RuntimeQuerySpec, rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
                      *, matched: int | None = None) -> tuple[dict[str, Any], ...]:
    """Verify the ANSWER (not just the plan) after a passed execution.

    `preflight_checks` validates the query *before* running it; nothing looked at whether the
    matched *population* actually supports the answer. These informational checks disclose that:
    they never change a correct answer, but they let a count/sum explain a discrepancy and let a
    factoid reveal a hidden multiplicity. `matched` is the EXACT match count (rows may be a capped
    sample from the fast count path — coverage is then over the sample and flagged).
    """
    checks: list[dict[str, Any]] = []
    rows = list(rows)
    total = matched if matched is not None else len(rows)
    sampled = len(rows) < total
    op = spec.answer_operation
    if op in {"count", "sum"}:
        without_supplier = sum(1 for row in rows if not _positive_count(row.get("supplier_count")))
        without_buyer = sum(1 for row in rows if not _positive_count(row.get("buyer_count")))
        checks.append(
            _check(
                "population_coverage",
                without_supplier == 0 and without_buyer == 0,
                matched=total,
                without_supplier=without_supplier,
                without_buyer=without_buyer,
                coverage_sampled=sampled,
                coverage_sample_size=len(rows) if sampled else total,
            )
        )
    if op == "select_unique":
        checks.append(_check("answer_uniqueness", total <= 1, matching_contracts=total))
    return tuple(checks)


def _positive_count(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def additive_value_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check whether all rows are additive when a value sum is requested."""
    if not rows:
        return _check("sum_additive_safety", True, reason="no rows")
    if not any("value_is_additive" in row for row in rows):
        return _check("sum_additive_safety", True, reason="value_is_additive not available")
    unsafe_ids = [
        str(row.get("contract_node_id") or row.get("record_id") or row.get("id") or idx)
        for idx, row in enumerate(rows)
        if not _truthy(row.get("value_is_additive"))
    ]
    return _check("sum_additive_safety", not unsafe_ids, unsafe_evidence_ids=unsafe_ids)


def _check(name: str, passed: bool, **metrics: Any) -> dict[str, Any]:
    return {"check": name, "passed": passed, **metrics}


def _stable_value(value: Any) -> str:
    return str(value).casefold()


def _to_decimal_or_none(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "y"}
    return bool(value)


__all__ = [
    "RuntimeQueryBackend",
    "additive_value_check",
    "backend_fields",
    "constraint_conflicts",
    "postflight_checks",
    "preflight_checks",
]
