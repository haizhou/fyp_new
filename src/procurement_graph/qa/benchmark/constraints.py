"""Constraint normalisation and conflict detection for answer specs.

Samplers can append fixed coverage constraints (for example ``supplier_count gte 1``)
on top of dynamically sampled field constraints (for example ``supplier_count eq 9``).
That produces redundant, and in principle contradictory, constraint sets. These helpers
keep generated specs clean and make any contradiction explicit rather than silent:

- :func:`normalize_constraints` removes exact duplicates and bounds made redundant by an
  ``eq`` on the same field. It never drops contradictions.
- :func:`detect_constraint_conflicts` reports mutually unsatisfiable constraints so Stage 1
  can reject the spec with a visible FAIL instead of producing an always-empty query.
"""

from __future__ import annotations

from typing import Any

from .models import Constraint


def _as_number(value: Any) -> float | None:
    """Return ``value`` as float, or None when it is not a plain number.

    Booleans are intentionally excluded: ``value_is_additive eq True`` is categorical,
    not a numeric bound, so it must not participate in gte/lte reasoning.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _value_key(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def normalize_constraints(constraints: tuple[Constraint, ...]) -> tuple[Constraint, ...]:
    """Drop exact duplicates and bounds made redundant by an ``eq`` on the same field.

    Example: ``supplier_count eq 9`` makes ``supplier_count gte 1`` redundant, so the
    gte is removed. Contradictions (``eq 9`` with ``gte 10``) are left in place and
    surfaced by :func:`detect_constraint_conflicts`.
    """
    seen: set[tuple[str, str, Any]] = set()
    unique: list[Constraint] = []
    for constraint in constraints:
        key = (constraint.field, constraint.op, _value_key(constraint.value))
        if key in seen:
            continue
        seen.add(key)
        unique.append(constraint)

    eq_values = {c.field: c.value for c in unique if c.op == "eq"}
    result: list[Constraint] = []
    for constraint in unique:
        if constraint.op in {"gte", "lte", "exists"} and constraint.field in eq_values:
            if _bound_redundant_under_eq(constraint, eq_values[constraint.field]):
                continue
        result.append(constraint)
    return tuple(result)


def _bound_redundant_under_eq(bound: Constraint, eq_value: Any) -> bool:
    if bound.op == "exists":
        return eq_value not in (None, "")
    eq_num = _as_number(eq_value)
    bound_num = _as_number(bound.value)
    if eq_num is None or bound_num is None:
        return False
    if bound.op == "gte":
        return eq_num >= bound_num
    if bound.op == "lte":
        return eq_num <= bound_num
    return False


def detect_constraint_conflicts(constraints: tuple[Constraint, ...]) -> list[str]:
    """Return human-readable descriptions of mutually unsatisfiable constraints."""
    conflicts: list[str] = []
    by_field: dict[str, list[Constraint]] = {}
    for constraint in constraints:
        by_field.setdefault(constraint.field, []).append(constraint)

    for field, items in by_field.items():
        eq_constraints = [c for c in items if c.op == "eq"]
        distinct_eq = {_value_key(c.value) for c in eq_constraints}
        if len(distinct_eq) > 1:
            rendered = sorted(str(c.value) for c in eq_constraints)
            conflicts.append(f"{field}: conflicting eq values {rendered}")

        lower = _max_bound(items, "gte")
        upper = _min_bound(items, "lte")
        if lower is not None and upper is not None and lower > upper:
            conflicts.append(f"{field}: gte {lower} conflicts with lte {upper}")

        eq_num = next((_as_number(c.value) for c in eq_constraints if _as_number(c.value) is not None), None)
        if eq_num is not None:
            if lower is not None and eq_num < lower:
                conflicts.append(f"{field}: eq {eq_num} violates gte {lower}")
            if upper is not None and eq_num > upper:
                conflicts.append(f"{field}: eq {eq_num} violates lte {upper}")
    return conflicts


def _max_bound(items: list[Constraint], op: str) -> float | None:
    values = [_as_number(c.value) for c in items if c.op == op]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _min_bound(items: list[Constraint], op: str) -> float | None:
    values = [_as_number(c.value) for c in items if c.op == op]
    values = [v for v in values if v is not None]
    return min(values) if values else None


__all__ = ["normalize_constraints", "detect_constraint_conflicts"]
