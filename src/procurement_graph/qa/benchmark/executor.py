"""Deterministic execution of answer specifications."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from .kg_interface import QueryBackend
from .models import AnswerSpec


class AnswerExecutionError(ValueError):
    """Raised when an answer specification is not deterministically executable."""


def execute_answer_spec(backend: QueryBackend, spec: AnswerSpec) -> Any:
    rows = backend.query(spec.constraints)
    return execute_answer_spec_rows(rows, spec)


def execute_answer_spec_rows(rows: list[dict[str, Any]], spec: AnswerSpec) -> Any:
    rows = _dedupe_rows(rows, spec.dedupe_key)
    if not rows:
        raise AnswerExecutionError("answer_spec matched no evidence records")

    if spec.answer_operation == "count":
        return len(rows)

    values = [_coerce_value(row.get(spec.answer_field), spec.answer_value_type) for row in rows]
    values = [value for value in values if value not in (None, "")]
    if not values:
        raise AnswerExecutionError(f"answer_field {spec.answer_field!r} has no non-empty values")

    if spec.answer_operation == "select_unique":
        unique_values = sorted(set(values), key=lambda value: str(value))
        if len(unique_values) != 1:
            raise AnswerExecutionError(
                f"answer_field {spec.answer_field!r} is not unique: {unique_values[:5]}"
            )
        return unique_values[0]

    if spec.answer_operation == "sum":
        return sum(_to_decimal(value) for value in values)

    raise AnswerExecutionError(f"Unsupported answer operation: {spec.answer_operation}")


def numeric_row_values(rows: list[dict[str, Any]], spec: AnswerSpec) -> list[Decimal]:
    """Per-row decimal values that a ``sum`` spec aggregates, after dedupe.

    Shares dedupe/coercion with :func:`execute_answer_spec_rows` so anomaly checks see
    exactly the values that produced the golden answer.
    """
    deduped = _dedupe_rows(rows, spec.dedupe_key)
    values: list[Decimal] = []
    for row in deduped:
        raw = row.get(spec.answer_field)
        if raw in (None, ""):
            continue
        values.append(_to_decimal(raw))
    return values


def _dedupe_rows(rows: list[dict[str, Any]], dedupe_key: str) -> list[dict[str, Any]]:
    if not dedupe_key:
        return rows
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get(dedupe_key, ""))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _coerce_value(value: Any, value_type: str) -> Any:
    if value is None:
        return None
    if value_type == "integer":
        return int(value)
    if value_type in {"number", "currency"}:
        return _to_decimal(value)
    if value_type == "boolean":
        return bool(value)
    return str(value) if value_type in {"string", "date"} else value


def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AnswerExecutionError(f"Cannot convert value to decimal: {value!r}") from exc


__all__ = [
    "AnswerExecutionError",
    "execute_answer_spec",
    "execute_answer_spec_rows",
    "numeric_row_values",
]
