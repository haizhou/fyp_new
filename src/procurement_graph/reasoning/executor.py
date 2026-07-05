"""Deterministic execution for runtime query specs."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from .models import EvidenceBundle, ExecutionResult, QueryConstraint, RuntimeQuerySpec
from .verifier import RuntimeQueryBackend, additive_value_check, preflight_checks


def execute_query_spec(backend: RuntimeQueryBackend, spec: RuntimeQuerySpec) -> ExecutionResult:
    """Run a RuntimeQuerySpec over a backend and return an auditable result."""
    checks = list(preflight_checks(backend, spec))
    failed_preflight = [check for check in checks if not check.get("passed")]
    if failed_preflight:
        status = "constraint_conflict"
        if any(check.get("check") == "schema_fields" and check.get("missing_fields") for check in failed_preflight):
            status = "schema_error"
        elif any(check.get("check") == "exhaustive_required" for check in failed_preflight):
            status = "incomplete_evidence"
        return ExecutionResult(query_spec=spec, status=status, checks=tuple(checks))

    _SUPPORTED = {"select_unique", "count", "sum", "exists", "argmax", "argmin", "distinct_set", "set", "top_k", "predicate"}
    if spec.answer_operation not in _SUPPORTED:
        checks.append({"check": "operation_supported", "passed": False, "operation": spec.answer_operation})
        return ExecutionResult(query_spec=spec, status="unsupported_operation", checks=tuple(checks))
    checks.append({"check": "operation_supported", "passed": True, "operation": spec.answer_operation})

    # Fast path: count/exists need only a count, not row dicts. Use the backend's vectorised count +
    # a capped sample for evidence, so a huge match set (bridge hop-2) is never fully materialised.
    if spec.answer_operation in {"count", "exists"} and hasattr(backend, "count") and hasattr(backend, "sample"):
        try:
            total = backend.count(spec.constraints, dedupe_field=spec.dedupe_key or "contract_node_id")
            sample_rows = backend.sample(spec.constraints, _EVIDENCE_ID_CAP)
        except Exception:  # pragma: no cover - defensive; fall back to the full path
            total = None
        if total is not None:
            evidence = _sampled_evidence_bundle(backend, sample_rows, total)
            checks.append({"check": "non_empty_evidence", "passed": total > 0, "evidence_count": total})
            answer = (total > 0) if spec.answer_operation == "exists" else total
            return ExecutionResult(query_spec=spec, status="passed", answer=answer, evidence=evidence,
                                   checks=tuple(checks), metrics={"evidence_count": total})

    if spec.answer_operation == "top_k" and hasattr(backend, "top_k") and hasattr(backend, "sample"):
        group_by = str(spec.metadata.get("group_by_field") or "")
        k = int(spec.metadata.get("k") or 3)
        metric = str(spec.metadata.get("metric") or "count")
        metric_field = str(spec.metadata.get("metric_field") or "value_amount")
        if group_by:
            try:
                ranked = backend.top_k(spec.constraints, group_by=group_by, k=k, metric=metric,
                                       metric_field=metric_field,
                                       dedupe_field=spec.dedupe_key or "contract_node_id")
            except Exception:  # pragma: no cover - defensive; fall back to full path
                ranked = {"passed": False, "reason": "backend_top_k_error"}
            if ranked.get("passed"):
                total = backend.count(spec.constraints, dedupe_field=spec.dedupe_key or "contract_node_id") \
                    if hasattr(backend, "count") else int(ranked.get("groups") or 0)
                sample_rows = backend.sample(spec.constraints, _EVIDENCE_ID_CAP)
                evidence = _sampled_evidence_bundle(backend, sample_rows, total)
                checks.append({"check": "top_k_group_field", "passed": True, "group_by_field": group_by})
                return ExecutionResult(query_spec=spec, status="passed", answer=ranked.get("answer") or [],
                                       evidence=evidence, checks=tuple(checks),
                                       metrics={"group_by_field": group_by, "k": k,
                                                "groups": ranked.get("groups"), "backend_top_k": True})
            if ranked.get("reason") == "non_additive_values":
                checks.append({"check": "sum_additive_safety", "passed": False, "reason": "non_additive_values"})
                return ExecutionResult(query_spec=spec, status="incomplete_evidence",
                                       checks=tuple(checks), metrics={"backend_top_k": True})

    try:
        projected_fields = _projection_fields(spec)
        if projected_fields and hasattr(backend, "project"):
            rows = backend.project(spec.constraints, projected_fields)
        else:
            rows = backend.query(spec.constraints)
    except Exception as exc:  # pragma: no cover - defensive boundary
        return ExecutionResult(query_spec=spec, status="error", checks=tuple(checks), error=str(exc))

    rows = _dedupe_rows(rows, spec.dedupe_key)
    evidence = _evidence_bundle(backend, rows)
    checks.append({"check": "non_empty_evidence", "passed": bool(rows), "evidence_count": len(rows)})

    # exists and count answer over the (possibly empty) match set (0 / False is a valid answer), so
    # they precede the empty gate. select_unique / sum / argmax / distinct_set / top_k need >=1 row.
    if spec.answer_operation == "exists":
        return ExecutionResult(query_spec=spec, status="passed", answer=bool(rows), evidence=evidence,
                               checks=tuple(checks), metrics={"evidence_count": len(rows)})
    if spec.answer_operation == "count":
        return ExecutionResult(query_spec=spec, status="passed", answer=len(rows), evidence=evidence,
                               checks=tuple(checks), metrics={"evidence_count": len(rows)})
    if not rows:
        return ExecutionResult(query_spec=spec, status="no_results", evidence=evidence, checks=tuple(checks))

    if spec.answer_operation == "top_k":
        return _execute_top_k(spec, rows, evidence, checks)
    if spec.answer_operation in {"argmax", "argmin"}:
        return _execute_extreme(backend, spec, rows, evidence, checks)
    if spec.answer_operation in {"distinct_set", "set"}:
        return _execute_distinct_set(spec, rows, evidence, checks)
    if spec.answer_operation == "predicate":
        return _execute_predicate(spec, rows, evidence, checks)

    values = [_coerce_value(row.get(spec.answer_field), spec.answer_value_type) for row in rows]
    values = [value for value in values if value not in (None, "")]
    checks.append({"check": "answer_field_non_empty", "passed": bool(values), "answer_field": spec.answer_field})
    if not values:
        return ExecutionResult(query_spec=spec, status="no_results", evidence=evidence, checks=tuple(checks))

    if spec.answer_operation == "select_unique":
        unique_values = sorted(set(values), key=lambda value: str(value))
        checks.append({"check": "select_unique", "passed": len(unique_values) == 1, "unique_values": [str(v) for v in unique_values[:10]]})
        if len(unique_values) != 1:
            return ExecutionResult(
                query_spec=spec,
                status="multiple_answers",
                evidence=evidence,
                checks=tuple(checks),
                metrics={"unique_value_count": len(unique_values)},
            )
        return ExecutionResult(query_spec=spec, status="passed", answer=unique_values[0], evidence=evidence, checks=tuple(checks))

    additive_check = additive_value_check(rows)
    checks.append(additive_check)
    if not additive_check.get("passed"):
        return ExecutionResult(query_spec=spec, status="incomplete_evidence", evidence=evidence, checks=tuple(checks))
    # values are already Decimals (currency/number coercion above); sum once, and carry a contributor
    # summary (max, count) so the sum-sanity gate need not re-scan every row.
    total = sum(values)
    metrics = {"contributor_count": len(values), "contributor_max": str(max(values)) if values else "0"}
    return ExecutionResult(query_spec=spec, status="passed", answer=total, evidence=evidence,
                           checks=tuple(checks), metrics=metrics)


def _execute_extreme(backend: RuntimeQueryBackend, spec: RuntimeQuerySpec, rows: list[dict[str, Any]],
                     evidence: EvidenceBundle, checks: list[dict[str, Any]]) -> ExecutionResult:
    """argmax/argmin: the record with the extreme numeric `sort_field`; answer is its answer_field
    (contract_node_id by default). Single exhaustive retrieval -> deterministic and exact."""
    metric_field = spec.sort_field or "value_amount"
    scored = [(row, _to_decimal_or_none(row.get(metric_field))) for row in rows]
    scored = [(row, value) for row, value in scored if value is not None]
    if spec.metadata.get("exclude_zero"):
        # "lowest NON-ZERO value": zero/placeholder values are not candidates for the extreme
        scored = [(row, value) for row, value in scored if value != 0]
    checks.append({"check": "extreme_metric_present", "passed": bool(scored), "metric_field": metric_field})
    if not scored:
        return ExecutionResult(query_spec=spec, status="no_results", evidence=evidence, checks=tuple(checks))
    pick = max if spec.answer_operation == "argmax" else min
    best_row, best_value = pick(scored, key=lambda rv: rv[1])
    answer_field = spec.answer_field or "contract_node_id"
    answer = best_row.get(answer_field)
    if answer in (None, "") or answer_field == "contract_node_id":
        answer = str(backend.record_id(best_row))
    return ExecutionResult(query_spec=spec, status="passed", answer=answer, evidence=evidence,
                           checks=tuple(checks), metrics={"metric_field": metric_field, "extreme_value": str(best_value)})


def _execute_distinct_set(spec: RuntimeQuerySpec, rows: list[dict[str, Any]],
                          evidence: EvidenceBundle, checks: list[dict[str, Any]]) -> ExecutionResult:
    """distinct_set/set: the sorted distinct values of answer_field over the exhaustive match set."""
    values = sorted({str(row.get(spec.answer_field)) for row in rows if row.get(spec.answer_field) not in (None, "")})
    checks.append({"check": "distinct_set_non_empty", "passed": bool(values), "distinct_count": len(values)})
    if not values:
        return ExecutionResult(query_spec=spec, status="no_results", evidence=evidence, checks=tuple(checks))
    return ExecutionResult(query_spec=spec, status="passed", answer=values, evidence=evidence,
                           checks=tuple(checks), metrics={"distinct_count": len(values)})


def _execute_predicate(spec: RuntimeQuerySpec, rows: list[dict[str, Any]],
                       evidence: EvidenceBundle, checks: list[dict[str, Any]]) -> ExecutionResult:
    """Boolean value-predicate: compute a subject (a single contract's field, or a sum/count over the
    match set) and compare it to a threshold from the question. subject=field needs a unique value."""
    md = spec.metadata
    subject = str(md.get("predicate_subject") or "field")
    comparator = str(md.get("comparator") or "eq")
    threshold = md.get("threshold")
    value_type = str(md.get("threshold_type") or "string")
    if subject == "count":
        subject_value: Any = len(rows)
    elif subject == "sum":
        additive = additive_value_check(rows)
        checks.append(additive)
        if not additive.get("passed"):
            return ExecutionResult(query_spec=spec, status="incomplete_evidence", evidence=evidence, checks=tuple(checks))
        values = [_to_decimal_or_none(row.get(spec.answer_field or "value_amount")) for row in rows]
        subject_value = sum(v for v in values if v is not None)
    else:  # field: needs a single consistent value across the (anchor) match set
        field = str(md.get("predicate_field") or spec.answer_field)
        distinct = {row.get(field) for row in rows if row.get(field) not in (None, "")}
        checks.append({"check": "predicate_subject_unique", "passed": len(distinct) == 1, "field": field, "n": len(distinct)})
        if len(distinct) != 1:
            status = "no_results" if not distinct else "multiple_answers"
            return ExecutionResult(query_spec=spec, status=status, evidence=evidence, checks=tuple(checks))
        subject_value = next(iter(distinct))
    answer = _compare_predicate(subject_value, comparator, threshold, value_type)
    return ExecutionResult(query_spec=spec, status="passed", answer=answer, evidence=evidence,
                           checks=tuple(checks), metrics={"subject": subject, "comparator": comparator,
                                                          "subject_value": str(subject_value)})


def _compare_predicate(value: Any, comparator: str, threshold: Any, value_type: str) -> bool:
    if value_type == "number":
        a, b = _to_decimal_or_none(value), _to_decimal_or_none(threshold)
    elif value_type == "date":
        a, b = _date10(value), _date10(threshold)
    else:
        a, b = str(value).strip().casefold(), str(threshold).strip().casefold()
    if a is None or b is None:
        return False
    if comparator in {"eq", "=="}:
        return a == b
    if comparator in {"gt", ">", "after"}:
        return a > b
    if comparator in {"gte", ">="}:
        return a >= b
    if comparator in {"lt", "<", "before"}:
        return a < b
    if comparator in {"lte", "<="}:
        return a <= b
    return False


def _date10(value: Any) -> str | None:
    import re as _re
    match = _re.search(r"\d{4}-\d{2}-\d{2}", str(value))
    return match.group(0) if match else None


def _execute_top_k(spec: RuntimeQuerySpec, rows: list[dict[str, Any]],
                   evidence: EvidenceBundle, checks: list[dict[str, Any]]) -> ExecutionResult:
    """top_k: group the exhaustive match set by group_by_field, aggregate (count|sum), rank, head-k."""
    group_by = str(spec.metadata.get("group_by_field") or "")
    k = int(spec.metadata.get("k") or 3)
    metric = str(spec.metadata.get("metric") or "count")
    metric_field = str(spec.metadata.get("metric_field") or "value_amount")
    checks.append({"check": "top_k_group_field", "passed": bool(group_by), "group_by_field": group_by})
    if not group_by:
        return ExecutionResult(query_spec=spec, status="unsupported_operation", checks=tuple(checks), evidence=evidence)
    if metric == "sum":
        additive = additive_value_check(rows)
        checks.append(additive)
        if not additive.get("passed"):
            return ExecutionResult(query_spec=spec, status="incomplete_evidence", evidence=evidence, checks=tuple(checks))
    groups: dict[str, Decimal] = {}
    for row in rows:
        key = str(row.get(group_by))
        if not key:
            continue
        if metric == "sum":
            value = _to_decimal_or_none(row.get(metric_field))
            groups[key] = groups.get(key, Decimal(0)) + (value or Decimal(0))
        else:
            groups[key] = groups.get(key, Decimal(0)) + Decimal(1)
    ranked = sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, k)]
    answer = [[key, int(value) if value == value.to_integral_value() else float(value)] for key, value in ranked]
    return ExecutionResult(query_spec=spec, status="passed", answer=answer, evidence=evidence,
                           checks=tuple(checks), metrics={"group_by_field": group_by, "k": k, "groups": len(groups)})


def _to_decimal_or_none(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() else None  # NaN/Inf must not enter min/max comparisons


def _projection_fields(spec: RuntimeQuerySpec) -> list[str]:
    """Fields needed to execute this operation, plus ids for evidence/dedupe."""
    fields: list[str] = [spec.dedupe_key or "contract_node_id", "contract_node_id", "ocid"]
    op = spec.answer_operation
    if op == "sum":
        fields += [spec.answer_field or "value_amount", "value_is_additive", "supplier_count", "buyer_count"]
    elif op == "select_unique":
        fields.append(spec.answer_field)
    elif op in {"distinct_set", "set"}:
        fields.append(spec.answer_field)
    elif op in {"argmax", "argmin"}:
        fields += [spec.sort_field or "value_amount", spec.answer_field or "contract_node_id"]
    elif op == "top_k":
        fields.append(str(spec.metadata.get("group_by_field") or ""))
        if str(spec.metadata.get("metric") or "count") == "sum":
            fields += [str(spec.metadata.get("metric_field") or "value_amount"),
                       "value_is_additive", "supplier_count", "buyer_count"]
    elif op == "predicate":
        subject = str(spec.metadata.get("predicate_subject") or "field")
        if subject == "sum":
            fields += [spec.answer_field or "value_amount", "value_is_additive", "supplier_count", "buyer_count"]
        elif subject == "field":
            fields.append(str(spec.metadata.get("predicate_field") or spec.answer_field))
    else:
        return []
    return [field for field in dict.fromkeys(fields) if field]


# Large reductions (esp. bridge hops) can match tens of thousands of rows. Materialising an id
# string + ocid + row dict for every one dominates execution time and is pure overhead: the ANSWER
# never needs it. Cap evidence samples; keep evidence_count EXACT and flag the cap for auditability.
_EVIDENCE_ID_CAP = 200


def _evidence_bundle(backend: RuntimeQueryBackend, rows: list[dict[str, Any]]) -> EvidenceBundle:
    total = len(rows)
    sample = rows[:_EVIDENCE_ID_CAP]
    evidence_ids = tuple(str(backend.record_id(row)) for row in sample)
    ocids = tuple(sorted({str(row.get("ocid")) for row in sample if row.get("ocid") not in (None, "")}))
    fields = {
        "evidence_count": total,  # exact
        "evidence_ids_capped": total > _EVIDENCE_ID_CAP,
        "evidence_ids_sampled": len(evidence_ids),
        "rows_capped": total > _EVIDENCE_ID_CAP,
        "rows_sampled": len(sample),
    }
    return EvidenceBundle(evidence_ids=evidence_ids, ocids=ocids, rows=tuple(sample), fields=fields, coverage=fields)


def _sampled_evidence_bundle(backend: RuntimeQueryBackend, sample_rows: list[dict[str, Any]], total: int) -> EvidenceBundle:
    """Evidence for the fast count/exists path: exact `total`, a capped row sample (no full set)."""
    evidence_ids = tuple(str(backend.record_id(row)) for row in sample_rows)
    ocids = tuple(sorted({str(row.get("ocid")) for row in sample_rows if row.get("ocid") not in (None, "")}))
    sampled = total > len(sample_rows)
    fields = {
        "evidence_count": total,  # exact
        "evidence_ids_capped": sampled,
        "evidence_ids_sampled": len(evidence_ids),
        "rows_sampled": sampled,
    }
    return EvidenceBundle(evidence_ids=evidence_ids, ocids=ocids, rows=tuple(sample_rows), fields=fields, coverage=fields)


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
        return _truthy(value)
    return str(value) if value_type in {"string", "date"} else value


def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Cannot convert value to decimal: {value!r}") from exc


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "y"}
    return bool(value)


__all__ = ["execute_query_spec"]
