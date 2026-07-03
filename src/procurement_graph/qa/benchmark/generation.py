"""Small schema-light helpers for generating prototype answer specs."""

from __future__ import annotations

from .kg_interface import QueryBackend
from .models import AnswerOperation, AnswerSpec, AnswerValueType, Constraint


def contract_field_spec(
    backend: QueryBackend,
    *,
    spec_id: str,
    contract_id_field: str,
    contract_id: str,
    answer_field: str,
    answer_value_type: str = "string",
    dedupe_key: str = "",
) -> AnswerSpec:
    """Create a one-contract select_unique spec for mock tests and bootstrapping."""

    constraints = (Constraint(contract_id_field, "eq", contract_id),)
    rows = backend.query(constraints)
    sampled_ids = tuple(sorted({backend.record_id(row) for row in rows if backend.record_id(row)}))
    return AnswerSpec(
        spec_id=spec_id,
        constraints=constraints,
        answer_operation="select_unique",
        answer_field=answer_field,
        answer_value_type=answer_value_type,  # type: ignore[arg-type]
        dedupe_key=dedupe_key,
        logic_chain=(f"{contract_id_field}={contract_id}", f"select {answer_field}"),
        sampled_evidence_ids=sampled_ids,
    )


def feature_set_spec(
    backend: QueryBackend,
    *,
    spec_id: str,
    constraints: tuple[Constraint, ...],
    answer_operation: AnswerOperation,
    answer_field: str,
    answer_value_type: AnswerValueType,
    dedupe_key: str = "",
    logic_label: str = "",
) -> AnswerSpec:
    """Create a spec from feature constraints that may match multiple records."""

    rows = backend.query(constraints)
    sampled_ids = tuple(sorted({backend.record_id(row) for row in rows if backend.record_id(row)}))
    logic_chain = tuple(
        [logic_label] if logic_label else [f"{constraint.field} {constraint.op} {constraint.value}" for constraint in constraints]
    )
    return AnswerSpec(
        spec_id=spec_id,
        constraints=constraints,
        answer_operation=answer_operation,
        answer_field=answer_field,
        answer_value_type=answer_value_type,
        dedupe_key=dedupe_key,
        logic_chain=(*logic_chain, f"{answer_operation} {answer_field}"),
        sampled_evidence_ids=sampled_ids,
    )


__all__ = ["contract_field_spec", "feature_set_spec"]
