"""Answer-card construction for verified runtime reasoning results."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .models import AnswerCard, EvidenceVerdict, ExecutionResult


def build_answer_card(
    result: ExecutionResult,
    *,
    question: str | None = None,
    evidence_verdict: EvidenceVerdict | None = None,
    trace_id: str = "",
) -> AnswerCard:
    """Create a user-facing answer card without changing the computed answer."""
    prompt = question or result.query_spec.question
    if not result.passed:
        return AnswerCard(
            question=prompt,
            answer=None,
            answer_text=_failure_text(result),
            query_spec=result.query_spec,
            execution=result,
            evidence_verdict=evidence_verdict,
            confidence_label="not_answered",
            limitations=tuple(result.warnings) or (result.status,),
            trace_id=trace_id,
        )
    limitations = tuple(result.warnings)
    if evidence_verdict is not None:
        limitations = limitations + tuple(evidence_verdict.limitations)
    return AnswerCard(
        question=prompt,
        answer=result.answer,
        answer_text=_answer_text(result),
        query_spec=result.query_spec,
        execution=result,
        evidence_verdict=evidence_verdict,
        confidence_label=_confidence_label(result, evidence_verdict),
        limitations=limitations,
        trace_id=trace_id,
    )


def _answer_text(result: ExecutionResult) -> str:
    operation = result.query_spec.answer_operation
    value = _format_value(result.answer)
    if operation == "count":
        return f"The answer is {value}."
    if operation == "sum":
        return f"The total is {value}."
    return f"The answer is {value}."


def _failure_text(result: ExecutionResult) -> str:
    if result.status == "multiple_answers":
        return "I could not answer uniquely from the KG evidence."
    if result.status == "no_results":
        return "I could not find matching KG evidence for this question."
    if result.status == "incomplete_evidence":
        return "I found matching KG evidence, but it is not safe to aggregate as requested."
    if result.status == "schema_error":
        return "The planned query referenced fields that are not available in the KG."
    if result.status == "constraint_conflict":
        return "The planned query contains conflicting constraints."
    return "I could not produce a verified answer from the KG evidence."


def _confidence_label(result: ExecutionResult, evidence_verdict: EvidenceVerdict | None) -> str:
    if not result.passed:
        return "not_answered"
    if evidence_verdict is None:
        return "medium"
    if evidence_verdict.status in {"kg_supported", "document_supported"}:
        return "high"
    if evidence_verdict.status == "partially_supported":
        return "medium"
    return "low"


def _format_value(value: Any) -> str:
    if isinstance(value, Decimal):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


__all__ = ["build_answer_card"]
