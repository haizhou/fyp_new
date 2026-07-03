"""Evidence verdict (design Stage 5): KG-first, documents-second.

The KG is the authoritative evidence source. For a verified execution this builds a
`kg_supported` verdict directly from the executor's evidence. Documents are passive and
opportunistic: an optional `DocumentInspector` may be consulted only when a claim needs
clause-level support, and its failure (blocked, empty, unreachable) is recorded as a
limitation rather than downgrading a KG-verified answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import DocumentEvidenceStatus, EvidenceVerdict, ExecutionResult


@dataclass(frozen=True)
class DocumentInspection:
    status: DocumentEvidenceStatus = "not_checked"
    support: tuple[dict[str, Any], ...] = ()
    contradictions: tuple[dict[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()


class DocumentInspector(Protocol):
    """Opportunistic, live document check for an executed answer. No offline corpus."""

    def inspect(self, result: ExecutionResult) -> DocumentInspection:
        ...


@dataclass(frozen=True)
class NullDocumentInspector:
    """Default: never fetch documents; the KG answer stands on its own."""

    def inspect(self, result: ExecutionResult) -> DocumentInspection:
        return DocumentInspection(status="not_needed")


def build_evidence_verdict(
    result: ExecutionResult,
    *,
    inspector: DocumentInspector | None = None,
    need_documents: bool = False,
) -> EvidenceVerdict:
    """KG-first verdict for an execution result, optionally augmented by a document check."""
    if not result.passed:
        return EvidenceVerdict(
            status="insufficient_evidence",
            document_status="not_checked",
            claim=_claim(result),
            limitations=tuple(result.warnings) or (f"execution status: {result.status}",),
            confidence=0.0,
        )

    evidence = result.evidence
    kg_support = (
        {
            "operation": result.query_spec.answer_operation,
            "answer": _jsonable(result.answer),
            "evidence_count": evidence.fields.get("evidence_count", len(evidence.evidence_ids)),  # exact (ids may be capped)
            "evidence_ids": list(evidence.evidence_ids[:50]),
            "evidence_ids_capped": bool(evidence.fields.get("evidence_ids_capped")),
            "ocids": list(evidence.ocids[:50]),
        },
    )
    limitations = list(result.warnings)

    document_status: DocumentEvidenceStatus = "not_needed"
    document_support: tuple[dict[str, Any], ...] = ()
    contradictions: tuple[dict[str, Any], ...] = ()
    status = "kg_supported"

    if need_documents and inspector is not None:
        inspection = inspector.inspect(result)
        document_status = inspection.status
        document_support = inspection.support
        contradictions = inspection.contradictions
        limitations.extend(inspection.limitations)
        if inspection.status == "checked_contradicted" and contradictions:
            status = "contradicted"
        elif inspection.status == "checked_supported":
            status = "document_supported"
        elif inspection.status in {"not_available", "access_blocked", "request_failed", "checked_no_relevant_text"}:
            limitations.append(f"document check inconclusive ({inspection.status}); relying on KG evidence")

    confidence = 0.9 if status in {"kg_supported", "document_supported"} else (0.3 if status == "contradicted" else 0.6)
    return EvidenceVerdict(
        status=status,
        document_status=document_status,
        claim=_claim(result),
        kg_support=kg_support,
        document_support=document_support,
        contradictions=contradictions,
        limitations=tuple(dict.fromkeys(limitations)),
        confidence=confidence,
    )


def _claim(result: ExecutionResult) -> str:
    spec = result.query_spec
    if not result.passed:
        return f"No verified answer ({result.status})."
    op = spec.answer_operation
    if op == "count":
        return f"{result.answer} contracts match the query."
    if op == "sum":
        return f"The total {spec.answer_field} over matching additive contracts is {result.answer}."
    return f"The {spec.answer_field} is {result.answer}."


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


__all__ = ["DocumentInspection", "DocumentInspector", "NullDocumentInspector", "build_evidence_verdict"]
