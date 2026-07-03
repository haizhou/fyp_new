"""Data models for QA benchmark construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ConstraintOp = Literal["eq", "in", "contains", "exists", "gte", "lte"]
AnswerOperation = Literal["select_unique", "count", "sum"]
AnswerValueType = Literal["string", "number", "integer", "date", "currency", "boolean"]


@dataclass(frozen=True)
class Constraint:
    field: str
    op: ConstraintOp
    value: Any = None


@dataclass(frozen=True)
class AnswerSpec:
    """Logical answer specification generated from a sampled KG subgraph."""

    spec_id: str
    constraints: tuple[Constraint, ...]
    answer_operation: AnswerOperation
    answer_field: str
    answer_value_type: AnswerValueType
    dedupe_key: str = ""
    logic_chain: tuple[str, ...] = ()
    sampled_evidence_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateReport:
    gate: str
    passed: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = ""

    @property
    def effective_status(self) -> str:
        """PASS/WARN/FAIL. Falls back to passed when status is unset."""
        if self.status:
            return self.status
        return "PASS" if self.passed else "FAIL"


@dataclass(frozen=True)
class SemanticCheckResult:
    passed: bool
    predicted_answer: Any
    reason: str
    raw_response: Any = None


@dataclass(frozen=True)
class BenchmarkExample:
    spec: AnswerSpec
    question: str
    golden_answer: Any
    evidence_ids: tuple[str, ...]
    gate_reports: tuple[GateReport, ...]
    semantic_check: SemanticCheckResult


__all__ = [
    "AnswerOperation",
    "AnswerSpec",
    "AnswerValueType",
    "BenchmarkExample",
    "Constraint",
    "ConstraintOp",
    "GateReport",
    "SemanticCheckResult",
]
