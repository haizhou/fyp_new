"""Data models for runtime procurement KGQA reasoning.

These classes define the contracts between the planner, KG executor, verifier,
evidence verdict, answer-card, and reflector stages. They intentionally mirror
the QA benchmark's AnswerSpec semantics without importing the benchmark package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


QueryIntent = Literal[
    "factoid",
    "aggregation_count",
    "aggregation_sum",
    "conjunction_constraint",
    "comparison",
    "superlative_top_k",
    "temporal",
    "categorical_cpv",
    "role_path",
    "unsupported",
    "ambiguous",
]
RuntimeAnswerOperation = Literal[
    "select_unique",
    "count",
    "sum",
    "exists",
    "argmax",
    "argmin",
    "distinct_set",
    "set",
    "predicate",
    "compare",
    "top_k",
    "duration",
    "unsupported",
]
ConstraintOp = Literal["eq", "in", "contains", "exists", "gte", "lte", "between"]
LinkedEntityType = Literal["organization", "contract", "award", "cpv", "region", "date", "value", "unknown"]
PlannerStatus = Literal["planned", "ambiguous", "unsupported", "invalid"]
ExecutionStatus = Literal[
    "not_run",
    "passed",
    "no_results",
    "multiple_answers",
    "incomplete_evidence",
    "constraint_conflict",
    "unsupported_operation",
    "schema_error",
    "error",
]
EvidenceVerdictStatus = Literal[
    "kg_supported",
    "document_supported",
    "partially_supported",
    "contradicted",
    "insufficient_evidence",
    "not_checked",
]
DocumentEvidenceStatus = Literal[
    "not_needed",
    "not_available",
    "not_checked",
    "checked_supported",
    "checked_contradicted",
    "checked_no_relevant_text",
    "access_blocked",
    "request_failed",
]
ReflectionActionType = Literal[
    "no_repair_needed",
    "ask_clarifying_question",
    "replan_query",
    "relax_non_answer_constraints",
    "run_document_verdict",
    "report_insufficient_evidence",
    "mark_unsupported",
    "escalate_for_manual_review",
]


@dataclass(frozen=True)
class QueryConstraint:
    """A structured constraint extracted from a user question."""

    field: str
    op: ConstraintOp
    value: Any = None
    visible_to_user: bool = True
    source_text: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityLinkCandidate:
    """Candidate link from a mention to a KG entity or controlled value."""

    mention: str
    linked_id: str
    linked_label: str
    entity_type: LinkedEntityType
    score: float
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeQuerySpec:
    """Executable query specification for runtime KG reasoning."""

    spec_id: str
    question: str
    intent: QueryIntent
    constraints: tuple[QueryConstraint, ...]
    answer_operation: RuntimeAnswerOperation
    answer_field: str
    answer_value_type: str
    dedupe_key: str = ""
    target_node_type: str = ""
    relation_path: tuple[str, ...] = ()
    linked_entities: tuple[EntityLinkCandidate, ...] = ()
    requires_exhaustive_retrieval: bool = True
    limit: int | None = None
    sort_field: str = ""
    sort_direction: Literal["asc", "desc", ""] = ""
    planner_version: str = ""
    schema_version: str = "runtime_query_spec_v1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidatePlan:
    """One planner candidate before deterministic execution."""

    plan_id: str
    query_spec: RuntimeQuerySpec
    status: PlannerStatus = "planned"
    confidence: float = 0.0
    rationale: str = ""
    warnings: tuple[str, ...] = ()
    raw_response: Any = None
    planner_source: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""
    fallback_planner: str = ""
    # Optional graph plan with reviewable A/B/C variables. When set, the pipeline routes to the
    # graph executor instead of treating query_spec as the planner-facing representation.
    graph_plan: Any = None
    # Optional multi-hop plan (a DecompositionPlan). When set, the pipeline routes to
    # execute_decomposition instead of the single-spec path. Typed Any to avoid a models<-decomposition
    # import cycle. query_spec still carries the terminal step's spec for card/metadata.
    decomposition: Any = None


@dataclass(frozen=True)
class EvidenceBundle:
    """Structured evidence returned by the KG executor."""

    evidence_ids: tuple[str, ...] = ()
    ocids: tuple[str, ...] = ()
    rows: tuple[dict[str, Any], ...] = ()
    triples: tuple[dict[str, Any], ...] = ()
    fields: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    """Deterministic result of running a RuntimeQuerySpec over the KG."""

    query_spec: RuntimeQuerySpec
    status: ExecutionStatus
    answer: Any = None
    evidence: EvidenceBundle = field(default_factory=EvidenceBundle)
    checks: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass(frozen=True)
class EvidenceVerdict:
    """KG-first, document-second evidence verdict for an executed answer."""

    status: EvidenceVerdictStatus
    document_status: DocumentEvidenceStatus = "not_needed"
    claim: str = ""
    kg_support: tuple[dict[str, Any], ...] = ()
    document_support: tuple[dict[str, Any], ...] = ()
    contradictions: tuple[dict[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerCard:
    """User-facing answer payload built only from verified execution output."""

    question: str
    answer: Any
    answer_text: str
    query_spec: RuntimeQuerySpec
    execution: ExecutionResult
    evidence_verdict: EvidenceVerdict | None = None
    confidence_label: Literal["high", "medium", "low", "not_answered"] = "not_answered"
    limitations: tuple[str, ...] = ()
    trace_id: str = ""
    confidence_breakdown: dict[str, Any] = field(default_factory=dict)
    sanity_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReflectionAction:
    """Structured repair route for failed or uncertain reasoning attempts."""

    action: ReflectionActionType
    reason: str
    rollback_to: str = ""
    message_to_user: str = ""
    suggested_constraints: tuple[QueryConstraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningTrace:
    """Audit trace for one end-to-end reasoning attempt."""

    question: str
    plans: tuple[CandidatePlan, ...] = ()
    selected_plan_id: str = ""
    execution: ExecutionResult | None = None
    evidence_verdict: EvidenceVerdict | None = None
    answer_card: AnswerCard | None = None
    reflection: ReflectionAction | None = None
    trace_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AnswerCard",
    "CandidatePlan",
    "ConstraintOp",
    "DocumentEvidenceStatus",
    "EntityLinkCandidate",
    "EvidenceBundle",
    "EvidenceVerdict",
    "EvidenceVerdictStatus",
    "ExecutionResult",
    "ExecutionStatus",
    "LinkedEntityType",
    "PlannerStatus",
    "QueryConstraint",
    "QueryIntent",
    "ReasoningTrace",
    "ReflectionAction",
    "ReflectionActionType",
    "RuntimeAnswerOperation",
    "RuntimeQuerySpec",
]
