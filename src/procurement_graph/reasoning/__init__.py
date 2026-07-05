"""Runtime reasoning interfaces for procurement KGQA.

Core, dependency-light stages are exported here. The KG adapter (`kg_backend`) and the LLM
planner (`llm_planner`) are intentionally NOT imported at package load: import them directly
when a real KG or a live model is needed, so importing the core reasoning logic stays free of
pandas / KG / LLM dependencies.
"""

from .models import (
    AnswerCard,
    CandidatePlan,
    ConstraintOp,
    DocumentEvidenceStatus,
    EntityLinkCandidate,
    EvidenceBundle,
    EvidenceVerdict,
    EvidenceVerdictStatus,
    ExecutionResult,
    ExecutionStatus,
    LinkedEntityType,
    PlannerStatus,
    QueryConstraint,
    QueryIntent,
    ReasoningTrace,
    ReflectionAction,
    ReflectionActionType,
    RuntimeAnswerOperation,
    RuntimeQuerySpec,
)
from .answer_card import build_answer_card
from .answer_sanity import SanityVerdict, check_sum_sanity, sanity_for_execution
from .evidence import (
    DocumentInspection,
    DocumentInspector,
    NullDocumentInspector,
    build_evidence_verdict,
)
from .executor import execute_query_spec
from .grounding import CANONICAL_FIELDS, FIELD_ALIASES, GroundingResult, ground_spec
from .graph_planning import (
    GraphExecutionPlan,
    GraphExecutionResult,
    GraphOperationUnitSpec,
    GraphRelationSpec,
    GraphReturnSpec,
    GraphVariableSpec,
    GraphVariableTrace,
    compile_graph_plan,
    execute_graph_plan,
)
from .linking import LinkingResult, OrgResolver, link_question
from .memory import ReflectorMemory
from .pipeline import ReasoningPipeline
from .verbalize import LLMVerbalizer, VerbalizationResult, answer_preserves_atoms
from .planner import (
    PLANNER_SCHEMA_VERSION,
    RULE_BASED_PLANNER_VERSION,
    FallbackChainPlanner,
    PlanningError,
    ReasoningPlanner,
    RuleBasedDryRunPlanner,
    candidate_from_payload,
    planner_output_schema,
)
from .reflector import reflect, reflect_plan
from .trace_reflector import LLMTraceAdvisor, TraceReflection, TraceReflector
from .typed_planning import (
    QUESTION_TYPES,
    ConsistencyVerdict,
    TypedLLMPlanner,
    compile_typed_plan,
    plan_consistency_check,
    question_understanding_messages,
    typed_plan_messages,
    typed_replan_messages,
)
from .diagnostics import LLMReflectionAnalyzer, LLMVerificationAnalyzer, ReflectionAnalyzer, VerificationAnalyzer
from .retrieval import (
    CandidateRetriever,
    CandidateSelection,
    CandidateSelector,
    LLMCandidateSelector,
    LexicalTopKCandidateRetriever,
    RetrievalCandidate,
    RetrievalPlan,
    TopScoreCandidateSelector,
    expansion_required,
    plan_retrieval,
    semantic_repair_spec,
)
from .verifier import (
    RuntimeQueryBackend,
    additive_value_check,
    backend_fields,
    constraint_conflicts,
    preflight_checks,
)

__all__ = [
    # models
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
    # planner
    "PLANNER_SCHEMA_VERSION",
    "RULE_BASED_PLANNER_VERSION",
    "FallbackChainPlanner",
    "PlanningError",
    "ReasoningPlanner",
    "RuleBasedDryRunPlanner",
    "candidate_from_payload",
    "planner_output_schema",
    # linking
    "LinkingResult",
    "OrgResolver",
    "link_question",
    # execution + verification
    "execute_query_spec",
    "RuntimeQueryBackend",
    "additive_value_check",
    "backend_fields",
    "constraint_conflicts",
    "preflight_checks",
    # retrieval
    "CandidateRetriever",
    "CandidateSelection",
    "CandidateSelector",
    "LLMCandidateSelector",
    "LexicalTopKCandidateRetriever",
    "RetrievalCandidate",
    "RetrievalPlan",
    "TopScoreCandidateSelector",
    "expansion_required",
    "plan_retrieval",
    "semantic_repair_spec",
    # optional LLM diagnostics
    "LLMReflectionAnalyzer",
    "LLMVerificationAnalyzer",
    "ReflectionAnalyzer",
    "VerificationAnalyzer",
    # evidence
    "DocumentInspection",
    "DocumentInspector",
    "NullDocumentInspector",
    "build_evidence_verdict",
    # grounding + answer sanity
    "CANONICAL_FIELDS",
    "FIELD_ALIASES",
    "GroundingResult",
    "ground_spec",
    "GraphExecutionPlan",
    "GraphExecutionResult",
    "GraphOperationUnitSpec",
    "GraphRelationSpec",
    "GraphReturnSpec",
    "GraphVariableSpec",
    "GraphVariableTrace",
    "compile_graph_plan",
    "execute_graph_plan",
    "SanityVerdict",
    "check_sum_sanity",
    "sanity_for_execution",
    # answer card + verbalization + reflector + memory + orchestrator
    "build_answer_card",
    "LLMVerbalizer",
    "VerbalizationResult",
    "answer_preserves_atoms",
    "reflect",
    "reflect_plan",
    "ReflectorMemory",
    "ReasoningPipeline",
    # trace-aware reflection
    "TraceReflector",
    "TraceReflection",
    "LLMTraceAdvisor",
    # typed planning DSL
    "QUESTION_TYPES",
    "ConsistencyVerdict",
    "TypedLLMPlanner",
    "compile_typed_plan",
    "plan_consistency_check",
    "question_understanding_messages",
    "typed_plan_messages",
    "typed_replan_messages",
]
