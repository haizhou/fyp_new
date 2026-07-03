"""End-to-end runtime reasoning orchestrator.

Ties the stages into one auditable closed loop:

    plan -> retrieval gating / entity resolution -> deterministic execution
         -> (on failure) reflector -> bounded relax/replan retry
         -> evidence verdict -> answer card -> ReasoningTrace

The LLM (planner) only proposes; the KG executor is the sole answer authority; the reflector
proposes deterministic repairs (broaden an over-specific filter, add the additive-sum guard,
dedupe contradictory constraints) for a bounded number of rounds. Every attempt is recorded in
the returned `ReasoningTrace`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

from .answer_card import build_answer_card
from .answer_sanity import SanityVerdict, sanity_for_execution
from .evidence import DocumentInspector, EvidenceVerdict, build_evidence_verdict
from .executor import execute_query_spec
from .grounding import CANONICAL_FIELDS, ground_spec
from .linking import OrgResolver
from .models import (
    AnswerCard,
    CandidatePlan,
    ExecutionResult,
    QueryConstraint,
    ReasoningTrace,
    ReflectionAction,
    RuntimeQuerySpec,
)
from .planner import ReasoningPlanner
from .reflector import reflect, reflect_plan
from .retrieval import CandidateRetriever, CandidateSelector, plan_retrieval, semantic_repair_spec
from .diagnostics import ReflectionAnalyzer, VerificationAnalyzer
from .verifier import RuntimeQueryBackend, backend_fields, postflight_checks, preflight_checks

_RETRY_ACTIONS = frozenset({"replan_query", "relax_non_answer_constraints"})


@dataclass
class ReasoningPipeline:
    backend: RuntimeQueryBackend
    planner: ReasoningPlanner
    inspector: DocumentInspector | None = None
    org_resolver: OrgResolver | None = None
    candidate_retriever: CandidateRetriever | None = None
    candidate_selector: CandidateSelector | None = None
    verification_analyzer: VerificationAnalyzer | None = None
    reflection_analyzer: ReflectionAnalyzer | None = None
    trace_reflector: Any = None
    max_rounds: int = 3
    need_documents: bool = False
    semantic_top_k: int = 8

    def run(self, question: str) -> ReasoningTrace:
        """One reasoning attempt plus (when configured) a trace-level reflection pass.

        The trace reflector judges the FINISHED trace (answer faithfulness, plan validity),
        may trigger at most one deterministic repair execution, and logs preference data.
        It is additive: without a `trace_reflector` the behaviour is unchanged.
        """
        trace = self._run_once(question)
        if self.trace_reflector is None:
            return trace
        reflection = self.trace_reflector.reflect_trace(trace)
        trace.metadata["trace_reflection"] = reflection.to_dict()
        repairable = trace.answer_card is None or trace.answer_card.answer in (None, 0, False)
        if reflection.repair_spec is not None and repairable:
            trace = self._execute_trace_repair(question, trace, reflection)
        elif reflection.faithfulness == "suspicious" and trace.answer_card is not None \
                and trace.answer_card.answer is not None:
            # the answer is kept (executor authority) but a not-evidence-entailed verdict is
            # never presented with confidence — downgrade and disclose.
            card = replace(trace.answer_card, confidence_label="low",
                           limitations=trace.answer_card.limitations
                           + ("trace reflector: answer not entailed by recorded evidence",))
            trace = replace(trace, answer_card=card)
        self.trace_reflector.log_preference(trace, reflection)
        return trace

    def _execute_trace_repair(self, question: str, trace: ReasoningTrace, reflection: Any) -> ReasoningTrace:
        """Run the reflector's repaired spec through ground -> execute exactly once."""
        allowed_fields = frozenset(backend_fields(self.backend)) or CANONICAL_FIELDS
        grounding = ground_spec(replace(reflection.repair_spec, requires_exhaustive_retrieval=True),
                                allowed_fields=allowed_fields)
        repair_meta: dict[str, Any] = {"action": reflection.action, "grounded": grounding.ok}
        if not grounding.ok:
            trace.metadata["trace_repair"] = {**repair_meta, "reason": grounding.reason}
            return trace
        result = execute_query_spec(self.backend, grounding.spec)
        repair_meta["status"] = result.status
        trace.metadata["trace_repair"] = repair_meta
        if not result.passed:
            return trace
        verdict = build_evidence_verdict(result, inspector=self.inspector, need_documents=self.need_documents)
        card = build_answer_card(result, question=question, evidence_verdict=verdict, trace_id=trace.trace_id)
        card = replace(card, confidence_label="medium",
                       limitations=card.limitations + (f"trace-reflector repair applied: {reflection.reason}",))
        action = ReflectionAction(reflection.action, reflection.reason)
        return replace(trace, execution=result, evidence_verdict=verdict, answer_card=card, reflection=action)

    def _run_once(self, question: str) -> ReasoningTrace:
        trace_id = _trace_id(question)
        allowed_fields = frozenset(backend_fields(self.backend)) or CANONICAL_FIELDS
        workflow = _workflow_trace(need_documents=self.need_documents)
        plans = tuple(self.planner.plan(question))
        plan_traces = [_plan_trace(plan) for plan in plans]
        executable = [plan for plan in plans if plan.status == "planned"]

        if not executable:
            lead = plans[0] if plans else None
            action = reflect_plan(lead)
            spec = lead.query_spec if lead else _placeholder_spec(question)
            card = _no_answer_card(question, spec, action, trace_id)
            return ReasoningTrace(
                question=question,
                plans=plans,
                reflection=action,
                answer_card=card,
                trace_id=trace_id,
                metadata={"workflow": workflow, "plans": plan_traces, "attempts": []},
            )

        candidate = max(executable, key=lambda plan: plan.confidence)
        if getattr(candidate, "decomposition", None) is not None:
            return self._run_decomposition(question, plans, plan_traces, workflow, candidate, trace_id, allowed_fields)
        attempts: list[dict[str, Any]] = []
        repairs: list[str] = []
        grounding_changes: list[str] = []
        last_result: ExecutionResult | None = None

        for round_index in range(self.max_rounds):
            spec = candidate.query_spec
            attempt: dict[str, Any] = {
                "round": round_index,
                "plan_id": candidate.plan_id,
                "selected_plan": _plan_trace(candidate),
                "pre_ground_spec": _spec_trace(spec),
            }
            # Generate-then-ground: repair/validate the (possibly LLM-proposed) spec before execution.
            grounding = ground_spec(spec, allowed_fields=allowed_fields)
            attempt["grounding"] = {
                "ok": grounding.ok,
                "reason": grounding.reason,
                "changes": list(grounding.changes),
                "issues": list(grounding.issues),
            }
            if not grounding.ok:
                attempts.append(attempt)
                action = ReflectionAction("mark_unsupported", grounding.reason)
                attempt["reflector"] = _reflection_trace(action)
                card = _no_answer_card(question, spec, action, trace_id)
                return ReasoningTrace(question=question, plans=plans, selected_plan_id=candidate.plan_id,
                                      reflection=action, answer_card=card, trace_id=trace_id,
                                      metadata={"workflow": workflow, "plans": plan_traces, "attempts": attempts})
            spec = grounding.spec
            attempt["grounded_spec"] = _spec_trace(spec)
            grounding_changes.extend(str(change) for change in grounding.changes)

            retrieval = plan_retrieval(spec, org_resolver=self.org_resolver)
            attempt["retrieval"] = {
                "mode": retrieval.mode,
                "reason": retrieval.reason,
                "changes": list(getattr(retrieval, "changes", ())),
            }
            if retrieval.mode != "exact_filters":
                attempts.append(attempt)
                action = ReflectionAction("mark_unsupported", retrieval.reason)
                attempt["reflector"] = _reflection_trace(action)
                card = _no_answer_card(question, spec, action, trace_id)
                return ReasoningTrace(question=question, plans=plans, selected_plan_id=candidate.plan_id,
                                      reflection=action, answer_card=card, trace_id=trace_id,
                                      metadata={"workflow": workflow, "plans": plan_traces, "attempts": attempts})
            spec = retrieval.resolved_spec or spec
            attempt["resolved_spec"] = _spec_trace(spec)

            preflight = tuple(preflight_checks(self.backend, spec))
            attempt["preflight"] = {
                "checks": [dict(check) for check in preflight],
                "failed_checks": _failed_check_dicts(preflight),
            }

            result = execute_query_spec(self.backend, spec)
            last_result = result
            attempt["status"] = result.status
            attempt["execution_status"] = result.status
            attempt["execution_checks"] = list(result.checks)
            attempt["failed_checks"] = _failed_checks(result)
            if not result.passed and self.verification_analyzer is not None:
                attempt["verifier_analysis"] = self.verification_analyzer.analyze_verification(
                    question=question,
                    spec=spec,
                    result=result,
                    failed_checks=tuple(attempt["failed_checks"]),
                )
            attempts.append(attempt)

            if result.passed:
                verdict = build_evidence_verdict(result, inspector=self.inspector, need_documents=self.need_documents)
                card = build_answer_card(result, question=question, evidence_verdict=verdict, trace_id=trace_id)
                sanity = sanity_for_execution(result)
                postflight = postflight_checks(result.query_spec, result.evidence.rows,
                                               matched=result.evidence.fields.get("evidence_count"))
                attempt["evidence_verdict"] = _evidence_verdict_trace(verdict)
                attempt["answer_sanity"] = _sanity_trace(sanity)
                attempt["postflight"] = {"checks": [dict(check) for check in postflight],
                                         "failed_checks": _failed_check_dicts(postflight)}
                limitations = card.limitations + tuple(f"auto-repair: {r}" for r in repairs)
                limitations += _postflight_disclosures(postflight)
                if not sanity.ok:
                    limitations += (f"answer sanity: {sanity.caveat}",)
                label, breakdown = _finalize_confidence(candidate, result, verdict, sanity, bool(repairs))
                card = replace(card, limitations=limitations, confidence_label=label,
                               confidence_breakdown=breakdown, sanity_flags=sanity.flags)
                action = ReflectionAction("no_repair_needed", "verified answer")
                attempt["reflector"] = _reflection_trace(action)
                attempt["answer_card"] = _answer_card_trace(card)
                return ReasoningTrace(question=question, plans=plans, selected_plan_id=candidate.plan_id,
                                      execution=result, evidence_verdict=verdict, answer_card=card,
                                      reflection=action, trace_id=trace_id,
                                      metadata={"workflow": workflow, "plans": plan_traces, "attempts": attempts, "repairs": repairs,
                                                "grounding_changes": grounding_changes})

            action = reflect(result, candidate, rounds_used=round_index, max_rounds=self.max_rounds)
            attempt["reflector"] = _reflection_trace(action)
            if (
                result.status == "no_results"
                and self.candidate_retriever is not None
                and self.candidate_selector is not None
                and round_index < self.max_rounds - 1
            ):
                repaired_spec, repair_trace = semantic_repair_spec(
                    spec,
                    question=question,
                    candidate_retriever=self.candidate_retriever,
                    candidate_selector=self.candidate_selector,
                    top_k=self.semantic_top_k,
                )
                attempt["semantic_candidate_repair"] = repair_trace
                if repaired_spec is not None:
                    repairs.append("semantic candidate retrieval repaired an exact-match constraint")
                    candidate = replace(
                        candidate,
                        plan_id=f"{candidate.plan_id}:semantic{round_index + 1}",
                        query_spec=repaired_spec,
                        rationale=f"{candidate.rationale} (semantic candidate repair)",
                    )
                    continue
            if self.reflection_analyzer is not None:
                attempt["reflector_analysis"] = self.reflection_analyzer.analyze_reflection(
                    question=question,
                    result=result,
                    action=action,
                    attempts=tuple(attempts),
                )
            if action.action in _RETRY_ACTIONS and action.suggested_constraints:
                repairs.append(action.reason)
                candidate = _apply_constraints(candidate, action.suggested_constraints, round_index)
                continue

            card = build_answer_card(result, question=question, trace_id=trace_id)
            attempt["answer_card"] = _answer_card_trace(card)
            return ReasoningTrace(question=question, plans=plans, selected_plan_id=candidate.plan_id,
                                  execution=result, answer_card=card, reflection=action, trace_id=trace_id,
                                  metadata={"workflow": workflow, "plans": plan_traces, "attempts": attempts})

        # Exhausted the repair budget without a verified answer.
        action = ReflectionAction("report_insufficient_evidence", "exhausted repair rounds without a verified answer")
        card = build_answer_card(last_result, question=question, trace_id=trace_id) if last_result else _no_answer_card(
            question, candidate.query_spec, action, trace_id
        )
        if attempts:
            attempts[-1]["reflector"] = _reflection_trace(action)
            attempts[-1]["answer_card"] = _answer_card_trace(card)
        return ReasoningTrace(question=question, plans=plans, selected_plan_id=candidate.plan_id,
                              execution=last_result, answer_card=card, reflection=action, trace_id=trace_id,
                              metadata={"workflow": workflow, "plans": plan_traces, "attempts": attempts})

    def _run_decomposition(self, question, plans, plan_traces, workflow, candidate, trace_id, allowed_fields):
        """Multi-hop path: run the candidate's DecompositionPlan through the shared decomposition
        executor and build an answer card from the result."""
        from .decomposition import execute_decomposition
        from .models import EvidenceBundle

        result = execute_decomposition(self.backend, candidate.decomposition, allowed_fields=allowed_fields)
        spec = candidate.query_spec
        hops = [{"step_id": h.step_id, "kind": h.kind, "status": h.status,
                 "output_size": h.output_size, "bound_from": list(h.bound_from)} for h in result.hops]
        evidence = EvidenceBundle(evidence_ids=tuple(result.evidence_ids),
                                  fields={"evidence_count": len(result.evidence_ids)})
        execution = ExecutionResult(query_spec=spec, status=result.status,
                                    answer=result.answer if result.passed else None, evidence=evidence)
        if result.passed:
            action = ReflectionAction("no_repair_needed", "verified multi-hop answer")
            card = AnswerCard(question=question, answer=result.answer,
                              answer_text=f"{result.answer}", query_spec=spec, execution=execution,
                              confidence_label="high", trace_id=trace_id,
                              confidence_breakdown={"decomposition_hops": len(result.hops), "execution": 1.0})
        else:
            action = ReflectionAction("mark_unsupported" if result.status == "unsupported" else "report_insufficient_evidence",
                                      result.reason or result.status)
            card = _no_answer_card(question, spec, action, trace_id)
        return ReasoningTrace(question=question, plans=plans, selected_plan_id=candidate.plan_id,
                              execution=execution, answer_card=card, reflection=action, trace_id=trace_id,
                              metadata={"workflow": workflow, "plans": plan_traces, "attempts": [],
                                        "decomposition": {"combine": candidate.decomposition.combine, "hops": hops,
                                                          "status": result.status}})


def _apply_constraints(
    candidate: CandidatePlan, constraints: tuple[QueryConstraint, ...], round_index: int
) -> CandidatePlan:
    spec = replace(candidate.query_spec, constraints=tuple(constraints))
    return replace(candidate, plan_id=f"{candidate.plan_id}:repair{round_index + 1}", query_spec=spec,
                   rationale=f"{candidate.rationale} (auto-repaired)")


def _finalize_confidence(
    candidate: CandidatePlan,
    result: ExecutionResult,
    verdict: EvidenceVerdict | None,
    sanity: SanityVerdict,
    repaired: bool,
) -> tuple[str, dict[str, Any]]:
    """Decompose confidence across planning / execution / evidence / sanity (audit trail).

    A sum that fails the answer-sanity gate can never be 'high' — a suspicious total is disclosed,
    not asserted. Applied repairs (relaxations) apply a penalty so a broadened answer is not 'high'.
    """
    planning = float(candidate.confidence or 0.5)
    execution = 1.0 if result.passed else 0.0
    evidence = float(verdict.confidence) if verdict is not None else 0.6
    sanity_score = 1.0 if sanity.ok else 0.4
    penalty = 0.85 if repaired else 1.0
    score = (0.25 * planning + 0.30 * execution + 0.25 * evidence + 0.20 * sanity_score) * penalty
    if not sanity.ok or repaired:
        label = "low" if not sanity.ok else "medium"
    elif score >= 0.8:
        label = "high"
    elif score >= 0.55:
        label = "medium"
    else:
        label = "low"
    breakdown = {
        "planning_confidence": round(planning, 3),
        "execution": execution,
        "evidence_support": round(evidence, 3),
        "answer_sanity": sanity_score,
        "repair_applied": repaired,
        "final_score": round(score, 3),
    }
    return label, breakdown


def _no_answer_card(question: str, spec: RuntimeQuerySpec, action: ReflectionAction, trace_id: str) -> AnswerCard:
    execution = ExecutionResult(query_spec=spec, status="not_run")
    text = action.message_to_user or action.reason or "I could not produce a verified answer from the KG."
    return AnswerCard(
        question=question,
        answer=None,
        answer_text=text,
        query_spec=spec,
        execution=execution,
        confidence_label="not_answered",
        limitations=(action.reason,) if action.reason else (),
        trace_id=trace_id,
    )


def _placeholder_spec(question: str) -> RuntimeQuerySpec:
    return RuntimeQuerySpec(
        spec_id=_trace_id(question),
        question=question,
        intent="ambiguous",
        constraints=(),
        answer_operation="unsupported",
        answer_field="",
        answer_value_type="",
        requires_exhaustive_retrieval=False,
    )


def _trace_id(question: str) -> str:
    return "trace_" + hashlib.sha1(question.strip().casefold().encode("utf-8")).hexdigest()[:12]


def _workflow_trace(*, need_documents: bool) -> dict[str, Any]:
    stages = [
        "planner",
        "grounding",
        "retrieval",
        "preflight_verifier",
        "kg_executor",
        "runtime_verifier",
        "evidence_verdict",
        "answer_sanity",
        "reflector",
        "answer_card",
    ]
    return {
        "principle": "LLM proposes; KG executes; verifier judges; reflector refines",
        "stages": stages,
        "llm_role": "planner_only",
        "answer_authority": "deterministic_kg_executor",
        "document_evidence": "opportunistic_verdict_time" if need_documents else "disabled",
    }


def _plan_trace(plan: CandidatePlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "status": plan.status,
        "confidence": plan.confidence,
        "rationale": plan.rationale,
        "warnings": list(plan.warnings),
        "planner_source": plan.planner_source,
        "fallback_used": plan.fallback_used,
        "fallback_reason": plan.fallback_reason,
        "fallback_planner": plan.fallback_planner,
        "raw_response": plan.raw_response,
        "query_spec": _spec_trace(plan.query_spec),
    }


def _spec_trace(spec: RuntimeQuerySpec) -> dict[str, Any]:
    return {
        "spec_id": spec.spec_id,
        "intent": spec.intent,
        "constraints": [_constraint_trace(constraint) for constraint in spec.constraints],
        "answer_operation": spec.answer_operation,
        "answer_field": spec.answer_field,
        "answer_value_type": spec.answer_value_type,
        "dedupe_key": spec.dedupe_key,
        "target_node_type": spec.target_node_type,
        "relation_path": list(spec.relation_path),
        "requires_exhaustive_retrieval": spec.requires_exhaustive_retrieval,
        "limit": spec.limit,
        "sort_field": spec.sort_field,
        "sort_direction": spec.sort_direction,
        "planner_version": spec.planner_version,
        "schema_version": spec.schema_version,
        "metadata": spec.metadata,
    }


def _constraint_trace(constraint: QueryConstraint) -> dict[str, Any]:
    return {
        "field": constraint.field,
        "op": constraint.op,
        "value": constraint.value,
        "visible_to_user": constraint.visible_to_user,
        "source_text": constraint.source_text,
        "confidence": constraint.confidence,
        "metadata": constraint.metadata,
    }


def _postflight_disclosures(checks: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    """Turn failed post-execution answer-quality checks into disclosed limitations.

    These never change the (correct) answer; they make it self-explaining — e.g. a count that
    includes notices with no recorded supplier says so, which is exactly the gap between a faithful
    count and a supplier/buyer-complete count.
    """
    notes: list[str] = []
    for check in checks:
        if check.get("passed"):
            continue
        if check.get("check") == "population_coverage":
            parts = []
            if check.get("without_supplier"):
                parts.append(f"{check['without_supplier']} with no recorded supplier")
            if check.get("without_buyer"):
                parts.append(f"{check['without_buyer']} with no recorded buyer")
            if parts:
                notes.append(
                    f"population coverage: of {check.get('matched')} matched contracts, "
                    + ", ".join(parts)
                    + " (all counted; a supplier/buyer-complete figure would be lower)"
                )
        elif check.get("check") == "answer_uniqueness":
            notes.append(
                f"answer confirmed across {check.get('matching_contracts')} matching contracts "
                "(they agree on this field)"
            )
    return tuple(notes)


def _failed_checks(result: ExecutionResult) -> list[dict[str, Any]]:
    return [dict(check) for check in result.checks if not check.get("passed")]


def _failed_check_dicts(checks: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    return [dict(check) for check in checks if not check.get("passed")]


def _evidence_verdict_trace(verdict: EvidenceVerdict) -> dict[str, Any]:
    return {
        "status": verdict.status,
        "document_status": verdict.document_status,
        "claim": verdict.claim,
        "kg_support": list(verdict.kg_support),
        "document_support": list(verdict.document_support),
        "contradictions": list(verdict.contradictions),
        "limitations": list(verdict.limitations),
        "confidence": verdict.confidence,
        "metadata": verdict.metadata,
    }


def _sanity_trace(sanity: SanityVerdict) -> dict[str, Any]:
    return {
        "ok": sanity.ok,
        "flags": list(sanity.flags),
        "dominant_share": sanity.dominant_share,
        "contributor_count": sanity.contributor_count,
        "caveat": sanity.caveat,
    }


def _reflection_trace(action: ReflectionAction) -> dict[str, Any]:
    return {
        "action": action.action,
        "reason": action.reason,
        "rollback_to": action.rollback_to,
        "message_to_user": action.message_to_user,
        "suggested_constraints": [_constraint_trace(constraint) for constraint in action.suggested_constraints],
        "metadata": action.metadata,
    }


def _answer_card_trace(card: AnswerCard) -> dict[str, Any]:
    return {
        "answer": card.answer,
        "answer_text": card.answer_text,
        "confidence_label": card.confidence_label,
        "confidence_breakdown": card.confidence_breakdown,
        "limitations": list(card.limitations),
        "sanity_flags": list(card.sanity_flags),
        "trace_id": card.trace_id,
    }


__all__ = ["ReasoningPipeline"]
