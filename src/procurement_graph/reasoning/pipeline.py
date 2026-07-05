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
from .graph_planning import execute_graph_plan, graph_execution_trace, graph_plan_trace
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


@dataclass(frozen=True)
class _StaticPlanner:
    candidates: tuple[CandidatePlan, ...]

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        return self.candidates


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
    oracle_matcher: Any = None
    max_rounds: int = 3
    # Verifier-guided repair budget: Attempt 0 is the initial plan; each feedback replan is one
    # more attempt, and EVERY repaired plan re-runs the full ground -> execute -> verify path
    # (the reflector never certifies its own fix). Teacher/data-generation runs set 2-3;
    # the runtime default stays 1.
    max_feedback_replans: int = 1
    need_documents: bool = False
    semantic_top_k: int = 8

    def run(self, question: str) -> ReasoningTrace:
        """One reasoning attempt plus (when configured) a trace-level reflection pass.

        The trace reflector judges the FINISHED trace (answer faithfulness, plan validity),
        may trigger at most one deterministic repair execution, and logs preference data.
        It is additive: without a `trace_reflector` the behaviour is unchanged.
        """
        trace = self._run_once(question)
        trace = self._fallback_if_unanswered(question, trace)
        trace = self._try_flagged_answer_repair(question, trace)
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
        oracle_match = None
        if callable(self.oracle_matcher):
            oracle_match = self.oracle_matcher(question, trace)
        self.trace_reflector.log_preference(trace, reflection, oracle_match=oracle_match)
        return trace

    def _try_flagged_answer_repair(self, question: str, trace: ReasoningTrace) -> ReasoningTrace:
        """Repair the only 'wrong answers' visible without an oracle: VERIFIER-FLAGGED ones.

        Triggers solely on a failed answer-sanity gate (not on informational postflight
        disclosures). The repaired answer replaces the original only when it is strictly
        cleaner — answered AND sanity-clean; otherwise the original stands (disclosed)."""
        if self.max_feedback_replans <= 0 or trace.answer_card is None                 or trace.answer_card.answer is None or not trace.answer_card.sanity_flags:
            return trace
        replan = getattr(self.planner, "replan_with_feedback", None)
        if not callable(replan):
            return trace
        feedback = _feedback_from_trace(trace)
        feedback["failure_stage"] = "verifier_flagged"
        feedback["failure_reason"] = "answer_sanity:" + ",".join(trace.answer_card.sanity_flags)
        feedback["submitted_answer"] = _jsonable_value(trace.answer_card.answer)
        replans = tuple(replan(question, feedback))
        executable = tuple(plan for plan in replans if plan.status == "planned")
        record: dict[str, Any] = {"attempted": True, "flags": list(trace.answer_card.sanity_flags),
                                  "replaced": False}
        if executable:
            sub_pipeline = replace(self, planner=_StaticPlanner((executable[0],)),
                                   trace_reflector=None, max_feedback_replans=0)
            sub_trace = sub_pipeline._run_once(question)
            cleaner = (sub_trace.answer_card is not None
                       and sub_trace.answer_card.answer is not None
                       and not sub_trace.answer_card.sanity_flags)
            if cleaner:
                metadata = dict(sub_trace.metadata)
                metadata["flagged_answer_repair"] = {**record, "replaced": True}
                return replace(sub_trace, plans=trace.plans + replans, metadata=metadata)
        trace.metadata["flagged_answer_repair"] = record
        return trace

    def _fallback_if_unanswered(self, question: str, trace: ReasoningTrace) -> ReasoningTrace:
        """If the selected executable plan produced no answer, try later planner candidates.

        Planner confidence values are not comparable across rule / typed / LLM planners. We treat the
        planner-provided order as the policy order and only move to the next candidate after the
        current one failed to produce an answer. Valid falsey answers (0 / False) are kept.
        """
        if trace.answer_card is None or trace.answer_card.answer is not None:
            return trace
        feedback_trace = self._try_feedback_replan(question, trace)
        if feedback_trace is not trace:
            return feedback_trace
        ordered = [plan for plan in trace.plans if plan.status == "planned"]
        if len(ordered) <= 1:
            return trace
        selected = trace.selected_plan_id
        selected_index = next((idx for idx, plan in enumerate(ordered) if plan.plan_id == selected), -1)
        remaining = ordered[selected_index + 1:] if selected_index >= 0 else ordered[1:]
        if not remaining:
            return trace

        fallback_attempts: list[dict[str, Any]] = []
        original_plan_traces = trace.metadata.get("plans", [_plan_trace(plan) for plan in trace.plans])
        for candidate in remaining:
            sub_pipeline = replace(self, planner=_StaticPlanner((candidate,)), trace_reflector=None)
            sub_trace = sub_pipeline._run_once(question)
            fallback_attempts.append({
                "plan_id": candidate.plan_id,
                "status": sub_trace.execution.status if sub_trace.execution is not None else "not_run",
                "answered": bool(sub_trace.answer_card and sub_trace.answer_card.answer is not None),
            })
            if sub_trace.answer_card is not None and sub_trace.answer_card.answer is not None:
                metadata = dict(sub_trace.metadata)
                metadata["plans"] = original_plan_traces
                parent_workflow = trace.metadata.get("workflow") or {}
                if "pre_execution_review" in parent_workflow:
                    workflow = dict(metadata.get("workflow") or {})
                    workflow["pre_execution_review"] = parent_workflow["pre_execution_review"]
                    metadata["workflow"] = workflow
                metadata["plan_fallbacks"] = fallback_attempts
                return replace(sub_trace, plans=trace.plans, metadata=metadata)

        metadata = dict(trace.metadata)
        metadata["plan_fallbacks"] = fallback_attempts
        return replace(trace, metadata=metadata)

    def _try_feedback_replan(self, question: str, trace: ReasoningTrace) -> ReasoningTrace:
        """Bounded verifier-guided repair loop (attempt protocol).

        Attempt 0 is the initial plan; each iteration here is one repair attempt driven by the
        LATEST failure's feedback. A repaired plan is only accepted after re-running the full
        ground -> execute -> verify path. Per-attempt records land in
        metadata['feedback_replan']['attempts'] so a teacher run can compute Repair@k and build
        DPO pairs (chosen = first verified attempt, rejected = the nearest preceding failure).
        """
        replan = getattr(self.planner, "replan_with_feedback", None)
        if not callable(replan):
            return trace
        eligible, gate_reason = _replan_eligibility(trace)
        if not eligible:
            # A clean no_results — grounded, executed, every literal traceable to the question —
            # IS the KG's answer. Replanning it is answer-shopping: measured on dev_smoke it
            # turned 6 correct abstentions into hallucinated answers (84% -> 66%).
            trace.metadata["feedback_replan"] = {"attempted": False, "skipped": gate_reason}
            return trace
        current = trace
        all_plans = trace.plans
        attempt_records: list[dict[str, Any]] = []
        last: dict[str, Any] = {"planned": False, "plan_id": "", "status": "not_run",
                                "answered": False, "feedback": None}
        for attempt_index in range(1, self.max_feedback_replans + 1):
            feedback = _feedback_from_trace(current)
            feedback["attempt_index"] = attempt_index
            replans = tuple(replan(question, feedback))
            executable = tuple(plan for plan in replans if plan.status == "planned")
            all_plans = all_plans + replans
            if not executable:
                attempt_records.append({"attempt": attempt_index, "planned": False,
                                        "feedback": feedback})
                last = {"planned": last["planned"], "plan_id": last["plan_id"],
                        "status": last["status"], "answered": False, "feedback": feedback}
                break  # the reflector produced no executable repair; more feedback won't differ
            candidate = executable[0]
            sub_pipeline = replace(self, planner=_StaticPlanner((candidate,)), trace_reflector=None)
            sub_trace = sub_pipeline._run_once(question)
            answered = bool(sub_trace.answer_card and sub_trace.answer_card.answer is not None)
            status = sub_trace.execution.status if sub_trace.execution is not None else "not_run"
            attempt_records.append({"attempt": attempt_index, "planned": True,
                                    "plan_id": candidate.plan_id, "status": status,
                                    "answered": answered, "feedback": feedback})
            last = {"planned": True, "plan_id": candidate.plan_id, "status": status,
                    "answered": answered, "feedback": feedback}
            if answered:
                metadata = dict(sub_trace.metadata)
                metadata["plans"] = [_plan_trace(plan) for plan in all_plans]
                metadata["feedback_replan"] = {"attempted": True, **last,
                                               "attempts": attempt_records,
                                               "first_verified_attempt": attempt_index}
                return replace(sub_trace, plans=all_plans, metadata=metadata)
            current = replace(sub_trace, plans=all_plans)
        metadata = dict(trace.metadata)
        if len(all_plans) > len(trace.plans):
            metadata["plans"] = [_plan_trace(plan) for plan in all_plans]
        metadata["feedback_replan"] = {"attempted": True, **last, "attempts": attempt_records}
        return replace(trace, plans=all_plans, metadata=metadata)

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

    def _pre_execution_review_replan(
        self,
        question: str,
        plans: tuple[CandidatePlan, ...],
        plan_traces: list[dict[str, Any]],
        workflow: dict[str, Any],
        candidate: CandidatePlan,
    ) -> tuple[tuple[CandidatePlan, ...], list[dict[str, Any]], CandidatePlan]:
        """Use soft plan review as a reflector trigger, not as a veto.

        The reviewer can flag semantic mismatch before KG execution. That diagnosis is fed into
        the same `replan_with_feedback` repair path used after executor/verifier failures. If the
        repair is unavailable or non-executable, the original plan still runs, because review is a
        diagnostic signal rather than an authority.
        """
        review = _plan_review_note(candidate)
        if review.get("verdict") != "mismatch":
            if review:
                workflow["pre_execution_review"] = {
                    "triggered": False,
                    "plan_id": candidate.plan_id,
                    "verdict": review.get("verdict", ""),
                    "reason": review.get("reason", ""),
                }
            return plans, plan_traces, candidate

        feedback = _pre_execution_review_feedback(question, candidate, review)
        record: dict[str, Any] = {
            "triggered": True,
            "plan_id": candidate.plan_id,
            "verdict": review.get("verdict", ""),
            "reason": review.get("reason", ""),
            "feedback": feedback,
            "repair_attempted": False,
            "repair_planned": False,
        }
        replan = getattr(self.planner, "replan_with_feedback", None)
        if callable(replan) and self.max_feedback_replans > 0:
            record["repair_attempted"] = True
            replans = tuple(replan(question, feedback))
            record["repair_candidates"] = [_plan_trace(plan) for plan in replans]
            executable = tuple(plan for plan in replans if plan.status == "planned")
            if executable:
                repaired = executable[0]
                remaining_replans = tuple(plan for plan in replans if plan.plan_id != repaired.plan_id)
                plans = (repaired,) + plans + remaining_replans
                candidate = repaired
                record["repair_planned"] = True
                record["selected_repair_plan_id"] = repaired.plan_id
        workflow["pre_execution_review"] = record
        return plans, [_plan_trace(plan) for plan in plans], candidate

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

        candidate = executable[0]
        plans, plan_traces, candidate = self._pre_execution_review_replan(question, plans, plan_traces,
                                                                          workflow, candidate)
        if getattr(candidate, "graph_plan", None) is not None:
            return self._run_graph(question, plans, plan_traces, workflow, candidate, trace_id)
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
            deterministic_guards = _deterministic_guard_trace(spec, grounding.spec, grounding.changes)
            attempt["grounding"] = {
                "ok": grounding.ok,
                "reason": grounding.reason,
                "changes": list(grounding.changes),
                "issues": list(grounding.issues),
                "deterministic_guards_added": deterministic_guards,
            }
            if deterministic_guards:
                attempt["deterministic_guards_added"] = deterministic_guards
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
                                    answer=result.answer if result.passed else None, evidence=evidence,
                                    metrics=dict(result.metrics))
        attempt: dict[str, Any] = {
            "round": 0,
            "plan_id": candidate.plan_id,
            "selected_plan": _plan_trace(candidate),
            "decomposition": {"combine": candidate.decomposition.combine, "hops": hops, "status": result.status},
            "status": result.status,
            "execution_status": result.status,
        }
        if result.passed:
            verdict = build_evidence_verdict(execution, inspector=self.inspector, need_documents=self.need_documents)
            card = build_answer_card(execution, question=question, evidence_verdict=verdict, trace_id=trace_id)
            sanity = sanity_for_execution(execution)
            postflight = postflight_checks(execution.query_spec, execution.evidence.rows,
                                           matched=execution.evidence.fields.get("evidence_count"))
            attempt["evidence_verdict"] = _evidence_verdict_trace(verdict)
            attempt["answer_sanity"] = _sanity_trace(sanity)
            attempt["postflight"] = {"checks": [dict(check) for check in postflight],
                                     "failed_checks": _failed_check_dicts(postflight)}
            limitations = card.limitations + _postflight_disclosures(postflight)
            if not sanity.ok:
                limitations += (f"answer sanity: {sanity.caveat}",)
            label, breakdown = _finalize_confidence(candidate, execution, verdict, sanity, repaired=False)
            breakdown["decomposition_hops"] = len(result.hops)
            card = replace(card, limitations=limitations, confidence_label=label,
                           confidence_breakdown=breakdown, sanity_flags=sanity.flags)
            action = ReflectionAction("no_repair_needed", "verified multi-hop answer")
            attempt["reflector"] = _reflection_trace(action)
            attempt["answer_card"] = _answer_card_trace(card)
            return ReasoningTrace(question=question, plans=plans, selected_plan_id=candidate.plan_id,
                                  execution=execution, evidence_verdict=verdict, answer_card=card,
                                  reflection=action, trace_id=trace_id,
                                  metadata={"workflow": workflow, "plans": plan_traces, "attempts": [attempt],
                                            "decomposition": {"combine": candidate.decomposition.combine, "hops": hops,
                                                              "status": result.status}})
        else:
            action = ReflectionAction("mark_unsupported" if result.status == "unsupported" else "report_insufficient_evidence",
                                      result.reason or result.status)
            card = _no_answer_card(question, spec, action, trace_id)
            attempt["reflector"] = _reflection_trace(action)
            attempt["answer_card"] = _answer_card_trace(card)
        return ReasoningTrace(question=question, plans=plans, selected_plan_id=candidate.plan_id,
                              execution=execution, answer_card=card, reflection=action, trace_id=trace_id,
                              metadata={"workflow": workflow, "plans": plan_traces, "attempts": [attempt],
                                        "decomposition": {"combine": candidate.decomposition.combine, "hops": hops,
                                                          "status": result.status}})

    def _run_graph(self, question, plans, plan_traces, workflow, candidate, trace_id):
        """Run a reviewable A/B/C graph plan through graph grounding/execution."""
        graph_result = execute_graph_plan(self.backend, candidate.graph_plan)
        execution = graph_result.low_level_result or ExecutionResult(
            query_spec=candidate.query_spec,
            status=graph_result.status if graph_result.status in {
                "not_run", "passed", "no_results", "multiple_answers", "incomplete_evidence",
                "constraint_conflict", "unsupported_operation", "schema_error", "error",
            } else "error",
            answer=graph_result.answer,
            error=graph_result.reason,
            metrics=dict(graph_result.metrics),
        )
        attempt: dict[str, Any] = {
            "round": 0,
            "plan_id": candidate.plan_id,
            "selected_plan": _plan_trace(candidate),
            "graph_plan": graph_plan_trace(candidate.graph_plan),
            "graph_execution": graph_execution_trace(graph_result),
            "status": graph_result.status,
            "execution_status": execution.status,
            "execution_checks": list(execution.checks),
            "failed_checks": _failed_checks(execution),
        }
        if graph_result.passed and execution.passed:
            verdict = build_evidence_verdict(execution, inspector=self.inspector, need_documents=self.need_documents)
            card = build_answer_card(execution, question=question, evidence_verdict=verdict, trace_id=trace_id)
            sanity = sanity_for_execution(execution)
            postflight = postflight_checks(execution.query_spec, execution.evidence.rows,
                                           matched=execution.evidence.fields.get("evidence_count"))
            attempt["evidence_verdict"] = _evidence_verdict_trace(verdict)
            attempt["answer_sanity"] = _sanity_trace(sanity)
            attempt["postflight"] = {"checks": [dict(check) for check in postflight],
                                     "failed_checks": _failed_check_dicts(postflight)}
            limitations = card.limitations + _postflight_disclosures(postflight)
            if not sanity.ok:
                limitations += (f"answer sanity: {sanity.caveat}",)
            caveats = ((candidate.graph_plan.understanding_network or {}).get("caveats")
                       if getattr(candidate, "graph_plan", None) is not None else None) or ()
            limitations += tuple(f"planner caveat: {c}" for c in caveats)
            label, breakdown = _finalize_confidence(candidate, execution, verdict, sanity, repaired=False)
            breakdown["graph_variables"] = len(graph_result.variable_traces)
            card = replace(card, limitations=limitations, confidence_label=label,
                           confidence_breakdown=breakdown, sanity_flags=sanity.flags)
            action = ReflectionAction("no_repair_needed", "verified graph answer")
            attempt["reflector"] = _reflection_trace(action)
            attempt["answer_card"] = _answer_card_trace(card)
            return ReasoningTrace(question=question, plans=plans, selected_plan_id=candidate.plan_id,
                                  execution=execution, evidence_verdict=verdict, answer_card=card,
                                  reflection=action, trace_id=trace_id,
                                  metadata={"workflow": workflow, "plans": plan_traces, "attempts": [attempt],
                                            "graph": graph_execution_trace(graph_result)})
        action = ReflectionAction("report_insufficient_evidence", graph_result.reason or graph_result.status)
        card = _no_answer_card(question, candidate.query_spec, action, trace_id)
        attempt["reflector"] = _reflection_trace(action)
        attempt["answer_card"] = _answer_card_trace(card)
        return ReasoningTrace(question=question, plans=plans, selected_plan_id=candidate.plan_id,
                              execution=execution, answer_card=card, reflection=action, trace_id=trace_id,
                              metadata={"workflow": workflow, "plans": plan_traces, "attempts": [attempt],
                                        "graph": graph_execution_trace(graph_result)})


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


def _deterministic_guard_trace(
    original: RuntimeQuerySpec, grounded: RuntimeQuerySpec, changes: tuple[str, ...]
) -> list[dict[str, Any]]:
    original_guarded = any(
        c.field == "value_is_additive" and c.op == "eq" and bool(c.value)
        for c in original.constraints
    )
    grounded_guarded = any(
        c.field == "value_is_additive" and c.op == "eq" and bool(c.value)
        for c in grounded.constraints
    )
    if original_guarded or not grounded_guarded:
        return []
    reason = next((change for change in changes if "value_is_additive guard" in change),
                  "money aggregation requires additive contract values")
    return [{
        "field": "value_is_additive",
        "op": "eq",
        "value": True,
        "reason": reason,
    }]


def _plan_review_note(candidate: CandidatePlan) -> dict[str, Any]:
    raw = candidate.raw_response
    if isinstance(raw, dict) and isinstance(raw.get("plan_review"), dict):
        return dict(raw["plan_review"])
    return {}


def _pre_execution_review_feedback(
    question: str, candidate: CandidatePlan, review: dict[str, Any]
) -> dict[str, Any]:
    return {
        "selected_plan_id": candidate.plan_id,
        "execution_status": "not_run",
        "reflection_action": "pre_execution_review",
        "reflection_reason": str(review.get("reason") or "semantic plan review mismatch"),
        "question": question,
        "failed_plan": _plan_trace(candidate),
        "failure_stage": "pre_execution_review",
        "failure_reason": str(review.get("reason") or "semantic plan review mismatch"),
        "submitted_answer": None,
        "plan_review": review,
        "allowed_repair_actions": [
            "fix_question_type",
            "repair_constraints",
            "swap_buyer_supplier",
            "change_operator",
            "change_operation",
            "abstain",
        ],
    }


def _replan_eligibility(trace: ReasoningTrace) -> tuple[bool, str]:
    """Deterministic gate: is this failure a repairable DEFECT or a legitimate abstention?

    - never executed (compile/consistency rejection)  -> repair (structured defect reason exists)
    - executor error / conflict / schema failure      -> repair
    - no_results where some literal was INVENTED by the planner (not in the question) -> repair
    - no_results where every literal traces to the question -> the emptiness is the answer
    """
    attempts = trace.metadata.get("attempts") or ()
    if not attempts:
        return True, "plan_compile"
    last = attempts[-1]
    status = str(last.get("execution_status") or last.get("status") or "")
    if status != "no_results":
        return True, status or "unknown_failure"
    invented = _invented_literals(trace.question, last)
    if invented:
        return True, "no_results_with_invented_literals:" + ",".join(invented[:3])
    return False, "clean_no_results_is_an_answer"


def _invented_literals(question: str, attempt: dict[str, Any]) -> list[str]:
    """User-visible literal filter values that do NOT occur in the question text."""
    question_low = " ".join(str(question).casefold().split())
    constraints: list[dict[str, Any]] = []
    for key in ("resolved_spec", "grounded_spec", "pre_ground_spec"):
        spec = attempt.get(key)
        if isinstance(spec, dict) and spec.get("constraints"):
            constraints = [c for c in spec["constraints"] if isinstance(c, dict)]
            break
    graph_execution = attempt.get("graph_execution") or {}
    for var in graph_execution.get("variables", ()):
        if isinstance(var, dict):
            constraints.extend(c for c in (var.get("constraints") or ()) if isinstance(c, dict))
    invented: list[str] = []
    for item in constraints:
        if item.get("visible_to_user") is False:
            continue
        if str(item.get("source_text") or "").startswith("bound from"):
            continue
        field = str(item.get("field") or "")
        if field in ("value_is_additive", "has_award_signed_date", "contract_node_id"):
            continue
        value = item.get("value")
        values = value if isinstance(value, (list, tuple)) else [value]
        if len(values) > 4:
            continue  # long in-lists are derived sets, not question literals
        for v in values:
            if isinstance(v, bool) or v in (None, ""):
                continue
            text = " ".join(str(v).casefold().split())
            if text and text not in question_low and text not in invented:
                invented.append(text)
    return invented


def _feedback_from_trace(trace: ReasoningTrace) -> dict[str, Any]:
    attempts = trace.metadata.get("attempts") or ()
    last = attempts[-1] if attempts else {}
    failed_plan = last.get("selected_plan")
    if not attempts and trace.plans:
        # The plan never reached execution (compile/consistency rejection). Without this branch
        # the reflector got failed_plan=None and failure_stage="verifier" — exactly the cases
        # where the structured compile reason (invalid_graph_structure:..., ungroundable_variable:
        # ...) is the repair signal, and the failed graph JSON is the thing to show the repairer.
        lead = next((plan for plan in trace.plans if plan.plan_id == trace.selected_plan_id),
                    trace.plans[0])
        return {
            "selected_plan_id": lead.plan_id,
            "execution_status": "not_run",
            "reflection_action": trace.reflection.action if trace.reflection is not None else "",
            "reflection_reason": trace.reflection.reason if trace.reflection is not None else "",
            "question": trace.question,
            "failed_plan": _plan_trace(lead),
            "failure_stage": "plan_compile",
            "failure_reason": lead.rationale or (trace.reflection.reason if trace.reflection else ""),
            "submitted_answer": None,
            "allowed_repair_actions": [
                "fix_question_type",
                "repair_constraints",
                "swap_buyer_supplier",
                "change_operator",
                "change_operation",
                "abstain",
            ],
        }
    failed_checks = last.get("failed_checks") or last.get("execution_checks") or ()
    preflight = last.get("preflight") or {}
    postflight = last.get("postflight") or {}
    sanity = last.get("answer_sanity") or {}
    grounding = last.get("grounding") or {}
    retrieval = last.get("retrieval") or {}
    graph_execution = last.get("graph_execution") or {}
    failed_graph_vars = [v for v in graph_execution.get("variables", ()) if v.get("status") != "passed"]
    failure_stage = _failure_stage_from_attempt(last, trace)
    failure_reason = _failure_reason_from_attempt(last, trace)
    submitted_answer = None
    if trace.answer_card is not None:
        submitted_answer = trace.answer_card.answer
    elif trace.execution is not None:
        submitted_answer = trace.execution.answer
    return {
        "selected_plan_id": trace.selected_plan_id,
        "execution_status": trace.execution.status if trace.execution is not None else "not_run",
        "reflection_action": trace.reflection.action if trace.reflection is not None else "",
        "reflection_reason": trace.reflection.reason if trace.reflection is not None else "",
        "question": trace.question,
        "failed_plan": failed_plan,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "submitted_answer": _jsonable_value(submitted_answer),
        "grounding_issues": list(grounding.get("issues") or ()),
        "schema_errors": list(preflight.get("failed_checks") or ()),
        "failed_checks": failed_checks,
        "preflight_failed_checks": list(preflight.get("failed_checks") or ()),
        "postflight_failed_checks": list(postflight.get("failed_checks") or ()),
        "answer_sanity": sanity,
        "grounding": grounding,
        "retrieval": retrieval,
        "graph_execution": graph_execution,
        "graph_failure_kind": graph_execution.get("failure_kind", ""),
        "failed_operation_unit": failed_graph_vars[-1] if failed_graph_vars else {},
        "deterministic_guards_added": list(last.get("deterministic_guards_added")
                                           or grounding.get("deterministic_guards_added") or ()),
        "allowed_repair_actions": [
            "fix_question_type",
            "repair_constraints",
            "swap_buyer_supplier",
            "change_operator",
            "change_operation",
            "abstain",
        ],
    }


def _failure_stage_from_attempt(attempt: dict[str, Any], trace: ReasoningTrace) -> str:
    graph_execution = attempt.get("graph_execution") or {}
    if graph_execution.get("status") and graph_execution.get("status") != "passed":
        kind = graph_execution.get("failure_kind")
        if kind == "syntax_error":
            return "schema"
        if kind in {"empty_result", "empty_or_semantic_result"}:
            return "executor"
        return "graph_executor"
    grounding = attempt.get("grounding") or {}
    if grounding and not grounding.get("ok", True):
        return "grounding"
    preflight = attempt.get("preflight") or {}
    if preflight.get("failed_checks"):
        return "schema"
    if attempt.get("execution_status") not in (None, "", "passed"):
        return "executor"
    sanity = attempt.get("answer_sanity") or {}
    if sanity and not sanity.get("ok", True):
        return "sanity"
    postflight = attempt.get("postflight") or {}
    if postflight.get("failed_checks"):
        return "verifier"
    if trace.answer_card is not None and trace.answer_card.answer is None:
        return "verifier"
    return "verifier"


def _failure_reason_from_attempt(attempt: dict[str, Any], trace: ReasoningTrace) -> str:
    graph_execution = attempt.get("graph_execution") or {}
    if graph_execution.get("status") and graph_execution.get("status") != "passed":
        failed = [v for v in graph_execution.get("variables", ()) if v.get("status") != "passed"]
        if failed:
            unit = failed[-1]
            return f"{graph_execution.get('failure_kind') or 'graph_failure'}:{unit.get('operation_unit')}:{unit.get('reason')}"
        return str(graph_execution.get("reason") or graph_execution.get("status"))
    grounding = attempt.get("grounding") or {}
    if grounding and not grounding.get("ok", True):
        return str(grounding.get("reason") or "grounding_failed")
    retrieval = attempt.get("retrieval") or {}
    if retrieval.get("mode") and retrieval.get("mode") != "exact_filters":
        return str(retrieval.get("reason") or "retrieval_not_exact")
    status = attempt.get("execution_status")
    if status and status != "passed":
        return str(status)
    sanity = attempt.get("answer_sanity") or {}
    if sanity and not sanity.get("ok", True):
        return str(sanity.get("caveat") or "answer_sanity_failed")
    reflection = attempt.get("reflector") or {}
    if reflection.get("reason"):
        return str(reflection.get("reason"))
    if trace.reflection is not None:
        return trace.reflection.reason
    return "unknown_failure"


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    return str(value)


def _trace_id(question: str) -> str:
    return "trace_" + hashlib.sha1(question.strip().casefold().encode("utf-8")).hexdigest()[:12]


def _workflow_trace(*, need_documents: bool) -> dict[str, Any]:
    stages = [
        "planner",
        "pre_execution_review",
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
        "graph_plan": graph_plan_trace(plan.graph_plan) if getattr(plan, "graph_plan", None) is not None else None,
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
