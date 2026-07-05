"""Trace-aware reflection (Stage 8): judge the WHOLE reasoning trace, not one stage.

The per-stage reflector (`reflector.py`) routes repairs while the loop is running; this module
looks at the finished `ReasoningTrace` and answers four questions the dissertation cares about:

1. **Answer faithfulness** — is the reported answer actually entailed by the recorded execution
   evidence (count == evidence_count, sum has contributors and dominates its max contributor,
   factoid uniqueness held, every decomposition hop passed)? This is independent of any gold
   answer: it catches an answer the executor produced but the evidence does not support.
2. **Plan validity** — do the question-type cues (count / sum / factoid / boolean / superlative)
   agree with the operation that was executed, and are the entity links confident and grounded in
   the question text?
3. **Targeted repair** — when the trace failed or verification is weak, choose ONE next action
   from a closed set: `re_link_entity` | `re_plan_question_type` | `relax_constraints` |
   `ask_clarifying_question` | `abstain` | `mark_ambiguous` | `no_action`. Repairs that change
   a spec are re-executed by the deterministic executor (bounded to one extra round by the
   pipeline); the reflector never produces an answer itself.
4. **Preference logging** — append the chosen plan (and the alternatives it beat) with the
   faithfulness verdict as good/bad preference-style JSONL for future planner fine-tuning.

An optional LLM advisor is consulted ONLY when the deterministic verdicts are uncertain; its
proposal must name an action from the closed set (and an entity from the candidate list we hand
it) or it is ignored. The KG executor remains the sole answer authority.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .entity_resolution import resolve_confident_org
from .linking import link_question
from .models import QueryConstraint, ReasoningTrace, RuntimeQuerySpec

REPAIR_ACTIONS = frozenset({
    "re_link_entity", "re_plan_question_type", "relax_constraints",
    "ask_clarifying_question", "abstain", "mark_ambiguous", "no_action",
})
_ORG_FIELDS = ("buyer_name", "supplier_name")
_TYPE_OPS = {
    "count": frozenset({"count"}),
    "sum": frozenset({"sum"}),
    "factoid": frozenset({"select_unique"}),
    "boolean": frozenset({"exists", "predicate", "compare"}),
    "superlative": frozenset({"argmax", "argmin", "top_k"}),
    "set": frozenset({"distinct_set", "set", "select_unique"}),
}


@dataclass(frozen=True)
class TraceReflection:
    """Verdict + decision for one finished trace (attached under metadata['trace_reflection'])."""

    faithfulness: str                      # faithful | abstained | suspicious | not_run
    faithfulness_checks: tuple[dict[str, Any], ...]
    plan_valid: bool
    plan_issues: tuple[str, ...]
    action: str                            # one of REPAIR_ACTIONS
    reason: str
    repair_spec: RuntimeQuerySpec | None = None
    advisor: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "faithfulness": self.faithfulness,
            "faithfulness_checks": [dict(check) for check in self.faithfulness_checks],
            "plan_valid": self.plan_valid,
            "plan_issues": list(self.plan_issues),
            "action": self.action,
            "reason": self.reason,
            "repaired": self.repair_spec is not None,
        }
        if self.advisor is not None:
            payload["advisor"] = self.advisor
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class TraceReflector:
    """Deterministic trace verifier + repair policy, with an optional LLM advisor.

    `advisor` is any object with `advise(payload: dict) -> dict` (see `LLMTraceAdvisor`); it is
    consulted only when the deterministic policy lands on abstain/ask with weak signals, and its
    output is validated against the closed action set before it can influence the decision.
    """

    org_resolver: Any = None
    advisor: Any = None
    preference_log: Path | str | None = None

    # ------------------------------------------------------------------ main entry
    def reflect_trace(self, trace: ReasoningTrace) -> TraceReflection:
        faithfulness, checks = self._verify_faithfulness(trace)
        plan_valid, issues = self._verify_plan(trace)
        action, reason, repair_spec, extra = self._decide(trace, faithfulness, plan_valid, issues)

        advisor_payload: dict[str, Any] | None = None
        if self.advisor is not None and action in {"abstain", "ask_clarifying_question"}:
            advisor_payload = self._consult_advisor(trace, faithfulness, issues, action)
            proposed = (advisor_payload or {}).get("recommended_action", "")
            if proposed in REPAIR_ACTIONS and proposed != action:
                # the advisor may re-route between conservative actions or nominate a repair the
                # deterministic policy can build; it cannot mint an answer.
                if proposed == "re_link_entity":
                    choice = (advisor_payload or {}).get("entity_choice") or {}
                    repaired = self._relink_spec(trace, preferred=choice)
                    if repaired is not None:
                        action, reason, repair_spec = proposed, "advisor proposed entity re-link", repaired
                elif proposed in {"mark_ambiguous", "ask_clarifying_question", "abstain"}:
                    action, reason = proposed, f"advisor re-routed from {action}"

        return TraceReflection(
            faithfulness=faithfulness,
            faithfulness_checks=tuple(checks),
            plan_valid=plan_valid,
            plan_issues=tuple(issues),
            action=action,
            reason=reason,
            repair_spec=repair_spec,
            advisor=advisor_payload,
            metadata=extra,
        )

    # ------------------------------------------------------------------ 1. faithfulness
    def _verify_faithfulness(self, trace: ReasoningTrace) -> tuple[str, list[dict[str, Any]]]:
        checks: list[dict[str, Any]] = []
        card = trace.answer_card
        execution = trace.execution
        if card is None or execution is None or execution.status == "not_run":
            return "not_run", checks
        if card.answer is None:
            return "abstained", checks
        spec = execution.query_spec
        evidence_count = None
        if execution.evidence is not None:
            evidence_count = execution.evidence.fields.get("evidence_count")
            if evidence_count is None:
                evidence_count = len(execution.evidence.evidence_ids)
        metrics = execution.metrics or {}
        # A decomposition trace carries CAPPED evidence ids (its exact totals live per hop), so the
        # scalar-vs-evidence checks below would misfire; hop statuses are its faithfulness signal.
        is_decomposition = bool((trace.metadata or {}).get("decomposition"))

        ok = True
        if is_decomposition:
            evidence_count = None
        if spec.answer_operation == "count" and evidence_count is not None:
            passed = int(card.answer) == int(evidence_count)
            checks.append({"check": "count_equals_evidence", "passed": passed,
                           "answer": card.answer, "evidence_count": evidence_count})
            ok &= passed
        elif spec.answer_operation == "exists" and evidence_count is not None:
            passed = bool(card.answer) == (int(evidence_count) > 0)
            checks.append({"check": "exists_consistent_with_evidence", "passed": passed,
                           "answer": card.answer, "evidence_count": evidence_count})
            ok &= passed
        elif spec.answer_operation == "sum" and not is_decomposition:
            contributors = int(metrics.get("contributor_count") or 0)
            checks.append({"check": "sum_has_contributors", "passed": contributors > 0,
                           "contributor_count": contributors})
            ok &= contributors > 0
            max_contribution = metrics.get("contributor_max")
            if max_contribution is not None:
                try:
                    passed = float(card.answer) >= float(max_contribution)
                except (TypeError, ValueError):
                    passed = False
                checks.append({"check": "sum_not_below_max_contributor", "passed": passed})
                ok &= passed
        elif spec.answer_operation == "select_unique":
            unique = [c for c in execution.checks if c.get("check") == "select_unique"]
            passed = bool(unique) and all(c.get("passed") for c in unique)
            checks.append({"check": "uniqueness_held", "passed": passed})
            ok &= passed

        decomposition = (trace.metadata or {}).get("decomposition")
        if decomposition:
            hops = decomposition.get("hops", [])
            passed = bool(hops) and all(h.get("status") == "passed" for h in hops)
            checks.append({"check": "all_hops_passed", "passed": passed, "hops": len(hops)})
            ok &= passed

        return ("faithful" if ok else "suspicious"), checks

    # ------------------------------------------------------------------ 2. plan validity
    def _verify_plan(self, trace: ReasoningTrace) -> tuple[bool, list[str]]:
        issues: list[str] = []
        execution = trace.execution
        if execution is None:
            return True, issues
        spec = execution.query_spec
        expected = self._expected_type(trace.question)
        if expected and spec.answer_operation not in _TYPE_OPS[expected]:
            issues.append(f"question reads as {expected!r} but executed operation is "
                          f"{spec.answer_operation!r}")
        lowered = trace.question.casefold()
        for constraint in spec.constraints:
            if not constraint.visible_to_user:
                continue
            if constraint.confidence < 0.6:
                issues.append(f"weak entity link on {constraint.field} "
                              f"({constraint.source_text!r} -> {constraint.value!r}, "
                              f"confidence {constraint.confidence})")
            source = str(constraint.source_text or "").casefold()
            if constraint.field in _ORG_FIELDS and source and source not in lowered \
                    and str(constraint.value).casefold() not in lowered:
                issues.append(f"constraint on {constraint.field} is not grounded in the question "
                              f"text ({constraint.source_text!r})")
        return not issues, issues

    @staticmethod
    def _expected_type(question: str) -> str:
        lowered = " ".join(question.casefold().split())
        link = link_question(question)
        if any(cue in lowered for cue in ("top ", "highest", "lowest", "most expensive", "cheapest")):
            return "superlative"
        if link.asks_sum:
            return "sum"
        if link.asks_count:
            return "count"
        if lowered.startswith(("did ", "is ", "was ", "were ", "are ")) or " did " in lowered:
            return "boolean"
        if link.factoid_field:
            return "factoid"
        return ""

    # ------------------------------------------------------------------ 3. repair decision
    def _decide(self, trace: ReasoningTrace, faithfulness: str, plan_valid: bool,
                issues: list[str]) -> tuple[str, str, RuntimeQuerySpec | None, dict[str, Any]]:
        execution = trace.execution
        status = execution.status if execution is not None else "not_run"
        extra: dict[str, Any] = {}

        if faithfulness == "faithful" and status == "passed":
            # A zero/False over an org constraint whose mention ALSO matches another surface
            # variant of the same organisation (case/punctuation) is the entity-variant trap:
            # re-link to the sibling variant. Genuinely different orgs are never swapped in.
            card = trace.answer_card
            if card is not None and card.answer in (0, False):
                repaired = self._relink_spec(trace, variant_only=True)
                if repaired is not None:
                    return ("re_link_entity",
                            "zero result over an organisation with sibling surface variants",
                            repaired, extra)
            if plan_valid:
                return "no_action", "answer is evidence-faithful and the plan matches the question", None, extra
            # answer stands (executor verified it) but the plan smells — surface for audit/logging.
            return "no_action", "answer is evidence-faithful; plan issues recorded for audit", None, extra

        if faithfulness == "suspicious":
            return "abstain", "produced answer is not entailed by the recorded evidence", None, extra

        if status == "multiple_answers":
            unique = next((c for c in (execution.checks if execution else ())
                           if c.get("check") == "select_unique"), {})
            extra["distinct_values"] = list(unique.get("unique_values", []))[:10]
            return ("mark_ambiguous",
                    "the question underdetermines the contract: several distinct values match",
                    None, extra)

        if status == "no_results":
            repaired = self._relink_spec(trace)
            if repaired is not None:
                return ("re_link_entity",
                        "no results; an organisation constraint has a plausible alternative link",
                        repaired, extra)

        if not plan_valid and any("executed operation" in issue for issue in issues):
            repaired = self._retype_spec(trace)
            if repaired is not None:
                return ("re_plan_question_type",
                        "execution failed and the executed operation disagrees with the question type",
                        repaired, extra)

        if status == "no_results":
            return "relax_constraints", "no matching rows; a broader filter set may recover evidence", None, extra
        if status in {"not_run", "unsupported_operation", "schema_error"}:
            return "abstain", f"execution status {status}: nothing further to repair deterministically", None, extra
        return "abstain", f"no targeted repair available for status {status!r}", None, extra

    def _relink_spec(self, trace: ReasoningTrace, preferred: dict[str, Any] | None = None,
                     *, variant_only: bool = False) -> RuntimeQuerySpec | None:
        """Swap ONE org constraint to its next-best resolver candidate (or an advisor-validated one).

        With `variant_only`, an alternative is accepted only when it normalises to the same name as
        the current value (case/punctuation sibling), so a verified answer can never be re-routed
        to a different real-world organisation.
        """
        if self.org_resolver is None or trace.execution is None:
            return None
        spec = trace.execution.query_spec
        for constraint in spec.constraints:
            if constraint.field not in _ORG_FIELDS or constraint.op != "eq":
                continue
            mention = str(constraint.source_text or constraint.value)
            if preferred:
                choice = str(preferred.get("value", ""))
                resolved = resolve_confident_org(self.org_resolver, mention,
                                                 exclude_ids={str(constraint.value)})
                if preferred.get("field") == constraint.field and resolved.ok \
                        and resolved.hit is not None and choice == resolved.hit.linked_id:
                    return _swap_constraint(spec, constraint, choice)
                continue
            resolved = resolve_confident_org(
                self.org_resolver,
                mention,
                exclude_ids={str(constraint.value)},
                variant_only_against=str(constraint.value) if variant_only else "",
            )
            if resolved.ok and resolved.hit is not None:
                return _swap_constraint(spec, constraint, resolved.hit.linked_id)
        return None

    def _retype_spec(self, trace: ReasoningTrace) -> RuntimeQuerySpec | None:
        """Re-issue the spec under the operation the question cues actually indicate."""
        if trace.execution is None:
            return None
        spec = trace.execution.query_spec
        expected = self._expected_type(trace.question)
        if expected == "count":
            return replace(spec, answer_operation="count", answer_field="contract_node_id",
                           answer_value_type="integer")
        if expected == "sum":
            return replace(spec, answer_operation="sum", answer_field="value_amount",
                           answer_value_type="currency")
        if expected == "boolean" and spec.answer_operation == "count":
            return replace(spec, answer_operation="exists", answer_value_type="boolean")
        return None

    # ------------------------------------------------------------------ advisor boundary
    def _consult_advisor(self, trace: ReasoningTrace, faithfulness: str,
                         issues: list[str], tentative: str) -> dict[str, Any]:
        execution = trace.execution
        spec = execution.query_spec if execution is not None else None
        org_candidates: dict[str, list[str]] = {}
        if self.org_resolver is not None and spec is not None:
            for constraint in spec.constraints:
                if constraint.field in _ORG_FIELDS:
                    mention = str(constraint.source_text or constraint.value)
                    org_candidates[constraint.field] = [
                        hit.linked_id for hit in self.org_resolver.resolve(mention)][:5]
        payload = {
            "question": trace.question,
            "execution_status": execution.status if execution is not None else "not_run",
            "faithfulness": faithfulness,
            "plan_issues": issues,
            "tentative_action": tentative,
            "entity_candidates": org_candidates,
            "allowed_actions": sorted(REPAIR_ACTIONS - {"no_action"}),
            "instruction": ("Choose the best next action from allowed_actions. If re_link_entity, "
                            "set entity_choice to {field, value} with value FROM entity_candidates. "
                            "Never answer the question itself."),
        }
        try:
            verdict = self.advisor.advise(payload)
        except Exception as exc:  # pragma: no cover - live boundary
            return {"error": repr(exc)}
        return verdict if isinstance(verdict, dict) else {"raw": str(verdict)}

    # ------------------------------------------------------------------ 4. preference logging
    def log_preference(self, trace: ReasoningTrace, reflection: TraceReflection,
                       *, oracle_match: bool | None = None) -> dict[str, Any] | None:
        """Append one preference-style record; returns the record (or None when logging is off)."""
        if self.preference_log is None:
            return None
        card = trace.answer_card
        answered = card is not None and card.answer is not None
        if oracle_match is False or reflection.faithfulness == "suspicious":
            label = "bad"
        elif answered and reflection.faithfulness == "faithful" and (oracle_match in (True, None)):
            label = "good"
        elif not answered:
            label = "safe_abstain"  # no hallucination happened, whatever repair was recommended
        else:
            label = "uncertain"
        chosen = next((p for p in trace.plans if p.plan_id == trace.selected_plan_id),
                      trace.plans[0] if trace.plans else None)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "trace_id": trace.trace_id,
            "question": trace.question,
            "label": label,
            "oracle_match": oracle_match,
            "faithfulness": reflection.faithfulness,
            "plan_valid": reflection.plan_valid,
            "plan_issues": list(reflection.plan_issues),
            "action": reflection.action,
            "execution_status": trace.execution.status if trace.execution else "not_run",
            "answer": _jsonable(card.answer if card else None),
            "chosen_plan": _plan_summary(chosen),
            "rejected_plans": [_plan_summary(p) for p in trace.plans
                               if chosen is None or p.plan_id != chosen.plan_id][:3],
        }
        path = Path(self.preference_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


@dataclass
class LLMTraceAdvisor:
    """LLM advisor over the closed repair-action set (generate-then-validate boundary)."""

    client: Any
    model: str

    def advise(self, payload: dict[str, Any]) -> dict[str, Any]:
        system = ("You audit a knowledge-graph QA reasoning trace. Diagnose, pick ONE action from "
                  "allowed_actions, and never answer the user question. Return strict JSON "
                  '{"assessment": str, "recommended_action": str, "entity_choice": {"field": str, "value": str} | null}.')
        result = self.client.complete_json(model=self.model, system=system,
                                           user=json.dumps(payload, ensure_ascii=False))
        parsed = getattr(result, "parsed", result)
        return parsed if isinstance(parsed, dict) else {"raw": str(parsed)}


def _normalize_org(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def _swap_constraint(spec: RuntimeQuerySpec, target: QueryConstraint, value: str) -> RuntimeQuerySpec:
    constraints = tuple(
        QueryConstraint(c.field, c.op, value, visible_to_user=c.visible_to_user,
                        source_text=c.source_text, confidence=c.confidence, metadata=c.metadata)
        if c is target else c
        for c in spec.constraints
    )
    return replace(spec, constraints=constraints)


def _plan_summary(plan: Any) -> dict[str, Any]:
    if plan is None:
        return {}
    spec = plan.query_spec
    return {
        "plan_id": plan.plan_id,
        "status": plan.status,
        "planner_source": plan.planner_source,
        "confidence": plan.confidence,
        "rationale": plan.rationale,
        "answer_operation": spec.answer_operation,
        "answer_field": spec.answer_field,
        "constraints": [{"field": c.field, "op": c.op, "value": _jsonable(c.value)}
                        for c in spec.constraints],
        "has_decomposition": getattr(plan, "decomposition", None) is not None,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


__all__ = ["TraceReflection", "TraceReflector", "LLMTraceAdvisor", "REPAIR_ACTIONS"]
