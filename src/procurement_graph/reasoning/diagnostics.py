"""Optional LLM/rule diagnostics for verification and reflection stages.

These analyzers are advisory only. They may explain a failed verifier check or
suggest what kind of repair would be useful, but they do not compute answers and
do not override deterministic verifier/reflector decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .models import ExecutionResult, ReflectionAction, RuntimeQuerySpec


class VerificationAnalyzer(Protocol):
    def analyze_verification(
        self,
        *,
        question: str,
        spec: RuntimeQuerySpec,
        result: ExecutionResult,
        failed_checks: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        ...


class ReflectionAnalyzer(Protocol):
    def analyze_reflection(
        self,
        *,
        question: str,
        result: ExecutionResult | None,
        action: ReflectionAction,
        attempts: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        ...


@dataclass
class LLMVerificationAnalyzer:
    """LLM explanation of failed verifier checks, constrained to diagnosis."""

    client: Any
    model: str

    def analyze_verification(
        self,
        *,
        question: str,
        spec: RuntimeQuerySpec,
        result: ExecutionResult,
        failed_checks: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "spec": _spec_payload(spec),
            "execution_status": result.status,
            "failed_checks": list(failed_checks),
            "instruction": "Diagnose why verification failed. Do not answer the user question.",
            "output_schema": {
                "diagnosis": "short string",
                "likely_layer": "planner|grounding|retrieval|executor|data_coverage|unsupported",
                "suggested_next_action": "abstain|clarify|candidate_retrieval|repair_constraints|unsupported",
            },
        }
        parsed = _complete_json(self.client, self.model, "You diagnose KGQA verifier failures.", payload)
        return parsed if isinstance(parsed, dict) else {"diagnosis": "parse_error", "raw_response": parsed}


@dataclass
class LLMReflectionAnalyzer:
    """LLM explanation of reflector decisions, constrained to repair diagnosis."""

    client: Any
    model: str

    def analyze_reflection(
        self,
        *,
        question: str,
        result: ExecutionResult | None,
        action: ReflectionAction,
        attempts: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "execution_status": result.status if result else "not_run",
            "reflector_action": action.action,
            "reflector_reason": action.reason,
            "attempts": list(attempts)[-3:],
            "instruction": "Explain whether the reflector action is appropriate. Do not invent an answer.",
            "output_schema": {
                "assessment": "short string",
                "risk": "low|medium|high",
                "suggested_next_action": "keep_action|clarify|candidate_retrieval|unsupported|manual_review",
            },
        }
        parsed = _complete_json(self.client, self.model, "You diagnose KGQA reflector decisions.", payload)
        return parsed if isinstance(parsed, dict) else {"assessment": "parse_error", "raw_response": parsed}


def _complete_json(client: Any, model: str, system: str, payload: dict[str, Any]) -> Any:
    try:
        result = client.complete_json(model=model, system=system, user=json.dumps(payload, ensure_ascii=False))
    except Exception as exc:  # pragma: no cover - live boundary
        return {"error": repr(exc)}
    return getattr(result, "parsed", result)


def _spec_payload(spec: RuntimeQuerySpec) -> dict[str, Any]:
    return {
        "intent": spec.intent,
        "answer_operation": spec.answer_operation,
        "answer_field": spec.answer_field,
        "answer_value_type": spec.answer_value_type,
        "constraints": [
            {"field": constraint.field, "op": constraint.op, "value": constraint.value}
            for constraint in spec.constraints
        ],
        "dedupe_key": spec.dedupe_key,
        "requires_exhaustive_retrieval": spec.requires_exhaustive_retrieval,
    }


__all__ = [
    "LLMReflectionAnalyzer",
    "LLMVerificationAnalyzer",
    "ReflectionAnalyzer",
    "VerificationAnalyzer",
]
