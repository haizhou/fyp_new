"""Stage 2b: independent semantic verification (Gate B).

Two modes, chosen by operation and evidence size:

- recompute: the verifier answers the question from the supplied evidence; the answer is
  compared to the golden answer (type-aware). Used for select_unique and small counts.
- faithfulness: the verifier converts the question into a structured query using only the
  allowed field vocabulary; the extracted operation + filters are compared to the spec. Used
  for sums and large aggregations, where dumping evidence for the model to recompute is
  unreliable and expensive. The spec values are never shown to the verifier.

`LLMGateBVerifier` calls the verification model (a different family from the generator, e.g.
grok-4-1-fast-non-reasoning). `DryRunGateBVerifier` returns the spec-consistent answer offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .chat import ChatClient
from .models import AnswerSpec
from .prompts import (
    FIELD_VOCABULARY,
    VERIFY_PROMPT_VERSION,
    VERIFY_SCHEMA_VERSION,
    build_verify_faithfulness_messages,
    build_verify_recompute_messages,
    semantic_filters,
)

_OPERATION_ALIASES = {"select_unique": "select", "count": "count", "sum": "sum"}


def verification_mode(spec: AnswerSpec, evidence_count: int, cap: int) -> str:
    """recompute only for select_unique (reads one value from one row, reliable);
    faithfulness for count and sum.

    Counts are NOT recompute-verified: the count is already deterministically correct from
    Gate A, and LLMs miscount moderately long evidence lists (pilot: grok read 34 rows as 28).
    Faithfulness instead checks the question decodes to the right operation + filters, which is
    what Gate B is for, and needs no per-row evidence."""
    if spec.answer_operation == "select_unique":
        return "recompute"
    return "faithfulness"


@dataclass(frozen=True)
class VerificationOutcome:
    verified: bool
    mode: str
    predicted_answer: Any = None
    reason: str = ""
    raw: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    def provenance(self, *, model: str, sampled: bool) -> dict[str, Any]:
        return {
            "model": model,
            "mode": self.mode,
            "sampled": sampled,
            "verified": self.verified,
            "predicted_answer": _jsonable(self.predicted_answer),
            "reason": self.reason,
            "prompt_version": VERIFY_PROMPT_VERSION,
            "schema_version": VERIFY_SCHEMA_VERSION,
            "raw": self.raw,
            "usage": self.usage,
        }


@dataclass
class LLMGateBVerifier:
    client: ChatClient
    model: str

    def verify_recompute(
        self, spec: AnswerSpec, question: str, evidence_sample: list[dict[str, Any]], golden_answer: Any
    ) -> VerificationOutcome:
        system, user = build_verify_recompute_messages(
            question, evidence_sample, answer_value_type=spec.answer_value_type
        )
        try:
            result = self.client.complete_json(model=self.model, system=system, user=user)
        except Exception as exc:  # pragma: no cover - live only
            return VerificationOutcome(False, "recompute", reason=f"verify_call_failed: {exc}")
        parsed = result.parsed if isinstance(result.parsed, dict) else {}
        predicted = parsed.get("answer")
        ok, kind = answer_match(predicted, golden_answer, spec.answer_value_type)
        return VerificationOutcome(
            verified=ok,
            mode="recompute",
            predicted_answer=predicted,
            reason=f"{kind}: {parsed.get('reason', '')}",
            raw=result.raw_text,
            usage=result.usage,
        )

    def verify_faithfulness(self, spec: AnswerSpec, question: str) -> VerificationOutcome:
        system, user = build_verify_faithfulness_messages(question)
        try:
            result = self.client.complete_json(model=self.model, system=system, user=user)
        except Exception as exc:  # pragma: no cover - live only
            return VerificationOutcome(False, "faithfulness", reason=f"verify_call_failed: {exc}")
        parsed = result.parsed if isinstance(result.parsed, dict) else {}
        ok, reason = faithfulness_match(parsed, spec)
        return VerificationOutcome(
            verified=ok,
            mode="faithfulness",
            predicted_answer={"operation": parsed.get("operation"), "filters": parsed.get("filters")},
            reason=reason,
            raw=result.raw_text,
            usage=result.usage,
        )


@dataclass
class DryRunGateBVerifier:
    """Offline stand-in: returns the spec-consistent result so the PASS path is exercised."""

    model: str = "dry-run-verifier"

    def verify_recompute(
        self, spec: AnswerSpec, question: str, evidence_sample: list[dict[str, Any]], golden_answer: Any
    ) -> VerificationOutcome:
        build_verify_recompute_messages(question, evidence_sample, answer_value_type=spec.answer_value_type)
        return VerificationOutcome(True, "recompute", predicted_answer=golden_answer, reason="dry-run pass")

    def verify_faithfulness(self, spec: AnswerSpec, question: str) -> VerificationOutcome:
        build_verify_faithfulness_messages(question)
        return VerificationOutcome(
            True,
            "faithfulness",
            predicted_answer={"operation": _OPERATION_ALIASES.get(spec.answer_operation), "filters": []},
            reason="dry-run pass",
        )


def answer_match(predicted: Any, golden: Any, value_type: str) -> tuple[bool, str]:
    if predicted is None:
        return False, "no_answer"
    text = str(predicted).strip()
    if text.casefold() == "uncertain":
        return False, "uncertain"
    if value_type == "integer":
        try:
            return int(float(_num(text))) == int(float(_num(str(golden)))), "int"
        except ValueError:
            return False, "parse_error"
    if value_type in {"number", "currency"}:
        try:
            pv, gv = float(_num(text)), float(_num(str(golden)))
        except ValueError:
            return False, "parse_error"
        return abs(pv - gv) <= max(1e-6, abs(gv) * 1e-6), "numeric"
    if value_type == "date":
        return _norm_date(text) == _norm_date(str(golden)), "date"
    return text.casefold() == str(golden).strip().casefold(), "string"


def faithfulness_match(parsed: dict[str, Any], spec: AnswerSpec) -> tuple[bool, str]:
    hidden_violations = hidden_semantic_constraint_violations(spec)
    if hidden_violations:
        return False, "hidden semantic constraint not expressible in question: " + "; ".join(hidden_violations)
    if parsed.get("ambiguous"):
        return False, f"verifier flagged ambiguous: {parsed.get('reason', '')}"
    want_op = _OPERATION_ALIASES.get(spec.answer_operation, spec.answer_operation)
    got_op = str(parsed.get("operation", "")).strip().casefold()
    if got_op != want_op:
        return False, f"operation mismatch: question->{got_op!r} spec->{want_op!r}"
    spec_filters = {(field, _norm_value(field, value)) for field, value in semantic_filters(spec)}
    got_filters = set()
    for item in parsed.get("filters", []) or []:
        if isinstance(item, dict) and item.get("field") in {f for f, _ in spec_filters} | _vocab_fields():
            got_filters.add((str(item["field"]), _norm_value(str(item["field"]), item.get("value"))))
    # restrict the verifier's filters to the question vocabulary for a like-for-like compare
    got_semantic = {(f, v) for f, v in got_filters if f in _vocab_fields()}
    if got_semantic != spec_filters:
        return False, f"filter mismatch: question->{sorted(got_semantic)} spec->{sorted(spec_filters)}"
    return True, "question decodes to the spec operation and filters"


def hidden_semantic_constraint_violations(spec: AnswerSpec) -> list[str]:
    """Return constraints that affect the answer set but are not allowed to be hidden.

    Stage 2 faithfulness compares only user-visible semantic filters. Some internal
    constraints are legitimate execution guards (for example ``value_is_additive=True``
    for sums, or role coverage ``supplier_count>=1``). Exact role-count constraints such
    as ``supplier_count=9`` change the answer set and must either be verbalised or
    rejected; they are not safe internal guards.
    """
    violations: list[str] = []
    for constraint in spec.constraints:
        field = constraint.field
        if field in FIELD_VOCABULARY:
            continue
        if field == "value_is_additive":
            if spec.answer_operation == "sum" and constraint.op == "eq" and bool(constraint.value) is True:
                continue
            violations.append(_constraint_label(constraint))
            continue
        if field in {"supplier_count", "buyer_count"}:
            if constraint.op == "gte" and _as_float(constraint.value) == 1:
                continue
            violations.append(_constraint_label(constraint))
            continue
        if field == "contract_node_id" and spec.answer_operation == "select_unique":
            continue
        violations.append(_constraint_label(constraint))
    return violations


def _vocab_fields() -> set[str]:
    return set(FIELD_VOCABULARY)


def _norm_value(field: str, value: Any) -> str:
    text = str(value).strip()
    if field == "release_year":
        digits = re.sub(r"\D", "", text)
        return digits or text
    if field == "tender_cpv_id":
        return re.sub(r"\s", "", text)
    return text.casefold()


def _num(text: str) -> str:
    cleaned = re.sub(r"[^0-9eE+\-.]", "", str(text))
    return cleaned or "nan"


def _norm_date(text: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(text))
    return match.group(0) if match else str(text).strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None), dict, list)):
        return value
    return str(value)


def _constraint_label(constraint: Any) -> str:
    return f"{constraint.field} {constraint.op} {constraint.value}"


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "VerificationOutcome",
    "LLMGateBVerifier",
    "DryRunGateBVerifier",
    "verification_mode",
    "answer_match",
    "faithfulness_match",
    "hidden_semantic_constraint_violations",
]
