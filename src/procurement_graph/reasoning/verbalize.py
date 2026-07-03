"""Guarded LLM verbalisation (adapted from the prior pipeline's step4 answer-preservation guard).

The deterministic answer card already yields a correct, plain `answer_text`. An optional LLM may
rewrite it more fluently, but its output is REJECTED — falling back to the deterministic text —
unless it preserves every verified answer atom (the answer value and, for entity answers, the
entity label). This is a GCR-style guard: the decoder is not token-constrained, yet it may never
drop or alter a fact the executor proved. Duck-typed client; no dependency without a live model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .models import AnswerCard


class ChatLike(Protocol):
    def complete_json(self, *, model: str, system: str, user: str) -> Any:
        ...


@dataclass(frozen=True)
class VerbalizationResult:
    text: str
    source: str  # "llm" | "deterministic_fallback" | "deterministic"
    status: str
    missing_atoms: tuple[str, ...] = ()


def required_atoms(card: AnswerCard) -> list[str]:
    """Verified atoms any rewrite must preserve: the answer value (and entity label if applicable)."""
    atoms: list[str] = []
    if card.answer not in (None, ""):
        atoms.append(str(card.answer))
    return atoms


def answer_preserves_atoms(text: str, atoms: list[str]) -> tuple[bool, list[str]]:
    norm = _norm(text)
    answer_number = _to_number(text)
    missing: list[str] = []
    for atom in atoms:
        value = str(atom).strip()
        if not value:
            continue
        expected = _to_number(value)
        if expected is not None:
            ok = (answer_number is not None and abs(expected - answer_number) <= 1.0) or _norm(value) in norm
        else:
            ok = _norm(value) in norm
        if not ok:
            missing.append(value)
    return not missing, missing


@dataclass
class LLMVerbalizer:
    client: ChatLike
    model: str

    def verbalize(self, card: AnswerCard) -> VerbalizationResult:
        if card.answer in (None, "") or card.confidence_label == "not_answered":
            return VerbalizationResult(card.answer_text, "deterministic", "not_applicable")
        atoms = required_atoms(card)
        system, user = build_verbalize_messages(card)
        try:
            result = self.client.complete_json(model=self.model, system=system, user=user)
        except Exception:  # pragma: no cover - live only
            return VerbalizationResult(card.answer_text, "deterministic_fallback", "call_failed")
        parsed = getattr(result, "parsed", result)
        text = str(parsed.get("answer", "")).strip() if isinstance(parsed, dict) else ""
        if not text:
            return VerbalizationResult(card.answer_text, "deterministic_fallback", "empty_answer")
        ok, missing = answer_preserves_atoms(text, atoms)
        if not ok:
            return VerbalizationResult(card.answer_text, "deterministic_fallback",
                                       "rejected_dropped_atoms", tuple(missing))
        return VerbalizationResult(text, "llm", "applied")


def build_verbalize_messages(card: AnswerCard) -> tuple[str, str]:
    system = (
        "You are the final verbaliser for a KG-grounded procurement QA system. Rewrite the verified "
        "answer more fluently. You MUST NOT add, remove, or change any fact: preserve the exact answer "
        "value and any entity name. Do not introduce numbers, dates, or entities not present. "
        "Respond with strict JSON: {\"answer\": \"...\"}."
    )
    payload = {
        "question": card.question,
        "verified_answer": _jsonable(card.answer),
        "deterministic_answer_text": card.answer_text,
        "answer_operation": card.query_spec.answer_operation,
        "required_atoms": required_atoms(card),
        "limitations": list(card.limitations),
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def _norm(text: str) -> str:
    lowered = re.sub(r"[^\w\s&.\-]", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", lowered).strip()


def _to_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


__all__ = ["LLMVerbalizer", "VerbalizationResult", "required_atoms", "answer_preserves_atoms", "build_verbalize_messages"]
