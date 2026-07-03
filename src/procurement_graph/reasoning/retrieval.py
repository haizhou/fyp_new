"""Candidate retrieval planning (design Stage 2).

Runtime retrieval starts from exact structured filters over the KG rather than learned graph
search. This module (1) gates operations/intents the deterministic executor cannot yet serve
(role/path expansion, comparison, top-k), so the orchestrator can mark them unsupported instead
of silently returning a wrong answer, and (2) resolves organisation-name mentions to canonical
filter values when an `OrgResolver` is available. Bounded graph expansion for open-ended
role/path questions is a documented extension point, not part of the exhaustive-retrieval core.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .linking import OrgResolver
from .models import QueryConstraint, RuntimeQuerySpec

_DETERMINISTIC_OPERATIONS = frozenset(
    {"select_unique", "count", "sum", "exists", "argmax", "argmin", "distinct_set", "set", "top_k", "predicate"}
)
_EXPANSION_INTENTS = frozenset({"role_path"})


@dataclass(frozen=True)
class RetrievalPlan:
    mode: str  # "exact_filters" | "needs_expansion" | "unsupported"
    reason: str = ""
    resolved_spec: RuntimeQuerySpec | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalCandidate:
    """A possible KG value for a failed natural-language mention."""

    field: str
    value: Any
    label: str
    score: float
    source: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CandidateSelection:
    """LLM/rule choice among retrieval candidates. It may abstain."""

    selected: RetrievalCandidate | None
    action: str = "abstain"  # "select" | "abstain"
    reason: str = ""
    raw_response: Any = None


class CandidateRetriever(Protocol):
    """Top-k retriever for natural mentions; may be lexical or embedding-backed."""

    def retrieve(self, *, field: str, mention: str, top_k: int = 8) -> tuple[RetrievalCandidate, ...]:
        ...


class CandidateSelector(Protocol):
    """Select a candidate from top-k. LLM implementations must not invent values."""

    def select(
        self,
        *,
        question: str,
        spec: RuntimeQuerySpec,
        constraint: QueryConstraint,
        candidates: tuple[RetrievalCandidate, ...],
    ) -> CandidateSelection:
        ...


@dataclass(frozen=True)
class LexicalTopKCandidateRetriever:
    """Tiny local top-k retriever used for tests and as a no-dependency baseline.

    This is deliberately not an embedding model. It implements the same interface
    an embedding retriever should implement, so later we can swap in a vector
    index without changing the pipeline.
    """

    values_by_field: dict[str, tuple[str, ...]]

    def retrieve(self, *, field: str, mention: str, top_k: int = 8) -> tuple[RetrievalCandidate, ...]:
        query_tokens = _tokens(mention)
        if not query_tokens:
            return ()
        hits: list[RetrievalCandidate] = []
        for value in self.values_by_field.get(field, ()):
            label = str(value)
            label_tokens = _tokens(label)
            if not label_tokens:
                continue
            overlap = len(query_tokens & label_tokens)
            substring_bonus = 1.0 if str(mention).strip().casefold() in label.casefold() else 0.0
            score = (2.0 * overlap + substring_bonus) / max(len(query_tokens | label_tokens), 1)
            if score > 0:
                hits.append(RetrievalCandidate(field=field, value=value, label=label,
                                               score=round(score, 4), source="lexical_topk"))
        hits.sort(key=lambda item: (-item.score, len(item.label), item.label.casefold()))
        return tuple(hits[:top_k])


@dataclass(frozen=True)
class TopScoreCandidateSelector:
    """Deterministic selector for tests; production can use an LLM selector."""

    min_score: float = 0.35
    ambiguity_margin: float = 0.03

    def select(
        self,
        *,
        question: str,
        spec: RuntimeQuerySpec,
        constraint: QueryConstraint,
        candidates: tuple[RetrievalCandidate, ...],
    ) -> CandidateSelection:
        if not candidates:
            return CandidateSelection(None, reason="no candidates")
        if candidates[0].score < self.min_score:
            return CandidateSelection(None, reason=f"top score below threshold: {candidates[0].score}")
        if len(candidates) > 1 and candidates[0].score - candidates[1].score < self.ambiguity_margin:
            return CandidateSelection(None, reason="top candidates are too close")
        return CandidateSelection(candidates[0], action="select", reason="top score selected")


@dataclass
class LLMCandidateSelector:
    """LLM selector constrained to a supplied top-k list.

    The model may only return a candidate index or abstain. The chosen value is
    taken from the candidate list, never from model text.
    """

    client: Any
    model: str

    def select(
        self,
        *,
        question: str,
        spec: RuntimeQuerySpec,
        constraint: QueryConstraint,
        candidates: tuple[RetrievalCandidate, ...],
    ) -> CandidateSelection:
        if not candidates:
            return CandidateSelection(None, reason="no candidates")
        payload = {
            "question": question,
            "intent": spec.intent,
            "constraint": {"field": constraint.field, "value": constraint.value},
            "candidates": [
                {"index": idx, "field": item.field, "label": item.label, "score": item.score, "source": item.source}
                for idx, item in enumerate(candidates)
            ],
            "instruction": "Choose one candidate index only if it clearly denotes the user's mention; otherwise abstain.",
            "output_schema": {"action": "select|abstain", "selected_index": "integer|null", "reason": "short string"},
        }
        result = self.client.complete_json(
            model=self.model,
            system="You select KG retrieval candidates. You cannot invent values; choose from the list or abstain.",
            user=json.dumps(payload, ensure_ascii=False),
        )
        parsed = getattr(result, "parsed", result)
        if not isinstance(parsed, dict):
            return CandidateSelection(None, reason="parse_error", raw_response=parsed)
        if parsed.get("action") != "select":
            return CandidateSelection(None, action="abstain", reason=str(parsed.get("reason", "")),
                                      raw_response=parsed)
        try:
            index = int(parsed.get("selected_index"))
        except (TypeError, ValueError):
            return CandidateSelection(None, reason="schema_error", raw_response=parsed)
        if index < 0 or index >= len(candidates):
            return CandidateSelection(None, reason="index_out_of_range", raw_response=parsed)
        return CandidateSelection(candidates[index], action="select", reason=str(parsed.get("reason", "")),
                                  raw_response=parsed)


def expansion_required(spec: RuntimeQuerySpec) -> str:
    """Return a reason string if the spec needs capability the runtime does not yet provide."""
    if spec.intent in _EXPANSION_INTENTS:
        return "role/path questions need bounded graph expansion, which the deterministic runtime does not implement yet"
    if spec.answer_operation not in _DETERMINISTIC_OPERATIONS:
        return f"answer_operation '{spec.answer_operation}' is not supported by the deterministic runtime yet"
    if spec.relation_path:
        return "multi-hop relation paths are not supported by the deterministic runtime yet"
    return ""


def plan_retrieval(spec: RuntimeQuerySpec, *, org_resolver: OrgResolver | None = None) -> RetrievalPlan:
    reason = expansion_required(spec)
    if reason:
        mode = "needs_expansion" if spec.intent in _EXPANSION_INTENTS or spec.relation_path else "unsupported"
        return RetrievalPlan(mode=mode, reason=reason, resolved_spec=None)
    resolved, notes = _resolve_entities(spec, org_resolver)
    return RetrievalPlan(mode="exact_filters", resolved_spec=resolved, notes=tuple(notes))


def semantic_repair_spec(
    spec: RuntimeQuerySpec,
    *,
    question: str,
    candidate_retriever: CandidateRetriever,
    candidate_selector: CandidateSelector,
    top_k: int = 8,
) -> tuple[RuntimeQuerySpec | None, dict[str, Any]]:
    """Try top-k candidate repair for failed exact matching.

    This is called only after an exact KG attempt fails. It rewrites at most one
    text equality constraint per call, using values selected from retriever
    candidates. It does not broaden filters or compute answers.
    """
    attempts: list[dict[str, Any]] = []
    repairable = {"buyer_name", "supplier_name", "tender_title", "award_title", "tender_cpv_description"}
    for index, constraint in enumerate(spec.constraints):
        if constraint.op != "eq" or constraint.field not in repairable:
            continue
        candidates = candidate_retriever.retrieve(field=constraint.field, mention=str(constraint.value), top_k=top_k)
        selection = candidate_selector.select(
            question=question,
            spec=spec,
            constraint=constraint,
            candidates=candidates,
        )
        attempts.append({
            "field": constraint.field,
            "mention": constraint.value,
            "candidates": [_candidate_trace(candidate) for candidate in candidates],
            "selection": _selection_trace(selection),
        })
        if selection.selected is None:
            continue
        replacement = replace(
            constraint,
            value=selection.selected.value,
            metadata={
                **constraint.metadata,
                "semantic_repair": True,
                "original_value": constraint.value,
                "selected_label": selection.selected.label,
                "selector_reason": selection.reason,
            },
        )
        constraints = list(spec.constraints)
        constraints[index] = replacement
        repaired = replace(
            spec,
            constraints=tuple(constraints),
            metadata={**spec.metadata, "semantic_repair": True},
        )
        return repaired, {"attempted": True, "repaired": True, "attempts": attempts}
    return None, {"attempted": bool(attempts), "repaired": False, "attempts": attempts}


def _resolve_entities(
    spec: RuntimeQuerySpec, org_resolver: OrgResolver | None
) -> tuple[RuntimeQuerySpec, list[str]]:
    """Resolve organisation-name constraints to canonical filter values when possible.

    buyer_name / supplier_name equality already queries the KG directly, so with no resolver
    this is identity. With a resolver, an ambiguous or unresolved mention is annotated so the
    verifier/reflector can act on it.
    """
    if org_resolver is None:
        return spec, []
    notes: list[str] = []
    new_constraints: list[QueryConstraint] = []
    for constraint in spec.constraints:
        if constraint.field in {"buyer_name", "supplier_name"} and constraint.op == "eq" and "canonical_id" not in constraint.metadata:
            candidates = org_resolver.resolve(str(constraint.value))
            if not candidates:
                notes.append(f"unresolved organisation: {constraint.value!r}")
                new_constraints.append(constraint)
                continue
            if len(candidates) > 1 and candidates[0].score - candidates[1].score < 0.05:
                notes.append(f"ambiguous organisation: {constraint.value!r} ({len(candidates)} candidates)")
            best = candidates[0]
            new_constraints.append(
                replace(constraint, value=best.linked_label,
                        metadata={**constraint.metadata, "canonical_id": best.linked_id})
            )
        else:
            new_constraints.append(constraint)
    return replace(spec, constraints=tuple(new_constraints)), notes


def _candidate_trace(candidate: RetrievalCandidate) -> dict[str, Any]:
    return {
        "field": candidate.field,
        "value": candidate.value,
        "label": candidate.label,
        "score": candidate.score,
        "source": candidate.source,
        "metadata": candidate.metadata or {},
    }


def _selection_trace(selection: CandidateSelection) -> dict[str, Any]:
    return {
        "action": selection.action,
        "reason": selection.reason,
        "selected": _candidate_trace(selection.selected) if selection.selected else None,
        "raw_response": selection.raw_response,
    }


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(text).casefold()) if len(token) > 1}


__all__ = [
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
]
