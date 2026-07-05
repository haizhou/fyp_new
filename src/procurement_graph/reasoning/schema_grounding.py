"""Semantic schema grounding for LLM intent programs.

The LLM is allowed to describe fields with natural language (`field_text`). This layer maps those
descriptions onto canonical intent slots before deterministic compile. The design is deliberately
two-stage:

1. candidate generation: exact/alias candidates plus embedding-style semantic candidates
2. validation: type gate, confidence threshold, and top-2 margin

The default embedder is a local character n-gram vectorizer so tests and offline runs need no model
download. A sentence-transformer/Azure encoder can later be plugged in by passing an embedder with
an `.encode(texts)` method.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroundingCandidate:
    slot: str
    score: float
    method: str
    matched: str


@dataclass(frozen=True)
class GroundedField:
    slot: str
    confidence: float
    method: str
    grounded_from: str
    reason: str = ""
    candidates: tuple[GroundingCandidate, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.slot)


@dataclass(frozen=True)
class SchemaField:
    slot: str
    description: str
    aliases: tuple[str, ...]
    allowed_value_types: frozenset[str]


_ALIASES: dict[str, tuple[str, ...]] = {
    "year": (
        "year", "published year", "publication year", "release year", "notice year",
        "contract notice year", "published in", "in year",
    ),
    "year_range": ("year range", "between years", "from year to year"),
    "cpv_code": ("cpv", "cpv code", "cpv number", "common procurement vocabulary code"),
    "cpv_label": ("cpv label", "cpv description", "cpv title", "cpv parenthetical label"),
    "procurement_category": (
        "notice type", "notice category", "type of notice",
        "procurement category", "high-level procurement category", "tender category",
        "goods services works category", "category goods services works",
    ),
    "buyer": ("buyer", "contracting authority", "authority", "awarding authority", "awarded by", "published by"),
    "supplier": ("supplier", "winner", "award winner", "awarded supplier", "awarded to", "vendor"),
    "title": ("title", "contract title", "notice title", "tender title", "quoted title"),
    "date": ("date", "award date", "signed award date", "award signed date", "date signed"),
    "amount": ("amount", "value", "contract value", "award value", "threshold", "money"),
    "has_signed_date": ("has signed award date", "award has been signed", "signed date exists"),
}

_DESCRIPTIONS: dict[str, str] = {
    "year": "publication release notice year",
    "year_range": "range of publication years between two years",
    "cpv_code": "numeric CPV common procurement vocabulary code",
    "cpv_label": "text label or parenthetical description of a CPV code",
    "procurement_category": "high level procurement category goods services or works",
    "buyer": "buyer contracting authority awarding organisation",
    "supplier": "supplier winner vendor awarded organisation",
    "title": "contract notice tender title",
    "date": "award signed date or explicit date",
    "amount": "money contract value award value threshold",
    "has_signed_date": "boolean existence of signed award date",
}

_VALUE_TYPE_ALLOWED: dict[str, frozenset[str]] = {
    "year": frozenset({"year", "number", "unknown"}),
    "year_range": frozenset({"year", "number", "unknown"}),
    "cpv_code": frozenset({"cpv", "code", "string", "unknown"}),
    "cpv_label": frozenset({"text", "string", "unknown"}),
    "procurement_category": frozenset({"category", "string", "unknown"}),
    "buyer": frozenset({"entity", "organisation", "organization", "string", "unknown"}),
    "supplier": frozenset({"entity", "organisation", "organization", "string", "unknown"}),
    "title": frozenset({"text", "string", "unknown"}),
    "date": frozenset({"date", "string", "unknown"}),
    "amount": frozenset({"money", "number", "amount", "unknown"}),
    "has_signed_date": frozenset({"boolean", "unknown"}),
}

_CATALOG: tuple[SchemaField, ...] = tuple(
    SchemaField(slot, _DESCRIPTIONS[slot], aliases, _VALUE_TYPE_ALLOWED[slot])
    for slot, aliases in _ALIASES.items()
)


class NgramEmbedder:
    """Tiny deterministic embedder for short schema phrases.

    This is not meant to beat a sentence embedding model; it gives us a no-dependency semantic
    candidate source and the same top-k/margin plumbing that a real embedder will use.
    """

    def __init__(self, *, n_min: int = 3, n_max: int = 5):
        self.n_min = n_min
        self.n_max = n_max

    def encode(self, texts: list[str]) -> list[dict[str, float]]:
        return [_char_ngrams(text, self.n_min, self.n_max) for text in texts]


_DEFAULT_EMBEDDER = NgramEmbedder()


# "awarded/won ... TO X" names X as the RECEIVER (supplier); "awarded/published BY X" names X as
# the buyer. The relation phrasing outranks the role NOUN in the field text: 1600's field_text
# 'buyer has awarded a contract to <org>' contains "buyer" yet its VALUE is the supplier.
_AWARDED_TO_RE = re.compile(r"\b(?:awarded?|award|won|contract(?:ed)?)\b[^.]{0,40}\bto\b|\bto\s*$")
_AWARDED_BY_RE = re.compile(r"\b(?:awarded?|published?|issued?|put\s+out)\b[^.]{0,40}\bby\b|\bby\s*$")


def ground_field_text(field_text: Any, *, value: Any = "", value_type: str = "unknown",
                      op: str = "", embedder: Any = None, min_confidence: float = 0.72,
                      margin: float = 0.08) -> GroundedField:
    text = " ".join(str(field_text or "").casefold().replace("_", " ").split())
    if not text:
        return GroundedField("", 0.0, "missing", str(field_text or ""), "missing_field_text")
    raw_candidates = _rank_candidates(text, embedder=embedder)
    candidates = tuple(c for c in raw_candidates if _type_gate(c.slot, value=value, value_type=value_type, op=op))
    if not candidates:
        return GroundedField("", 0.0, "candidate+type_gate", str(field_text),
                             _candidate_reason("type_gate_rejected", raw_candidates), tuple(raw_candidates[:5]))
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    # Direction cue outranks the role NOUN whenever an org slot leads: the field text describes
    # the relation from the OTHER party's perspective ('buyer has awarded a contract to <org>'
    # contains "buyer" yet its VALUE is the supplier).
    if best.slot in {"buyer", "supplier"} \
            and str(value_type).casefold() in ("entity", "organisation", "organization", "unknown"):
        if _AWARDED_TO_RE.search(text):
            return GroundedField("supplier", max(best.score, 0.9), "direction_cue", str(field_text),
                                 candidates=candidates[:5])
        if _AWARDED_BY_RE.search(text):
            return GroundedField("buyer", max(best.score, 0.9), "direction_cue", str(field_text),
                                 candidates=candidates[:5])
    if best.score < min_confidence:
        return GroundedField("", best.score, best.method, str(field_text),
                             _candidate_reason("no_confident_schema_match", candidates), candidates[:5])
    if second and best.score - second.score < margin and best.slot != second.slot:
        return GroundedField("", best.score, best.method, str(field_text),
                             _candidate_reason("ambiguous_schema_match", candidates), candidates[:5])
    return GroundedField(best.slot, best.score, best.method, str(field_text), candidates=candidates[:5])


def canonical_slot_from_text(field_text: Any, *, value: Any = "", value_type: str = "unknown",
                             op: str = "") -> tuple[str, str]:
    grounded = ground_field_text(field_text, value=value, value_type=value_type, op=op)
    if grounded.ok:
        return grounded.slot, ""
    return "", grounded.reason


def grounding_candidates(field_text: Any, *, embedder: Any = None, top_k: int = 5) -> tuple[GroundingCandidate, ...]:
    text = " ".join(str(field_text or "").casefold().replace("_", " ").split())
    return tuple(_rank_candidates(text, embedder=embedder)[:top_k])


def schema_catalog() -> tuple[SchemaField, ...]:
    return _CATALOG


def _rank_candidates(text: str, *, embedder: Any = None) -> list[GroundingCandidate]:
    by_slot: dict[str, GroundingCandidate] = {}
    for field in _CATALOG:
        for alias in field.aliases:
            score = _alias_score(text, alias)
            _keep_best(by_slot, GroundingCandidate(field.slot, score, "alias", alias))
    embedder = embedder or _DEFAULT_EMBEDDER
    documents = [f"{field.slot} {field.description} {' '.join(field.aliases)}" for field in _CATALOG]
    try:
        vectors = embedder.encode([text] + documents)
        query_vec, doc_vecs = vectors[0], vectors[1:]
        for field, vec in zip(_CATALOG, doc_vecs):
            score = _cosine(query_vec, vec)
            _keep_best(by_slot, GroundingCandidate(field.slot, score, "embedding", field.description))
    except Exception:
        # Grounding must fail closed through alias candidates if an optional embedding provider is
        # unavailable; provider errors should not produce arbitrary schema matches.
        pass
    return sorted(by_slot.values(), key=lambda c: c.score, reverse=True)


def _alias_score(text: str, alias: str) -> float:
    alias = " ".join(alias.casefold().split())
    if text == alias:
        return 1.0
    if alias in text or text in alias:
        return 0.9
    text_tokens = set(text.split())
    alias_tokens = set(alias.split())
    if not text_tokens or not alias_tokens:
        return 0.0
    return len(text_tokens & alias_tokens) / len(text_tokens | alias_tokens)


def _keep_best(candidates: dict[str, GroundingCandidate], cand: GroundingCandidate) -> None:
    prev = candidates.get(cand.slot)
    if prev is None or cand.score > prev.score:
        candidates[cand.slot] = cand


_DECOR_RE = re.compile(r"\b(?:notices?|contracts?|tenders?|category)\b")


def _type_gate(slot: str, *, value: Any, value_type: str, op: str) -> bool:
    vt = str(value_type or "unknown").casefold()
    if vt and vt not in _VALUE_TYPE_ALLOWED.get(slot, {"unknown"}):
        return False
    value_text = str(value or "").strip().casefold()
    if slot == "procurement_category":
        # accept decorated category values ("services notice", "goods contracts") but never a
        # CPV-ish description ("IT-service contracts"): strip decoration words, then demand the
        # remainder BE a canonical category token
        cleaned = " ".join(_DECOR_RE.sub(" ", value_text).split())
        return cleaned in {"goods", "services", "service", "works", "work", "good"}
    if slot == "cpv_code":
        return value_text.isdigit() and 5 <= len(value_text) <= 8
    return True


def _candidate_reason(prefix: str, candidates: tuple[GroundingCandidate, ...] | list[GroundingCandidate]) -> str:
    preview = ",".join(f"{c.slot}:{c.score:.2f}" for c in list(candidates)[:3])
    return f"{prefix}:{preview}" if preview else prefix


def _char_ngrams(text: str, n_min: int, n_max: int) -> dict[str, float]:
    norm = f"  {' '.join(str(text or '').casefold().replace('_', ' ').split())}  "
    counts: dict[str, float] = {}
    for n in range(n_min, n_max + 1):
        if len(norm) < n:
            continue
        for i in range(0, len(norm) - n + 1):
            gram = norm[i:i + n]
            counts[gram] = counts.get(gram, 0.0) + 1.0
    return counts


def _cosine(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        common = set(left) & set(right)
        dot = sum(left[k] * right[k] for k in common)
        lnorm = math.sqrt(sum(v * v for v in left.values()))
        rnorm = math.sqrt(sum(v * v for v in right.values()))
        return dot / (lnorm * rnorm) if lnorm and rnorm else 0.0
    try:
        lvals, rvals = list(left), list(right)
        dot = sum(float(a) * float(b) for a, b in zip(lvals, rvals))
        lnorm = math.sqrt(sum(float(v) * float(v) for v in lvals))
        rnorm = math.sqrt(sum(float(v) * float(v) for v in rvals))
        return dot / (lnorm * rnorm) if lnorm and rnorm else 0.0
    except Exception:
        return 0.0


__all__ = [
    "GroundedField", "GroundingCandidate", "NgramEmbedder", "SchemaField",
    "canonical_slot_from_text", "ground_field_text", "grounding_candidates", "schema_catalog",
]
