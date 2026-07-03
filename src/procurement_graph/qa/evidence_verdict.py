"""Evidence verdict interfaces for KG-first, document-second QA.

Documents are deliberately passive here: the caller must already know the
contract/OCID from KG evidence. This prevents document chunks from becoming a
parallel full-corpus retrieval system.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/]{2,}")


@dataclass(frozen=True)
class KGEvidence:
    ocid: str
    triples: list[dict[str, Any]] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceVerdictRequest:
    question: str
    claim: str
    kg_evidence: KGEvidence
    required_facets: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceChunkHit:
    chunk_id: str
    score: float
    heading: str
    text: str
    metadata: dict[str, Any]


def query_terms(request: EvidenceVerdictRequest) -> set[str]:
    raw = [request.question, request.claim, *request.required_facets]
    for triple in request.kg_evidence.triples:
        raw.extend(str(value) for value in triple.values())
    for value in request.kg_evidence.fields.values():
        raw.append(str(value))
    terms = {token.upper() for text in raw for token in TOKEN_RE.findall(str(text or ""))}
    return {term for term in terms if len(term) >= 3}


def _chunk_terms(chunk: dict[str, Any]) -> set[str]:
    text = " ".join(
        str(chunk.get(key, ""))
        for key in ["heading", "text", "document_type", "source"]
    )
    return {token.upper() for token in TOKEN_RE.findall(text)}


def score_chunk(chunk: dict[str, Any], terms: set[str]) -> float:
    chunk_terms = _chunk_terms(chunk)
    if not chunk_terms or not terms:
        return 0.0
    overlap = terms & chunk_terms
    if not overlap:
        return 0.0
    precision = len(overlap) / math.sqrt(len(chunk_terms))
    recall = len(overlap) / math.sqrt(len(terms))
    heading_bonus = 0.25 if any(term in str(chunk.get("heading", "")).upper() for term in overlap) else 0.0
    return round(precision + recall + heading_bonus, 6)


def select_document_chunks(
    request: EvidenceVerdictRequest,
    chunks: Iterable[dict[str, Any]],
    *,
    top_k: int = 8,
) -> list[EvidenceChunkHit]:
    """Select chunks for a known KG contract/OCID only."""
    terms = query_terms(request)
    scored: list[EvidenceChunkHit] = []
    for chunk in chunks:
        if str(chunk.get("ocid", "")) != request.kg_evidence.ocid:
            continue
        score = score_chunk(chunk, terms)
        if score <= 0:
            continue
        scored.append(
            EvidenceChunkHit(
                chunk_id=str(chunk.get("chunk_id", "")),
                score=score,
                heading=str(chunk.get("heading", "")),
                text=str(chunk.get("text", "")),
                metadata={key: value for key, value in chunk.items() if key not in {"text"}},
            )
        )
    return sorted(scored, key=lambda hit: (hit.score, hit.chunk_id), reverse=True)[:top_k]


__all__ = [
    "EvidenceChunkHit",
    "EvidenceVerdictRequest",
    "KGEvidence",
    "query_terms",
    "score_chunk",
    "select_document_chunks",
]

