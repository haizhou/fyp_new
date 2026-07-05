"""Shared conservative entity resolution policy for runtime planners/repair.

Resolvers may return substring candidates that are real KG organisations but not necessarily the
organisation meant by the user. This module centralises the threshold/margin policy so no caller
silently takes ``hits[0]`` under a weak match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import EntityLinkCandidate


@dataclass(frozen=True)
class EntityResolution:
    ok: bool
    hit: EntityLinkCandidate | None = None
    reason: str = ""
    candidates: tuple[EntityLinkCandidate, ...] = ()
    # All KG surface forms of the SAME organisation (decoration variants: "The X", "X (BEIS)").
    # len > 1 means the caller should filter with an IN over every variant, not eq on one — the
    # org's rows are split across the variants.
    variants: tuple[str, ...] = ()


# Evidence-driven, unambiguous abbreviation expansions only (see worklog 2026-07-04: the KG
# stores "... Integrated Care Board" while questions say "... ICB"). Applied to the MENTION when
# the literal lookup finds nothing.
_MENTION_EXPANSIONS: tuple[tuple[str, str], ...] = (
    (r"\bICB\b", "Integrated Care Board"),
)


def _mention_variants(mention: str) -> list[str]:
    out = [mention]
    for pattern, expansion in _MENTION_EXPANSIONS:
        expanded = re.sub(pattern, expansion, mention)
        if expanded != mention:
            out.append(expanded)
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", mention).strip()
    if stripped and stripped != mention:
        out.append(stripped)
    return out


def _org_equivalence_key(value: str) -> str:
    """Canonical key treating decoration variants of one org as equal: leading 'the', a trailing
    parenthetical alias '(BEIS)', '&' vs 'and', case and punctuation."""
    text = str(value).casefold().strip()
    text = re.sub(r"^the\s+", "", text)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = text.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", text)


def resolve_confident_org(
    resolver: Any,
    mention: str,
    *,
    min_score: float = 0.85,
    min_margin: float = 0.10,
    exclude_ids: set[str] | None = None,
    variant_only_against: str = "",
) -> EntityResolution:
    """Resolve one organisation mention only when the top candidate is sufficiently safe."""
    if resolver is None:
        return EntityResolution(False, reason="no_resolver")
    hits: tuple = ()
    for candidate_mention in _mention_variants(str(mention)):
        hits = tuple(resolver.resolve(candidate_mention) or ())
        if hits:
            break
    if not hits:
        return EntityResolution(False, reason="entity_not_found")

    exclude_ids = exclude_ids or set()
    filtered = tuple(
        hit for hit in hits
        if hit.linked_id not in exclude_ids
        and (not variant_only_against or _normalize_org(hit.linked_id) == _normalize_org(variant_only_against))
    )
    if not filtered:
        return EntityResolution(False, reason="no_alternative_candidate", candidates=hits)

    ranked = tuple(sorted(filtered, key=lambda h: h.score, reverse=True))
    top = ranked[0]
    if top.source == "records_exact" or top.score >= 0.999:
        return EntityResolution(True, top, candidates=ranked)
    if top.score < min_score:
        return EntityResolution(False, reason=f"low_confidence:{top.score:.3f}", candidates=ranked)
    if len(ranked) > 1 and ranked[1].score >= min_score and (top.score - ranked[1].score) < min_margin:
        # Decoration variants of ONE org ("The X" / "X (BEIS)" / case variants) are not an
        # ambiguity — but the org's rows are split across them, so surface every variant for an
        # IN filter instead of picking one and silently undercounting.
        keys = {_org_equivalence_key(hit.linked_id) for hit in ranked}
        if len(keys) == 1:
            return EntityResolution(True, top, candidates=ranked,
                                    variants=tuple(dict.fromkeys(hit.linked_id for hit in ranked)))
        return EntityResolution(False, reason="ambiguous_entity_candidates", candidates=ranked)
    return EntityResolution(True, top, candidates=ranked)


def _normalize_org(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


__all__ = ["EntityResolution", "resolve_confident_org"]
