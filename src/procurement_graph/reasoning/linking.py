"""Deterministic linking of a user question to KG fields, values, and entities.

Turns free text into structured `QueryConstraint`s plus `EntityLinkCandidate`s, and flags
concepts the frozen KG v0.1 schema cannot answer. This is the entity/CPV/field/date/value
linking layer (design Stage 1.5); the planner consumes it so classification and linking stay
separated. Organization linking is optional and pluggable via `OrgResolver`, so linking works
with no KG loaded (regex only) or with a canonical-name index when one is provided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .entity_resolution import resolve_confident_org
from .models import EntityLinkCandidate, QueryConstraint

CPV_RE = re.compile(r"\b(?:CPV\s*)?(\d{8})\b", flags=re.IGNORECASE)
YEAR_RE = re.compile(r"\b(20[2-3]\d)\b")
# "between 2022 and 2024" / "from 2022 to 2024": a RANGE, not two contradictory eq filters.
YEAR_RANGE_RE = re.compile(r"\b(?:between|from)\s+(20[2-3]\d)\s+(?:and|to|through|until)\s+(20[2-3]\d)\b")
CONTRACT_ID_RE = re.compile(r"\b(contract:[A-Za-z0-9:_\-.]+|c\d+)\b")
CATEGORIES = ("goods", "services", "works")
# A bare category word is not evidence of a category filter: org names ("Atos IT Services UK
# Limited") and CPV descriptions ("Health services") contain them. Require category context.
CATEGORY_CONTEXT_RE = re.compile(
    r"\bfor\s+(goods|services|works)\b"
    r"|\b(goods|services?|works)\s+(?:contract|notice|tender)s?\b"
    r"|\b(?:categori[sz]ed|classified)\s+as\s+(goods|services|works)\b"
    r"|\bin\s+the\s+(goods|services|works)\s+(?:\w+\s+)?category\b"
    r"|\bcategory\s+(?:is\s+|of\s+)?[\"']?(goods|services|works)\b"
)

UNSUPPORTED_TERMS: dict[str, str] = {
    "social value": "social-value clauses are not available in KG v0.1",
    "payment term": "payment-term text is not available in KG v0.1",
    "subcontract": "subcontracting details are not available in KG v0.1",
    "framework lot": "lot-level breakdowns are not available in KG v0.1",
    # non-KG attributes probed by the unanswerable subset. Phrases are specific enough not to
    # false-positive on answerable count/sum/factoid/predicate questions (or quoted tender titles).
    "bidder": "bidder / tender-participation counts are not in KG v0.1",
    "framework agreement reference": "framework agreement references are not in KG v0.1",
    "an sme": "supplier SME status is not in KG v0.1",
    "sme status": "supplier SME status is not in KG v0.1",
    "contract manager": "named contract managers are not in KG v0.1",
    "evaluation criteria": "evaluation-criteria weightings are not in KG v0.1",
    "evaluation score": "evaluation scores are not in KG v0.1",
    "delivery performance": "delivery-performance metrics are not in KG v0.1",
    "invoice or payment": "invoice / payment dates are not in KG v0.1",
}

# Single common words matched as WHOLE WORDS (not substrings) so "deCARBONisation", "Fairfield",
# titles, etc. don't false-trigger. Includes subjective-judgement questions
# ("was the contract fair/reasonable/reliable?"), which are unanswerable from structured KG facts.
_JUDGEMENT_TERMS: dict[str, str] = {
    "carbon": "carbon and emissions clauses are not available in KG v0.1",
    "emission": "carbon and emissions clauses are not available in KG v0.1",
    "emissions": "carbon and emissions clauses are not available in KG v0.1",
    "fair": "a fairness/quality judgement is not answerable from KG v0.1 facts",
    "reasonable": "a value/quality judgement is not answerable from KG v0.1 facts",
    "reliable": "a supplier-reliability judgement is not answerable from KG v0.1 facts",
    "value for money": "a value-for-money judgement is not answerable from KG v0.1 facts",
    "good value": "a value judgement is not answerable from KG v0.1 facts",
}


class OrgResolver(Protocol):
    """Maps an organisation mention to candidate KG organisations."""

    def resolve(self, mention: str) -> list[EntityLinkCandidate]:
        ...


@dataclass(frozen=True)
class LinkingResult:
    constraints: tuple[QueryConstraint, ...] = ()
    entities: tuple[EntityLinkCandidate, ...] = ()
    unsupported_reason: str = ""
    asks_count: bool = False
    asks_sum: bool = False
    factoid_field: str = ""
    explicit_contract_id: str = ""
    notes: tuple[str, ...] = ()


def link_question(question: str, *, org_resolver: OrgResolver | None = None) -> LinkingResult:
    text = " ".join(str(question or "").split())
    lowered = text.casefold()

    unsupported = _unsupported_reason(lowered)
    constraints: list[QueryConstraint] = []
    entities: list[EntityLinkCandidate] = []
    notes: list[str] = []

    # Years: a range phrase becomes ONE between filter; several bare years become an `in` union
    # ("in 2022 and 2023" counts both). Two eq filters on the same field would be a contradiction
    # that the reflector can only repair by narrowing — i.e. a silently wrong scope.
    range_match = YEAR_RANGE_RE.search(lowered)
    years = sorted({int(year) for year in YEAR_RE.findall(text)})
    if range_match:
        low, high = sorted((int(range_match.group(1)), int(range_match.group(2))))
        constraints.append(QueryConstraint("release_year", "between", [low, high], source_text=range_match.group(0)))
    elif len(years) > 1:
        constraints.append(QueryConstraint("release_year", "in", years, source_text=", ".join(str(y) for y in years)))
    elif years:
        constraints.append(QueryConstraint("release_year", "eq", years[0], source_text=str(years[0])))
    category_match = CATEGORY_CONTEXT_RE.search(lowered)
    if category_match:
        category = next(group for group in category_match.groups() if group)
        if category == "service":  # "service contract notices" -> the 'services' category
            category = "services"
        constraints.append(QueryConstraint("tender_category", "eq", category, source_text=category_match.group(0)))
    cpv_match = CPV_RE.search(text)
    if cpv_match:
        constraints.append(QueryConstraint("tender_cpv_id", "eq", cpv_match.group(1), source_text=cpv_match.group(0)))
    if "signed award date" in lowered or "award signed date" in lowered or "signed date" in lowered:
        constraints.append(QueryConstraint("has_award_signed_date", "eq", True, source_text="signed award date"))

    factoid_field = _factoid_field(lowered)
    explicit_contract_id = _explicit_contract_id(text)

    # Organisation linking (optional). Buyer/supplier role words steer which field the
    # resolved organisation constrains; without a resolver, org mentions are left to the
    # planner/backend as buyer_name/supplier_name text filters.
    if org_resolver is not None:
        for role_field, mention in _org_mentions(text, lowered):
            resolve_mention = mention
            candidates = org_resolver.resolve(resolve_mention)
            if not candidates and mention.rstrip(". ") != mention:
                resolve_mention = mention.rstrip(". ")
                candidates = org_resolver.resolve(resolve_mention)  # sentence-final period
            if not candidates:
                notes.append(f"unresolved organisation mention: {mention!r}")
                continue
            resolved = resolve_confident_org(org_resolver, resolve_mention)
            if not resolved.ok or resolved.hit is None:
                # a weak fuzzy hit is a hallucinated entity link waiting to happen — leave the
                # mention unresolved (the planner/LLM layer can still handle it) rather than
                # silently constraining on the wrong organisation.
                notes.append(f"low-confidence organisation link dropped: {mention!r}: {resolved.reason}")
                continue
            best = resolved.hit
            entities.append(best)
            constraints.append(
                QueryConstraint(role_field, "eq", best.linked_label, source_text=mention,
                                confidence=best.score, metadata={"canonical_id": best.linked_id})
            )

    return LinkingResult(
        constraints=tuple(constraints),
        entities=tuple(entities),
        unsupported_reason=unsupported,
        asks_count=_asks_count(lowered),
        asks_sum=_asks_sum(lowered),
        factoid_field=factoid_field,
        explicit_contract_id=explicit_contract_id,
        notes=tuple(notes),
    )


def _org_mentions(text: str, lowered: str) -> list[tuple[str, str]]:
    """Very small heuristic for 'awarded to <supplier>' / 'buyer <name>' patterns."""
    mentions: list[tuple[str, str]] = []
    # '.' must not be a stop character: initialisms ("The C.S.P (...) JOINT VENTURE") would be cut
    # at the first period, and the stub then fuzzy-links to an unrelated org. Parens are part of
    # many stored names, so they are allowed inside the mention.
    _stop = r"(?:\s+(?:in|under|for)\b|[?]|,\s|$)"
    for pattern, role_field in (
        (rf"awarded to ([A-Z][\w&.'()\- ]+?){_stop}", "supplier_name"),
        (rf"supplier ([A-Z][\w&.'()\- ]+?){_stop}", "supplier_name"),
        (rf"(?:buyer|contracting authority) ([A-Z][\w&.'()\- ]+?){_stop}", "buyer_name"),
    ):
        for match in re.finditer(pattern, text):
            # keep a trailing '.' — "EDS Ltd." and "EDS Ltd" can be two DIFFERENT stored orgs, so
            # stripping the dot exact-matches the wrong one. The resolver caller retries dot-less.
            mention = match.group(1).strip(" ,")
            if len(mention.strip(".")) >= 3:
                mentions.append((role_field, mention))
    return mentions


def _asks_count(lowered: str) -> bool:
    return "how many" in lowered or lowered.startswith("count ") or " number of " in lowered


SUM_PHRASE_RE = re.compile(r"\btotal\s+(?:\w+\s+){0,2}value\b")  # total [recorded] [contract] value


def _asks_sum(lowered: str) -> bool:
    return bool(SUM_PHRASE_RE.search(lowered)) or lowered.startswith("sum ") or "combined value" in lowered


def _factoid_field(lowered: str) -> str:
    if "who is the buyer" in lowered or "which buyer" in lowered or "contracting authority" in lowered:
        return "buyer_name"
    if "who is the supplier" in lowered or "which supplier" in lowered or "who supplied" in lowered:
        return "supplier_name"
    if "what category" in lowered or "which category" in lowered:
        return "tender_category"
    return ""


def _explicit_contract_id(text: str) -> str:
    match = CONTRACT_ID_RE.search(text)
    return match.group(1) if match else ""


def _unsupported_reason(lowered: str) -> str:
    # An unsupported concept is the question's SUBJECT, never inside a quoted tender title; strip
    # quoted spans so a title like "... Carbon C Autoanalyser" doesn't false-trigger.
    scrubbed = re.sub(r'["“][^"”]*["”]', " ", lowered)
    for term, reason in UNSUPPORTED_TERMS.items():
        if term in scrubbed:
            return reason
    for term, reason in _JUDGEMENT_TERMS.items():
        if re.search(rf"\b{re.escape(term)}\b", scrubbed):
            return reason
    return ""


__all__ = ["LinkingResult", "OrgResolver", "link_question", "UNSUPPORTED_TERMS"]
