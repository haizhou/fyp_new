"""Lightweight schema retrieval for graph planning prompts.

This is deliberately small and dependency-free. It selects a local procurement ontology slice from
question keywords so prompts get relevant node/relation/field hints without dumping the full KG
schema. A dense retriever can replace this interface later.
"""

from __future__ import annotations

import re
from typing import Any


_FRAGMENTS: dict[str, dict[str, Any]] = {
    "contract": {
        "keywords": ("contract", "notice", "record", "award"),
        "node_types": ("contract", "award"),
        "fields": ("contract_node_id", "tender_title", "award_title", "tender_category"),
        "relations": ("buyer publishes contract", "supplier is awarded contract"),
    },
    "buyer": {
        "keywords": ("buyer", "authority", "contracting authority", "published by", "issued by"),
        "node_types": ("buyer", "organization"),
        "fields": ("buyer_name",),
        "relations": ("buyer publishes contract", "buyer awards contract to supplier"),
    },
    "supplier": {
        "keywords": ("supplier", "winner", "awarded to", "vendor", "worked with", "supplied"),
        "node_types": ("supplier", "organization"),
        "fields": ("supplier_name",),
        "relations": ("supplier is awarded contract", "supplier works with buyer via contract"),
    },
    "value": {
        "keywords": ("value", "amount", "total", "sum", "spend", "spent", "£", "gbp"),
        "node_types": ("contract", "value"),
        "fields": ("value_amount", "value_is_additive"),
        "relations": ("contract has monetary value",),
        "guards": ("value_is_additive for monetary sums",),
    },
    "date": {
        "keywords": ("date", "signed", "published", "release", "before", "after", "between"),
        "node_types": ("date",),
        "fields": ("release_year", "release_date", "award_date_signed"),
        "relations": ("contract has release date", "award has signed date"),
    },
    "cpv": {
        "keywords": ("cpv", "category", "goods", "services", "works"),
        "node_types": ("cpv", "category"),
        "fields": ("tender_cpv_id", "tender_cpv_description", "tender_category"),
        "relations": ("contract has CPV classification", "contract has procurement category"),
    },
    "rank": {
        "keywords": ("top", "rank", "highest", "lowest", "most", "least", "largest", "smallest"),
        "operation_units": ("rank_top_k", "find_extreme"),
        "fields": ("value_amount", "buyer_name", "supplier_name", "contract_node_id"),
    },
    "compare": {
        "keywords": ("more than", "less than", "fewer than", "compare", "compared", "than"),
        "operation_units": ("count", "sum", "compare"),
    },
}


def retrieve_schema_context(question: str, *, max_fragments: int = 5) -> dict[str, Any]:
    low = " ".join(str(question).casefold().split())
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for name, fragment in _FRAGMENTS.items():
        score = sum(1 for keyword in fragment.get("keywords", ()) if keyword in low)
        if name == "date" and re.search(r"\b20\d{2}\b", low):
            score += 1
        if name == "cpv" and re.search(r"\b\d{5,8}\b", low):
            score += 1
        if score:
            scored.append((score, name, fragment))
    if not scored:
        scored.append((1, "contract", _FRAGMENTS["contract"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [{"name": name, **{k: v for k, v in fragment.items() if k != "keywords"}}
                for _score, name, fragment in scored[:max_fragments]]
    return {
        "retrieval_method": "keyword_schema_slice_v1",
        "selected_fragments": selected,
        "allowed_operation_units": (
            "filter_records", "distinct_set", "count", "sum", "select",
            "exists", "argmax", "argmin", "top_k", "compare", "abstain",
        ),
        "note": "Use these as local ontology hints only; question surface facts remain authoritative.",
    }


__all__ = ["retrieve_schema_context"]
