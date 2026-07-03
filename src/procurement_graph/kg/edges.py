"""Build deterministic KG v0 edge tables."""

from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

from .nodes import contract_node_id
from .schema import BUYER_EDGE_COLUMNS, CATEGORIZED_BY_EDGE_COLUMNS, EVIDENCE_FOR_EDGE_COLUMNS, SUPPLIER_EDGE_COLUMNS


def _alias_lookup(alias_map: pd.DataFrame) -> dict[str, str]:
    if alias_map.empty:
        return {}
    clean = alias_map[["raw_id", "canonical_id"]].dropna().copy()
    clean["raw_id"] = clean["raw_id"].astype(str)
    clean["canonical_id"] = clean["canonical_id"].astype(str)
    return dict(zip(clean["raw_id"], clean["canonical_id"]))


def _parse_json_list(value: object) -> list[str]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item)]


def _contract_lot_map(contract_nodes: pd.DataFrame) -> dict[str, dict[str, list[tuple[str, str]]]]:
    by_ocid: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for row in contract_nodes[["ocid", "award_id", "contract_node_id", "related_lots"]].itertuples(index=False):
        ocid = str(row.ocid)
        award_id = str(row.award_id)
        node_id = str(row.contract_node_id)
        entry = by_ocid.setdefault(ocid, {"__all__": [], "__lots__": {}})
        entry["__all__"].append((node_id, award_id))
        for lot_id in _parse_json_list(row.related_lots):
            entry["__lots__"].setdefault(str(lot_id), []).append((node_id, award_id))
    return by_ocid


def build_buyer_edges(awards: pd.DataFrame, releases: pd.DataFrame, alias_map: pd.DataFrame) -> pd.DataFrame:
    lookup = _alias_lookup(alias_map)
    release_cols = ["ocid", "buyer_raw_id"]
    merged = awards[["ocid", "award_id"]].merge(
        releases[release_cols],
        on="ocid",
        how="left",
        validate="many_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        raw_id = str(getattr(row, "buyer_raw_id") or "")
        canonical_id = lookup.get(raw_id, "")
        if not raw_id or not canonical_id:
            continue
        ocid = str(row.ocid)
        award_id = str(row.award_id)
        node_id = contract_node_id(ocid, award_id)
        rows.append(
            {
                "edge_id": f"buyer:{canonical_id}:{node_id}",
                "canonical_id": canonical_id,
                "contract_node_id": node_id,
                "raw_id": raw_id,
                "ocid": ocid,
                "award_id": award_id,
                "edge_source": "releases.buyer_raw_id",
            }
        )
    return pd.DataFrame(rows, columns=BUYER_EDGE_COLUMNS).drop_duplicates().reset_index(drop=True)


def build_supplier_edges(awards: pd.DataFrame, alias_map: pd.DataFrame) -> pd.DataFrame:
    lookup = _alias_lookup(alias_map)
    rows = []
    for row in awards[["ocid", "award_id", "supplier_raw_ids"]].itertuples(index=False):
        ocid = str(row.ocid)
        award_id = str(row.award_id)
        node_id = contract_node_id(ocid, award_id)
        for raw_id in _parse_json_list(row.supplier_raw_ids):
            canonical_id = lookup.get(raw_id, "")
            if not canonical_id:
                continue
            rows.append(
                {
                    "edge_id": f"supplier:{canonical_id}:{node_id}",
                    "canonical_id": canonical_id,
                    "contract_node_id": node_id,
                    "raw_id": raw_id,
                    "ocid": ocid,
                    "award_id": award_id,
                    "edge_source": "awards.supplier_raw_ids",
                }
            )
    edges = pd.DataFrame(rows, columns=SUPPLIER_EDGE_COLUMNS).drop_duplicates()
    if edges.empty:
        return edges.reset_index(drop=True)
    group_cols = [
        "edge_id",
        "canonical_id",
        "contract_node_id",
        "ocid",
        "award_id",
        "edge_source",
    ]
    return (
        edges.groupby(group_cols, as_index=False)
        .agg(raw_id=("raw_id", lambda values: json.dumps(sorted(set(map(str, values))))))
        [SUPPLIER_EDGE_COLUMNS]
        .reset_index(drop=True)
    )


def build_categorized_by_edges(awards: pd.DataFrame, releases: pd.DataFrame) -> pd.DataFrame:
    merged = awards[["ocid", "award_id"]].merge(
        releases[["ocid", "tender_cpv_id"]],
        on="ocid",
        how="left",
        validate="many_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        cpv_code = str(getattr(row, "tender_cpv_id") or "").strip()
        if not cpv_code:
            continue
        ocid = str(row.ocid)
        award_id = str(row.award_id)
        node_id = contract_node_id(ocid, award_id)
        rows.append(
            {
                "edge_id": f"categorized_by:{node_id}:{cpv_code}",
                "contract_node_id": node_id,
                "cpv_code": cpv_code,
                "ocid": ocid,
                "award_id": award_id,
                "edge_source": "releases.tender_cpv_id",
            }
        )
    return pd.DataFrame(rows, columns=CATEGORIZED_BY_EDGE_COLUMNS).drop_duplicates().reset_index(drop=True)


def build_evidence_for_edges(evidence_nodes: pd.DataFrame, contract_nodes: pd.DataFrame) -> pd.DataFrame:
    if evidence_nodes.empty or contract_nodes.empty:
        return pd.DataFrame(columns=EVIDENCE_FOR_EDGE_COLUMNS)
    by_ocid = _contract_lot_map(contract_nodes)
    rows = []
    for row in evidence_nodes[["evidence_id", "ocid", "lot_id"]].itertuples(index=False):
        ocid = str(row.ocid)
        lot_id = str(row.lot_id or "").strip()
        mapping = by_ocid.get(ocid)
        if not mapping:
            continue
        matches = []
        match_scope = "ocid"
        if lot_id:
            matches = mapping["__lots__"].get(lot_id, [])
            match_scope = "lot"
        if not matches:
            matches = mapping["__all__"]
            match_scope = "ocid_fallback" if lot_id else "ocid"
        for node_id, award_id in matches:
            rows.append(
                {
                    "edge_id": f"evidence_for:{row.evidence_id}:{node_id}",
                    "evidence_id": row.evidence_id,
                    "contract_node_id": node_id,
                    "ocid": ocid,
                    "award_id": award_id,
                    "match_scope": match_scope,
                    "edge_source": "text_evidence.ocid_lot_id",
                }
            )
    return pd.DataFrame(rows, columns=EVIDENCE_FOR_EDGE_COLUMNS).drop_duplicates().reset_index(drop=True)


def iter_raw_supplier_ids(awards: pd.DataFrame) -> Iterable[str]:
    for value in awards.get("supplier_raw_ids", pd.Series(dtype="object")):
        yield from _parse_json_list(value)
