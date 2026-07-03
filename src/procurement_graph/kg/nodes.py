"""Build deterministic KG v0 node tables."""

from __future__ import annotations

import json

import pandas as pd

from .schema import CONTRACT_NODE_COLUMNS, CPV_NODE_COLUMNS, EVIDENCE_NODE_COLUMNS, ORG_NODE_COLUMNS


def contract_node_id(ocid: object, award_id: object) -> str:
    return f"contract:{str(ocid)}:{str(award_id)}"


def _string_series(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[column].fillna(default).astype(str)


def _value_is_additive(source: object) -> bool:
    return str(source or "") in {"award", "contract"}


def _non_empty(value: object) -> bool:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    return bool(str(value).strip())


def _days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    start_dt = pd.to_datetime(start.replace("", pd.NA), errors="coerce", utc=True)
    end_dt = pd.to_datetime(end.replace("", pd.NA), errors="coerce", utc=True)
    return (end_dt - start_dt).dt.days.astype("Int64")


def build_org_nodes(
    canonical_orgs: pd.DataFrame,
    alias_map: pd.DataFrame | None = None,
    releases: pd.DataFrame | None = None,
) -> pd.DataFrame:
    nodes = canonical_orgs.copy()
    for column in ORG_NODE_COLUMNS:
        if column not in nodes.columns:
            nodes[column] = 0 if column.endswith("_count") or column.endswith("_sum") else ""
    if alias_map is not None and not alias_map.empty:
        alias_counts = alias_map.groupby("canonical_id").size().rename("alias_raw_id_count")
        nodes = nodes.drop(columns=["alias_raw_id_count"], errors="ignore").merge(
            alias_counts,
            on="canonical_id",
            how="left",
        )
        if releases is not None and not releases.empty:
            region_sets = _address_region_sets(releases, alias_map)
            nodes = nodes.drop(columns=["address_regions"], errors="ignore").merge(
                region_sets,
                on="canonical_id",
                how="left",
            )
    nodes["alias_raw_id_count"] = pd.to_numeric(nodes["alias_raw_id_count"], errors="coerce").fillna(0).astype("Int64")
    nodes["address_regions"] = [
        _normalise_region_set(primary, regions)
        for primary, regions in zip(nodes.get("address_region", ""), nodes.get("address_regions", ""))
    ]
    nodes = nodes[ORG_NODE_COLUMNS].drop_duplicates(subset=["canonical_id"], keep="first")
    return nodes.sort_values("canonical_id").reset_index(drop=True)


def build_contract_nodes(awards: pd.DataFrame, releases: pd.DataFrame) -> pd.DataFrame:
    release_cols = [
        "ocid",
        "date",
        "year",
        "buyer_raw_id",
        "tender_title",
        "tender_value_amount",
        "tender_value_currency",
        "tender_cpv_id",
        "tender_cpv_description",
        "tender_method",
        "tender_category",
        "tender_period_end",
        "contract_period_start",
        "contract_period_end",
    ]
    available_release_cols = [c for c in release_cols if c in releases.columns]
    merged = awards.merge(
        releases[available_release_cols],
        on="ocid",
        how="left",
        validate="many_to_one",
    )

    nodes = pd.DataFrame(index=merged.index)
    nodes["ocid"] = _string_series(merged, "ocid")
    nodes["award_id"] = _string_series(merged, "award_id")
    nodes["contract_node_id"] = [
        contract_node_id(ocid, award_id)
        for ocid, award_id in zip(nodes["ocid"], nodes["award_id"])
    ]
    nodes["contract_id"] = _string_series(merged, "contract_id")
    nodes["release_date"] = _string_series(merged, "date")
    nodes["release_year"] = merged["year"] if "year" in merged.columns else ""
    nodes["buyer_raw_id"] = _string_series(merged, "buyer_raw_id")
    nodes["tender_title"] = _string_series(merged, "tender_title")
    nodes["award_title"] = _string_series(merged, "award_title")
    nodes["award_status"] = _string_series(merged, "award_status")
    nodes["tender_method"] = _string_series(merged, "tender_method")
    nodes["tender_category"] = _string_series(merged, "tender_category")
    nodes["tender_cpv_id"] = _string_series(merged, "tender_cpv_id")
    nodes["tender_cpv_description"] = _string_series(merged, "tender_cpv_description")
    nodes["value_amount"] = pd.to_numeric(
        merged.get("award_value_best_amount", merged.get("award_value_amount")),
        errors="coerce",
    )
    nodes["value_currency"] = _string_series(
        merged,
        "award_value_best_currency"
        if "award_value_best_currency" in merged.columns
        else "award_value_currency",
    )
    nodes["value_source"] = _string_series(merged, "award_value_source")
    nodes["value_is_additive"] = nodes["value_source"].map(_value_is_additive)
    nodes["award_date_signed"] = _string_series(merged, "award_date_signed")
    nodes["award_period_start"] = _string_series(merged, "award_period_start")
    nodes["award_period_end"] = _string_series(merged, "award_period_end")
    nodes["tender_period_end"] = _string_series(merged, "tender_period_end")
    nodes["contract_period_start"] = _string_series(merged, "contract_period_start")
    nodes["contract_period_end"] = _string_series(merged, "contract_period_end")
    nodes["has_award_signed_date"] = nodes["award_date_signed"].map(_non_empty)
    nodes["has_contract_period"] = nodes["contract_period_start"].map(_non_empty) & nodes["contract_period_end"].map(_non_empty)
    nodes["days_release_to_tender_end"] = _days_between(nodes["release_date"], nodes["tender_period_end"])
    nodes["days_release_to_award_signed"] = _days_between(nodes["release_date"], nodes["award_date_signed"])
    nodes["contract_duration_days"] = _days_between(nodes["contract_period_start"], nodes["contract_period_end"])
    nodes["related_lots"] = _string_series(merged, "related_lots")
    nodes["above_threshold"] = merged["above_threshold"] if "above_threshold" in merged.columns else ""

    return nodes[CONTRACT_NODE_COLUMNS].reset_index(drop=True)


def _cpv_prefix(code: str, n: int) -> str:
    code = str(code or "").strip()
    return code[:n] if len(code) >= n else ""


def build_cpv_nodes(releases: pd.DataFrame) -> pd.DataFrame:
    if "tender_cpv_id" not in releases.columns:
        return pd.DataFrame(columns=CPV_NODE_COLUMNS)
    cpv = releases[["tender_cpv_id", "tender_cpv_description"]].copy()
    cpv = cpv.rename(
        columns={
            "tender_cpv_id": "cpv_code",
            "tender_cpv_description": "cpv_description",
        }
    )
    cpv["cpv_code"] = cpv["cpv_code"].fillna("").astype(str).str.strip()
    cpv = cpv[cpv["cpv_code"] != ""]
    cpv["cpv_description"] = cpv["cpv_description"].fillna("").astype(str)
    cpv = cpv.sort_values(["cpv_code", "cpv_description"])
    cpv = cpv.drop_duplicates(subset=["cpv_code"], keep="first")
    cpv["cpv_division"] = cpv["cpv_code"].map(lambda value: _cpv_prefix(value, 2))
    cpv["cpv_group"] = cpv["cpv_code"].map(lambda value: _cpv_prefix(value, 3))
    cpv["cpv_class"] = cpv["cpv_code"].map(lambda value: _cpv_prefix(value, 4))
    cpv["contract_count"] = 0
    cpv["additive_value_sum"] = 0.0
    cpv["first_activity_year"] = pd.NA
    cpv["last_activity_year"] = pd.NA
    return cpv[CPV_NODE_COLUMNS].reset_index(drop=True)


def build_evidence_nodes(text_evidence: pd.DataFrame) -> pd.DataFrame:
    if text_evidence.empty:
        return pd.DataFrame(columns=EVIDENCE_NODE_COLUMNS)
    nodes = text_evidence[["ocid", "field_path", "lot_id", "text"]].copy()
    for column in ["ocid", "field_path", "lot_id", "text"]:
        nodes[column] = nodes[column].fillna("").astype(str)
    nodes = nodes.reset_index(drop=True)
    nodes["evidence_id"] = "evidence:" + nodes.index.astype(str)
    nodes["text_length"] = nodes["text"].str.len()
    return nodes[EVIDENCE_NODE_COLUMNS]


def _address_region_sets(releases: pd.DataFrame, alias_map: pd.DataFrame) -> pd.DataFrame:
    if "parties_json" not in releases.columns:
        return pd.DataFrame(columns=["canonical_id", "address_regions"])
    lookup = dict(
        zip(
            alias_map["raw_id"].fillna("").astype(str),
            alias_map["canonical_id"].fillna("").astype(str),
        )
    )
    region_map: dict[str, set[str]] = {}
    for parties_text in releases["parties_json"]:
        try:
            parties = json.loads(parties_text or "[]")
        except Exception:
            continue
        if not isinstance(parties, list):
            continue
        for party in parties:
            if not isinstance(party, dict):
                continue
            raw_id = str(party.get("raw_id") or "")
            region = str(party.get("address_region") or "").strip()
            canonical_id = lookup.get(raw_id, "")
            if not canonical_id or not region:
                continue
            region_map.setdefault(canonical_id, set()).add(region)
    rows = [
        {"canonical_id": canonical_id, "address_regions": json.dumps(sorted(regions))}
        for canonical_id, regions in region_map.items()
    ]
    return pd.DataFrame(rows, columns=["canonical_id", "address_regions"])


def _normalise_region_set(primary: object, regions_json: object) -> str:
    regions: set[str] = set()
    primary_text = str(primary or "").strip()
    if primary_text:
        regions.add(primary_text)
    if isinstance(regions_json, str) and regions_json.strip():
        try:
            parsed = json.loads(regions_json)
        except Exception:
            parsed = []
        if isinstance(parsed, list):
            regions.update(str(region).strip() for region in parsed if str(region).strip())
    return json.dumps(sorted(regions))
