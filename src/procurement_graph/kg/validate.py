"""Validation checks for deterministic KG v0 outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json

import pandas as pd


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass(frozen=True)
class ValidationCheck:
    area: str
    name: str
    status: str
    detail: str


def _check(area: str, name: str, condition: bool, detail: str) -> ValidationCheck:
    return ValidationCheck(area=area, name=name, status=PASS if condition else FAIL, detail=detail)


def _warn(area: str, name: str, detail: str) -> ValidationCheck:
    return ValidationCheck(area=area, name=name, status=WARN, detail=detail)


def _duplicate_count(df: pd.DataFrame, columns: list[str]) -> int:
    if df.empty:
        return 0
    return int(df.duplicated(subset=columns, keep=False).sum())


def validate_kg(
    *,
    org_nodes: pd.DataFrame,
    contract_nodes: pd.DataFrame,
    cpv_nodes: pd.DataFrame,
    buyer_edges: pd.DataFrame,
    supplier_edges: pd.DataFrame,
    categorized_by_edges: pd.DataFrame,
    evidence_nodes: pd.DataFrame,
    evidence_for_edges: pd.DataFrame,
    awards: pd.DataFrame,
    alias_map: pd.DataFrame,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []

    org_ids = set(org_nodes.get("canonical_id", pd.Series(dtype="object")).dropna().astype(str))
    contract_ids = set(contract_nodes.get("contract_node_id", pd.Series(dtype="object")).dropna().astype(str))
    cpv_codes = set(cpv_nodes.get("cpv_code", pd.Series(dtype="object")).dropna().astype(str))
    evidence_ids = set(evidence_nodes.get("evidence_id", pd.Series(dtype="object")).dropna().astype(str))
    alias_targets = set(alias_map.get("canonical_id", pd.Series(dtype="object")).dropna().astype(str))

    checks.append(_check("nodes", "org node primary key", _duplicate_count(org_nodes, ["canonical_id"]) == 0, "duplicate canonical_id rows = 0"))
    checks.append(_check("nodes", "contract node primary key", _duplicate_count(contract_nodes, ["contract_node_id"]) == 0, "duplicate contract_node_id rows = 0"))
    checks.append(_check("nodes", "award key uniqueness", _duplicate_count(contract_nodes, ["ocid", "award_id"]) == 0, "duplicate (ocid, award_id) rows = 0"))
    checks.append(_check("nodes", "cpv node primary key", _duplicate_count(cpv_nodes, ["cpv_code"]) == 0, "duplicate cpv_code rows = 0"))
    checks.append(_check("nodes", "evidence node primary key", _duplicate_count(evidence_nodes, ["evidence_id"]) == 0, "duplicate evidence_id rows = 0"))
    invalid_region_sets, missing_primary_region = _region_set_issues(org_nodes)
    checks.append(_check("nodes", "org address_regions JSON", invalid_region_sets == 0, f"invalid address_regions rows = {invalid_region_sets:,}"))
    checks.append(_check("nodes", "primary address_region included in region set", missing_primary_region == 0, f"missing primary region rows = {missing_primary_region:,}"))

    missing_alias_targets = alias_targets - org_ids
    checks.append(
        _check(
            "entities",
            "alias targets exist as org nodes",
            not missing_alias_targets,
            f"missing alias targets = {len(missing_alias_targets):,}",
        )
    )

    expected_contract_rows = len(awards)
    checks.append(
        _check(
            "nodes",
            "award rows represented as contract nodes",
            len(contract_nodes) == expected_contract_rows,
            f"contract_nodes={len(contract_nodes):,}; awards={expected_contract_rows:,}",
        )
    )

    for edge_name, edges in [
        ("buyer_of", buyer_edges),
        ("supplier_of", supplier_edges),
    ]:
        missing_org = set(edges.get("canonical_id", pd.Series(dtype="object")).dropna().astype(str)) - org_ids
        missing_contract = set(edges.get("contract_node_id", pd.Series(dtype="object")).dropna().astype(str)) - contract_ids
        checks.append(_check("edges", f"{edge_name} org endpoints", not missing_org, f"missing org endpoints = {len(missing_org):,}"))
        checks.append(_check("edges", f"{edge_name} contract endpoints", not missing_contract, f"missing contract endpoints = {len(missing_contract):,}"))
        checks.append(_check("edges", f"{edge_name} edge_id primary key", _duplicate_count(edges, ["edge_id"]) == 0, "duplicate edge_id rows = 0"))

    missing_cpv_contracts = set(
        categorized_by_edges.get("contract_node_id", pd.Series(dtype="object")).dropna().astype(str)
    ) - contract_ids
    missing_cpv_codes = set(categorized_by_edges.get("cpv_code", pd.Series(dtype="object")).dropna().astype(str)) - cpv_codes
    checks.append(_check("edges", "categorized_by contract endpoints", not missing_cpv_contracts, f"missing contract endpoints = {len(missing_cpv_contracts):,}"))
    checks.append(_check("edges", "categorized_by cpv endpoints", not missing_cpv_codes, f"missing cpv endpoints = {len(missing_cpv_codes):,}"))
    checks.append(_check("edges", "categorized_by edge_id primary key", _duplicate_count(categorized_by_edges, ["edge_id"]) == 0, "duplicate edge_id rows = 0"))

    missing_evidence_ids = set(evidence_for_edges.get("evidence_id", pd.Series(dtype="object")).dropna().astype(str)) - evidence_ids
    missing_evidence_contracts = set(
        evidence_for_edges.get("contract_node_id", pd.Series(dtype="object")).dropna().astype(str)
    ) - contract_ids
    checks.append(_check("edges", "evidence_for evidence endpoints", not missing_evidence_ids, f"missing evidence endpoints = {len(missing_evidence_ids):,}"))
    checks.append(_check("edges", "evidence_for contract endpoints", not missing_evidence_contracts, f"missing contract endpoints = {len(missing_evidence_contracts):,}"))
    checks.append(_check("edges", "evidence_for edge_id primary key", _duplicate_count(evidence_for_edges, ["edge_id"]) == 0, "duplicate edge_id rows = 0"))

    if "value_source" in contract_nodes.columns and "value_is_additive" in contract_nodes.columns:
        additive_sources = contract_nodes["value_source"].isin(["award", "contract"])
        bad_additive = contract_nodes[additive_sources & ~contract_nodes["value_is_additive"].fillna(False)]
        bad_tender = contract_nodes[(contract_nodes["value_source"] == "tender") & contract_nodes["value_is_additive"].fillna(False)]
        checks.append(_check("values", "award/contract values are additive", bad_additive.empty, f"bad additive rows = {len(bad_additive):,}"))
        checks.append(_check("values", "tender fallback values are non-additive", bad_tender.empty, f"bad tender fallback rows = {len(bad_tender):,}"))
    else:
        checks.append(_warn("values", "value additivity columns", "missing value_source or value_is_additive"))

    buyer_coverage = len(buyer_edges) / len(contract_nodes) * 100 if len(contract_nodes) else 0.0
    supplier_awards = supplier_edges["contract_node_id"].nunique() if not supplier_edges.empty else 0
    supplier_coverage = supplier_awards / len(contract_nodes) * 100 if len(contract_nodes) else 0.0
    evidence_contracts = evidence_for_edges["contract_node_id"].nunique() if not evidence_for_edges.empty else 0
    evidence_coverage = evidence_contracts / len(contract_nodes) * 100 if len(contract_nodes) else 0.0
    checks.append(_warn("coverage", "buyer edge coverage", f"{len(buyer_edges):,}/{len(contract_nodes):,} contract nodes have buyer edges ({buyer_coverage:.1f}%)"))
    checks.append(_warn("coverage", "supplier award coverage", f"{supplier_awards:,}/{len(contract_nodes):,} contract nodes have supplier edges ({supplier_coverage:.1f}%)"))
    checks.append(_warn("coverage", "evidence contract coverage", f"{evidence_contracts:,}/{len(contract_nodes):,} contract nodes have text evidence pointers ({evidence_coverage:.1f}%)"))

    return checks


def _region_set_issues(org_nodes: pd.DataFrame) -> tuple[int, int]:
    if "address_regions" not in org_nodes.columns:
        return len(org_nodes), len(org_nodes)
    invalid = 0
    missing_primary = 0
    for row in org_nodes[["address_region", "address_regions"]].itertuples(index=False):
        try:
            regions = json.loads(row.address_regions or "[]")
        except Exception:
            invalid += 1
            continue
        if not isinstance(regions, list) or any(not isinstance(region, str) for region in regions):
            invalid += 1
            continue
        primary = str(row.address_region or "").strip()
        if primary and primary not in set(regions):
            missing_primary += 1
    return invalid, missing_primary


def validation_summary(checks: list[ValidationCheck]) -> dict[str, int]:
    return {
        PASS: sum(1 for check in checks if check.status == PASS),
        WARN: sum(1 for check in checks if check.status == WARN),
        FAIL: sum(1 for check in checks if check.status == FAIL),
    }
