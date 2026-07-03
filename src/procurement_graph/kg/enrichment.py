"""Deterministic enrichment for KG v0 node tables."""

from __future__ import annotations

import pandas as pd


def enrich_org_nodes(
    org_nodes: pd.DataFrame,
    contract_nodes: pd.DataFrame,
    buyer_edges: pd.DataFrame,
    supplier_edges: pd.DataFrame,
) -> pd.DataFrame:
    nodes = org_nodes.copy()
    recalculated_columns = [
        "is_buyer",
        "is_supplier",
        "is_mixed_role",
        "buyer_contract_count",
        "supplier_contract_count",
        "buyer_additive_value_sum",
        "supplier_additive_value_sum",
        "first_activity_year",
        "last_activity_year",
    ]
    nodes = nodes.drop(columns=recalculated_columns, errors="ignore")
    contract_facts = contract_nodes[
        ["contract_node_id", "release_year", "value_amount", "value_is_additive"]
    ].copy()
    contract_facts["release_year"] = pd.to_numeric(contract_facts["release_year"], errors="coerce")
    contract_facts["value_amount"] = pd.to_numeric(contract_facts["value_amount"], errors="coerce")
    additive = contract_facts["value_is_additive"].fillna(False)
    contract_facts["additive_value"] = contract_facts["value_amount"].where(additive, 0.0)

    def role_summary(edges: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if edges.empty:
            return pd.DataFrame(columns=["canonical_id", f"{prefix}_contract_count", f"{prefix}_additive_value_sum", f"{prefix}_first_year", f"{prefix}_last_year"])
        merged = edges[["canonical_id", "contract_node_id"]].merge(contract_facts, on="contract_node_id", how="left")
        return (
            merged.groupby("canonical_id", as_index=False)
            .agg(
                **{
                    f"{prefix}_contract_count": ("contract_node_id", "nunique"),
                    f"{prefix}_additive_value_sum": ("additive_value", "sum"),
                    f"{prefix}_first_year": ("release_year", "min"),
                    f"{prefix}_last_year": ("release_year", "max"),
                }
            )
        )

    buyer = role_summary(buyer_edges, "buyer")
    supplier = role_summary(supplier_edges, "supplier")
    nodes = nodes.merge(buyer, on="canonical_id", how="left").merge(supplier, on="canonical_id", how="left")

    for column in ["buyer_contract_count", "supplier_contract_count"]:
        nodes[column] = pd.to_numeric(nodes[column], errors="coerce").fillna(0).astype("Int64")
    for column in ["buyer_additive_value_sum", "supplier_additive_value_sum"]:
        nodes[column] = pd.to_numeric(nodes[column], errors="coerce").fillna(0.0)

    nodes["is_buyer"] = nodes["buyer_contract_count"] > 0
    nodes["is_supplier"] = nodes["supplier_contract_count"] > 0
    nodes["is_mixed_role"] = nodes["is_buyer"] & nodes["is_supplier"]
    first_year = pd.concat([nodes.get("buyer_first_year"), nodes.get("supplier_first_year")], axis=1).min(axis=1)
    last_year = pd.concat([nodes.get("buyer_last_year"), nodes.get("supplier_last_year")], axis=1).max(axis=1)
    nodes["first_activity_year"] = first_year.astype("Int64")
    nodes["last_activity_year"] = last_year.astype("Int64")
    return nodes.drop(columns=["buyer_first_year", "buyer_last_year", "supplier_first_year", "supplier_last_year"], errors="ignore")


def enrich_cpv_nodes(
    cpv_nodes: pd.DataFrame,
    contract_nodes: pd.DataFrame,
    categorized_by_edges: pd.DataFrame,
) -> pd.DataFrame:
    nodes = cpv_nodes.copy()
    if categorized_by_edges.empty:
        return nodes
    facts = contract_nodes[
        ["contract_node_id", "release_year", "value_amount", "value_is_additive"]
    ].copy()
    facts["release_year"] = pd.to_numeric(facts["release_year"], errors="coerce")
    facts["value_amount"] = pd.to_numeric(facts["value_amount"], errors="coerce")
    facts["additive_value"] = facts["value_amount"].where(facts["value_is_additive"].fillna(False), 0.0)
    merged = categorized_by_edges[["cpv_code", "contract_node_id"]].merge(facts, on="contract_node_id", how="left")
    summary = (
        merged.groupby("cpv_code", as_index=False)
        .agg(
            contract_count=("contract_node_id", "nunique"),
            additive_value_sum=("additive_value", "sum"),
            first_activity_year=("release_year", "min"),
            last_activity_year=("release_year", "max"),
        )
    )
    nodes = nodes.drop(
        columns=["contract_count", "additive_value_sum", "first_activity_year", "last_activity_year"],
        errors="ignore",
    ).merge(summary, on="cpv_code", how="left")
    nodes["contract_count"] = pd.to_numeric(nodes["contract_count"], errors="coerce").fillna(0).astype("Int64")
    nodes["additive_value_sum"] = pd.to_numeric(nodes["additive_value_sum"], errors="coerce").fillna(0.0)
    nodes["first_activity_year"] = pd.to_numeric(nodes["first_activity_year"], errors="coerce").astype("Int64")
    nodes["last_activity_year"] = pd.to_numeric(nodes["last_activity_year"], errors="coerce").astype("Int64")
    return nodes
