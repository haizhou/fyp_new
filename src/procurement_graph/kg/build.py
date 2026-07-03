"""Orchestrate deterministic KG v0 construction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .edges import build_buyer_edges, build_categorized_by_edges, build_evidence_for_edges, build_supplier_edges
from .enrichment import enrich_cpv_nodes, enrich_org_nodes
from .nodes import build_contract_nodes, build_cpv_nodes, build_evidence_nodes, build_org_nodes
from .schema import EDGE_FILES, NODE_FILES
from .validate import FAIL, ValidationCheck, validate_kg, validation_summary


def load_kg_inputs(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "releases": pd.read_parquet(root / "data" / "interim" / "releases.parquet"),
        "awards": pd.read_parquet(root / "data" / "extracted" / "awards.parquet"),
        "text_evidence": pd.read_parquet(root / "data" / "extracted" / "text_evidence.parquet"),
        "canonical_orgs": pd.read_parquet(root / "data" / "entities" / "canonical_orgs.parquet"),
        "alias_map": pd.read_parquet(root / "data" / "entities" / "alias_map.parquet"),
    }


def build_kg_tables(inputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    releases = inputs["releases"]
    awards = inputs["awards"]
    canonical_orgs = inputs["canonical_orgs"]
    alias_map = inputs["alias_map"]
    text_evidence = inputs["text_evidence"]

    org_nodes = build_org_nodes(canonical_orgs, alias_map, releases)
    contract_nodes = build_contract_nodes(awards, releases)
    cpv_nodes = build_cpv_nodes(releases)
    evidence_nodes = build_evidence_nodes(text_evidence)
    buyer_edges = build_buyer_edges(awards, releases, alias_map)
    supplier_edges = build_supplier_edges(awards, alias_map)
    categorized_by_edges = build_categorized_by_edges(awards, releases)
    evidence_for_edges = build_evidence_for_edges(evidence_nodes, contract_nodes)
    org_nodes = enrich_org_nodes(org_nodes, contract_nodes, buyer_edges, supplier_edges)
    cpv_nodes = enrich_cpv_nodes(cpv_nodes, contract_nodes, categorized_by_edges)

    return {
        "org_nodes": org_nodes,
        "contract_nodes": contract_nodes,
        "cpv_nodes": cpv_nodes,
        "evidence_nodes": evidence_nodes,
        "buyer_of": buyer_edges,
        "supplier_of": supplier_edges,
        "categorized_by": categorized_by_edges,
        "evidence_for": evidence_for_edges,
    }


def write_kg_tables(tables: dict[str, pd.DataFrame], kg_dir: Path) -> None:
    for subdir in ["nodes", "edges", "exports"]:
        (kg_dir / subdir).mkdir(parents=True, exist_ok=True)
    for name, rel_path in {**NODE_FILES, **EDGE_FILES}.items():
        tables[name].to_parquet(kg_dir / rel_path, index=False)


def run_validation(tables: dict[str, pd.DataFrame], inputs: dict[str, pd.DataFrame]) -> list[ValidationCheck]:
    return validate_kg(
        org_nodes=tables["org_nodes"],
        contract_nodes=tables["contract_nodes"],
        cpv_nodes=tables["cpv_nodes"],
        buyer_edges=tables["buyer_of"],
        supplier_edges=tables["supplier_of"],
        categorized_by_edges=tables["categorized_by"],
        evidence_nodes=tables["evidence_nodes"],
        evidence_for_edges=tables["evidence_for"],
        awards=inputs["awards"],
        alias_map=inputs["alias_map"],
    )


def write_validation_report(checks: list[ValidationCheck], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = validation_summary(checks)
    payload = {
        "summary": summary,
        "checks": [check.__dict__ for check in checks],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_and_validate(root: Path, kg_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], list[ValidationCheck], bool]:
    kg_dir = kg_dir or (root / "data" / "kg")
    inputs = load_kg_inputs(root)
    tables = build_kg_tables(inputs)
    checks = run_validation(tables, inputs)
    write_validation_report(checks, root / "reports" / "kg" / "kg_validation_summary.json")
    did_write = validation_summary(checks)[FAIL] == 0
    if did_write:
        write_kg_tables(tables, kg_dir)
    return tables, checks, did_write
