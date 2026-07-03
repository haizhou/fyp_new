"""Tests for deterministic KG v0 construction and validation."""

from __future__ import annotations

import pandas as pd

from procurement_graph.kg.build import build_kg_tables, run_validation
from procurement_graph.kg.validate import FAIL


def _inputs() -> dict[str, pd.DataFrame]:
    return {
        "canonical_orgs": pd.DataFrame(
            [
                {
                    "canonical_id": "ORG-BUYER",
                    "canonical_name": "Buyer Authority",
                    "org_type": "buyer",
                    "address_region": "UK",
                    "address_regions": '["UK"]',
                    "org_category": "BODY_PUBLIC",
                    "official_scheme": "",
                    "official_id": "",
                    "er_status": "deterministic",
                    "n_aliases": 1,
                },
                {
                    "canonical_id": "ORG-SUPPLIER",
                    "canonical_name": "Supplier Ltd",
                    "org_type": "supplier",
                    "address_region": "UK",
                    "address_regions": '["UK"]',
                    "org_category": "",
                    "official_scheme": "GB-COH",
                    "official_id": "12345678",
                    "er_status": "deterministic",
                    "n_aliases": 1,
                },
            ]
        ),
        "alias_map": pd.DataFrame(
            [
                {"raw_id": "RAW-BUYER", "canonical_id": "ORG-BUYER", "alias_source": "test"},
                {"raw_id": "RAW-SUPPLIER", "canonical_id": "ORG-SUPPLIER", "alias_source": "test"},
            ]
        ),
        "releases": pd.DataFrame(
            [
                {
                    "ocid": "ocds-test-1",
                    "date": "2025-01-01T00:00:00Z",
                    "year": 2025,
                    "buyer_raw_id": "RAW-BUYER",
                    "tender_title": "Test Tender",
                    "tender_value_amount": 1000.0,
                    "tender_value_currency": "GBP",
                    "tender_cpv_id": "72000000",
                    "tender_cpv_description": "IT services",
                    "tender_method": "open",
                    "tender_category": "services",
                    "tender_period_end": "2025-02-01T00:00:00Z",
                    "contract_period_start": "2025-03-01T00:00:00Z",
                    "contract_period_end": "2026-03-01T00:00:00Z",
                }
            ]
        ),
        "awards": pd.DataFrame(
            [
                {
                    "ocid": "ocds-test-1",
                    "award_id": "award-1",
                    "contract_id": "contract-1",
                    "award_title": "Award One",
                    "award_status": "active",
                    "award_value_amount": None,
                    "award_value_currency": "",
                    "award_value_best_amount": 1000.0,
                    "award_value_best_currency": "GBP",
                    "award_value_source": "contract",
                    "award_date_signed": "2025-03-02T00:00:00Z",
                    "award_period_start": "",
                    "award_period_end": "",
                    "related_lots": '["1"]',
                    "above_threshold": True,
                    "supplier_raw_ids": '["RAW-SUPPLIER"]',
                }
            ]
        ),
        "text_evidence": pd.DataFrame(
            [
                {
                    "ocid": "ocds-test-1",
                    "field_path": "tender.description",
                    "lot_id": "",
                    "text": "A compact description of the test tender.",
                }
            ]
        ),
    }


def test_builds_minimum_kg_tables() -> None:
    inputs = _inputs()
    tables = build_kg_tables(inputs)

    assert len(tables["org_nodes"]) == 2
    assert len(tables["contract_nodes"]) == 1
    assert len(tables["cpv_nodes"]) == 1
    assert len(tables["evidence_nodes"]) == 1
    assert len(tables["buyer_of"]) == 1
    assert len(tables["supplier_of"]) == 1
    assert len(tables["categorized_by"]) == 1
    assert len(tables["evidence_for"]) == 1
    assert bool(tables["contract_nodes"].iloc[0]["value_is_additive"]) is True
    assert bool(tables["contract_nodes"].iloc[0]["has_award_signed_date"]) is True
    assert tables["org_nodes"].set_index("canonical_id").loc["ORG-BUYER", "buyer_contract_count"] == 1
    assert tables["org_nodes"].set_index("canonical_id").loc["ORG-BUYER", "address_regions"] == '["UK"]'
    assert tables["cpv_nodes"].iloc[0]["contract_count"] == 1

    checks = run_validation(tables, inputs)
    assert not [check for check in checks if check.status == FAIL]


def test_validation_catches_missing_edge_endpoint() -> None:
    inputs = _inputs()
    tables = build_kg_tables(inputs)
    tables["supplier_of"].loc[0, "canonical_id"] = "MISSING-ORG"

    checks = run_validation(tables, inputs)
    failures = [check for check in checks if check.status == FAIL]
    assert any(check.name == "supplier_of org endpoints" for check in failures)


def test_tender_fallback_is_not_additive() -> None:
    inputs = _inputs()
    inputs["awards"].loc[0, "award_value_source"] = "tender"
    tables = build_kg_tables(inputs)
    row = tables["contract_nodes"].iloc[0]

    assert row["value_source"] == "tender"
    assert bool(row["value_is_additive"]) is False
    checks = run_validation(tables, inputs)
    assert not [check for check in checks if check.status == FAIL]
