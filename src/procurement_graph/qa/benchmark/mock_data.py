"""Mock KG records for QA benchmark framework tests."""

from __future__ import annotations


def mock_contract_records() -> list[dict[str, object]]:
    return [
        {
            "record_id": "r1",
            "ocid": "ocds-mock-001",
            "buyer_name": "Alpha NHS Trust",
            "supplier_name": "MedSupply Ltd",
            "award_value": 1000,
            "award_date": "2025-01-10",
            "region": "UKI",
        },
        {
            "record_id": "r2",
            "ocid": "ocds-mock-002",
            "buyer_name": "Alpha NHS Trust",
            "supplier_name": "CareWorks Ltd",
            "award_value": 2500,
            "award_date": "2025-03-12",
            "region": "UKI",
        },
        {
            "record_id": "r3",
            "ocid": "ocds-mock-003",
            "buyer_name": "Beta Council",
            "supplier_name": "RoadBuild PLC",
            "award_value": 7500,
            "award_date": "2024-11-02",
            "region": "UKJ",
        },
        {
            "record_id": "r4",
            "ocid": "ocds-mock-004",
            "buyer_name": "Gamma University",
            "supplier_name": "LabSupply Ltd",
            "award_value": 3200,
            "award_date": "2026-02-20",
            "region": "UKK",
        },
    ]


__all__ = ["mock_contract_records"]
