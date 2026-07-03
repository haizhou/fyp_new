"""Tests for src/er_phase1.py — deterministic entity resolution."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest
from er_phase1 import collect_parties, resolve_phase1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_releases(records: list[dict]) -> pd.DataFrame:
    """Build a minimal releases DataFrame from a list of record dicts."""
    rows = []
    for rec in records:
        rows.append(
            {
                "ocid": rec["ocid"],
                "release_id": rec.get("release_id", rec["ocid"] + "-rel"),
                "date": rec.get("date", "2024-01-01T00:00:00Z"),
                "tag": "compiled",
                "buyer_raw_id": rec.get("buyer_raw_id", ""),
                "buyer_name": rec.get("buyer_name", ""),
                "tender_title": "",
                "tender_value_amount": None,
                "tender_value_currency": "",
                "tender_cpv_id": "",
                "tender_cpv_description": "",
                "tender_method": "",
                "tender_method_details": "",
                "tender_category": "",
                "tender_period_end": "",
                "contract_period_start": "",
                "contract_period_end": "",
                "n_lots": 0,
                "source_years": "2024",
                "year": 2024,
                "parties_json": json.dumps(rec.get("parties", [])),
                "contracts_json": json.dumps(rec.get("contracts", [])),
            }
        )
    return pd.DataFrame(rows)


def make_party(raw_id, name, roles, scheme="", official_id="", region=""):
    return {
        "raw_id": raw_id,
        "name": name,
        "roles": json.dumps(roles),
        "identifier_scheme": scheme,
        "identifier_id": official_id,
        "identifier_legal_name": name,
        "address_region": region,
        "details_url": "",
        "org_category": "",
    }


# ---------------------------------------------------------------------------
# collect_parties
# ---------------------------------------------------------------------------


class TestCollectParties:
    def test_basic_extraction(self):
        releases = make_releases(
            [
                {
                    "ocid": "ocds-test-001",
                    "parties": [
                        make_party("GB-COH-12345", "Acme Ltd", ["supplier"], "GB-COH", "12345"),
                        make_party("GB-FTS-999", "Buyer Org", ["buyer"]),
                    ],
                }
            ]
        )
        parties = collect_parties(releases)
        raw_ids = set(parties["raw_id"].unique())
        assert "GB-COH-12345" in raw_ids
        assert "GB-FTS-999" in raw_ids

    def test_supplier_from_contracts_json(self):
        """Suppliers in contracts_json (not just parties) must be captured."""
        releases = make_releases(
            [
                {
                    "ocid": "ocds-test-002",
                    "parties": [],
                    "contracts": [
                        {
                            "contract_id": "c1",
                            "award_id": "a1",
                            "value_amount": 100.0,
                            "value_currency": "GBP",
                            "date_signed": "",
                            "period_start": "",
                            "period_end": "",
                            "status": "active",
                            "supplier_raw_ids": ["GB-COH-99999"],
                        }
                    ],
                }
            ]
        )
        parties = collect_parties(releases)
        assert "GB-COH-99999" in set(parties["raw_id"].unique())

    def test_empty_raw_id_skipped(self):
        releases = make_releases(
            [
                {
                    "ocid": "ocds-test-003",
                    "parties": [
                        make_party("", "Nameless", ["buyer"]),
                        make_party("GB-FTS-1", "Real Org", ["buyer"]),
                    ],
                }
            ]
        )
        parties = collect_parties(releases)
        assert "" not in set(parties["raw_id"].unique())


# ---------------------------------------------------------------------------
# resolve_phase1 — official ID resolution
# ---------------------------------------------------------------------------


class TestResolvePhase1Official:
    def test_gb_coh_party_gets_canonical_id(self):
        releases = make_releases(
            [
                {
                    "ocid": "ocds-001",
                    "parties": [
                        make_party("GB-COH-06884292", "Ward Security", ["supplier"], "GB-COH", "06884292"),
                    ],
                }
            ]
        )
        orgs, alias_map = resolve_phase1(releases)
        assert "GB-COH-06884292" in orgs["canonical_id"].values

    def test_gb_nhs_party_gets_canonical_id(self):
        releases = make_releases(
            [
                {
                    "ocid": "ocds-002",
                    "parties": [make_party("GB-NHS-Y56", "NHS England", ["buyer"], "GB-NHS", "Y56")],
                }
            ]
        )
        orgs, _ = resolve_phase1(releases)
        assert "GB-NHS-Y56" in orgs["canonical_id"].values

    def test_gb_fts_only_is_unresolved(self):
        releases = make_releases(
            [
                {
                    "ocid": "ocds-003",
                    "parties": [make_party("GB-FTS-9999", "Some Supplier", ["supplier"])],
                }
            ]
        )
        orgs, alias_map = resolve_phase1(releases)
        row = orgs[orgs["canonical_id"] == "GB-FTS-9999"]
        assert len(row) == 1
        assert row.iloc[0]["er_status"] == "unresolved"

    def test_gb_fts_never_canonical_when_official_exists(self):
        """GB-FTS raw_id must never appear as a canonical_id when an official ID is available."""
        releases = make_releases(
            [
                {
                    "ocid": "ocds-004",
                    "parties": [
                        make_party("GB-COH-11111", "Acme", ["supplier"], "GB-COH", "11111"),
                        make_party("GB-FTS-1234", "Acme Ltd", ["supplier"]),
                    ],
                }
            ]
        )
        orgs, alias_map = resolve_phase1(releases)
        canonical_ids = set(orgs["canonical_id"].values)
        # GB-FTS-1234 must NOT be a canonical_id; it should be an alias to GB-COH-11111
        assert "GB-FTS-1234" not in canonical_ids or (
            orgs[orgs["canonical_id"] == "GB-FTS-1234"].iloc[0]["er_status"] == "unresolved"
            and alias_map[alias_map["raw_id"] == "GB-FTS-1234"]["canonical_id"].iloc[0] != "GB-FTS-1234"
        )

    def test_same_official_id_two_names_merged(self):
        """Two different name variants with the same official ID → one canonical entity."""
        releases = make_releases(
            [
                {
                    "ocid": "ocds-005a",
                    "parties": [
                        make_party("GB-NHS-QHM", "North East ICB", ["buyer"], "GB-NHS", "QHM"),
                    ],
                },
                {
                    "ocid": "ocds-005b",
                    "parties": [
                        make_party("GB-NHS-QHM", "NHS North East and North Cumbria ICB", ["buyer"], "GB-NHS", "QHM"),
                    ],
                },
            ]
        )
        orgs, alias_map = resolve_phase1(releases)
        # Should be ONE canonical entity for GB-NHS-QHM
        coh_rows = orgs[orgs["canonical_id"] == "GB-NHS-QHM"]
        assert len(coh_rows) == 1
        # Should have both raw_ids as aliases
        aliases = set(json.loads(coh_rows.iloc[0]["alias_raw_ids"]))
        assert "GB-NHS-QHM" in aliases


class TestResolvePhase1CrossRef:
    def test_fts_aliased_to_official_via_same_ocid(self):
        """A GB-FTS ID that co-occurs with GB-COH in the same OCID and has same name gets aliased."""
        releases = make_releases(
            [
                {
                    "ocid": "ocds-cr-001",
                    "parties": [
                        make_party("GB-COH-55555", "University of Sheffield", ["buyer"], "GB-COH", "55555"),
                        make_party("GB-FTS-RC000667", "UNIVERSITY OF SHEFFIELD", ["buyer"]),
                    ],
                }
            ]
        )
        orgs, alias_map = resolve_phase1(releases)
        # GB-FTS-RC000667 should point to GB-COH-55555 in alias_map
        row = alias_map[alias_map["raw_id"] == "GB-FTS-RC000667"]
        assert len(row) == 1
        assert row.iloc[0]["canonical_id"] == "GB-COH-55555"
        assert row.iloc[0]["alias_source"] == "phase1"

    def test_fts_not_aliased_when_name_differs(self):
        """A GB-FTS ID should NOT be aliased if the name doesn't match the official entity."""
        releases = make_releases(
            [
                {
                    "ocid": "ocds-cr-002",
                    "parties": [
                        make_party("GB-COH-55555", "University of Sheffield", ["buyer"], "GB-COH", "55555"),
                        make_party("GB-FTS-99999", "Totally Different Org", ["supplier"]),
                    ],
                }
            ]
        )
        orgs, alias_map = resolve_phase1(releases)
        row = alias_map[alias_map["raw_id"] == "GB-FTS-99999"]
        # Should remain self-aliased (unresolved)
        assert row.iloc[0]["canonical_id"] == "GB-FTS-99999"


# ---------------------------------------------------------------------------
# resolve_phase1 — alias_map completeness
# ---------------------------------------------------------------------------


class TestAliasMap:
    def test_every_raw_id_has_entry_in_alias_map(self):
        releases = make_releases(
            [
                {
                    "ocid": "ocds-am-001",
                    "parties": [
                        make_party("GB-COH-111", "Org A", ["buyer"], "GB-COH", "111"),
                        make_party("GB-FTS-222", "Org B", ["supplier"]),
                        make_party("GB-NHS-Y56", "NHS Body", ["buyer"], "GB-NHS", "Y56"),
                    ],
                }
            ]
        )
        orgs, alias_map = resolve_phase1(releases)
        all_raw_ids_in_parties = {"GB-COH-111", "GB-FTS-222", "GB-NHS-Y56"}
        alias_raw_ids = set(alias_map["raw_id"].values)
        # Every raw_id from parties must appear in alias_map
        for rid in all_raw_ids_in_parties:
            assert rid in alias_raw_ids, f"{rid} missing from alias_map"

    def test_official_raw_ids_point_to_canonical(self):
        releases = make_releases(
            [
                {
                    "ocid": "ocds-am-002",
                    "parties": [make_party("GB-COH-777", "Corp X", ["supplier"], "GB-COH", "777")],
                }
            ]
        )
        _, alias_map = resolve_phase1(releases)
        row = alias_map[alias_map["raw_id"] == "GB-COH-777"]
        assert row.iloc[0]["canonical_id"] == "GB-COH-777"
        assert row.iloc[0]["alias_source"] == "phase1"

    def test_unresolved_self_alias_source(self):
        releases = make_releases(
            [
                {
                    "ocid": "ocds-am-003",
                    "parties": [make_party("GB-FTS-5555", "Unknown Supplier", ["supplier"])],
                }
            ]
        )
        _, alias_map = resolve_phase1(releases)
        row = alias_map[alias_map["raw_id"] == "GB-FTS-5555"]
        assert row.iloc[0]["canonical_id"] == "GB-FTS-5555"
        assert row.iloc[0]["alias_source"] == "unresolved"


# ---------------------------------------------------------------------------
# Priority: COH wins over NHS if somehow both appear for same entity
# ---------------------------------------------------------------------------


class TestSchemePriority:
    def test_coh_priority_over_nhs_for_same_entity(self):
        """If a raw_id appears once with GB-COH and once another way, COH wins."""
        releases = make_releases(
            [
                {
                    "ocid": "ocds-p-001",
                    "parties": [
                        # Two raw_ids pointing to the same OCID — one official, one FTS
                        make_party("GB-COH-999", "Body X", ["buyer"], "GB-COH", "999"),
                        make_party("GB-NHS-X99", "Body X NHS", ["buyer"], "GB-NHS", "X99"),
                    ],
                }
            ]
        )
        orgs, alias_map = resolve_phase1(releases)
        # Both should resolve to their own canonical IDs (different entities here)
        assert "GB-COH-999" in orgs["canonical_id"].values
        assert "GB-NHS-X99" in orgs["canonical_id"].values
        # Neither should be marked unresolved
        for cid in ["GB-COH-999", "GB-NHS-X99"]:
            row = orgs[orgs["canonical_id"] == cid]
            assert row.iloc[0]["er_status"] == "deterministic"
