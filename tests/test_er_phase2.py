"""Tests for src/er_phase2.py and src/er_candidates.py."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest
from er_phase2 import (
    apply_gov_lookup,
    merge_by_name_region,
    consolidate_merged_groups,
    build_audit_log,
    _make_merged_id,
    _consolidate_gov_groups,
    consolidate_duplicate_canonical_orgs,
    resolve_phase2,
)
from er_candidates import generate_candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_unresolved(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal canonical_orgs-like DataFrame of unresolved entities."""
    defaults = {
        "org_type": "buyer",
        "address_region": "",
        "org_category": "",
        "official_scheme": "",
        "official_id": "",
        "alias_raw_ids": "[]",
        "alias_names": "[]",
        "n_aliases": 1,
        "er_status": "unresolved",
    }
    out = []
    for r in rows:
        row = {**defaults, **r}
        if "alias_raw_ids" not in r:
            row["alias_raw_ids"] = json.dumps([row["canonical_id"]])
        if "alias_names" not in r:
            row["alias_names"] = json.dumps([row.get("canonical_name", "")])
        out.append(row)
    return pd.DataFrame(out)


SAMPLE_GOV_LOOKUP = {
    "MINISTRY OF DEFENCE": {"canonical_id": "GOV-MOD", "_source": "test"},
    "NHS ENGLAND": {"canonical_id": "GOV-NHSE", "_source": "test"},
    "SCOTTISH GOVERNMENT": {"canonical_id": "GOV-SCOTGOV", "_source": "test"},
}


# ---------------------------------------------------------------------------
# apply_gov_lookup
# ---------------------------------------------------------------------------


class TestApplyGovLookup:
    def test_mod_matched(self):
        df = make_unresolved([
            {"canonical_id": "GB-FTS-1", "canonical_name": "Ministry of Defence"},
            {"canonical_id": "GB-FTS-2", "canonical_name": "Some Unrelated Org"},
        ])
        matched, remaining = apply_gov_lookup(df, SAMPLE_GOV_LOOKUP)
        assert len(matched) == 1
        assert matched.iloc[0]["canonical_id"] == "GOV-MOD"
        assert matched.iloc[0]["er_status"] == "gov_lookup"
        assert len(remaining) == 1
        assert remaining.iloc[0]["canonical_id"] == "GB-FTS-2"

    def test_case_insensitive_match(self):
        df = make_unresolved([
            {"canonical_id": "GB-FTS-3", "canonical_name": "ministry of defence"},
        ])
        matched, _ = apply_gov_lookup(df, SAMPLE_GOV_LOOKUP)
        assert len(matched) == 1
        assert matched.iloc[0]["canonical_id"] == "GOV-MOD"

    def test_nhs_england_matched(self):
        df = make_unresolved([
            {"canonical_id": "GB-FTS-10", "canonical_name": "NHS England"},
        ])
        matched, _ = apply_gov_lookup(df, SAMPLE_GOV_LOOKUP)
        assert matched.iloc[0]["canonical_id"] == "GOV-NHSE"

    def test_unmatched_stays_in_remaining(self):
        df = make_unresolved([
            {"canonical_id": "GB-FTS-99", "canonical_name": "Totally Unknown Corp"},
        ])
        matched, remaining = apply_gov_lookup(df, SAMPLE_GOV_LOOKUP)
        assert len(matched) == 0
        assert len(remaining) == 1

    def test_empty_input(self):
        df = make_unresolved([])
        matched, remaining = apply_gov_lookup(df, SAMPLE_GOV_LOOKUP)
        assert len(matched) == 0
        assert len(remaining) == 0

    def test_all_matched(self):
        df = make_unresolved([
            {"canonical_id": "GB-FTS-1", "canonical_name": "Ministry of Defence"},
            {"canonical_id": "GB-FTS-2", "canonical_name": "NHS England"},
        ])
        matched, remaining = apply_gov_lookup(df, SAMPLE_GOV_LOOKUP)
        assert len(matched) == 2
        assert len(remaining) == 0

    def test_gov_lookup_does_not_auto_merge_ambiguous(self):
        """An entity NOT in the lookup must NOT get a GOV-* id."""
        df = make_unresolved([
            {"canonical_id": "GB-FTS-77", "canonical_name": "Ministry of Something Else"},
        ])
        matched, remaining = apply_gov_lookup(df, SAMPLE_GOV_LOOKUP)
        assert len(matched) == 0  # "Ministry of Something Else" ≠ "MINISTRY OF DEFENCE"
        assert len(remaining) == 1


class TestConsolidateGovGroups:
    def test_multiple_raw_entities_collapsed_to_one_canonical_row(self):
        """Bug regression: multiple unresolved entities sharing the same GOV-* id after
        gov_lookup must produce exactly ONE canonical row in canonical_orgs, not one per
        raw entity. This was the production bug: 2,739 duplicate canonical_id rows."""
        gov_matched = pd.DataFrame([
            {"canonical_id": "GOV-MOD", "canonical_name": "Ministry of Defence",
             "er_status": "gov_lookup", "org_type": "buyer", "address_region": "UK",
             "org_category": "MINISTRY", "official_scheme": "", "official_id": "",
             "alias_raw_ids": '["GB-FTS-1"]', "alias_names": '["Ministry of Defence"]',
             "n_aliases": 1, "_gov_source": "test source"},
            {"canonical_id": "GOV-MOD", "canonical_name": "Ministry of Defence",
             "er_status": "gov_lookup", "org_type": "unknown", "address_region": "",
             "org_category": "", "official_scheme": "", "official_id": "",
             "alias_raw_ids": '["GB-FTS-2"]', "alias_names": '["Ministry of Defence"]',
             "n_aliases": 1, "_gov_source": "test source"},
            {"canonical_id": "GOV-MOD", "canonical_name": "Ministry of Defence",
             "er_status": "gov_lookup", "org_type": "buyer", "address_region": "UK",
             "org_category": "MINISTRY", "official_scheme": "", "official_id": "",
             "alias_raw_ids": '["GB-FTS-3"]', "alias_names": '["MoD"]',
             "n_aliases": 1, "_gov_source": "test source"},
        ])
        result = _consolidate_gov_groups(gov_matched)
        # Must produce exactly ONE row for GOV-MOD
        assert len(result) == 1
        assert result.iloc[0]["canonical_id"] == "GOV-MOD"
        # All three raw IDs must be in alias_raw_ids
        raw_ids = json.loads(result.iloc[0]["alias_raw_ids"])
        assert set(raw_ids) == {"GB-FTS-1", "GB-FTS-2", "GB-FTS-3"}
        assert result.iloc[0]["n_aliases"] == 3

    def test_resolve_phase2_no_duplicate_canonical_ids(self, tmp_path):
        """End-to-end regression: resolve_phase2 must never produce duplicate canonical_id rows."""
        # Build gov_lookup.json with one entry that will match multiple raw entities
        import json as _json
        gov_lookup = {
            "MINISTRY OF DEFENCE": {"canonical_id": "GOV-MOD", "_source": "test"},
        }
        gov_path = tmp_path / "gov_lookup.json"
        gov_path.write_text(_json.dumps(gov_lookup), encoding="utf-8")

        # Five distinct unresolved entities all named "Ministry of Defence"
        phase1_orgs = pd.DataFrame([
            {"canonical_id": f"GB-FTS-{i}", "canonical_name": "Ministry of Defence",
             "er_status": "unresolved", "org_type": "buyer", "address_region": "UK",
             "org_category": "MINISTRY", "official_scheme": "", "official_id": "",
             "alias_raw_ids": f'["GB-FTS-{i}"]', "alias_names": '["Ministry of Defence"]',
             "n_aliases": 1, "_gov_source": ""}
            for i in range(1, 6)
        ])
        phase1_alias = pd.DataFrame([
            {"raw_id": f"GB-FTS-{i}", "canonical_id": f"GB-FTS-{i}", "alias_source": "unresolved"}
            for i in range(1, 6)
        ])

        updated_orgs, updated_alias, audit = resolve_phase2(phase1_orgs, phase1_alias, gov_path)

        # canonical_orgs must have exactly one GOV-MOD row
        gov_rows = updated_orgs[updated_orgs["canonical_id"] == "GOV-MOD"]
        assert len(gov_rows) == 1, (
            f"Expected 1 GOV-MOD row, got {len(gov_rows)} — duplicate canonical_id bug"
        )
        assert updated_orgs["canonical_id"].duplicated().sum() == 0, \
            "canonical_orgs must have no duplicate canonical_id values"

    def test_existing_duplicate_gov_rows_are_repaired_on_rerun(self, tmp_path):
        """Phase 2 must be idempotent over older outputs that already contain
        one gov_lookup row per raw alias."""
        gov_path = tmp_path / "gov_lookup.json"
        gov_path.write_text("{}", encoding="utf-8")

        existing_orgs = pd.DataFrame([
            {"canonical_id": "GOV-MOD", "canonical_name": "Ministry of Defence",
             "er_status": "gov_lookup", "org_type": "buyer", "address_region": "UK",
             "org_category": "MINISTRY", "official_scheme": "", "official_id": "",
             "alias_raw_ids": '["GB-FTS-1"]', "alias_names": '["Ministry of Defence"]',
             "n_aliases": 1, "_gov_source": "test"},
            {"canonical_id": "GOV-MOD", "canonical_name": "Ministry of Defence",
             "er_status": "gov_lookup", "org_type": "buyer", "address_region": "UK",
             "org_category": "MINISTRY", "official_scheme": "", "official_id": "",
             "alias_raw_ids": '["GB-FTS-2"]', "alias_names": '["Ministry of Defence"]',
             "n_aliases": 1, "_gov_source": "test"},
            {"canonical_id": "GB-COH-111", "canonical_name": "Supplier Ltd",
             "er_status": "deterministic", "org_type": "supplier", "address_region": "",
             "org_category": "", "official_scheme": "GB-COH", "official_id": "111",
             "alias_raw_ids": '["GB-COH-111"]', "alias_names": '["Supplier Ltd"]',
             "n_aliases": 1, "_gov_source": ""},
        ])
        alias_map = pd.DataFrame([
            {"raw_id": "GB-FTS-1", "canonical_id": "GOV-MOD", "alias_source": "phase2"},
            {"raw_id": "GB-FTS-2", "canonical_id": "GOV-MOD", "alias_source": "phase2"},
            {"raw_id": "GB-COH-111", "canonical_id": "GB-COH-111", "alias_source": "phase1"},
        ])

        updated_orgs, updated_alias, audit = resolve_phase2(existing_orgs, alias_map, gov_path)

        assert updated_orgs["canonical_id"].duplicated().sum() == 0
        gov_row = updated_orgs[updated_orgs["canonical_id"] == "GOV-MOD"].iloc[0]
        assert set(json.loads(gov_row["alias_raw_ids"])) == {"GB-FTS-1", "GB-FTS-2"}
        assert gov_row["n_aliases"] == 2


# ---------------------------------------------------------------------------
# merge_by_name_region
# ---------------------------------------------------------------------------


class TestMergeByNameRegion:
    def test_same_name_same_region_merged(self):
        df = make_unresolved([
            {"canonical_id": "GB-FTS-A", "canonical_name": "Staffordshire County Council", "address_region": "UKG23"},
            {"canonical_id": "GB-FTS-B", "canonical_name": "Staffordshire County Council", "address_region": "UKG23"},
            {"canonical_id": "GB-FTS-C", "canonical_name": "Staffordshire County Council", "address_region": "UKG23"},
        ])
        merged, singletons = merge_by_name_region(df)
        assert len(merged) == 3
        assert len(singletons) == 0
        # All should have the same new canonical_id
        new_ids = merged["canonical_id"].unique()
        assert len(new_ids) == 1
        assert new_ids[0].startswith("MERGED-")

    def test_same_name_different_region_not_merged(self):
        df = make_unresolved([
            {"canonical_id": "GB-FTS-A", "canonical_name": "Southampton City Council", "address_region": "UKJ32"},
            {"canonical_id": "GB-FTS-B", "canonical_name": "Southampton City Council", "address_region": "UKJ33"},
        ])
        merged, singletons = merge_by_name_region(df)
        # Different regions → each is its own group of size 1 → singletons
        assert len(singletons) == 2
        assert len(merged) == 0

    def test_same_name_no_region_still_merged(self):
        """When both entities lack a region, they should still merge by name alone."""
        df = make_unresolved([
            {"canonical_id": "GB-FTS-A", "canonical_name": "National Highways", "address_region": ""},
            {"canonical_id": "GB-FTS-B", "canonical_name": "National Highways", "address_region": ""},
        ])
        merged, singletons = merge_by_name_region(df)
        assert len(merged) == 2
        assert len(singletons) == 0
        assert merged["canonical_id"].unique()[0].startswith("MERGED-")

    def test_unique_name_is_singleton(self):
        df = make_unresolved([
            {"canonical_id": "GB-FTS-X", "canonical_name": "Unique Supplier Ltd"},
        ])
        merged, singletons = merge_by_name_region(df)
        assert len(merged) == 0
        assert len(singletons) == 1

    def test_merged_id_is_deterministic(self):
        """Same name+region should always produce the same MERGED-* id."""
        id1 = _make_merged_id("STAFFORDSHIRE COUNTY COUNCIL", "UKG23")
        id2 = _make_merged_id("STAFFORDSHIRE COUNTY COUNCIL", "UKG23")
        assert id1 == id2

    def test_merged_id_differs_for_different_groups(self):
        id1 = _make_merged_id("STAFFORDSHIRE COUNTY COUNCIL", "UKG23")
        id2 = _make_merged_id("SOUTHAMPTON CITY COUNCIL", "UKJ32")
        assert id1 != id2

    def test_empty_input(self):
        df = make_unresolved([])
        merged, singletons = merge_by_name_region(df)
        assert len(merged) == 0
        assert len(singletons) == 0


# ---------------------------------------------------------------------------
# consolidate_merged_groups
# ---------------------------------------------------------------------------


class TestConsolidateMergedGroups:
    def test_groups_consolidated_to_one_row(self):
        merged = pd.DataFrame([
            {
                "canonical_id": "MERGED-abc123",
                "canonical_name": "Southampton City Council",
                "org_type": "buyer",
                "address_region": "UKJ32",
                "org_category": "BODY_PUBLIC",
                "official_scheme": "",
                "official_id": "",
                "alias_raw_ids": json.dumps(["GB-FTS-A"]),
                "alias_names": json.dumps(["Southampton City Council"]),
                "n_aliases": 1,
                "er_status": "name_region_merge",
            },
            {
                "canonical_id": "MERGED-abc123",
                "canonical_name": "Southampton City Council",
                "org_type": "buyer",
                "address_region": "UKJ32",
                "org_category": "BODY_PUBLIC",
                "official_scheme": "",
                "official_id": "",
                "alias_raw_ids": json.dumps(["GB-FTS-B"]),
                "alias_names": json.dumps(["Southampton City Council"]),
                "n_aliases": 1,
                "er_status": "name_region_merge",
            },
        ])
        consolidated = consolidate_merged_groups(merged)
        assert len(consolidated) == 1
        row = consolidated.iloc[0]
        assert row["canonical_id"] == "MERGED-abc123"
        raw_ids = json.loads(row["alias_raw_ids"])
        assert "GB-FTS-A" in raw_ids
        assert "GB-FTS-B" in raw_ids
        assert row["n_aliases"] == 2


# ---------------------------------------------------------------------------
# build_audit_log
# ---------------------------------------------------------------------------


class TestBuildAuditLog:
    def test_audit_has_one_row_per_entity(self):
        orgs = pd.DataFrame([
            {"canonical_id": "GB-COH-111", "canonical_name": "Corp A", "er_status": "deterministic",
             "official_scheme": "GB-COH", "n_aliases": 1, "alias_raw_ids": '["GB-COH-111"]'},
            {"canonical_id": "GOV-MOD", "canonical_name": "Ministry of Defence", "er_status": "gov_lookup",
             "official_scheme": "", "n_aliases": 5, "alias_raw_ids": '[]'},
        ])
        alias_map = pd.DataFrame([
            {"raw_id": "GB-COH-111", "canonical_id": "GB-COH-111", "alias_source": "phase1"},
            {"raw_id": "GB-FTS-1", "canonical_id": "GOV-MOD", "alias_source": "phase2"},
        ])
        log = build_audit_log(orgs, alias_map)
        assert len(log) == 2
        assert set(log["canonical_id"].values) == {"GB-COH-111", "GOV-MOD"}

    def test_audit_shows_correct_alias_count(self):
        orgs = pd.DataFrame([
            {"canonical_id": "GOV-MOD", "canonical_name": "Ministry of Defence",
             "er_status": "gov_lookup", "official_scheme": "", "n_aliases": 3,
             "alias_raw_ids": '[]'},
        ])
        alias_map = pd.DataFrame([
            {"raw_id": "GB-FTS-1", "canonical_id": "GOV-MOD", "alias_source": "phase2"},
            {"raw_id": "GB-FTS-2", "canonical_id": "GOV-MOD", "alias_source": "phase2"},
            {"raw_id": "GB-FTS-3", "canonical_id": "GOV-MOD", "alias_source": "phase2"},
        ])
        log = build_audit_log(orgs, alias_map)
        assert log.iloc[0]["n_aliases"] == 3


# ---------------------------------------------------------------------------
# er_candidates — no auto-merges
# ---------------------------------------------------------------------------


class TestERCandidates:
    def test_similar_names_produce_candidates(self):
        orgs = pd.DataFrame([
            {"canonical_id": "GB-FTS-A", "canonical_name": "Staffordshire County Council",
             "er_status": "singleton", "address_region": "UKG23"},
            {"canonical_id": "GB-FTS-B", "canonical_name": "Staffordshire County Councill",  # typo
             "er_status": "singleton", "address_region": "UKG23"},
        ])
        candidates = generate_candidates(orgs, threshold=0.90)
        assert len(candidates) > 0
        assert candidates.iloc[0]["similarity"] > 0.90

    def test_candidates_never_modify_input(self):
        """generate_candidates must not mutate the input DataFrame."""
        orgs = pd.DataFrame([
            {"canonical_id": "GB-FTS-A", "canonical_name": "Org Alpha",
             "er_status": "singleton", "address_region": ""},
            {"canonical_id": "GB-FTS-B", "canonical_name": "Org Alph",
             "er_status": "singleton", "address_region": ""},
        ])
        original_ids = list(orgs["canonical_id"].values)
        _ = generate_candidates(orgs, threshold=0.85)
        assert list(orgs["canonical_id"].values) == original_ids

    def test_dissimilar_names_below_threshold(self):
        orgs = pd.DataFrame([
            {"canonical_id": "GB-FTS-A", "canonical_name": "Apple Computers Ltd",
             "er_status": "singleton", "address_region": ""},
            {"canonical_id": "GB-FTS-B", "canonical_name": "Zeppelin Manufacturing",
             "er_status": "singleton", "address_region": ""},
        ])
        candidates = generate_candidates(orgs, threshold=0.92)
        # These names are very different; should produce zero candidates
        assert len(candidates) == 0

    def test_deterministic_entities_excluded_by_default(self):
        """generate_candidates should only look at unresolved/singleton entities."""
        orgs = pd.DataFrame([
            {"canonical_id": "GB-COH-111", "canonical_name": "Acme Ltd",
             "er_status": "deterministic", "address_region": ""},
            {"canonical_id": "GB-COH-112", "canonical_name": "Acme Ltd",
             "er_status": "deterministic", "address_region": ""},
        ])
        candidates = generate_candidates(orgs, threshold=0.92)
        assert len(candidates) == 0

    def test_empty_orgs(self):
        orgs = pd.DataFrame(columns=["canonical_id", "canonical_name", "er_status", "address_region"])
        candidates = generate_candidates(orgs)
        assert len(candidates) == 0

    def test_output_has_block_key_column(self):
        """Result must include block_key so the blocking strategy is auditable."""
        orgs = pd.DataFrame([
            {"canonical_id": "GB-FTS-A", "canonical_name": "Staffordshire County Council",
             "er_status": "singleton", "address_region": ""},
            {"canonical_id": "GB-FTS-B", "canonical_name": "Staffordshire County Councils",
             "er_status": "singleton", "address_region": ""},
        ])
        candidates = generate_candidates(orgs, threshold=0.90)
        assert len(candidates) > 0
        assert "block_key" in candidates.columns

    def test_max_pairs_total_is_a_hard_global_cap(self):
        """max_pairs_total must cap total output, not just per-block."""
        # Build many similar pairs across different prefix blocks so the cap fires globally.
        rows = []
        for i in range(10):
            prefix = f"Z{i:02d}"  # distinct prefix per group → distinct blocks
            rows.append({"canonical_id": f"ID-{i}-A", "canonical_name": f"{prefix} Council Foo",
                         "er_status": "singleton", "address_region": ""})
            rows.append({"canonical_id": f"ID-{i}-B", "canonical_name": f"{prefix} Council Fob",
                         "er_status": "singleton", "address_region": ""})
            rows.append({"canonical_id": f"ID-{i}-C", "canonical_name": f"{prefix} Council Foc",
                         "er_status": "singleton", "address_region": ""})
        orgs = pd.DataFrame(rows)
        candidates = generate_candidates(orgs, threshold=0.85, max_pairs_total=3)
        assert len(candidates) <= 3

    def test_block_too_large_is_skipped(self):
        """Blocks exceeding max_block_size must produce zero pairs (skipped entirely)."""
        rows = [
            {"canonical_id": f"ID-{i}", "canonical_name": f"National Health Service Unit {i}",
             "er_status": "singleton", "address_region": ""}
            for i in range(30)  # all share "NAT" prefix
        ]
        orgs = pd.DataFrame(rows)
        # max_block_size=5 means a block of 30 is skipped
        candidates = generate_candidates(orgs, threshold=0.80, max_block_size=5)
        # The "NAT" block should be skipped; no other blocks exist → zero pairs
        assert len(candidates) == 0

    def test_max_pairs_per_block_limits_single_block_output(self):
        """max_pairs_per_block must cap pairs from a single block."""
        # Build a block of 10 very similar names (all "STA" prefix) that would
        # produce 10*9/2 = 45 pairs above threshold without the per-block cap.
        rows = [
            {"canonical_id": f"ID-{i}", "canonical_name": f"Staffordshire Council {i:03d}",
             "er_status": "singleton", "address_region": ""}
            for i in range(10)
        ]
        orgs = pd.DataFrame(rows)
        candidates = generate_candidates(
            orgs, threshold=0.80, max_block_size=20, max_pairs_per_block=5, max_pairs_total=1000
        )
        assert len(candidates) <= 5
