"""Tests for src/normalise.py — the shared name/ID normalisation layer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from normalise import (
    normalise_name,
    normalise_org_id,
    scheme_of,
    value_of,
    is_official,
    is_fts,
    canonical_id_from_raw,
    priority,
    OFFICIAL_SCHEMES,
)


# ---------------------------------------------------------------------------
# normalise_name
# ---------------------------------------------------------------------------

class TestNormaliseName:
    def test_uppercase(self):
        assert normalise_name("University of Sheffield") == "UNIVERSITY OF SHEFFIELD"

    def test_strips_whitespace(self):
        assert normalise_name("  NHS England  ") == "NHS ENGLAND"

    def test_ampersand_expansion(self):
        assert normalise_name("Department for Energy Security & Net Zero") == \
            "DEPARTMENT FOR ENERGY SECURITY AND NET ZERO"

    def test_ampersand_no_spaces(self):
        assert normalise_name("Smith&Jones") == "SMITH AND JONES"

    def test_strips_limited(self):
        assert normalise_name("Ingeus UK Limited") == "INGEUS UK"

    def test_strips_ltd(self):
        assert normalise_name("Ward Security Limited") == "WARD SECURITY"

    def test_strips_plc(self):
        assert normalise_name("BT PLC") == "BT"

    def test_strips_llp(self):
        assert normalise_name("Deloitte LLP") == "DELOITTE"

    def test_strips_cic(self):
        assert normalise_name("TKB Housing CIC") == "TKB HOUSING"

    def test_strips_inc(self):
        assert normalise_name("10X Genomics INC") == "10X GENOMICS"

    def test_collapses_internal_spaces(self):
        assert "  " not in normalise_name("NHS  England")

    def test_removes_punctuation(self):
        result = normalise_name("Dept. for Energy")
        assert "." not in result

    def test_none_returns_empty(self):
        assert normalise_name(None) == ""

    def test_empty_string_returns_empty(self):
        assert normalise_name("") == ""

    def test_unicode_transliteration(self):
        # curly quotes and accented chars should not crash
        result = normalise_name("Café Ltd")
        assert isinstance(result, str)
        assert result != ""

    def test_mod_fragmentation_case(self):
        # All of these should normalise to the same string
        variants = [
            "Ministry of Defence",
            "MINISTRY OF DEFENCE",
            "Ministry Of Defence",
        ]
        norms = {normalise_name(v) for v in variants}
        assert len(norms) == 1

    def test_nhs_england_variants(self):
        variants = [
            "NHS England",
            "NHS ENGLAND",
            "nhs england",
        ]
        norms = {normalise_name(v) for v in variants}
        assert len(norms) == 1

    def test_does_not_strip_mid_word_ltd(self):
        # "MIDLTD" should not lose the "ltd" part — suffix strip is word-boundary only
        result = normalise_name("Midltd Corp")
        assert "MIDLTD" in result


# ---------------------------------------------------------------------------
# normalise_org_id
# ---------------------------------------------------------------------------

class TestNormaliseOrgId:
    def test_gbnhs_uppercased(self):
        assert normalise_org_id("GB-NHS", "y56") == "GB-NHS-Y56"

    def test_gbcoh_preserves_number(self):
        assert normalise_org_id("GB-COH", "06884292") == "GB-COH-06884292"

    def test_strips_whitespace_from_id(self):
        assert normalise_org_id("GB-COH", "  06884292  ") == "GB-COH-06884292"

    def test_scheme_uppercased(self):
        assert normalise_org_id("gb-coh", "123") == "GB-COH-123"


# ---------------------------------------------------------------------------
# scheme_of / value_of
# ---------------------------------------------------------------------------

class TestSchemeAndValue:
    def test_scheme_of_gbcoh(self):
        assert scheme_of("GB-COH-06884292") == "GB-COH"

    def test_scheme_of_gbnhs(self):
        assert scheme_of("GB-NHS-Y56") == "GB-NHS"

    def test_scheme_of_gbfts(self):
        assert scheme_of("GB-FTS-18165") == "GB-FTS"

    def test_scheme_of_short_returns_empty(self):
        assert scheme_of("GB") == ""

    def test_scheme_of_empty_returns_empty(self):
        assert scheme_of("") == ""

    def test_value_of_gbcoh(self):
        assert value_of("GB-COH-06884292") == "06884292"

    def test_value_of_gbnhs_with_dash_in_value(self):
        # e.g. GB-COH-HRB 304054 — value is "HRB 304054"
        assert value_of("GB-COH-HRB 304054") == "HRB 304054"


# ---------------------------------------------------------------------------
# is_official / is_fts
# ---------------------------------------------------------------------------

class TestIsOfficial:
    @pytest.mark.parametrize("scheme", list(OFFICIAL_SCHEMES))
    def test_official_schemes_are_official(self, scheme):
        assert is_official(scheme)

    def test_gbfts_is_not_official(self):
        assert not is_official("GB-FTS")

    def test_unknown_scheme_is_not_official(self):
        assert not is_official("XX-UNKNOWN")

    def test_gbfts_is_fts(self):
        assert is_fts("GB-FTS")

    def test_gbcoh_is_not_fts(self):
        assert not is_fts("GB-COH")

    def test_case_insensitive(self):
        assert is_official("gb-coh")
        assert is_fts("gb-fts")


# ---------------------------------------------------------------------------
# canonical_id_from_raw
# ---------------------------------------------------------------------------

class TestCanonicalIdFromRaw:
    def test_gbcoh_raw_id(self):
        assert canonical_id_from_raw("GB-COH-06884292") == "GB-COH-06884292"

    def test_gbnhs_raw_id_uppercased(self):
        result = canonical_id_from_raw("GB-NHS-y56")
        assert result == "GB-NHS-Y56"

    def test_gbfts_returns_none(self):
        # GB-FTS must NEVER become a canonical ID
        assert canonical_id_from_raw("GB-FTS-18165") is None

    def test_empty_returns_none(self):
        assert canonical_id_from_raw("") is None

    def test_unknown_scheme_returns_none(self):
        assert canonical_id_from_raw("XX-UNK-999") is None


# ---------------------------------------------------------------------------
# priority
# ---------------------------------------------------------------------------

class TestPriority:
    def test_gbcoh_highest(self):
        assert priority("GB-COH") == 1

    def test_gbfts_lowest_known(self):
        assert priority("GB-FTS") == 99

    def test_gbcoh_beats_gbnhs(self):
        assert priority("GB-COH") < priority("GB-NHS")

    def test_gbnhs_beats_gbukprn(self):
        assert priority("GB-NHS") < priority("GB-UKPRN")
