"""Tests for src/extract.py — rich OCDS field extraction."""
import gzip
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from extract import extract_release, extract_all


# ---------------------------------------------------------------------------
# Test fixture: a realistic minimal OCDS release
# ---------------------------------------------------------------------------

SAMPLE_RELEASE = {
    "ocid": "ocds-h6vhtk-test01",
    "id": "ocds-h6vhtk-test01-2024-01-15T10:00:00Z",
    "date": "2024-01-15T10:00:00Z",
    "tag": ["compiled"],
    "initiationType": "tender",
    "language": "en",
    "buyer": {"id": "GB-COH-12345678", "name": "Test Authority"},
    "parties": [
        {
            "id": "GB-COH-12345678",
            "name": "Test Authority",
            "roles": ["buyer"],
            "identifier": {"scheme": "GB-COH", "id": "12345678", "legalName": "Test Authority Ltd"},
            "address": {
                "region": "UKJ32",
                "postalCode": "SO14 7FJ",
                "locality": "Southampton",
                "streetAddress": "1 Test Street",
                "countryName": "United Kingdom",
            },
            "details": {
                "url": "https://example.com/profile",
                "buyerProfile": "https://example.com/buyer",
                "classifications": [
                    {"scheme": "UK_CA_TYPE", "id": "publicAuthoritySubCentralGovernment",
                     "description": "Public authority - sub-central government"},
                ],
            },
            "contactPoint": {"email": "procurement@example.com"},
        },
        {
            "id": "GB-COH-87654321",
            "name": "Supplier Co Ltd",
            "roles": ["supplier"],
            "identifier": {"scheme": "GB-COH", "id": "87654321", "legalName": "Supplier Co Ltd"},
            "address": {"region": "UKJ12"},
            "details": {},
        },
    ],
    "tender": {
        "id": "TENDER-001",
        "title": "Supply of Widgets",
        "description": "This tender covers the supply of high-quality widgets for the authority's operational needs over a 3-year period.",
        "status": "complete",
        "value": {"amount": 500000.0, "currency": "GBP"},
        "procurementMethod": "open",
        "procurementMethodDetails": "Open procedure",
        "mainProcurementCategory": "goods",
        "classification": {"id": "44000000", "scheme": "CPV", "description": "Construction structures"},
        "legalBasis": {"id": "2023/54", "scheme": "UKPGA"},
        "coveredBy": ["GPA"],
        "submissionMethod": ["electronicSubmission"],
        "submissionTerms": {"variantPolicy": "notAllowed", "languages": ["en"]},
        "hasRecurrence": False,
        "tenderPeriod": {"endDate": "2024-03-01T12:00:00Z", "startDate": "2024-01-15T10:00:00Z"},
        "awardPeriod": {"startDate": "2024-03-15T00:00:00Z"},
        "bidOpening": {"date": "2024-03-01T12:01:00Z"},
        "enquiryPeriod": {"endDate": "2024-02-15T12:00:00Z"},
        "communication": {"futureNoticeDate": "2024-04-01T00:00:00Z"},
        "submissionMethodDetails": "https://example.com/submit",
        "techniques": {
            "hasFrameworkAgreement": False,
            "hasDynamicPurchasingSystem": False,
        },
        "secondStage": {"minimumCandidates": 3, "maximumCandidates": 5},
        "reviewDetails": "Review by the High Court.",
        "contractTerms": {"performanceTerms": "Deliveries must occur within 5 working days."},
        "selectionCriteria": {
            "criteria": [
                {"type": "suitability", "description": "ISO 9001 certification required."},
            ]
        },
        "lots": [
            {
                "id": "1",
                "title": "Lot 1 - Standard Widgets",
                "description": "Supply of standard widgets in sizes A, B and C for general use across all depots.",
                "status": "complete",
                "hasOptions": False,
                "hasRenewal": True,
                "value": {"amount": 300000.0, "currency": "GBP"},
                "contractPeriod": {
                    "startDate": "2024-04-01T00:00:00Z",
                    "endDate": "2027-03-31T23:59:59Z",
                    "durationInDays": 1095,
                },
                "submissionTerms": {"variantPolicy": "notAllowed"},
                "suitability": {"vcse": False},
                "awardCriteria": {
                    "criteria": [
                        {"type": "price", "name": "Price", "weight": 60},
                        {"type": "quality", "name": "Quality", "weight": 40,
                         "description": "Assessed via sample submission and technical evaluation."},
                    ]
                },
                "renewal": {"description": "Option to extend by 12 months, maximum 2 extensions."},
            },
            {
                "id": "2",
                "title": "Lot 2 - Premium Widgets",
                "description": "Premium grade widgets for specialist applications.",
                "status": "active",
                "hasOptions": True,
                "hasRenewal": False,
                "value": {"amount": 200000.0, "currency": "GBP"},
                "contractPeriod": {
                    "startDate": "2024-04-01T00:00:00Z",
                    "endDate": "2026-03-31T23:59:59Z",
                },
                "submissionTerms": {"variantPolicy": "allowed"},
                "awardCriteria": {
                    "criteria": [
                        {"type": "price", "name": "Price", "weight": 50},
                        {"type": "quality", "name": "Quality", "weight": 50},
                    ]
                },
                "options": {"description": "Option to purchase additional 20% volume at same rates."},
            },
        ],
        "items": [
            {
                "id": "1",
                "relatedLot": "1",
                "deliveryAddresses": [{"region": "UKJ32"}, {"region": "UKJ33"}],
                "additionalClassifications": [
                    {"id": "44100000", "scheme": "CPV", "description": "Construction materials"},
                ],
            },
            {
                "id": "2",
                "relatedLot": "2",
                "deliveryAddresses": [{"region": "UKK11"}],
                "additionalClassifications": [],
            },
        ],
        "documents": [
            {
                "id": "doc-001",
                "documentType": "tenderNotice",
                "url": "https://find-tender.service.gov.uk/Notice/001",
                "title": "Contract Notice",
                "format": "text/html",
                "datePublished": "2024-01-15T10:00:00Z",
            },
        ],
    },
    "awards": [
        {
            "id": "award-001",
            "status": "active",
            "title": "Widget Supply Award",
            "suppliers": [{"id": "GB-COH-87654321", "name": "Supplier Co Ltd"}],
            "relatedLots": ["1"],
            "value": {"amount": 280000.0, "currency": "GBP", "amountGross": 336000.0},
            "contractPeriod": {
                "startDate": "2024-04-01T00:00:00Z",
                "endDate": "2027-03-31T23:59:59Z",
            },
            "aboveThreshold": True,
            "documents": [
                {
                    "id": "award-doc-001",
                    "documentType": "awardNotice",
                    "url": "https://find-tender.service.gov.uk/Notice/award-001",
                    "noticeType": "UK6",
                    "format": "text/html",
                    "datePublished": "2024-03-20T00:00:00Z",
                    "description": "Contract award notice on Find a Tender",
                }
            ],
        }
    ],
    "contracts": [
        {
            "id": "contract-001",
            "awardID": "award-001",
            "status": "active",
            "value": {"amount": 280000.0, "currency": "GBP"},
            "dateSigned": "2024-04-01T00:00:00Z",
            "period": {"startDate": "2024-04-01T00:00:00Z", "endDate": "2027-03-31T23:59:59Z"},
            "title": "Widget Supply Contract",
        }
    ],
    "bids": {
        "statistics": [
            {"id": "1", "measure": "bids", "value": 7, "relatedLot": "1"},
            {"id": "2", "measure": "smeBids", "value": 4, "relatedLot": "1"},
            {"id": "3", "measure": "bids", "value": 3, "relatedLot": "2"},
        ]
    },
}


def get_result():
    return extract_release(SAMPLE_RELEASE)


# ---------------------------------------------------------------------------
# tender_core
# ---------------------------------------------------------------------------


class TestTenderCore:
    def test_produces_one_row(self):
        r = get_result()
        assert len(r["tender_core"]) == 1

    def test_ocid_correct(self):
        row = get_result()["tender_core"][0]
        assert row["ocid"] == "ocds-h6vhtk-test01"

    def test_tender_status(self):
        row = get_result()["tender_core"][0]
        assert row["tender_status"] == "complete"

    def test_legal_basis(self):
        row = get_result()["tender_core"][0]
        assert row["tender_legal_basis_id"] == "2023/54"
        assert row["tender_legal_basis_scheme"] == "UKPGA"

    def test_covered_by_is_json_list(self):
        row = get_result()["tender_core"][0]
        assert json.loads(row["tender_covered_by"]) == ["GPA"]

    def test_has_recurrence_false(self):
        row = get_result()["tender_core"][0]
        assert row["tender_has_recurrence"] is False

    def test_has_framework_false(self):
        row = get_result()["tender_core"][0]
        assert row["tender_has_framework"] is False

    def test_has_dps_false(self):
        row = get_result()["tender_core"][0]
        assert row["tender_has_dps"] is False

    def test_second_stage_candidates(self):
        row = get_result()["tender_core"][0]
        assert row["tender_second_stage_min"] == 3.0
        assert row["tender_second_stage_max"] == 5.0

    def test_submission_url(self):
        row = get_result()["tender_core"][0]
        assert row["tender_submission_url"] == "https://example.com/submit"

    def test_award_period_start(self):
        row = get_result()["tender_core"][0]
        assert row["tender_award_period_start"] == "2024-03-15T00:00:00Z"

    def test_bid_opening_date(self):
        row = get_result()["tender_core"][0]
        assert row["tender_bid_opening_date"] == "2024-03-01T12:01:00Z"

    def test_enquiry_period_end(self):
        row = get_result()["tender_core"][0]
        assert row["tender_enquiry_period_end"] == "2024-02-15T12:00:00Z"

    def test_future_notice_date(self):
        row = get_result()["tender_core"][0]
        assert row["tender_future_notice_date"] == "2024-04-01T00:00:00Z"

    def test_no_tender_missing_fields(self):
        """Release with no tender block returns empty strings, not errors."""
        result = extract_release({"ocid": "test", "date": "2024-01-01"})
        row = result["tender_core"][0]
        assert row["tender_status"] == ""
        assert row["tender_has_framework"] is None


# ---------------------------------------------------------------------------
# lots
# ---------------------------------------------------------------------------


class TestLots:
    def test_two_lots_extracted(self):
        r = get_result()
        assert len(r["lots"]) == 2

    def test_lot_ids(self):
        lots = {row["lot_id"]: row for row in get_result()["lots"]}
        assert "1" in lots
        assert "2" in lots

    def test_lot1_has_renewal(self):
        lots = {row["lot_id"]: row for row in get_result()["lots"]}
        assert lots["1"]["has_renewal"] is True
        assert lots["2"]["has_renewal"] is False

    def test_lot2_has_options(self):
        lots = {row["lot_id"]: row for row in get_result()["lots"]}
        assert lots["2"]["has_options"] is True
        assert lots["1"]["has_options"] is False

    def test_lot_value(self):
        lots = {row["lot_id"]: row for row in get_result()["lots"]}
        assert lots["1"]["lot_value_amount"] == 300000.0
        assert lots["1"]["lot_value_currency"] == "GBP"

    def test_contract_period(self):
        lots = {row["lot_id"]: row for row in get_result()["lots"]}
        assert lots["1"]["contract_start"] == "2024-04-01T00:00:00Z"
        assert lots["1"]["contract_end"] == "2027-03-31T23:59:59Z"
        assert lots["1"]["contract_duration_days"] == 1095.0

    def test_delivery_regions_from_items(self):
        lots = {row["lot_id"]: row for row in get_result()["lots"]}
        regions_1 = json.loads(lots["1"]["delivery_regions"])
        assert "UKJ32" in regions_1
        assert "UKJ33" in regions_1
        regions_2 = json.loads(lots["2"]["delivery_regions"])
        assert "UKK11" in regions_2

    def test_additional_cpvs_from_items(self):
        lots = {row["lot_id"]: row for row in get_result()["lots"]}
        cpvs = json.loads(lots["1"]["additional_cpv_ids"])
        assert "44100000" in cpvs
        # Lot 2 has no additional CPVs
        assert json.loads(lots["2"]["additional_cpv_ids"]) == []

    def test_vcse_flag(self):
        lots = {row["lot_id"]: row for row in get_result()["lots"]}
        assert lots["1"]["is_vcse"] is False

    def test_no_lots_returns_empty(self):
        release = {"ocid": "test", "tender": {"id": "T1", "title": "No lots"}}
        result = extract_release(release)
        assert result["lots"] == []


# ---------------------------------------------------------------------------
# award_criteria
# ---------------------------------------------------------------------------


class TestAwardCriteria:
    def test_four_criteria_total(self):
        # 2 criteria in lot 1, 2 criteria in lot 2
        r = get_result()
        assert len(r["award_criteria"]) == 4

    def test_criteria_types(self):
        ac = get_result()["award_criteria"]
        types = {(row["lot_id"], row["criterion_index"]): row["criterion_type"] for row in ac}
        assert types[("1", 0)] == "price"
        assert types[("1", 1)] == "quality"

    def test_criterion_weights(self):
        ac = {(row["lot_id"], row["criterion_index"]): row for row in get_result()["award_criteria"]}
        assert ac[("1", 0)]["criterion_weight"] == 60.0
        assert ac[("1", 1)]["criterion_weight"] == 40.0

    def test_criterion_names(self):
        ac = {(row["lot_id"], row["criterion_index"]): row for row in get_result()["award_criteria"]}
        assert ac[("2", 0)]["criterion_name"] == "Price"

    def test_no_criteria_returns_empty(self):
        release = {"ocid": "test", "tender": {"lots": [{"id": "1"}]}}
        assert extract_release(release)["award_criteria"] == []


# ---------------------------------------------------------------------------
# awards
# ---------------------------------------------------------------------------


class TestAwards:
    def test_one_award(self):
        assert len(get_result()["awards"]) == 1

    def test_award_fields(self):
        award = get_result()["awards"][0]
        assert award["ocid"] == "ocds-h6vhtk-test01"
        assert award["award_id"] == "award-001"
        assert award["award_status"] == "active"
        assert award["award_title"] == "Widget Supply Award"

    def test_award_value(self):
        award = get_result()["awards"][0]
        assert award["award_value_amount"] == 280000.0
        assert award["award_value_gross"] == 336000.0
        assert award["award_value_currency"] == "GBP"
        assert award["contract_value_amount"] == 280000.0
        assert award["contract_value_currency"] == "GBP"
        assert award["award_value_best_amount"] == 280000.0
        assert award["award_value_best_currency"] == "GBP"
        assert award["award_value_source"] == "award"

    def test_award_signed_date(self):
        award = get_result()["awards"][0]
        assert award["award_date_signed"] == "2024-04-01T00:00:00Z"

    def test_award_suppliers(self):
        award = get_result()["awards"][0]
        suppliers = json.loads(award["supplier_raw_ids"])
        assert suppliers == ["GB-COH-87654321"]

    def test_award_related_lots(self):
        award = get_result()["awards"][0]
        related = json.loads(award["related_lots"])
        assert related == ["1"]

    def test_above_threshold(self):
        award = get_result()["awards"][0]
        assert award["above_threshold"] is True

    def test_no_awards_returns_empty(self):
        assert extract_release({"ocid": "test"})["awards"] == []


# ---------------------------------------------------------------------------
# bid_stats
# ---------------------------------------------------------------------------


class TestBidStats:
    def test_three_stats(self):
        assert len(get_result()["bid_stats"]) == 3

    def test_stat_measures(self):
        stats = {(row["measure"], row["related_lot"]): row for row in get_result()["bid_stats"]}
        assert ("bids", "1") in stats
        assert ("smeBids", "1") in stats
        assert ("bids", "2") in stats

    def test_stat_values(self):
        stats = {(row["measure"], row["related_lot"]): row for row in get_result()["bid_stats"]}
        assert stats[("bids", "1")]["stat_value"] == 7.0
        assert stats[("smeBids", "1")]["stat_value"] == 4.0

    def test_no_bids_returns_empty(self):
        assert extract_release({"ocid": "test"})["bid_stats"] == []


# ---------------------------------------------------------------------------
# text_evidence
# ---------------------------------------------------------------------------


class TestTextEvidence:
    def test_tender_description_captured(self):
        ev = get_result()["text_evidence"]
        paths = [r["field_path"] for r in ev]
        assert "tender.description" in paths

    def test_review_details_captured(self):
        ev = get_result()["text_evidence"]
        paths = [r["field_path"] for r in ev]
        assert "tender.reviewDetails" in paths

    def test_performance_terms_captured(self):
        ev = get_result()["text_evidence"]
        paths = [r["field_path"] for r in ev]
        assert "tender.contractTerms.performanceTerms" in paths

    def test_selection_criteria_description_captured(self):
        ev = get_result()["text_evidence"]
        paths = [r["field_path"] for r in ev]
        assert any("selectionCriteria" in p for p in paths)

    def test_lot_descriptions_captured(self):
        ev = get_result()["text_evidence"]
        lot_desc = [r for r in ev if r["field_path"] == "tender.lots[].description"]
        assert len(lot_desc) == 2  # two lots
        lot_ids = {r["lot_id"] for r in lot_desc}
        assert lot_ids == {"1", "2"}

    def test_renewal_description_captured(self):
        ev = get_result()["text_evidence"]
        renewal = [r for r in ev if r["field_path"] == "tender.lots[].renewal.description"]
        assert len(renewal) == 1
        assert renewal[0]["lot_id"] == "1"

    def test_options_description_captured(self):
        ev = get_result()["text_evidence"]
        options = [r for r in ev if r["field_path"] == "tender.lots[].options.description"]
        assert len(options) == 1
        assert options[0]["lot_id"] == "2"

    def test_award_criteria_text_captured_when_not_just_number(self):
        ev = get_result()["text_evidence"]
        ac_texts = [r for r in ev if "awardCriteria" in r["field_path"]]
        # The quality criterion in lot 1 has a real description
        assert any("Assessed via sample submission" in r["text"] for r in ac_texts)

    def test_empty_text_not_captured(self):
        release = {"ocid": "test", "tender": {"description": ""}}
        ev = extract_release(release)["text_evidence"]
        assert not any(r["field_path"] == "tender.description" for r in ev)

    def test_numeric_only_award_criteria_description_excluded(self):
        """A description that is just a number (e.g. '75') should not be stored."""
        release = {
            "ocid": "test",
            "tender": {
                "lots": [{
                    "id": "1",
                    "awardCriteria": {
                        "criteria": [{"type": "price", "description": "75"}]
                    }
                }]
            }
        }
        ev = extract_release(release)["text_evidence"]
        ac_texts = [r for r in ev if "awardCriteria" in r["field_path"]]
        assert len(ac_texts) == 0

    def test_no_text_fields_no_rows(self):
        release = {"ocid": "test", "tender": {"id": "T1"}}
        ev = extract_release(release)["text_evidence"]
        assert ev == []


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------


class TestDocuments:
    def _docs_by_type(self):
        docs = get_result()["documents"]
        result = {}
        for d in docs:
            result.setdefault(d["document_type"], []).append(d)
        return result

    def test_tender_notice_captured(self):
        by_type = self._docs_by_type()
        assert "tenderNotice" in by_type
        assert by_type["tenderNotice"][0]["url"] == "https://find-tender.service.gov.uk/Notice/001"

    def test_award_notice_captured(self):
        by_type = self._docs_by_type()
        assert "awardNotice" in by_type
        award_doc = by_type["awardNotice"][0]
        assert award_doc["notice_type"] == "UK6"
        assert "find-tender" in award_doc["url"]

    def test_submission_url_captured(self):
        by_type = self._docs_by_type()
        assert "submissionUrl" in by_type
        assert by_type["submissionUrl"][0]["url"] == "https://example.com/submit"

    def test_party_profile_url_captured(self):
        by_type = self._docs_by_type()
        assert "profileUrl" in by_type
        urls = [d["url"] for d in by_type["profileUrl"]]
        assert "https://example.com/profile" in urls

    def test_buyer_profile_captured(self):
        by_type = self._docs_by_type()
        assert "buyerProfile" in by_type
        assert by_type["buyerProfile"][0]["url"] == "https://example.com/buyer"

    def test_contact_email_captured(self):
        by_type = self._docs_by_type()
        assert "contactEmail" in by_type
        emails = [d["description"] for d in by_type["contactEmail"]]
        assert "procurement@example.com" in emails

    def test_source_field_correct(self):
        docs = {d["document_type"]: d["source"] for d in get_result()["documents"]}
        assert docs["tenderNotice"] == "tender"
        assert docs["awardNotice"] == "award"
        assert docs["submissionUrl"] == "tender_submission"

    def test_no_docs_returns_empty_for_no_doc_records(self):
        release = {"ocid": "test", "tender": {}}
        docs = extract_release(release)["documents"]
        assert docs == []

    def test_party_without_url_not_captured(self):
        """A party with no URL fields should not produce a profileUrl doc row."""
        release = {
            "ocid": "test",
            "parties": [{"id": "GB-FTS-1", "name": "No URL Org", "details": {}}],
            "tender": {},
        }
        docs = extract_release(release)["documents"]
        profile_docs = [d for d in docs if d["document_type"] == "profileUrl"]
        assert len(profile_docs) == 0

    def test_doc_id_and_format_captured(self):
        docs = get_result()["documents"]
        tender_notice = next(d for d in docs if d["document_type"] == "tenderNotice")
        assert tender_notice["doc_id"] == "doc-001"
        assert tender_notice["format"] == "text/html"
        assert tender_notice["date_published"] == "2024-01-15T10:00:00Z"


# ---------------------------------------------------------------------------
# Cross-table consistency
# ---------------------------------------------------------------------------


class TestCrossTableConsistency:
    def test_all_tables_share_same_ocid(self):
        r = get_result()
        ocid = "ocds-h6vhtk-test01"
        for table_name, rows in r.items():
            for row in rows:
                assert row["ocid"] == ocid, \
                    f"Table {table_name} has wrong ocid: {row['ocid']}"

    def test_lot_ids_in_award_criteria_match_lots(self):
        r = get_result()
        lot_ids = {row["lot_id"] for row in r["lots"]}
        ac_lot_ids = {row["lot_id"] for row in r["award_criteria"]}
        assert ac_lot_ids.issubset(lot_ids)

    def test_lot_ids_in_text_evidence_match_lots(self):
        r = get_result()
        lot_ids = {row["lot_id"] for row in r["lots"]}
        ev_lot_ids = {row["lot_id"] for row in r["text_evidence"] if row["lot_id"]}
        assert ev_lot_ids.issubset(lot_ids)

    def test_empty_release_produces_no_errors(self):
        """An almost-empty release dict must not raise exceptions."""
        result = extract_release({"ocid": "empty-test", "date": "2024-01-01"})
        assert isinstance(result, dict)
        assert len(result["tender_core"]) == 1
        assert result["lots"] == []
        assert result["awards"] == []


# ---------------------------------------------------------------------------
# Snapshot alignment: extract_all must match on release_id, not file order
# ---------------------------------------------------------------------------


def _write_gz(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestSnapshotAlignment:
    """Verify that extract_all uses release_id matching, not file-order heuristics."""

    def test_correct_release_extracted_not_older_one(self, tmp_path):
        """When two releases share an OCID, only the one matching release_id is extracted."""
        older = {
            "ocid": "ocds-test-001",
            "id": "ocds-test-001-2023-01-01",
            "date": "2023-01-01T00:00:00Z",
            "tender": {"title": "OLD TITLE", "status": "planning"},
        }
        newer = {
            "ocid": "ocds-test-001",
            "id": "ocds-test-001-2024-06-01",
            "date": "2024-06-01T00:00:00Z",
            "tender": {"title": "NEW TITLE", "status": "complete"},
        }
        # Both releases are in the same file
        _write_gz(tmp_path / "2024.jsonl.gz", [older, newer])

        # releases.parquet selected the newer release
        ocid_to_release_id = {"ocds-test-001": "ocds-test-001-2024-06-01"}
        tables = extract_all(tmp_path, ocid_to_release_id, show_progress=False)

        assert len(tables["tender_core"]) == 1
        assert tables["tender_core"].iloc[0]["tender_status"] == "complete"

    def test_older_release_is_rejected(self, tmp_path):
        """The older release must NOT appear in outputs even if it comes first in the file."""
        older = {
            "ocid": "ocds-test-002",
            "id": "ocds-test-002-2022-01-01",
            "date": "2022-01-01T00:00:00Z",
            "tender": {"title": "STALE", "status": "cancelled"},
        }
        newer = {
            "ocid": "ocds-test-002",
            "id": "ocds-test-002-2025-03-01",
            "date": "2025-03-01T00:00:00Z",
            "tender": {"title": "CURRENT", "status": "active"},
        }
        # Older release appears first in the file — file-order heuristic would pick it
        _write_gz(tmp_path / "2025.jsonl.gz", [older, newer])

        ocid_to_release_id = {"ocds-test-002": "ocds-test-002-2025-03-01"}
        tables = extract_all(tmp_path, ocid_to_release_id, show_progress=False)

        assert len(tables["tender_core"]) == 1
        assert tables["tender_core"].iloc[0]["tender_status"] == "active"

    def test_ocid_not_in_map_is_skipped(self, tmp_path):
        """Releases whose OCID is not in ocid_to_release_id must be ignored."""
        rec = {
            "ocid": "ocds-not-in-map",
            "id": "ocds-not-in-map-2024-01-01",
            "date": "2024-01-01T00:00:00Z",
            "tender": {"title": "Should be skipped"},
        }
        _write_gz(tmp_path / "2024.jsonl.gz", [rec])

        tables = extract_all(tmp_path, {}, show_progress=False)
        assert len(tables["tender_core"]) == 0

    def test_extract_all_multi_ocid(self, tmp_path):
        """Multiple distinct OCIDs are all extracted when release_ids match."""
        records = [
            {"ocid": "ocds-A", "id": "ocds-A-2024", "date": "2024-01-01Z",
             "tender": {"title": "Alpha", "status": "complete"}},
            {"ocid": "ocds-B", "id": "ocds-B-2024", "date": "2024-01-01Z",
             "tender": {"title": "Beta", "status": "active"}},
            # A stale duplicate for ocds-A — must be rejected
            {"ocid": "ocds-A", "id": "ocds-A-2023", "date": "2023-01-01Z",
             "tender": {"title": "Alpha OLD", "status": "planning"}},
        ]
        _write_gz(tmp_path / "2024.jsonl.gz", records)

        ocid_to_release_id = {"ocds-A": "ocds-A-2024", "ocds-B": "ocds-B-2024"}
        tables = extract_all(tmp_path, ocid_to_release_id, show_progress=False)

        assert len(tables["tender_core"]) == 2
        statuses = set(tables["tender_core"]["tender_status"])
        assert statuses == {"complete", "active"}  # no "planning" from stale release

    def test_wrong_release_id_produces_zero_rows(self, tmp_path):
        """When the release_id in the map doesn't match any raw record, zero rows are extracted
        and no exception is raised — the OCID is simply not found."""
        rec = {
            "ocid": "ocds-C",
            "id": "ocds-C-2024-real",
            "date": "2024-01-01Z",
            "tender": {"title": "Real release"},
        }
        _write_gz(tmp_path / "2024.jsonl.gz", [rec])

        # Map points to a release_id that doesn't exist in the file
        tables = extract_all(tmp_path, {"ocds-C": "ocds-C-2024-DOES-NOT-EXIST"},
                             show_progress=False)
        assert len(tables["tender_core"]) == 0

    def test_years_filter_excludes_non_matching_files(self, tmp_path):
        """When --years=[2025] is given and only a 2024 file exists, FileNotFoundError is raised
        because the glob finds no matching files."""
        rec = {"ocid": "ocds-D", "id": "ocds-D-2024", "date": "2024-01-01Z"}
        _write_gz(tmp_path / "2024.jsonl.gz", [rec])

        with pytest.raises(FileNotFoundError):
            extract_all(tmp_path, {"ocds-D": "ocds-D-2024"},
                        years=[2025], show_progress=False)
