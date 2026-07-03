"""
Stage 1b: Rich OCDS field extraction from the interim releases parquet.

Reads data/interim/releases.parquet (one row per OCID, parties_json and
contracts_json already extracted by ingest.py).

This module re-reads the raw JSONL.gz files to access the full OCDS record
for fields not captured by ingest.py. The interim parquet is the deduplication
authority (latest release per OCID); we only extract fields for OCIDs that
survived deduplication.

Produces seven output tables in data/extracted/:
  tender_core.parquet    — structured tender attributes (one row per OCID)
  lots.parquet           — per-lot structured data
  award_criteria.parquet — per-lot award evaluation criteria
  awards.parquet         — award records
  bid_stats.parquet      — bid statistics (counts, values)
  text_evidence.parquet  — long free-text fields, indexed by ocid + field_path
  documents.parquet      — document URLs and metadata

Design principles:
  - Entity resolution lives in er_phase1/er_phase2; this module does not resolve entities.
  - Text evidence is separated from structured attributes; do not store paragraphs as KG nodes.
  - Document downloading is NOT done here; only metadata and URLs are captured.
  - All tables share `ocid` as the join key to releases.parquet.
  - Import rules: this module does NOT import from er_*.py or kg_*.py.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ingest import load_year  # re-use the JSONL.gz streamer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _s(obj: Any, *keys, default="") -> str:
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k)
        if obj is None:
            return default
    return str(obj) if obj is not None else default


def _f(obj: Any, *keys) -> float | None:
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
        if obj is None:
            return None
    try:
        return float(obj)
    except (TypeError, ValueError):
        return None


def _b(obj: Any, *keys) -> bool | None:
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
        if obj is None:
            return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj.lower() == "true"
    return None


def _list_str(obj: Any, *keys) -> str:
    """Drill into obj by keys, expect a list, return JSON string."""
    for k in keys:
        if not isinstance(obj, dict):
            return "[]"
        obj = obj.get(k)
        if obj is None:
            return "[]"
    if isinstance(obj, list):
        return json.dumps([str(x) for x in obj if x is not None])
    return "[]"


# ---------------------------------------------------------------------------
# Per-release extractors
# ---------------------------------------------------------------------------


def _extract_tender_core(ocid: str, release: dict) -> dict:
    """Extract structured tender attributes not already in releases.parquet."""
    t = release.get("tender") or {}
    tech = t.get("techniques") or {}
    fa = tech.get("frameworkAgreement") or {}
    ss = t.get("secondStage") or {}
    comm = t.get("communication") or {}
    sub = t.get("submissionTerms") or {}
    enq = t.get("enquiryPeriod") or {}
    ap = t.get("awardPeriod") or {}
    bo = t.get("bidOpening") or {}

    return {
        "ocid": ocid,
        "tender_status": _s(t, "status"),
        "tender_legal_basis_id": _s(t, "legalBasis", "id"),
        "tender_legal_basis_scheme": _s(t, "legalBasis", "scheme"),
        "tender_covered_by": _list_str(t, "coveredBy"),
        "tender_submission_method": _list_str(t, "submissionMethod"),
        "tender_variant_policy": _s(sub, "variantPolicy"),
        "tender_has_recurrence": _b(t, "hasRecurrence"),
        "tender_has_framework": _b(tech, "hasFrameworkAgreement"),
        "tender_framework_max_participants": _f(fa, "maximumParticipants"),
        "tender_has_dps": _b(tech, "hasDynamicPurchasingSystem"),
        "tender_second_stage_min": _f(ss, "minimumCandidates"),
        "tender_second_stage_max": _f(ss, "maximumCandidates"),
        "tender_award_period_start": _s(ap, "startDate"),
        "tender_bid_opening_date": _s(bo, "date"),
        "tender_enquiry_period_end": _s(enq, "endDate"),
        "tender_future_notice_date": _s(comm, "futureNoticeDate"),
        "tender_submission_url": _s(t, "submissionMethodDetails"),
    }


def _extract_lots(ocid: str, release: dict) -> list[dict]:
    """Extract per-lot structured fields. Award criteria go to a separate extractor."""
    t = release.get("tender") or {}
    lots = t.get("lots") or []
    items = t.get("items") or []

    # Pre-index items by relatedLot for delivery regions and additional CPVs
    items_by_lot: dict[str, list[dict]] = {}
    for item in items:
        lot_ref = _s(item, "relatedLot")
        items_by_lot.setdefault(lot_ref, []).append(item)

    rows = []
    for lot in lots:
        lot_id = _s(lot, "id")
        period = lot.get("contractPeriod") or {}
        value = lot.get("value") or {}
        sub = lot.get("submissionTerms") or {}
        suit = lot.get("suitability") or {}

        # Delivery regions from items linked to this lot
        lot_items = items_by_lot.get(lot_id, [])
        delivery_regions: list[str] = []
        additional_cpvs: list[str] = []
        for item in lot_items:
            for addr in item.get("deliveryAddresses") or []:
                r = _s(addr, "region")
                if r:
                    delivery_regions.append(r)
            for cls in item.get("additionalClassifications") or []:
                cpv = _s(cls, "id")
                if cpv and _s(cls, "scheme").upper() == "CPV":
                    additional_cpvs.append(cpv)

        rows.append({
            "ocid": ocid,
            "lot_id": lot_id,
            "lot_status": _s(lot, "status"),
            "lot_title": _s(lot, "title"),
            "has_options": _b(lot, "hasOptions"),
            "has_renewal": _b(lot, "hasRenewal"),
            "lot_value_amount": _f(value, "amount"),
            "lot_value_currency": _s(value, "currency"),
            "contract_start": _s(period, "startDate"),
            "contract_end": _s(period, "endDate"),
            "contract_duration_days": _f(period, "durationInDays"),
            "variant_policy": _s(sub, "variantPolicy"),
            "is_vcse": _b(suit, "vcse"),
            "delivery_regions": json.dumps(sorted(set(delivery_regions))),
            "additional_cpv_ids": json.dumps(sorted(set(additional_cpvs))),
        })
    return rows


def _extract_award_criteria(ocid: str, release: dict) -> list[dict]:
    """Extract per-lot award evaluation criteria."""
    t = release.get("tender") or {}
    rows = []
    for lot in t.get("lots") or []:
        lot_id = _s(lot, "id")
        ac = lot.get("awardCriteria") or {}
        for idx, crit in enumerate(ac.get("criteria") or []):
            weight_raw = crit.get("weight") or crit.get("numbers", [{}])[0].get("weight") if isinstance(crit.get("numbers"), list) and crit.get("numbers") else None
            # weight may be stored as a plain number OR inside numbers[]
            weight = None
            if "weight" in crit:
                try:
                    weight = float(crit["weight"])
                except (TypeError, ValueError):
                    pass
            if weight is None and isinstance(crit.get("numbers"), list):
                for num in crit["numbers"]:
                    if "weight" in num:
                        try:
                            weight = float(num["weight"])
                        except (TypeError, ValueError):
                            pass
                        break

            rows.append({
                "ocid": ocid,
                "lot_id": lot_id,
                "criterion_index": idx,
                "criterion_type": _s(crit, "type"),
                "criterion_name": _s(crit, "name"),
                "criterion_weight": weight,
            })
    return rows


def _contract_lookup_by_award(release: dict) -> dict[str, dict]:
    """Return award_id -> flattened contract fields for the first matching contract."""
    lookup: dict[str, dict] = {}
    for contract in release.get("contracts") or []:
        award_id = _s(contract, "awardID")
        if not award_id or award_id in lookup:
            continue
        value = contract.get("value") or {}
        lookup[award_id] = {
            "contract_id": _s(contract, "id"),
            "contract_value_amount": _f(value, "amount") or _f(value, "amountNet"),
            "contract_value_currency": _s(value, "currency"),
            "award_date_signed": _s(contract, "dateSigned"),
        }
    return lookup


def _best_award_value(
    award_value_amount: float | None,
    award_value_currency: str,
    contract_value_amount: float | None,
    contract_value_currency: str,
    tender_value_amount: float | None,
    tender_value_currency: str,
) -> tuple[float | None, str, str]:
    """Choose the best available award-level value without overwriting provenance."""
    if award_value_amount is not None:
        return award_value_amount, award_value_currency, "award"
    if contract_value_amount is not None:
        return contract_value_amount, contract_value_currency, "contract"
    if tender_value_amount is not None:
        return tender_value_amount, tender_value_currency, "tender"
    return None, "", ""


def _extract_awards(ocid: str, release: dict) -> list[dict]:
    """Extract award records."""
    rows = []
    contract_by_award = _contract_lookup_by_award(release)
    tender_value = (release.get("tender") or {}).get("value") or {}
    tender_value_amount = _f(tender_value, "amount")
    tender_value_currency = _s(tender_value, "currency")
    for award in release.get("awards") or []:
        value = award.get("value") or {}
        period = award.get("contractPeriod") or {}
        suppliers = [_s(s, "id") for s in award.get("suppliers") or [] if s.get("id")]
        related_lots = award.get("relatedLots") or []
        award_id = _s(award, "id")
        award_value_amount = _f(value, "amount") or _f(value, "amountNet")
        award_value_currency = _s(value, "currency")
        contract = contract_by_award.get(award_id, {})
        contract_value_amount = contract.get("contract_value_amount")
        contract_value_currency = str(contract.get("contract_value_currency") or "")
        best_amount, best_currency, value_source = _best_award_value(
            award_value_amount,
            award_value_currency,
            contract_value_amount,
            contract_value_currency,
            tender_value_amount,
            tender_value_currency,
        )
        rows.append({
            "ocid": ocid,
            "award_id": award_id,
            "award_status": _s(award, "status"),
            "award_title": _s(award, "title"),
            "award_date": _s(award, "date"),
            "award_date_published": _s(award, "datePublished"),
            "award_date_signed": str(contract.get("award_date_signed") or ""),
            "award_value_amount": award_value_amount,
            "award_value_gross": _f(value, "amountGross"),
            "award_value_currency": award_value_currency,
            "contract_value_amount": contract_value_amount,
            "contract_value_currency": contract_value_currency,
            "award_value_best_amount": best_amount,
            "award_value_best_currency": best_currency,
            "award_value_source": value_source,
            "award_period_start": _s(period, "startDate"),
            "award_period_end": _s(period, "endDate"),
            "related_lots": json.dumps([str(x) for x in related_lots]),
            "supplier_raw_ids": json.dumps(suppliers),
            "above_threshold": _b(award, "aboveThreshold"),
        })
    return rows


def _extract_bid_stats(ocid: str, release: dict) -> list[dict]:
    """Extract bid statistics."""
    rows = []
    bids = release.get("bids") or {}
    for stat in bids.get("statistics") or []:
        measure = _s(stat, "measure")
        if not measure:
            continue
        rows.append({
            "ocid": ocid,
            "measure": measure,
            "stat_value": _f(stat, "value"),
            "related_lot": _s(stat, "relatedLot"),
        })
    return rows


def _extract_text_evidence(ocid: str, release: dict) -> list[dict]:
    """Extract long free-text fields into a flat evidence table."""
    t = release.get("tender") or {}
    rows = []

    def _add(field_path: str, text: str, lot_id: str = "") -> None:
        text = (text or "").strip()
        if text:
            rows.append({
                "ocid": ocid,
                "field_path": field_path,
                "lot_id": lot_id,
                "text": text,
            })

    _add("tender.description", _s(t, "description"))
    _add("tender.reviewDetails", _s(t, "reviewDetails"))

    ct = t.get("contractTerms") or {}
    _add("tender.contractTerms.performanceTerms", _s(ct, "performanceTerms"))

    sc = t.get("selectionCriteria") or {}
    for idx, crit in enumerate(sc.get("criteria") or []):
        _add(f"tender.selectionCriteria.criteria[{idx}].description",
             _s(crit, "description"))

    for lot in t.get("lots") or []:
        lot_id = _s(lot, "id")
        _add("tender.lots[].description", _s(lot, "description"), lot_id)
        renewal = lot.get("renewal") or {}
        _add("tender.lots[].renewal.description", _s(renewal, "description"), lot_id)
        options = lot.get("options") or {}
        _add("tender.lots[].options.description", _s(options, "description"), lot_id)

        ac = lot.get("awardCriteria") or {}
        for idx, crit in enumerate(ac.get("criteria") or []):
            desc = _s(crit, "description")
            # Only store if it looks like actual text, not just a number
            if desc and not desc.replace(".", "").isdigit():
                _add(f"tender.lots[].awardCriteria.criteria[{idx}].description",
                     desc, lot_id)

    return rows


def _extract_documents(ocid: str, release: dict) -> list[dict]:
    """Extract document metadata and URLs from tender, awards, and parties."""
    rows = []
    t = release.get("tender") or {}

    def _doc_row(source: str, doc: dict, lot_id: str = "") -> dict:
        return {
            "ocid": ocid,
            "source": source,
            "lot_id": lot_id,
            "doc_id": _s(doc, "id"),
            "document_type": _s(doc, "documentType"),
            "url": _s(doc, "url"),
            "title": _s(doc, "title"),
            "description": _s(doc, "description"),
            "format": _s(doc, "format"),
            "date_published": _s(doc, "datePublished"),
            "notice_type": _s(doc, "noticeType"),
        }

    # Tender documents
    for doc in t.get("documents") or []:
        rows.append(_doc_row("tender", doc))

    # Award documents
    for award in release.get("awards") or []:
        for doc in award.get("documents") or []:
            rows.append(_doc_row("award", doc))

    # Submission URL as a pseudo-document
    sub_url = _s(t, "submissionMethodDetails")
    if sub_url:
        rows.append({
            "ocid": ocid,
            "source": "tender_submission",
            "lot_id": "",
            "doc_id": "",
            "document_type": "submissionUrl",
            "url": sub_url,
            "title": "",
            "description": "",
            "format": "",
            "date_published": "",
            "notice_type": "",
        })

    # Party URLs
    for party in release.get("parties") or []:
        raw_id = _s(party, "id")
        details = party.get("details") or {}
        contact = party.get("contactPoint") or {}

        profile_url = _s(details, "url") or _s(contact, "url")
        if profile_url:
            rows.append({
                "ocid": ocid,
                "source": "party_profile",
                "lot_id": "",
                "doc_id": raw_id,
                "document_type": "profileUrl",
                "url": profile_url,
                "title": _s(party, "name"),
                "description": "",
                "format": "",
                "date_published": "",
                "notice_type": "",
            })

        buyer_profile = _s(details, "buyerProfile")
        if buyer_profile:
            rows.append({
                "ocid": ocid,
                "source": "party_buyer_profile",
                "lot_id": "",
                "doc_id": raw_id,
                "document_type": "buyerProfile",
                "url": buyer_profile,
                "title": _s(party, "name"),
                "description": "",
                "format": "",
                "date_published": "",
                "notice_type": "",
            })

        email = _s(contact, "email")
        if email:
            rows.append({
                "ocid": ocid,
                "source": "party_contact",
                "lot_id": "",
                "doc_id": raw_id,
                "document_type": "contactEmail",
                "url": "",
                "title": _s(party, "name"),
                "description": email,
                "format": "",
                "date_published": "",
                "notice_type": "",
            })

    return rows


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def extract_all(
    raw_dir: Path,
    ocid_to_release_id: dict[str, str],
    years: list[int] | None = None,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Stream raw JSONL.gz files and extract all seven output tables.

    **Snapshot mode only.** Extracts exactly one record per OCID — the same
    release that 01_ingest.py selected (latest date per OCID). Alignment is
    enforced by matching on the release `id` field, not on file order.

    `ocid_to_release_id` must be built from releases.parquet:
        ocid_to_release_id = dict(zip(df["ocid"], df["release_id"]))

    A raw release is accepted only when:
        release["ocid"] is in ocid_to_release_id
        AND release["id"] == ocid_to_release_id[release["ocid"]]

    This guarantees the extracted tables are row-for-row consistent with
    releases.parquet. History mode (all releases per OCID) is not implemented;
    if needed it would require a separate function and a different output schema
    that includes release_id and date as part of every row key.

    Args:
        raw_dir: directory containing *.jsonl.gz files
        ocid_to_release_id: mapping ocid → release_id from releases.parquet
        years: if given, only read these year files (saves time when the winning
               release is known to be within specific years)
        show_progress: print file-level progress

    Returns:
        dict with keys: tender_core, lots, award_criteria, awards,
                        bid_stats, text_evidence, documents
        Values are DataFrames (may be empty if no records matched).
    """
    paths = sorted(raw_dir.glob("*.jsonl.gz"))
    if years:
        paths = [p for p in paths if any(str(y) in p.stem for y in years)]
    if not paths:
        raise FileNotFoundError(f"No .jsonl.gz files in {raw_dir}")

    accumulators: dict[str, list[dict]] = {
        "tender_core": [],
        "lots": [],
        "award_criteria": [],
        "awards": [],
        "bid_stats": [],
        "text_evidence": [],
        "documents": [],
    }

    # OCIDs still waiting to be matched (shrinks as we find their winning release)
    remaining: dict[str, str] = dict(ocid_to_release_id)  # ocid → release_id

    for path in paths:
        if not remaining:
            break  # all OCIDs already matched; skip remaining files
        n_matched = 0
        if show_progress:
            print(f"  Extracting {path.name} ...", end=" ", flush=True)
        for release in load_year(path):
            ocid = release.get("ocid") or ""
            release_id = release.get("id") or ""
            # Accept only the exact release that ingest selected
            if ocid not in remaining or remaining[ocid] != release_id:
                continue
            del remaining[ocid]  # mark as matched
            n_matched += 1

            accumulators["tender_core"].append(_extract_tender_core(ocid, release))
            accumulators["lots"].extend(_extract_lots(ocid, release))
            accumulators["award_criteria"].extend(_extract_award_criteria(ocid, release))
            accumulators["awards"].extend(_extract_awards(ocid, release))
            accumulators["bid_stats"].extend(_extract_bid_stats(ocid, release))
            accumulators["text_evidence"].extend(_extract_text_evidence(ocid, release))
            accumulators["documents"].extend(_extract_documents(ocid, release))

        if show_progress:
            print(f"{n_matched} matched")

    if remaining and show_progress:
        logger.warning(
            "%d OCIDs from releases.parquet were not found in raw files "
            "(they may be in year files excluded by --years): %s …",
            len(remaining),
            list(remaining)[:5],
        )

    return {k: pd.DataFrame(v) for k, v in accumulators.items()}


def write_extracted(tables: dict[str, pd.DataFrame], extracted_dir: Path) -> None:
    """Write all extraction tables to Parquet files."""
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = extracted_dir / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        size_kb = path.stat().st_size // 1024
        print(f"  Written: {path}  ({len(df):,} rows, {size_kb:,} KB)")


def load_extracted(extracted_dir: Path, table: str) -> pd.DataFrame:
    """Load one extracted table by name."""
    return pd.read_parquet(extracted_dir / f"{table}.parquet")


# ---------------------------------------------------------------------------
# Convenience: extract from a single release dict (used in tests)
# ---------------------------------------------------------------------------


def extract_release(release: dict) -> dict[str, list[dict]]:
    """Run all extractors on a single release dict. Returns raw row lists."""
    ocid = release.get("ocid") or ""
    return {
        "tender_core": [_extract_tender_core(ocid, release)],
        "lots": _extract_lots(ocid, release),
        "award_criteria": _extract_award_criteria(ocid, release),
        "awards": _extract_awards(ocid, release),
        "bid_stats": _extract_bid_stats(ocid, release),
        "text_evidence": _extract_text_evidence(ocid, release),
        "documents": _extract_documents(ocid, release),
    }


# Compatibility re-export. New code should import from
# procurement_graph.extract.tables; old flat imports keep working.
from procurement_graph.extract.tables import (  # noqa: E402,F401
    extract_all,
    extract_release,
    load_extracted,
    write_extracted,
)
