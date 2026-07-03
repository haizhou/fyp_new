"""
Stage 1: OCDS data loading, flattening, and OCID deduplication.

Reads raw .jsonl.gz files from data/raw/, extracts a flat row per release,
deduplicates by OCID (latest date wins), and writes data/interim/releases.parquet.

Import rules: this module does NOT import from er_*.py or kg_*.py.
"""

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level streaming
# ---------------------------------------------------------------------------


def load_year(path: Path) -> Iterator[dict]:
    """Stream one decompressed JSONL.gz file, yielding one dict per line.

    Skips blank lines and lines that fail JSON parsing (logs a warning).
    Never loads the full file into memory.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("JSON parse error in %s line %d: %s", path.name, lineno, exc)


# ---------------------------------------------------------------------------
# Per-release extraction
# ---------------------------------------------------------------------------


def _safe_str(obj, *keys, default=None):
    """Drill into a nested dict safely, returning default if any key is missing."""
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k)
    return obj if obj is not None else default


def _safe_float(obj, *keys) -> float | None:
    v = _safe_str(obj, *keys)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _extract_parties(release: dict) -> list[dict]:
    """Extract structured party records from a release.

    Each returned dict has a flat schema with all fields the ER stage needs.
    """
    parties = []
    for p in release.get("parties") or []:
        raw_id = p.get("id") or ""
        ident = p.get("identifier") or {}
        address = p.get("address") or {}
        details = p.get("details") or {}
        contact = p.get("contactPoint") or {}

        # Extract TED_CA_TYPE classification if present
        org_category = None
        for cls in details.get("classifications") or []:
            if cls.get("scheme") == "TED_CA_TYPE":
                org_category = cls.get("id")
                break

        parties.append(
            {
                "raw_id": raw_id,
                "name": p.get("name") or "",
                "roles": json.dumps(p.get("roles") or []),
                "identifier_scheme": ident.get("scheme") or "",
                "identifier_id": ident.get("id") or "",
                "identifier_legal_name": ident.get("legalName") or "",
                "address_region": address.get("region") or "",
                "address_postcode": address.get("postalCode") or "",
                "details_url": details.get("url") or contact.get("url") or "",
                "org_category": org_category or "",
            }
        )
    return parties


def _extract_contracts(release: dict) -> list[dict]:
    """Extract structured contract records from a release.

    Supplier raw_ids come from the matching award (joined by award_id).
    """
    # Build award_id → supplier_raw_ids lookup
    award_suppliers: dict[str, list[str]] = {}
    for award in release.get("awards") or []:
        aid = award.get("id") or ""
        suppliers = [s.get("id") or "" for s in award.get("suppliers") or []]
        award_suppliers[aid] = suppliers

    contracts = []
    for c in release.get("contracts") or []:
        award_id = c.get("awardID") or ""
        value = c.get("value") or {}
        period = c.get("period") or {}
        contracts.append(
            {
                "contract_id": c.get("id") or "",
                "award_id": award_id,
                "value_amount": _safe_float(value, "amount"),
                "value_currency": value.get("currency") or "GBP",
                "date_signed": c.get("dateSigned") or "",
                "period_start": period.get("startDate") or "",
                "period_end": period.get("endDate") or "",
                "status": c.get("status") or "",
                "supplier_raw_ids": award_suppliers.get(award_id, []),
            }
        )
    return contracts


def extract_release_row(release: dict) -> dict | None:
    """Convert a single OCDS release dict into one flat row dict.

    Returns None for malformed records that lack an ocid or date.
    Nested arrays (parties, contracts) are JSON-serialised strings so that
    the interim parquet file has a simple, universally readable schema.
    """
    ocid = release.get("ocid")
    date_str = release.get("date")
    if not ocid or not date_str:
        return None

    buyer = release.get("buyer") or {}
    tender = release.get("tender") or {}
    tender_value = tender.get("value") or {}
    tender_cpv = tender.get("classification") or {}
    tender_period = tender.get("tenderPeriod") or {}
    contract_period = tender.get("contractPeriod") or {}

    parties = _extract_parties(release)
    contracts = _extract_contracts(release)

    # Detect multi-lot (potential framework indicator)
    lots = tender.get("lots") or []
    n_lots = len(lots)

    return {
        "ocid": ocid,
        "release_id": release.get("id") or "",
        "date": date_str,
        "tag": release.get("tag") or "",
        "buyer_raw_id": buyer.get("id") or "",
        "buyer_name": buyer.get("name") or "",
        "tender_title": tender.get("title") or "",
        "tender_value_amount": _safe_float(tender_value, "amount"),
        "tender_value_currency": tender_value.get("currency") or "",
        "tender_cpv_id": tender_cpv.get("id") or "",
        "tender_cpv_description": tender_cpv.get("description") or "",
        "tender_method": tender.get("procurementMethod") or "",
        "tender_method_details": tender.get("procurementMethodDetails") or "",
        "tender_category": tender.get("mainProcurementCategory") or "",
        "tender_period_end": tender_period.get("endDate") or "",
        "contract_period_start": contract_period.get("startDate") or "",
        "contract_period_end": contract_period.get("endDate") or "",
        "n_lots": n_lots,
        "parties_json": json.dumps(parties),
        "contracts_json": json.dumps(contracts),
    }


# ---------------------------------------------------------------------------
# Multi-year flattening with OCID deduplication
# ---------------------------------------------------------------------------


def flatten_all_years(
    raw_dir: Path,
    years: list[int] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Load all .jsonl.gz files in raw_dir, flatten, and deduplicate by OCID.

    Deduplication rule: for each ocid, keep the row with the latest `date`
    (string ISO comparison is safe because dates are all ISO-8601 format).
    A `source_years` column records which year file(s) each OCID appeared in.

    Args:
        raw_dir: directory containing *.jsonl.gz files
        years: if given, only load files for these years (e.g. [2022, 2023])
        show_progress: whether to print a simple progress line per file

    Returns:
        DataFrame with one row per unique OCID.
    """
    paths = sorted(raw_dir.glob("*.jsonl.gz"))
    if years:
        paths = [p for p in paths if any(str(y) in p.stem for y in years)]
    if not paths:
        raise FileNotFoundError(f"No .jsonl.gz files found in {raw_dir}")

    # Collect: ocid → {best_row, source_years_set}
    best: dict[str, dict] = {}
    source_years: dict[str, set] = {}
    total_parsed = 0
    total_skipped = 0

    for path in paths:
        year_tag = path.stem  # e.g. "2024"
        n_file = 0
        if show_progress:
            print(f"  Loading {path.name} ...", end=" ", flush=True)
        for release in load_year(path):
            row = extract_release_row(release)
            if row is None:
                total_skipped += 1
                continue
            total_parsed += 1
            n_file += 1
            ocid = row["ocid"]
            if ocid not in best or row["date"] > best[ocid]["date"]:
                best[ocid] = row
            source_years.setdefault(ocid, set()).add(year_tag)
        if show_progress:
            print(f"{n_file} records")

    if not best:
        raise ValueError("No valid releases found in any input file.")

    # Build DataFrame
    rows = list(best.values())
    df = pd.DataFrame(rows)

    # Attach source_years
    df["source_years"] = df["ocid"].map(lambda o: "|".join(sorted(source_years[o])))

    # Parse date to datetime; derive year
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["year"] = df["date"].dt.year.astype("Int64")

    # Numeric columns
    df["n_lots"] = df["n_lots"].astype("Int64")
    df["tender_value_amount"] = pd.to_numeric(df["tender_value_amount"], errors="coerce")

    logger.info(
        "Ingest complete: %d unique OCIDs from %d releases (%d skipped)",
        len(df),
        total_parsed,
        total_skipped,
    )
    print(
        f"\n  Total: {total_parsed} releases parsed → {len(df)} unique OCIDs "
        f"({total_parsed - len(df)} duplicates resolved, {total_skipped} skipped)"
    )
    return df


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_interim(df: pd.DataFrame, out_path: Path) -> None:
    """Write the flattened releases DataFrame to Parquet (snappy compression)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  Written: {out_path}  ({size_mb:.1f} MB,  {len(df):,} rows)")


def load_interim(path: Path) -> pd.DataFrame:
    """Load the interim releases parquet file."""
    return pd.read_parquet(path)
