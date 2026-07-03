"""
Pipeline step 00 (optional): Fetch and cache official reference datasets.

NOT part of the default pipeline run. Run this separately when you want to refresh
the reference cache used by src/reference_lookup.py.

Writes to data/reference/:
    govuk_orgs.json              ← GOV.UK Organisations API snapshot
    nhs_ods.json                 ← NHS ODS API snapshot
    contracts_finder_buyers.json ← buyer name summary derived from CF published notices
    _cache_meta.json             ← updated with timestamps and record counts

Usage:
    python pipelines/00_fetch_reference.py [--source govuk_orgs] [--source nhs_ods]
    python pipelines/00_fetch_reference.py --refresh   # refresh all

IMPORTANT: This script makes live HTTP requests. Run it only when intentionally
refreshing the cache, not as part of the normal 01→02→03 pipeline. The main
pipeline reads from the cached files and is fully reproducible without network access.

Source tiers (for reference_lookup.py):
    govuk_orgs        — Tier 2 STRONG: central gov departments, agencies, NDPBs
    nhs_ods           — Tier 2 STRONG: buyer-level NHS trusts, ICBs, CCGs, health boards
    contracts_finder  — Tier 3 MEDIUM: buyer names from published notices (no direct API)
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from normalise import normalise_name

ROOT = Path(__file__).parent.parent
REFERENCE_DIR = ROOT / "data" / "reference"
META_PATH = REFERENCE_DIR / "_cache_meta.json"
INTERIM_PATH = ROOT / "data" / "interim" / "releases.parquet"

# Each fetcher returns a list of dicts (records).

FETCHERS = {
    "govuk_orgs": {
        "description": "GOV.UK Organisations API — central government & public bodies",
        "implemented": True,
        "cache_file": "govuk_orgs.json",
    },
    "nhs_ods": {
        "description": "NHS ODS API — NHS/health organisations",
        "implemented": True,
        "cache_file": "nhs_ods.json",
    },
    "contracts_finder": {
        "description": "Contracts Finder buyer summary (derived from published notices)",
        "implemented": True,
        "cache_file": "contracts_finder_buyers.json",
    },
}


def _get_json(url: str, timeout: int = 60, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_govuk_orgs() -> list[dict]:
    """Fetch all GOV.UK organisations from the GOV.UK Organisations API.

    Returns:
        List of org dicts with keys: slug, title, web_url, details.
    """
    url = "https://www.gov.uk/api/organisations"
    records: list[dict] = []

    while url:
        data = _get_json(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "fyp_new-reference-cache/1.0",
            },
        )
        records.extend(data.get("results") or [])
        url = data.get("next_page_url")
        if url:
            time.sleep(0.1)

    return records


def fetch_nhs_ods() -> list[dict]:
    """Fetch NHS organisations from the NHS ODS API.

    Returns:
        List of org dicts with keys: OrgId, Name, Status, PrimaryRoleId.
    """
    base_url = "https://directory.spineservices.nhs.uk/ORD/2-0-0/organisations"
    limit = 1000
    by_org_id: dict[str, dict] = {}

    primary_roles = [
        "RO98",   # Clinical Commissioning Group
        "RO107",  # Care Trust
        "RO132",  # Health Authority
        "RO142",  # Local Health Board
        "RO144",  # Welsh Local Health Board
        "RO153",  # Northern Ireland Health & Social Care Board
        "RO154",  # Northern Ireland Health and Social Care Trust
        "RO155",  # Northern Ireland Local Commissioning Group
        "RO179",  # Primary Care Trust
        "RO189",  # Special Health Authority
        "RO190",  # Scottish Health Board
        "RO197",  # NHS Trust
        "RO209",  # NHS England Region
        "RO210",  # NHS England Region/Local Office
        "RO213",  # Commissioning Support Unit
    ]
    non_primary_roles = [
        "RO57",   # Foundation Trust
        "RO67",   # Specialised Commissioning Group
        "RO211",  # Specialised Commissioning Hub
        "RO218",  # Commissioning Hub
        "RO318",  # Integrated Care Board
        "RO326",  # ICB Commissioning Proxy
    ]

    def fetch_role(param_name: str, role_id: str) -> None:
        offset = 0
        role_count = 0
        while True:
            params = {"Limit": limit, param_name: role_id}
            if offset:
                params["Offset"] = offset
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            data = _get_json(url, headers={"Accept": "application/json"})
            batch = data.get("Organisations") or []
            for org in batch:
                org_id = str(org.get("OrgId") or "").strip()
                if not org_id:
                    continue
                record = by_org_id.setdefault(org_id, dict(org))
                roles = set(record.get("_matched_reference_roles") or [])
                roles.add(f"{param_name}:{role_id}")
                record["_matched_reference_roles"] = sorted(roles)
            role_count += len(batch)
            if len(batch) < limit:
                break
            offset += limit
            time.sleep(0.1)
        print(f"  NHS ODS: {param_name}={role_id} returned {role_count:,}", flush=True)

    for role_id in primary_roles:
        fetch_role("PrimaryRoleId", role_id)
    for role_id in non_primary_roles:
        fetch_role("NonPrimaryRoleId", role_id)

    return sorted(by_org_id.values(), key=lambda org: str(org.get("OrgId") or ""))


def fetch_contracts_finder_buyers() -> list[dict]:
    """Derive a buyer name registry from Contracts Finder published notices.

    This is not a direct buyer-register API. Instead it aggregates buyer names
    and IDs from the published notices data already ingested in data/raw/.
    """
    if not INTERIM_PATH.exists():
        raise FileNotFoundError(
            f"{INTERIM_PATH} not found. Run pipelines/01_ingest.py before "
            "deriving the Contracts Finder buyer cache."
        )

    releases = pd.read_parquet(INTERIM_PATH, columns=["buyer_raw_id", "buyer_name", "year"])
    releases = releases[
        releases["buyer_raw_id"].notna()
        & releases["buyer_raw_id"].astype(str).str.strip().ne("")
    ].copy()
    releases["norm_name"] = releases["buyer_name"].apply(normalise_name)

    records = []
    for raw_id, grp in releases.groupby("buyer_raw_id", sort=False):
        names = (
            grp["buyer_name"].dropna().astype(str)
            .loc[lambda s: s.str.strip().ne("")]
            .value_counts()
        )
        norm_names = grp["norm_name"].dropna().astype(str).loc[lambda s: s.str.strip().ne("")]
        records.append({
            "raw_id": str(raw_id),
            "canonical_name": names.index[0] if len(names) else str(raw_id),
            "normalised_names": sorted(set(norm_names.tolist())),
            "name_variants": names.head(10).to_dict(),
            "n_releases": int(len(grp)),
            "first_year": int(grp["year"].min()) if grp["year"].notna().any() else None,
            "last_year": int(grp["year"].max()) if grp["year"].notna().any() else None,
        })

    return records


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _write_cache(name: str, records: list[dict], cache_file: str | None = None) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REFERENCE_DIR / (cache_file or f"{name}.json")
    serialised = json.dumps(records, ensure_ascii=False, indent=2)
    out_path.write_text(serialised, encoding="utf-8")

    meta_raw = META_PATH.read_text(encoding="utf-8") if META_PATH.exists() else "{}"
    meta = json.loads(meta_raw)
    if name == "contracts_finder":
        meta.pop("contracts_finder_buyers", None)
    meta[name] = {
        **meta.get(name, {}),
        "last_fetched": datetime.now(tz=timezone.utc).isoformat(),
        "record_count": len(records),
        "sha256": _sha256(serialised),
        "cache_file": out_path.name,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Cached {len(records):,} records → {out_path}")


def main(sources: list[str]) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "govuk_orgs": fetch_govuk_orgs,
        "nhs_ods": fetch_nhs_ods,
        "contracts_finder": fetch_contracts_finder_buyers,
    }

    for source in sources:
        if source not in dispatch:
            print(f"Unknown source: {source}. Valid: {list(dispatch)}")
            continue
        info = FETCHERS[source]
        print(f"\nFetching {source}: {info['description']}")
        if not info["implemented"]:
            print(f"  NOT YET IMPLEMENTED. Skipping.")
            continue
        try:
            records = dispatch[source]()
            _write_cache(source, records, cache_file=info.get("cache_file"))
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone. Update data/reference/_cache_meta.json for audit trail.")


if __name__ == "__main__":
    from procurement_graph.reference.fetchers import cli_main

    cli_main()
