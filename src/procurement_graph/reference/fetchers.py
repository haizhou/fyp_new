"""Fetch and cache official reference datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from procurement_graph.common.normalise import normalise_name

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_DIR = ROOT / "data" / "reference"
META_PATH = REFERENCE_DIR / "_cache_meta.json"
INTERIM_PATH = ROOT / "data" / "interim" / "releases.parquet"

FETCHERS = {
    "govuk_orgs": {
        "description": "GOV.UK Organisations API - central government & public bodies",
        "implemented": True,
        "cache_file": "govuk_orgs.json",
        "source_url": "https://www.gov.uk/api/organisations",
    },
    "nhs_ods": {
        "description": "NHS ODS API - buyer-level NHS/health organisations",
        "implemented": True,
        "cache_file": "nhs_ods.json",
        "source_url": "https://directory.spineservices.nhs.uk/ORD/2-0-0/organisations",
    },
    "contracts_finder": {
        "description": "Contracts Finder buyer summary (derived from published notices)",
        "implemented": True,
        "cache_file": "contracts_finder_buyers.json",
        "source_url": "data/interim/releases.parquet",
    },
}


def _get_json(url: str, timeout: int = 60, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_govuk_orgs() -> list[dict]:
    """Fetch all GOV.UK organisations from the GOV.UK Organisations API."""
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
    """Fetch buyer-level NHS organisations from the NHS ODS API."""
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


def fetch_contracts_finder_buyers(interim_path: Path = INTERIM_PATH) -> list[dict]:
    """Derive a buyer-name registry from ingested Contracts Finder notices."""
    if not interim_path.exists():
        raise FileNotFoundError(
            f"{interim_path} not found. Run the ingest pipeline before deriving "
            "the Contracts Finder buyer cache."
        )

    releases = pd.read_parquet(interim_path, columns=["buyer_raw_id", "buyer_name", "year"])
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


def write_cache(
    name: str,
    records: list[dict],
    cache_file: str | None = None,
    reference_dir: Path = REFERENCE_DIR,
) -> Path:
    """Write a reference cache and update `_cache_meta.json`."""
    reference_dir.mkdir(parents=True, exist_ok=True)
    out_path = reference_dir / (cache_file or f"{name}.json")
    meta_path = reference_dir / "_cache_meta.json"
    serialised = json.dumps(records, ensure_ascii=False, indent=2)
    out_path.write_text(serialised, encoding="utf-8")

    meta_raw = meta_path.read_text(encoding="utf-8") if meta_path.exists() else "{}"
    meta = json.loads(meta_raw)
    if name == "contracts_finder":
        meta.pop("contracts_finder_buyers", None)
    meta[name] = {
        **meta.get(name, {}),
        "source_url": FETCHERS.get(name, {}).get("source_url"),
        "last_fetched": datetime.now(tz=timezone.utc).isoformat(),
        "record_count": len(records),
        "sha256": _sha256(serialised),
        "cache_file": out_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Cached {len(records):,} records -> {out_path}")
    return out_path


def fetch_source(source: str) -> list[dict]:
    """Fetch one configured reference source."""
    dispatch = {
        "govuk_orgs": fetch_govuk_orgs,
        "nhs_ods": fetch_nhs_ods,
        "contracts_finder": fetch_contracts_finder_buyers,
    }
    return dispatch[source]()


def run(sources: list[str]) -> None:
    """Fetch/cache the requested reference sources."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    for source in sources:
        if source not in FETCHERS:
            print(f"Unknown source: {source}. Valid: {list(FETCHERS)}")
            continue
        info = FETCHERS[source]
        print(f"\nFetching {source}: {info['description']}")
        if not info["implemented"]:
            print("  NOT YET IMPLEMENTED. Skipping.")
            continue
        try:
            records = fetch_source(source)
            write_cache(source, records, cache_file=info.get("cache_file"))
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print("\nDone. Update data/reference/_cache_meta.json for audit trail.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and cache official reference data")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Which source(s) to fetch (can repeat). Default: all.",
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh all sources")
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    chosen = list(FETCHERS.keys()) if args.refresh or not args.sources else args.sources
    run(chosen)


__all__ = [
    "FETCHERS",
    "cli_main",
    "fetch_contracts_finder_buyers",
    "fetch_govuk_orgs",
    "fetch_nhs_ods",
    "fetch_source",
    "run",
    "write_cache",
]

