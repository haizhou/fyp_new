"""
Reference lookup / enrichment layer (scaffold — API clients not yet implemented).

This module resolves GB-FTS-only (unresolved) entities against official government
and public-sector reference datasets. It provides a higher-confidence resolution
path than name-only matching, and sits between Phase 1 (deterministic, official IDs)
and name/region merge in Phase 2.

## Source hierarchy

    Tier 1 – CANONICAL (official OCDS schemes, handled in er_phase1.py):
        GB-COH (Companies House), GB-NHS (NHS ODS), GB-UKPRN, GB-CHC, GB-SC, GB-NIC, GB-MPR

    Tier 2 – STRONG reference evidence (this module, when implemented):
        GOV.UK Organisations API  → central government departments, executive agencies, NDPBs
        NHS ODS API               → NHS trusts, CCGs, ICBs, health boards
        Contracts Finder buyer register → Contracts Finder buyer IDs → official org names

    Tier 3 – MANUAL override (configs/gov_lookup.json):
        Small reviewed file (~55 entries), each with _source audit field.
        Used as fallback for entities not covered by Tier 2 APIs.
        Risky merges are flagged; no bulk auto-merges.

    Tier 4 – HEURISTIC (er_phase2.py):
        Exact normalised name + region merge
        Singleton fallback

    Tier 5 – CANDIDATE ONLY (er_candidates.py):
        Jaro-Winkler fuzzy pairs for human review; never auto-merged

## Reproducibility

    All API responses are cached to data/reference/ as JSON snapshots.
    The pipeline reads from cache; it never calls live APIs during normal runs.
    Cache is refreshed by running: python pipelines/00_fetch_reference.py --refresh
    Each cache file is checksummed in data/reference/_cache_meta.json.

## Directory layout

    data/reference/
        govuk_orgs.json              ← GOV.UK Organisations API snapshot
        nhs_ods.json                 ← NHS ODS API snapshot
        contracts_finder_buyers.json ← Contracts Finder buyer register snapshot
        _cache_meta.json             ← timestamps, source URLs, checksums

## Usage (when implemented)

    from reference_lookup import ReferenceStore, enrich_from_reference

    store = ReferenceStore.load(Path("data/reference"))
    enriched = enrich_from_reference(unresolved_df, store)
    # enriched contains: canonical_id, reference_source, reference_confidence columns

## Interface contract

    enrich_from_reference(unresolved_df, store) -> pd.DataFrame
        Input:  unresolved entities (canonical_name, address_region, alias_raw_ids, ...)
        Output: same columns + [reference_source, reference_confidence, reference_canonical_id]
            reference_source: "govuk_orgs" | "nhs_ods" | "contracts_finder" | None
            reference_confidence: "high" | "medium" | None
            reference_canonical_id: e.g. "GOV-MOD" or None if no match
        NEVER modifies alias_map or canonical_orgs directly.
        Always returns a plain DataFrame; caller decides whether to merge.
"""

import json
from pathlib import Path

import pandas as pd

from normalise import normalise_name


class ReferenceStore:
    """Holder for cached reference datasets.

    Currently a stub; will load from data/reference/ when implemented.
    """

    def __init__(self) -> None:
        self.govuk_orgs: dict = {}
        self.nhs_ods: dict = {}
        self.contracts_finder: dict = {}
        self._loaded = False

    @classmethod
    def load(cls, reference_dir: Path) -> "ReferenceStore":
        """Load cached reference data from reference_dir.

        If no cache exists, returns an empty store (no-op enrichment).
        """
        store = cls()
        govuk_path = reference_dir / "govuk_orgs.json"
        nhs_path = reference_dir / "nhs_ods.json"
        cf_path = reference_dir / "contracts_finder_buyers.json"

        if govuk_path.exists():
            import json
            with open(govuk_path, encoding="utf-8") as f:
                store.govuk_orgs = json.load(f)

        if nhs_path.exists():
            import json
            with open(nhs_path, encoding="utf-8") as f:
                store.nhs_ods = json.load(f)

        if cf_path.exists():
            import json
            with open(cf_path, encoding="utf-8") as f:
                store.contracts_finder = json.load(f)

        store._loaded = any([store.govuk_orgs, store.nhs_ods, store.contracts_finder])
        return store

    @property
    def is_empty(self) -> bool:
        return not self._loaded


def enrich_from_reference(
    unresolved_df: pd.DataFrame,
    store: ReferenceStore,
) -> pd.DataFrame:
    """Enrich entity rows with cached reference matches.

    This is read-only: it returns candidate reference IDs and evidence columns,
    but never mutates canonical_orgs or alias_map on disk.
    """
    result = unresolved_df.copy()
    for col in [
        "reference_source",
        "reference_confidence",
        "reference_canonical_id",
        "reference_matched_name",
        "reference_status",
    ]:
        result[col] = None

    if store.is_empty:
        return result

    govuk_index = _build_govuk_index(store.govuk_orgs)
    nhs_index = _build_nhs_index(store.nhs_ods)
    cf_index = _build_contracts_finder_index(store.contracts_finder)

    for idx, row in result.iterrows():
        names = _entity_names(row)
        raw_ids = _entity_raw_ids(row)

        match = _match_names(names, govuk_index)
        if match is None:
            match = _match_names(names, nhs_index)
        if match is None:
            match = _match_raw_ids(raw_ids, cf_index)
        if match is None:
            continue

        result.loc[idx, "reference_source"] = match["source"]
        result.loc[idx, "reference_confidence"] = match["confidence"]
        result.loc[idx, "reference_canonical_id"] = match["canonical_id"]
        result.loc[idx, "reference_matched_name"] = match["matched_name"]
        result.loc[idx, "reference_status"] = match.get("status", "")

    return result


def _safe_json_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(value or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _entity_names(row: pd.Series) -> list[str]:
    names = [row.get("canonical_name", "")]
    names.extend(_safe_json_list(row.get("alias_names")))
    out = []
    for name in names:
        if name is None or pd.isna(name):
            continue
        text = str(name).strip()
        if text:
            out.append(text)
    return list(dict.fromkeys(out))


def _entity_raw_ids(row: pd.Series) -> list[str]:
    raw_ids = _safe_json_list(row.get("alias_raw_ids"))
    cid = row.get("canonical_id", "")
    if isinstance(cid, str) and cid.startswith("GB-"):
        raw_ids.append(cid)
    return list(dict.fromkeys(str(r) for r in raw_ids if r))


def _slug_id(slug: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in str(slug).upper()).strip("-")
    while "--" in safe:
        safe = safe.replace("--", "-")
    return f"GOVUK-{safe}"


def _add_index_entry(index: dict[str, dict], norm_name: str, entry: dict) -> None:
    if not norm_name:
        return
    existing = index.get(norm_name)
    if existing is None:
        index[norm_name] = entry
        return
    existing_rank = 0 if str(existing.get("status", "")).lower() in {"active", "live"} else 1
    new_rank = 0 if str(entry.get("status", "")).lower() in {"active", "live"} else 1
    if new_rank < existing_rank:
        index[norm_name] = entry


def _build_govuk_index(records) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for rec in records or []:
        details = rec.get("details") or {}
        slug = details.get("slug") or (rec.get("web_url") or "").rstrip("/").split("/")[-1]
        status = details.get("govuk_status") or ""
        names = [
            rec.get("title"),
            details.get("abbreviation"),
            details.get("logo_formatted_name"),
        ]
        for name in names:
            norm = normalise_name(name)
            _add_index_entry(index, norm, {
                "canonical_id": _slug_id(slug or rec.get("title", "")),
                "matched_name": name,
                "source": "govuk_orgs",
                "confidence": "high" if str(status).lower() in {"live", "exempt"} else "medium",
                "status": status,
            })
    return index


def _build_nhs_index(records) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for rec in records or []:
        org_id = rec.get("OrgId") or rec.get("org_id")
        name = rec.get("Name") or rec.get("name")
        if not org_id or not name:
            continue
        status = rec.get("Status") or rec.get("status") or ""
        _add_index_entry(index, normalise_name(name), {
            "canonical_id": f"GB-NHS-{str(org_id).upper()}",
            "matched_name": name,
            "source": "nhs_ods",
            "confidence": "high" if str(status).lower() == "active" else "medium",
            "status": status,
        })
    return index


def _build_contracts_finder_index(records) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for rec in records or []:
        raw_id = rec.get("raw_id")
        if not raw_id:
            continue
        index[str(raw_id)] = {
            "canonical_id": f"CF-BUYER-{str(raw_id).replace('GB-FTS-', '')}",
            "matched_name": rec.get("canonical_name") or raw_id,
            "source": "contracts_finder",
            "confidence": "medium",
            "status": f"{rec.get('n_releases', 0)} releases",
        }
    return index


def _match_names(names: list[str], index: dict[str, dict]) -> dict | None:
    for name in names:
        norm = normalise_name(name)
        if norm in index:
            return index[norm]
    return None


def _match_raw_ids(raw_ids: list[str], index: dict[str, dict]) -> dict | None:
    for raw_id in raw_ids:
        if raw_id in index:
            return index[raw_id]
    return None


# Compatibility re-export. New code should import from
# procurement_graph.reference.lookup; old flat imports keep working.
from procurement_graph.reference.lookup import (  # noqa: E402,F401
    ReferenceStore,
    enrich_from_reference,
)
