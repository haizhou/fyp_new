"""
Stage 2a: Deterministic Entity Resolution using official identifiers.

Reads data/interim/releases.parquet.
Produces:
  data/entities/canonical_orgs.parquet  (initial, official-ID-only entries)
  data/entities/alias_map.parquet       (initial, official+FTS aliases for resolved entities)

Resolution rules (in order):
  1. Any party with an official scheme (GB-COH, GB-NHS, GB-UKPRN, GB-CHC, GB-SC, GB-NIC, GB-MPR)
     gets that as their canonical_id.
  2. When a single OCID has both an official-scheme party and a GB-FTS party playing the
     same role with matching normalised name, the GB-FTS ID is aliased to the canonical_id.
  3. GB-FTS-only entities (no matching official ID found) are marked er_status='unresolved'
     and deferred to Phase 2.

Import rules: this module imports ONLY from normalise.py (no kg_*.py, no er_phase2.py).
"""

import json
import logging
from pathlib import Path

import pandas as pd

from procurement_graph.common.normalise import (
    canonical_id_from_raw,
    is_fts,
    is_official,
    normalise_name,
    priority,
    scheme_of,
    value_of,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Collect all party observations from all releases
# ---------------------------------------------------------------------------


def collect_parties(releases_df: pd.DataFrame) -> pd.DataFrame:
    """Explode parties_json into a flat party-observation table.

    Returns one row per (ocid, raw_id) observation.
    Multiple rows for the same raw_id across different OCIDs are expected
    and are aggregated later.

    Columns returned:
        ocid, raw_id, name, roles_json, identifier_scheme, identifier_id,
        identifier_legal_name, address_region, details_url, org_category
    """
    rows = []
    for _, release in releases_df.iterrows():
        ocid = release["ocid"]
        parties = json.loads(release["parties_json"] or "[]")
        for p in parties:
            raw_id = p.get("raw_id") or ""
            if not raw_id:
                continue
            rows.append(
                {
                    "ocid": ocid,
                    "raw_id": raw_id,
                    "name": p.get("name") or "",
                    "roles_json": p.get("roles") or "[]",
                    "identifier_scheme": p.get("identifier_scheme") or "",
                    "identifier_id": p.get("identifier_id") or "",
                    "identifier_legal_name": p.get("identifier_legal_name") or "",
                    "address_region": p.get("address_region") or "",
                    "details_url": p.get("details_url") or "",
                    "org_category": p.get("org_category") or "",
                }
            )

    # Also collect supplier raw_ids from contracts_json
    # (suppliers appear in awards, not always as full party entries)
    for _, release in releases_df.iterrows():
        ocid = release["ocid"]
        contracts = json.loads(release["contracts_json"] or "[]")
        for c in contracts:
            for sid in c.get("supplier_raw_ids") or []:
                if sid:
                    rows.append(
                        {
                            "ocid": ocid,
                            "raw_id": sid,
                            "name": "",  # name not available here; will be enriched below
                            "roles_json": '["supplier"]',
                            "identifier_scheme": "",
                            "identifier_id": "",
                            "identifier_legal_name": "",
                            "address_region": "",
                            "details_url": "",
                            "org_category": "",
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ocid", "raw_id", "name", "roles_json", "identifier_scheme",
                "identifier_id", "identifier_legal_name", "address_region",
                "details_url", "org_category",
            ]
        )

    df = pd.DataFrame(rows)
    # Add derived columns
    df["norm_name"] = df["name"].apply(normalise_name)
    df["scheme"] = df["raw_id"].apply(scheme_of)
    # Enrich scheme/id from identifier fields when raw_id scheme is FTS but identifier has official
    mask = df["scheme"].apply(is_fts) & df["identifier_scheme"].apply(is_official)
    df.loc[mask, "scheme"] = df.loc[mask, "identifier_scheme"]
    return df


# ---------------------------------------------------------------------------
# Step 2: Assign canonical_id to officially-identified parties
# ---------------------------------------------------------------------------


def _best_canonical_for_raw_id(group: pd.DataFrame) -> str | None:
    """Given all observations of a raw_id, return its canonical_id if official.

    Prefers the highest-priority official scheme seen across all observations.
    Returns None if the raw_id is only ever seen with GB-FTS scheme.
    """
    # Sort by scheme priority ascending (1 = best)
    official_rows = group[group["scheme"].apply(is_official)].copy()
    if official_rows.empty:
        return None
    official_rows["prio"] = official_rows["scheme"].apply(priority)
    best = official_rows.sort_values("prio").iloc[0]
    # Build canonical_id from the raw_id in that row (which should already have official scheme)
    raw = best["raw_id"]
    cid = canonical_id_from_raw(raw)
    if cid:
        return cid
    # Fallback: build from identifier fields if raw_id scheme differs
    if is_official(best["identifier_scheme"]) and best["identifier_id"]:
        from procurement_graph.common.normalise import normalise_org_id
        return normalise_org_id(best["identifier_scheme"], best["identifier_id"])
    return None


# ---------------------------------------------------------------------------
# Step 3: Cross-reference — link FTS aliases within the same OCID
# ---------------------------------------------------------------------------


def _cross_ref_fts_aliases(
    parties_df: pd.DataFrame,
    official_map: dict[str, str],  # raw_id → canonical_id for officially resolved
) -> dict[str, str]:
    """Find GB-FTS IDs that co-occur with an official-scheme ID in the same OCID.

    If, within a single OCID, a GB-FTS party and an official-scheme party share
    the same role AND the same normalised name, the GB-FTS ID is an alias.

    Returns a dict of additional fts_raw_id → canonical_id mappings.
    """
    extra: dict[str, str] = {}

    # Build per-OCID lookup: norm_name+role → canonical_id
    # Only include parties that are already officially resolved
    resolved = parties_df[parties_df["raw_id"].isin(official_map)].copy()
    resolved["canonical_id"] = resolved["raw_id"].map(official_map)

    # Group by (ocid, norm_name) → set of canonical_ids
    ocid_name_to_cid: dict[tuple, str] = {}
    for _, row in resolved.iterrows():
        key = (row["ocid"], row["norm_name"])
        if key not in ocid_name_to_cid:
            ocid_name_to_cid[key] = row["canonical_id"]

    # Find FTS parties whose (ocid, norm_name) matches a resolved entity
    fts_parties = parties_df[
        parties_df["scheme"].apply(is_fts) & ~parties_df["raw_id"].isin(official_map)
    ]
    for _, row in fts_parties.iterrows():
        if not row["norm_name"]:
            continue
        key = (row["ocid"], row["norm_name"])
        cid = ocid_name_to_cid.get(key)
        if cid and row["raw_id"] not in extra:
            extra[row["raw_id"]] = cid

    return extra


# ---------------------------------------------------------------------------
# Step 4: Build canonical_orgs and alias_map DataFrames
# ---------------------------------------------------------------------------


def _aggregate_org_metadata(
    parties_df: pd.DataFrame, canonical_id: str, raw_ids: list[str]
) -> dict:
    """Aggregate metadata for all raw_ids that map to a single canonical_id."""
    subset = parties_df[parties_df["raw_id"].isin(raw_ids)]

    # Collect all names seen (non-empty)
    names_seen = subset["name"].dropna().unique().tolist()
    names_seen = [n for n in names_seen if n]

    # Best display name: prefer identifier_legal_name, else most common name
    legal_names = subset["identifier_legal_name"].dropna()
    legal_names = [n for n in legal_names if n]
    if legal_names:
        canonical_name = max(set(legal_names), key=legal_names.count)
    elif names_seen:
        canonical_name = max(set(names_seen), key=names_seen.count)
    else:
        canonical_name = canonical_id

    # Roles
    all_roles: set[str] = set()
    for r in subset["roles_json"].dropna():
        try:
            all_roles.update(json.loads(r))
        except Exception:
            pass
    if "buyer" in all_roles and "supplier" in all_roles:
        org_type = "both"
    elif "buyer" in all_roles:
        org_type = "buyer"
    elif "supplier" in all_roles:
        org_type = "supplier"
    else:
        org_type = "unknown"

    # Region: most common non-empty
    regions = subset["address_region"].dropna()
    regions = [r for r in regions if r]
    address_region = max(set(regions), key=regions.count) if regions else ""

    # Org category
    cats = subset["org_category"].dropna()
    cats = [c for c in cats if c]
    org_category = max(set(cats), key=cats.count) if cats else ""

    return {
        "canonical_id": canonical_id,
        "canonical_name": canonical_name,
        "org_type": org_type,
        "address_region": address_region,
        "org_category": org_category,
        "alias_raw_ids": json.dumps(sorted(set(raw_ids))),
        "alias_names": json.dumps(sorted(set(names_seen))),
        "n_aliases": len(set(raw_ids)),
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def resolve_phase1(
    releases_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run deterministic entity resolution (Phase 1) on the releases DataFrame.

    Returns:
        canonical_orgs_df: one row per canonical entity (official + cross-ref resolved)
        alias_map_df:      many-to-one mapping raw_id → canonical_id (for ALL raw_ids,
                           including unresolved GB-FTS singletons with er_status='unresolved')
    """
    print("  Collecting party observations ...")
    parties_df = collect_parties(releases_df)
    print(f"  {len(parties_df):,} party observations from {parties_df['ocid'].nunique():,} OCIDs")
    print(f"  {parties_df['raw_id'].nunique():,} unique raw_ids")

    # --- Official ID resolution ---
    print("  Assigning canonical IDs from official schemes ...")
    # Group by raw_id and find best canonical_id
    official_map: dict[str, str] = {}  # raw_id → canonical_id
    for raw_id, grp in parties_df.groupby("raw_id"):
        cid = _best_canonical_for_raw_id(grp)
        if cid:
            official_map[raw_id] = cid

    print(f"  {len(official_map):,} raw_ids resolved via official scheme")

    # Invert: canonical_id → set of official raw_ids
    cid_to_raw: dict[str, set[str]] = {}
    for raw_id, cid in official_map.items():
        cid_to_raw.setdefault(cid, set()).add(raw_id)

    # --- Cross-reference: FTS aliases within same OCID ---
    print("  Cross-referencing GB-FTS aliases within same OCID ...")
    extra_aliases = _cross_ref_fts_aliases(parties_df, official_map)
    print(f"  {len(extra_aliases):,} GB-FTS IDs linked via OCID cross-reference")

    # Merge extra aliases into maps
    for raw_id, cid in extra_aliases.items():
        official_map[raw_id] = cid
        cid_to_raw.setdefault(cid, set()).add(raw_id)

    # --- Build canonical_orgs for officially resolved entities ---
    print("  Building canonical_orgs table ...")
    org_rows = []
    for cid, raw_ids in cid_to_raw.items():
        meta = _aggregate_org_metadata(parties_df, cid, list(raw_ids))
        # Determine official_scheme and official_id from the canonical_id
        parts = cid.split("-", 2)
        if len(parts) >= 3:
            meta["official_scheme"] = f"{parts[0]}-{parts[1]}"
            meta["official_id"] = parts[2]
        else:
            meta["official_scheme"] = ""
            meta["official_id"] = ""
        meta["er_status"] = "deterministic"
        org_rows.append(meta)

    # --- Identify unresolved GB-FTS-only raw_ids ---
    all_raw_ids = set(parties_df["raw_id"].unique())
    resolved_raw_ids = set(official_map.keys())
    unresolved_raw_ids = all_raw_ids - resolved_raw_ids

    print(f"  {len(unresolved_raw_ids):,} raw_ids unresolved (GB-FTS-only, deferred to Phase 2)")

    # Unresolved entities: each keeps its own raw_id as a provisional canonical_id
    for raw_id in unresolved_raw_ids:
        grp = parties_df[parties_df["raw_id"] == raw_id]
        meta = _aggregate_org_metadata(parties_df, raw_id, [raw_id])
        meta["official_scheme"] = ""
        meta["official_id"] = ""
        meta["er_status"] = "unresolved"
        org_rows.append(meta)

    canonical_orgs_df = pd.DataFrame(org_rows)

    # --- Build alias_map: every raw_id → its canonical_id ---
    alias_rows = []
    # Resolved (official + cross-ref)
    for raw_id, cid in official_map.items():
        alias_rows.append({"raw_id": raw_id, "canonical_id": cid, "alias_source": "phase1"})
    # Unresolved (self-alias)
    for raw_id in unresolved_raw_ids:
        alias_rows.append({"raw_id": raw_id, "canonical_id": raw_id, "alias_source": "unresolved"})

    alias_map_df = pd.DataFrame(alias_rows)

    return canonical_orgs_df, alias_map_df


def write_entities(
    canonical_orgs: pd.DataFrame,
    alias_map: pd.DataFrame,
    entities_dir: Path,
) -> None:
    """Write canonical_orgs.parquet and alias_map.parquet to entities_dir."""
    entities_dir.mkdir(parents=True, exist_ok=True)

    orgs_path = entities_dir / "canonical_orgs.parquet"
    alias_path = entities_dir / "alias_map.parquet"

    canonical_orgs.to_parquet(orgs_path, index=False, compression="snappy")
    alias_map.to_parquet(alias_path, index=False, compression="snappy")

    print(f"  Written: {orgs_path}  ({len(canonical_orgs):,} entities)")
    print(f"  Written: {alias_path}  ({len(alias_map):,} alias entries)")


def load_canonical_orgs(entities_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(entities_dir / "canonical_orgs.parquet")


def load_alias_map(entities_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(entities_dir / "alias_map.parquet")
