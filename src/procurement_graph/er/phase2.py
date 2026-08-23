"""
Stage 2b: Heuristic Entity Resolution for GB-FTS-only (unresolved) entities.

Reads the outputs of Phase 1 and applies two conservative merge strategies:

  Step A — Government entity lookup:
    Matches normalised name against configs/gov_lookup.json.
    Assigns a synthetic GOV-* canonical_id.
    Evidence is documented; every merge is auditable.

  Step B — Exact normalised-name + region merge:
    Groups unresolved entities by (normalised_name, address_region).
    If >1 raw_id share the same group → they are aliased together.
    A synthetic MERGED-* canonical_id is created from a hash.
    Falls back to name-only merge (without region) for entities with no region.

  Step C — Singleton fallback:
    Remaining unresolved entities with unique names keep their GB-FTS raw_id
    as provisional canonical_id (er_status='singleton').

After Phase 2, every entity in canonical_orgs has a final canonical_id.
The alias_map is extended with new mappings.
An audit log (er_audit.csv) is written documenting every decision.

Import rules: imports ONLY from normalise.py. Does NOT import from kg_*.py or er_phase1.py
(reads their parquet outputs instead).
"""

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from procurement_graph.common.normalise import normalise_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step A: Government lookup
# ---------------------------------------------------------------------------


def load_gov_lookup(path: Path) -> dict[str, dict]:
    """Load configs/gov_lookup.json.

    Returns a dict: normalised_name → {canonical_id, _source}.
    Skips the _meta key.
    """
    import json
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for name, val in raw.items():
        if name.startswith("_"):
            continue
        if isinstance(val, dict) and "canonical_id" in val:
            result[name] = val
    return result


def apply_gov_lookup(
    unresolved_df: pd.DataFrame,
    gov_lookup: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply government lookup to unresolved entities.

    Args:
        unresolved_df: subset of canonical_orgs where er_status == 'unresolved'
        gov_lookup: dict from load_gov_lookup()

    Returns:
        (matched_df, remaining_df) — both with updated er_status and canonical_id.
        matched_df: entities resolved by gov_lookup (er_status='gov_lookup')
        remaining_df: entities not matched (er_status still 'unresolved')
    """
    if unresolved_df.empty:
        return pd.DataFrame(columns=unresolved_df.columns), unresolved_df.copy()

    df = unresolved_df.copy()
    df["_norm_name"] = df["canonical_name"].fillna("").apply(normalise_name)

    matched_rows = []
    remaining_rows = []

    for _, row in df.iterrows():
        norm = row["_norm_name"]
        if norm in gov_lookup:
            entry = gov_lookup[norm]
            row = row.copy()
            row["canonical_id"] = entry["canonical_id"]
            row["er_status"] = "gov_lookup"
            row["_gov_source"] = entry.get("_source", "")
            matched_rows.append(row)
        else:
            remaining_rows.append(row)

    matched_df = pd.DataFrame(matched_rows).drop(columns=["_norm_name"], errors="ignore")
    remaining_df = pd.DataFrame(remaining_rows).drop(columns=["_norm_name"], errors="ignore")
    return matched_df, remaining_df


# ---------------------------------------------------------------------------
# Step B: Exact name + region merge
# ---------------------------------------------------------------------------


def _make_merged_id(norm_name: str, region: str = "") -> str:
    """Create a stable synthetic canonical_id for a name-based merge group."""
    key = f"{norm_name}|{region}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()[:12]
    return f"MERGED-{h}"


def merge_by_name_region(
    unresolved_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group unresolved entities by (norm_name, region) and merge groups of size > 1.

    Entities with no region are grouped by name alone.
    Singletons (no other entity shares name+region) remain unresolved.

    Returns:
        (merged_df, singleton_df)
    """
    if unresolved_df.empty:
        return pd.DataFrame(columns=unresolved_df.columns), pd.DataFrame(columns=unresolved_df.columns)

    df = unresolved_df.copy()
    df["_norm_name"] = df["canonical_name"].fillna("").apply(normalise_name)
    df["_region"] = df["address_region"].fillna("").str.strip()

    # Build group key: prefer (norm_name, region) if region exists, else (norm_name, "")
    df["_group_key"] = df.apply(
        lambda r: (r["_norm_name"], r["_region"]) if r["_region"] else (r["_norm_name"], ""),
        axis=1,
    )

    # Empty names cannot be merged safely, but they are still real entities.
    # Preserve them as singletons instead of dropping them from Phase 2 output.
    blank_name_rows = [row for _, row in df[df["_norm_name"] == ""].iterrows()]
    df = df[df["_norm_name"] != ""]

    group_sizes = df.groupby("_group_key")["canonical_id"].count()
    merge_groups = group_sizes[group_sizes > 1].index

    merged_rows = []
    singleton_rows = blank_name_rows

    for _, row in df.iterrows():
        key = row["_group_key"]
        if key in merge_groups:
            row = row.copy()
            # Assign MERGED-* id based on norm_name + region
            row["canonical_id"] = _make_merged_id(key[0], key[1])
            row["er_status"] = "name_region_merge" if key[1] else "name_only_merge"
            merged_rows.append(row)
        else:
            singleton_rows.append(row)

    merged_df = pd.DataFrame(merged_rows).drop(
        columns=["_norm_name", "_region", "_group_key"], errors="ignore"
    )
    singleton_df = pd.DataFrame(singleton_rows).drop(
        columns=["_norm_name", "_region", "_group_key"], errors="ignore"
    )
    return merged_df, singleton_df


# ---------------------------------------------------------------------------
# Step C: Consolidate merged groups
# ---------------------------------------------------------------------------


def consolidate_merged_groups(merged_df: pd.DataFrame) -> pd.DataFrame:
    """For each MERGED-* canonical_id, produce a single canonical_orgs row.

    All raw_ids that share the same new canonical_id are aggregated.
    """
    if merged_df.empty:
        return pd.DataFrame()

    consolidated = []
    for new_cid, grp in merged_df.groupby("canonical_id"):
        # Collect all original alias raw_ids across the merged group
        all_raw_ids: list[str] = []
        all_names: list[str] = []
        for _, row in grp.iterrows():
            try:
                all_raw_ids.extend(json.loads(row["alias_raw_ids"] or "[]"))
            except Exception:
                all_raw_ids.append(row.get("canonical_id", ""))
            try:
                all_names.extend(json.loads(row["alias_names"] or "[]"))
            except Exception:
                all_names.append(row.get("canonical_name", ""))

        # Best display name: most common non-empty
        all_names = [n for n in all_names if n]
        best_name = max(set(all_names), key=all_names.count) if all_names else new_cid

        # Best region
        regions = [r for r in grp["address_region"].dropna() if r]
        region = max(set(regions), key=regions.count) if regions else ""

        # Org category
        cats = [c for c in grp["org_category"].dropna() if c]
        org_category = max(set(cats), key=cats.count) if cats else ""

        # Org type: aggregate
        types = set(grp["org_type"].dropna())
        if "both" in types or ("buyer" in types and "supplier" in types):
            org_type = "both"
        elif "buyer" in types:
            org_type = "buyer"
        elif "supplier" in types:
            org_type = "supplier"
        else:
            org_type = "unknown"

        er_status = grp["er_status"].iloc[0]

        consolidated.append(
            {
                "canonical_id": new_cid,
                "canonical_name": best_name,
                "org_type": org_type,
                "address_region": region,
                "org_category": org_category,
                "official_scheme": "",
                "official_id": "",
                "alias_raw_ids": json.dumps(sorted(set(all_raw_ids))),
                "alias_names": json.dumps(sorted(set(all_names))),
                "n_aliases": len(set(all_raw_ids)),
                "er_status": er_status,
            }
        )
    return pd.DataFrame(consolidated)


# ---------------------------------------------------------------------------
# Audit log builder
# ---------------------------------------------------------------------------


def build_audit_log(
    canonical_orgs: pd.DataFrame,
    alias_map: pd.DataFrame,
) -> pd.DataFrame:
    """Produce one audit row per canonical entity.

    Documents: canonical_id, er_status, n_aliases, evidence (raw_ids), canonical_name.
    """
    aliases_by_cid = (
        alias_map.groupby("canonical_id")["raw_id"]
        .apply(lambda values: sorted(str(v) for v in values if v))
        .to_dict()
    )

    rows = []
    for _, org in canonical_orgs.iterrows():
        cid = org["canonical_id"]
        aliases_in_map = aliases_by_cid.get(cid, [])
        rows.append(
            {
                "canonical_id": cid,
                "canonical_name": org.get("canonical_name", ""),
                "er_status": org.get("er_status", ""),
                "official_scheme": org.get("official_scheme", ""),
                "n_aliases": len(aliases_in_map),
                "alias_raw_ids": "|".join(sorted(aliases_in_map)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Consolidate gov_lookup groups (mirror of consolidate_merged_groups)
# ---------------------------------------------------------------------------


def _consolidate_gov_groups(gov_matched: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple raw entities with the same GOV-* canonical_id into one row.

    Preserves _gov_source (the first non-empty value per group).
    All alias_raw_ids and alias_names are merged across source entities.
    """
    if gov_matched.empty:
        return pd.DataFrame(columns=gov_matched.columns)

    consolidated = []
    for new_cid, grp in gov_matched.groupby("canonical_id"):
        all_raw_ids: list[str] = []
        all_names: list[str] = []
        for _, row in grp.iterrows():
            try:
                all_raw_ids.extend(json.loads(row.get("alias_raw_ids") or "[]"))
            except Exception:
                pass
            try:
                all_names.extend(json.loads(row.get("alias_names") or "[]"))
            except Exception:
                pass

        all_names = [n for n in all_names if n]
        best_name = max(set(all_names), key=all_names.count) if all_names else new_cid

        regions = [r for r in grp["address_region"].dropna() if r]
        region = max(set(regions), key=regions.count) if regions else ""

        cats = [c for c in grp["org_category"].dropna() if c]
        org_category = max(set(cats), key=cats.count) if cats else ""

        types = set(grp["org_type"].dropna())
        if "both" in types or ("buyer" in types and "supplier" in types):
            org_type = "both"
        elif "buyer" in types:
            org_type = "buyer"
        elif "supplier" in types:
            org_type = "supplier"
        else:
            org_type = "unknown"

        gov_sources = [s for s in grp["_gov_source"].dropna() if s]
        gov_source = gov_sources[0] if gov_sources else ""

        consolidated.append({
            "canonical_id": new_cid,
            "canonical_name": best_name,
            "org_type": org_type,
            "address_region": region,
            "org_category": org_category,
            "official_scheme": "",
            "official_id": "",
            "alias_raw_ids": json.dumps(sorted(set(all_raw_ids))),
            "alias_names": json.dumps(sorted(set(all_names))),
            "n_aliases": len(set(all_raw_ids)),
            "er_status": "gov_lookup",
            "_gov_source": gov_source,
        })
    return pd.DataFrame(consolidated)


def _json_list(value) -> list:
    """Parse a JSON list field defensively."""
    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(value or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _non_empty_values(values) -> list[str]:
    """Return non-empty string values, skipping nulls."""
    cleaned: list[str] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _modal(values: list[str], default: str = "") -> str:
    """Most common value, preserving deterministic behaviour for ties."""
    if not values:
        return default
    counts = pd.Series(values).value_counts()
    return str(counts.index[0])


def _aggregate_org_type(values) -> str:
    types = set(_non_empty_values(values))
    if "both" in types or ("buyer" in types and "supplier" in types):
        return "both"
    if "buyer" in types:
        return "buyer"
    if "supplier" in types:
        return "supplier"
    return "unknown"


def consolidate_duplicate_canonical_orgs(canonical_orgs: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate canonical_id rows into one canonical_orgs row each.

    Phase 2 should be idempotent: rerunning it on already-processed outputs must
    not preserve an older bug where gov_lookup rows were written once per raw
    GB-FTS alias. This function repairs any duplicate canonical_id group by
    aggregating alias ids/names and keeping the strongest available metadata.
    """
    if canonical_orgs.empty or "canonical_id" not in canonical_orgs.columns:
        return canonical_orgs.copy()

    if canonical_orgs["canonical_id"].duplicated().sum() == 0:
        return canonical_orgs.copy()

    rows = []
    for cid, grp in canonical_orgs.groupby("canonical_id", sort=False):
        if len(grp) == 1:
            rows.append(grp.iloc[0].to_dict())
            continue

        all_raw_ids: list[str] = []
        all_names: list[str] = []
        for _, row in grp.iterrows():
            raw_ids = _json_list(row.get("alias_raw_ids"))
            names = _json_list(row.get("alias_names"))
            all_raw_ids.extend(str(v) for v in raw_ids if v)
            all_names.extend(str(v) for v in names if v)
            name = row.get("canonical_name", "")
            if name is not None and not pd.isna(name) and str(name).strip():
                all_names.append(str(name).strip())

        raw_ids_unique = sorted(set(all_raw_ids))
        names_unique = sorted(set(all_names))

        statuses = _non_empty_values(grp.get("er_status", pd.Series(dtype=str)))
        status_priority = [
            "deterministic", "gov_lookup", "name_region_merge",
            "name_only_merge", "singleton", "unresolved",
        ]
        er_status = next((s for s in status_priority if s in statuses), _modal(statuses))

        official_schemes = _non_empty_values(grp.get("official_scheme", pd.Series(dtype=str)))
        official_ids = _non_empty_values(grp.get("official_id", pd.Series(dtype=str)))
        gov_sources = _non_empty_values(grp.get("_gov_source", pd.Series(dtype=str)))

        rows.append({
            "canonical_id": cid,
            "canonical_name": _modal(all_names, default=str(cid)),
            "org_type": _aggregate_org_type(grp.get("org_type", pd.Series(dtype=str))),
            "address_region": _modal(_non_empty_values(grp.get("address_region", pd.Series(dtype=str)))),
            "org_category": _modal(_non_empty_values(grp.get("org_category", pd.Series(dtype=str)))),
            "alias_raw_ids": json.dumps(raw_ids_unique),
            "alias_names": json.dumps(names_unique),
            "n_aliases": len(raw_ids_unique),
            "official_scheme": official_schemes[0] if official_schemes else "",
            "official_id": official_ids[0] if official_ids else "",
            "er_status": er_status,
            "_gov_source": gov_sources[0] if gov_sources else "",
        })

    result = pd.DataFrame(rows)
    # Preserve the original column order, including any columns not explicitly
    # reconstructed above.
    for col in canonical_orgs.columns:
        if col not in result.columns:
            result[col] = ""
    return result[list(canonical_orgs.columns)]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def resolve_phase2(
    canonical_orgs: pd.DataFrame,
    alias_map: pd.DataFrame,
    gov_lookup_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Phase 2 heuristic resolution on unresolved entities.

    Args:
        canonical_orgs: output of Phase 1 (contains 'unresolved' rows)
        alias_map: output of Phase 1
        gov_lookup_path: path to configs/gov_lookup.json

    Returns:
        (updated_canonical_orgs, updated_alias_map, audit_log)
    """
    gov_lookup = load_gov_lookup(gov_lookup_path)

    unresolved_mask = canonical_orgs["er_status"] == "unresolved"
    resolved_part = canonical_orgs[~unresolved_mask].copy()
    unresolved_part = canonical_orgs[unresolved_mask].copy()

    total_unresolved = len(unresolved_part)
    print(f"  Phase 2 input: {total_unresolved:,} unresolved entities")

    before_resolved = len(resolved_part)
    resolved_part = consolidate_duplicate_canonical_orgs(resolved_part)
    collapsed_resolved = before_resolved - len(resolved_part)
    if collapsed_resolved:
        print(
            f"  Repaired existing duplicate canonical rows: "
            f"{before_resolved:,} 鈫?{len(resolved_part):,} "
            f"({collapsed_resolved:,} duplicate rows collapsed)"
        )

    # Step A: Government lookup
    print(f"  Step A: government lookup ({len(gov_lookup)} entries) ...")
    gov_matched, remaining = apply_gov_lookup(unresolved_part, gov_lookup)
    print(f"    {len(gov_matched):,} raw entities resolved via gov_lookup")

    # Consolidate gov_lookup rows: multiple raw entities may share the same GOV-* id.
    # Each must be collapsed into a single canonical row (same as MERGED-* groups).
    gov_consolidated = _consolidate_gov_groups(gov_matched)
    print(f"    {len(gov_matched):,} raw entities → {len(gov_consolidated):,} canonical gov entities")

    # Step B: Name + region merge
    print("  Step B: exact name + region merge ...")
    merged, singletons = merge_by_name_region(remaining)

    # Consolidate merged groups into single rows
    consolidated = consolidate_merged_groups(merged)
    n_merged_groups = len(consolidated)
    n_merged_raw = len(merged)
    print(f"    {n_merged_raw:,} raw entities merged into {n_merged_groups:,} canonical entities")
    print(f"    {len(singletons):,} singletons (no merge match)")

    # Singletons: er_status = 'singleton'
    singletons = singletons.copy()
    if len(singletons):
        singletons["er_status"] = "singleton"

    # --- Assemble updated canonical_orgs ---
    frames = [resolved_part, gov_consolidated]
    if len(consolidated):
        frames.append(consolidated)
    if len(singletons):
        frames.append(singletons)
    updated_orgs = pd.concat(frames, ignore_index=True)
    updated_orgs = consolidate_duplicate_canonical_orgs(updated_orgs)

    # --- Extend alias_map with new mappings ---
    # Build old_canonical_id → new_canonical_id for gov_lookup and merged entities
    old_to_new: dict[str, str] = {}

    for _, row in gov_matched.iterrows():
        # The gov_lookup match row was originally a singleton with canonical_id = its raw_id
        # We need to find all alias_raw_ids that mapped to this old canonical_id
        old_cid = row.get("_old_canonical_id", "")
        if not old_cid:
            # The old canonical_id was the raw_id itself (self-alias)
            # canonical_name is the name; we need to find original raw_ids
            # by looking at alias_raw_ids stored in the row
            try:
                raw_ids = json.loads(row.get("alias_raw_ids") or "[]")
            except Exception:
                raw_ids = []
            for rid in raw_ids:
                old_to_new[rid] = row["canonical_id"]

    for _, row in merged.iterrows():
        try:
            raw_ids = json.loads(row.get("alias_raw_ids") or "[]")
        except Exception:
            raw_ids = []
        new_cid = row["canonical_id"]
        for rid in raw_ids:
            old_to_new[rid] = new_cid

    # Also add gov_matched via alias_raw_ids
    for _, row in gov_matched.iterrows():
        try:
            raw_ids = json.loads(row.get("alias_raw_ids") or "[]")
        except Exception:
            raw_ids = []
        new_cid = row["canonical_id"]
        for rid in raw_ids:
            old_to_new[rid] = new_cid

    # Update alias_map: remap alias_source for updated entries
    updated_alias_map = alias_map.copy()
    new_rows = []

    for raw_id, new_cid in old_to_new.items():
        # Check if this raw_id already appears in alias_map
        existing = updated_alias_map[updated_alias_map["raw_id"] == raw_id]
        if len(existing):
            updated_alias_map.loc[existing.index, "canonical_id"] = new_cid
            updated_alias_map.loc[existing.index, "alias_source"] = "phase2"
        else:
            new_rows.append({"raw_id": raw_id, "canonical_id": new_cid, "alias_source": "phase2"})

    if new_rows:
        updated_alias_map = pd.concat(
            [updated_alias_map, pd.DataFrame(new_rows)], ignore_index=True
        )

    # Build audit log
    audit_log = build_audit_log(updated_orgs, updated_alias_map)

    return updated_orgs, updated_alias_map, audit_log


def write_phase2_outputs(
    canonical_orgs: pd.DataFrame,
    alias_map: pd.DataFrame,
    audit_log: pd.DataFrame,
    entities_dir: Path,
) -> None:
    """Overwrite canonical_orgs.parquet, alias_map.parquet; write er_audit.csv."""
    entities_dir.mkdir(parents=True, exist_ok=True)

    orgs_path = entities_dir / "canonical_orgs.parquet"
    alias_path = entities_dir / "alias_map.parquet"
    audit_path = entities_dir / "er_audit.csv"

    # Normalise n_aliases dtype — pd.concat of mixed frames can produce object
    canonical_orgs = canonical_orgs.copy()
    canonical_orgs["n_aliases"] = (
        pd.to_numeric(canonical_orgs["n_aliases"], errors="coerce")
        .fillna(0).astype(int)
    )

    canonical_orgs.to_parquet(orgs_path, index=False, compression="snappy")
    alias_map.to_parquet(alias_path, index=False, compression="snappy")
    audit_log.to_csv(audit_path, index=False)

    print(f"  Updated: {orgs_path}  ({len(canonical_orgs):,} entities)")
    print(f"  Updated: {alias_path}  ({len(alias_map):,} alias entries)")
    print(f"  Written: {audit_path}  ({len(audit_log):,} audit rows)")
