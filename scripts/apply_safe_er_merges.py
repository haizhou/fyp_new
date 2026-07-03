"""Apply cluster-checked ER merge edges to entity tables.

This script consumes `safe_merge_edges_after_cluster_check.parquet` and writes a
staged set of entity outputs. With `--apply`, it backs up the current entity
tables and replaces them with the staged outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from procurement_graph.common.normalise import is_fts, is_official, priority, scheme_of, value_of


ROOT = Path(__file__).resolve().parents[1]
ENTITIES_DIR = ROOT / "data" / "entities"
ABLATION_DIR = ROOT / "data" / "ablation" / "er"
REPORT_DIR = ROOT / "reports" / "ablation" / "er"


@dataclass
class UnionFind:
    parent: dict[str, str]

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[max(root_left, root_right)] = min(root_left, root_right)


def json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(value or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def stable_merged_id(members: list[str]) -> str:
    key = "|".join(sorted(members)).encode("utf-8")
    return f"MERGED-LLM-{hashlib.sha256(key).hexdigest()[:12]}"


def non_empty(values: Any) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return cleaned


def modal(values: list[str], default: str = "") -> str:
    if not values:
        return default
    counts = Counter(values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def aggregate_org_type(values: Any) -> str:
    types = set(non_empty(values))
    if "both" in types or {"buyer", "supplier"}.issubset(types):
        return "both"
    if "buyer" in types:
        return "buyer"
    if "supplier" in types:
        return "supplier"
    return "unknown"


def direct_official_key(canonical_id: str) -> tuple[int, str]:
    scheme = scheme_of(canonical_id)
    if is_official(scheme):
        return priority(scheme), canonical_id
    return 999, canonical_id


def choose_component_id(members: list[str], canonical: pd.DataFrame) -> str:
    rows = canonical[canonical["canonical_id"].isin(members)].copy()

    official_candidates: list[str] = []
    for cid in members:
        scheme = scheme_of(cid)
        if is_official(scheme):
            official_candidates.append(cid)
    if official_candidates:
        return sorted(official_candidates, key=direct_official_key)[0]

    gov_candidates = sorted(cid for cid in members if cid.startswith("GOV-"))
    if gov_candidates:
        return gov_candidates[0]

    merged_candidates = sorted(cid for cid in members if cid.startswith("MERGED-"))
    if merged_candidates:
        return merged_candidates[0]

    non_fts = sorted(cid for cid in members if not is_fts(scheme_of(cid)))
    if non_fts:
        return non_fts[0]

    return stable_merged_id(members)


def build_components(edges: pd.DataFrame) -> dict[str, list[str]]:
    uf = UnionFind(parent={})
    for row in edges.itertuples(index=False):
        left = str(getattr(row, "entity_a_id"))
        right = str(getattr(row, "entity_b_id"))
        uf.union(left, right)

    members_by_root: dict[str, set[str]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        for entity_id in [str(getattr(row, "entity_a_id")), str(getattr(row, "entity_b_id"))]:
            members_by_root[uf.find(entity_id)].add(entity_id)
    return {root: sorted(members) for root, members in members_by_root.items() if len(members) > 1}


def recover_missing_alias_canonicals(canonical: pd.DataFrame, alias_map: pd.DataFrame) -> pd.DataFrame:
    canonical_ids = set(canonical["canonical_id"].astype(str))
    missing = sorted(set(alias_map["canonical_id"].astype(str)) - canonical_ids)
    if not missing:
        return canonical
    rows = []
    for canonical_id in missing:
        rows.append({
            "canonical_id": canonical_id,
            "canonical_name": canonical_id,
            "org_type": "unknown",
            "address_region": "",
            "org_category": "",
            "alias_raw_ids": json.dumps([canonical_id]),
            "alias_names": json.dumps([canonical_id]),
            "n_aliases": 1,
            "official_scheme": scheme_of(canonical_id) if is_official(scheme_of(canonical_id)) else "",
            "official_id": value_of(canonical_id) if is_official(scheme_of(canonical_id)) else "",
            "er_status": "singleton",
            "_gov_source": "",
        })
    recovered = pd.DataFrame(rows)
    for column in canonical.columns:
        if column not in recovered.columns:
            recovered[column] = ""
    return pd.concat([canonical, recovered[canonical.columns]], ignore_index=True)


def aggregate_component_row(
    new_id: str,
    members: list[str],
    canonical: pd.DataFrame,
    alias_map: pd.DataFrame,
    component_sources: list[str],
) -> dict[str, Any]:
    grp = canonical[canonical["canonical_id"].isin(members)]
    aliases = alias_map[alias_map["canonical_id"].isin(members)]

    raw_ids: list[str] = []
    names: list[str] = []
    raw_ids.extend(aliases["raw_id"].astype(str).tolist())
    raw_ids.extend(members)
    for _, row in grp.iterrows():
        raw_ids.extend(str(value) for value in json_list(row.get("alias_raw_ids")) if value)
        names.extend(str(value) for value in json_list(row.get("alias_names")) if value)
        if str(row.get("canonical_name") or "").strip():
            names.append(str(row.get("canonical_name")).strip())

    chosen = grp[grp["canonical_id"].eq(new_id)]
    chosen_row = chosen.iloc[0].to_dict() if not chosen.empty else {}
    chosen_name = str(chosen_row.get("canonical_name") or "").strip()

    scheme = str(chosen_row.get("official_scheme") or "")
    official_id = str(chosen_row.get("official_id") or "")
    direct_scheme = scheme_of(new_id)
    if not scheme and is_official(direct_scheme):
        scheme = direct_scheme
        official_id = value_of(new_id)

    status = (
        "llm_safe_merge"
        if "llm" in set(component_sources)
        else "deterministic_safe_merge"
    )

    return {
        "canonical_id": new_id,
        "canonical_name": chosen_name or modal(names, default=new_id),
        "org_type": aggregate_org_type(grp.get("org_type", pd.Series(dtype=str))),
        "address_region": modal(non_empty(grp.get("address_region", pd.Series(dtype=str)))),
        "org_category": modal(non_empty(grp.get("org_category", pd.Series(dtype=str)))),
        "alias_raw_ids": json.dumps(sorted(set(raw_ids)), ensure_ascii=False),
        "alias_names": json.dumps(sorted(set(names)), ensure_ascii=False),
        "n_aliases": len(set(raw_ids)),
        "official_scheme": scheme,
        "official_id": official_id,
        "er_status": status,
        "_gov_source": modal(non_empty(grp.get("_gov_source", pd.Series(dtype=str)))),
    }


def build_merged_tables(
    canonical: pd.DataFrame,
    alias_map: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical = recover_missing_alias_canonicals(canonical.copy(), alias_map.copy())
    components = build_components(edges)

    source_by_component: dict[str, list[str]] = {}
    for root, members in components.items():
        member_set = set(members)
        component_edges = edges[
            edges["entity_a_id"].astype(str).isin(member_set)
            & edges["entity_b_id"].astype(str).isin(member_set)
        ]
        source_by_component[root] = sorted(set(component_edges["decision_source"].astype(str)))

    old_to_new: dict[str, str] = {}
    component_rows: list[dict[str, Any]] = []
    for root, members in sorted(components.items()):
        new_id = choose_component_id(members, canonical)
        for member in members:
            old_to_new[member] = new_id
        component_rows.append(
            {
                "component_root": root,
                "new_canonical_id": new_id,
                "component_size": len(members),
                "member_ids": json.dumps(members, ensure_ascii=False),
                "decision_sources": json.dumps(source_by_component[root], ensure_ascii=False),
            }
        )

    merged_member_ids = set(old_to_new)
    unchanged = canonical[~canonical["canonical_id"].astype(str).isin(merged_member_ids)].copy()
    merged_rows = [
        aggregate_component_row(
            row["new_canonical_id"],
            json_list(row["member_ids"]),
            canonical,
            alias_map,
            json_list(row["decision_sources"]),
        )
        for row in component_rows
    ]
    merged = pd.DataFrame(merged_rows)
    for column in canonical.columns:
        if column not in merged.columns:
            merged[column] = ""
    output_canonical = pd.concat([unchanged, merged[canonical.columns]], ignore_index=True)
    output_canonical["n_aliases"] = pd.to_numeric(output_canonical["n_aliases"], errors="coerce").fillna(0).astype(int)

    output_alias = alias_map.copy()
    output_alias["canonical_id"] = output_alias["canonical_id"].astype(str).map(lambda cid: old_to_new.get(cid, cid))
    changed = alias_map["canonical_id"].astype(str) != output_alias["canonical_id"].astype(str)
    output_alias.loc[changed, "alias_source"] = "er_llm_safe_merge"

    existing_raw = set(output_alias["raw_id"].astype(str))
    extra_alias_rows = []
    for old_id, new_id in old_to_new.items():
        if old_id in existing_raw:
            output_alias.loc[output_alias["raw_id"].astype(str).eq(old_id), "canonical_id"] = new_id
            output_alias.loc[output_alias["raw_id"].astype(str).eq(old_id), "alias_source"] = "er_llm_safe_merge"
        else:
            extra_alias_rows.append({"raw_id": old_id, "canonical_id": new_id, "alias_source": "er_llm_safe_merge"})
    if extra_alias_rows:
        output_alias = pd.concat([output_alias, pd.DataFrame(extra_alias_rows)], ignore_index=True)
    output_alias = output_alias.drop_duplicates(subset=["raw_id"], keep="last").reset_index(drop=True)

    audit = build_audit(output_canonical, output_alias)
    components_df = pd.DataFrame(component_rows)
    return output_canonical, output_alias, audit, components_df


def build_audit(canonical: pd.DataFrame, alias_map: pd.DataFrame) -> pd.DataFrame:
    aliases_by_cid = (
        alias_map.groupby("canonical_id")["raw_id"]
        .apply(lambda values: sorted(str(value) for value in values if str(value)))
        .to_dict()
    )
    rows = []
    for _, org in canonical.iterrows():
        cid = str(org["canonical_id"])
        aliases = aliases_by_cid.get(cid, [])
        rows.append(
            {
                "canonical_id": cid,
                "canonical_name": org.get("canonical_name", ""),
                "er_status": org.get("er_status", ""),
                "official_scheme": org.get("official_scheme", ""),
                "n_aliases": len(aliases),
                "alias_raw_ids": "|".join(aliases),
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(canonical: pd.DataFrame, alias_map: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if canonical["canonical_id"].isna().any():
        errors.append("canonical_orgs.canonical_id has nulls")
    if alias_map["raw_id"].isna().any():
        errors.append("alias_map.raw_id has nulls")
    if alias_map["canonical_id"].isna().any():
        errors.append("alias_map.canonical_id has nulls")
    dup_canonical = int(canonical["canonical_id"].duplicated().sum())
    dup_alias = int(alias_map["raw_id"].duplicated().sum())
    if dup_canonical:
        errors.append(f"canonical_orgs has {dup_canonical} duplicate canonical_id rows")
    if dup_alias:
        errors.append(f"alias_map has {dup_alias} duplicate raw_id rows")
    missing_alias_targets = sorted(set(alias_map["canonical_id"].astype(str)) - set(canonical["canonical_id"].astype(str)))
    if missing_alias_targets:
        errors.append(f"alias_map has {len(missing_alias_targets)} canonical_id values absent from canonical_orgs")
    fts_bad = canonical[
        canonical["canonical_id"].astype(str).str.startswith("GB-FTS-", na=False)
        & ~canonical["er_status"].astype(str).isin(["singleton", "unresolved"])
    ]
    if len(fts_bad):
        errors.append(f"GB-FTS canonical_id appears on {len(fts_bad)} non-singleton rows")
    return errors


def write_outputs(out_dir: Path, canonical: pd.DataFrame, alias_map: pd.DataFrame, audit: pd.DataFrame, components: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(out_dir / "canonical_orgs.parquet", index=False, compression="snappy")
    alias_map.to_parquet(out_dir / "alias_map.parquet", index=False, compression="snappy")
    audit.to_csv(out_dir / "er_audit.csv", index=False)
    components.to_csv(out_dir / "safe_merge_components.csv", index=False)


def backup_current_entities(backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ["canonical_orgs.parquet", "alias_map.parquet", "er_audit.csv"]:
        src = ENTITIES_DIR / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)


def apply_outputs(staging_dir: Path, backup_dir: Path) -> None:
    backup_current_entities(backup_dir)
    for name in ["canonical_orgs.parquet", "alias_map.parquet", "er_audit.csv"]:
        shutil.copy2(staging_dir / name, ENTITIES_DIR / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, default=ENTITIES_DIR / "canonical_orgs.parquet")
    parser.add_argument("--alias-map", type=Path, default=ENTITIES_DIR / "alias_map.parquet")
    parser.add_argument("--safe-edges", type=Path, default=ABLATION_DIR / "safe_merge_edges_after_cluster_check.parquet")
    parser.add_argument("--staging-dir", type=Path, default=ENTITIES_DIR / "staging" / "er_llm_safe_merge")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    canonical = pd.read_parquet(args.canonical)
    alias_map = pd.read_parquet(args.alias_map)
    edges = pd.read_parquet(args.safe_edges)

    output_canonical, output_alias, audit, components = build_merged_tables(canonical, alias_map, edges)
    errors = validate_outputs(output_canonical, output_alias)
    write_outputs(args.staging_dir, output_canonical, output_alias, audit, components)

    summary = {
        "input_canonical_rows": len(canonical),
        "input_alias_rows": len(alias_map),
        "safe_merge_edges": len(edges),
        "merge_components": len(components),
        "output_canonical_rows": len(output_canonical),
        "output_alias_rows": len(output_alias),
        "canonical_row_delta": len(output_canonical) - len(canonical),
        "alias_row_delta": len(output_alias) - len(alias_map),
        "validation_errors": errors,
        "staging_dir": str(args.staging_dir),
    }

    if errors:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise SystemExit("Validation failed; staged outputs were not applied.")

    if args.apply:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = ENTITIES_DIR / "backups" / f"er_llm_merge_{timestamp}"
        apply_outputs(args.staging_dir, backup_dir)
        summary["applied"] = True
        summary["backup_dir"] = str(backup_dir)
    else:
        summary["applied"] = False

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "safe_merge_apply_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
