"""
Reference/API entity ablation runner.

This script is intentionally read-only with respect to the main entity layer:
it reads data/entities/* and data/reference/*, then writes separate experiment
outputs under data/ablation/ and reports/ablation/.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from normalise import normalise_name
from reference_lookup import ReferenceStore, enrich_from_reference


ENTITIES_DIR = ROOT / "data" / "entities"
REFERENCE_DIR = ROOT / "data" / "reference"
ABLATION_DIR = ROOT / "data" / "ablation"
REPORT_DIR = ROOT / "reports" / "ablation"

TARGET_STATUSES = ["singleton", "name_region_merge", "name_only_merge", "gov_lookup"]
PROBE_NAMES = [
    "MINISTRY OF DEFENCE",
    "NHS ENGLAND",
    "UK RESEARCH AND INNOVATION",
    "CROWN COMMERCIAL SERVICE",
    "NATIONAL HIGHWAYS",
    "SCOTTISH GOVERNMENT",
    "STAFFORDSHIRE COUNTY COUNCIL",
    "SOUTHAMPTON CITY COUNCIL",
]


@dataclass
class Variant:
    name: str
    sources: tuple[str, ...]


VARIANTS = [
    Variant("govuk_only", ("govuk_orgs",)),
    Variant("nhs_only", ("nhs_ods",)),
    Variant("cf_buyer_only", ("contracts_finder",)),
    Variant("all_references", ("govuk_orgs", "nhs_ods", "contracts_finder")),
]


def _variant_store(base: ReferenceStore, sources: tuple[str, ...]) -> ReferenceStore:
    store = ReferenceStore()
    if "govuk_orgs" in sources:
        store.govuk_orgs = base.govuk_orgs
    if "nhs_ods" in sources:
        store.nhs_ods = base.nhs_ods
    if "contracts_finder" in sources:
        store.contracts_finder = base.contracts_finder
    store._loaded = any([store.govuk_orgs, store.nhs_ods, store.contracts_finder])
    return store


def _json_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(value or "[]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _load() -> tuple[pd.DataFrame, pd.DataFrame, ReferenceStore]:
    canonical = pd.read_parquet(ENTITIES_DIR / "canonical_orgs.parquet")
    alias_map = pd.read_parquet(ENTITIES_DIR / "alias_map.parquet")
    store = ReferenceStore.load(REFERENCE_DIR)
    return canonical, alias_map, store


def _baseline_metrics(canonical: pd.DataFrame, alias_map: pd.DataFrame) -> dict:
    status_counts = canonical["er_status"].value_counts()
    return {
        "variant": "current_clean_baseline",
        "sources": "current_pipeline",
        "scope": "all_entities",
        "canonical_entities_total": len(canonical),
        "raw_ids_total": len(alias_map),
        "duplicate_canonical_id_count": int(canonical["canonical_id"].duplicated().sum()),
        "alias_map_raw_id_duplicate_count": int(alias_map["raw_id"].duplicated().sum()),
        "target_entities": len(canonical),
        "singleton_count": int(status_counts.get("singleton", 0)),
        "singleton_rate": float(status_counts.get("singleton", 0) / len(canonical)),
        "merged_entity_count": int((canonical["n_aliases"] > 1).sum()),
        "avg_aliases_per_entity": float(canonical["n_aliases"].mean()),
        "max_aliases_per_entity": int(canonical["n_aliases"].max()),
        "gb_fts_as_canonical_count": int(canonical["canonical_id"].str.startswith("GB-FTS-", na=False).sum()),
        "candidate_entities": 0,
        "candidate_aliases": 0,
        "candidate_rate": 0.0,
        "candidate_singletons": 0,
        "candidate_name_region": 0,
        "candidate_name_only": 0,
        "candidate_gov_lookup": 0,
    }


def _candidate_metrics(variant: Variant, enriched: pd.DataFrame, target: pd.DataFrame) -> dict:
    candidates = enriched[enriched["reference_source"].notna()].copy()
    return {
        "variant": variant.name,
        "sources": "+".join(variant.sources),
        "scope": "target_entities_for_reference_ablation",
        "canonical_entities_total": None,
        "raw_ids_total": None,
        "duplicate_canonical_id_count": None,
        "alias_map_raw_id_duplicate_count": None,
        "target_entities": len(target),
        "singleton_count": int((target["er_status"] == "singleton").sum()),
        "singleton_rate": float((target["er_status"] == "singleton").mean()) if len(target) else 0.0,
        "merged_entity_count": int((target["n_aliases"] > 1).sum()),
        "avg_aliases_per_entity": float(target["n_aliases"].mean()) if len(target) else 0.0,
        "max_aliases_per_entity": int(target["n_aliases"].max()) if len(target) else 0,
        "gb_fts_as_canonical_count": int(target["canonical_id"].str.startswith("GB-FTS-", na=False).sum()),
        "candidate_entities": len(candidates),
        "candidate_rate": float(len(candidates) / len(target)) if len(target) else 0.0,
        "candidate_aliases": int(candidates["n_aliases"].sum()) if len(candidates) else 0,
        "candidate_singletons": int((candidates["er_status"] == "singleton").sum()) if len(candidates) else 0,
        "candidate_name_region": int((candidates["er_status"] == "name_region_merge").sum()) if len(candidates) else 0,
        "candidate_name_only": int((candidates["er_status"] == "name_only_merge").sum()) if len(candidates) else 0,
        "candidate_gov_lookup": int((candidates["er_status"] == "gov_lookup").sum()) if len(candidates) else 0,
    }


def _probe_rows(canonical: pd.DataFrame, all_candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    canonical = canonical.copy()
    canonical["_norm"] = canonical["canonical_name"].apply(normalise_name)

    for probe in PROBE_NAMES:
        norm = normalise_name(probe)
        matches = canonical[canonical["_norm"] == norm]
        if matches.empty:
            matches = canonical[canonical["_norm"].str.contains(norm, regex=False, na=False)].head(5)

        for _, row in matches.iterrows():
            base = {
                "probe": probe,
                "canonical_id": row["canonical_id"],
                "canonical_name": row["canonical_name"],
                "er_status": row["er_status"],
                "n_aliases": row["n_aliases"],
            }
            cand = all_candidates[all_candidates["canonical_id"] == row["canonical_id"]]
            if cand.empty:
                rows.append({**base, "variant": "", "reference_source": "", "reference_canonical_id": ""})
            else:
                for _, crow in cand.iterrows():
                    rows.append({
                        **base,
                        "variant": crow["variant"],
                        "reference_source": crow.get("reference_source", ""),
                        "reference_canonical_id": crow.get("reference_canonical_id", ""),
                    })

    return pd.DataFrame(rows)


def _high_risk(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    risky = candidates[
        (candidates["n_aliases"] >= 20)
        | candidates["canonical_name"].astype(str).str.contains("@|http|www\\.", regex=True, case=False, na=False)
        | ((candidates["reference_source"] == "contracts_finder") & (candidates["n_aliases"] > 1))
    ].copy()
    risky["alias_names_sample"] = risky["alias_names"].apply(lambda value: _json_list(value)[:5])
    risky["alias_raw_ids_sample"] = risky["alias_raw_ids"].apply(lambda value: _json_list(value)[:5])
    review_cols = [
        "variant",
        "canonical_id",
        "canonical_name",
        "er_status",
        "n_aliases",
        "org_type",
        "org_category",
        "address_region",
        "reference_source",
        "reference_confidence",
        "reference_canonical_id",
        "reference_matched_name",
        "reference_status",
        "alias_names_sample",
        "alias_raw_ids_sample",
    ]
    review_cols = [col for col in review_cols if col in risky.columns]
    return risky.sort_values(["variant", "n_aliases"], ascending=[True, False])[review_cols]


def _write_summary(metrics: pd.DataFrame, candidates: pd.DataFrame, high_risk: pd.DataFrame) -> None:
    def _md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        display = df.fillna("")
        cols = list(display.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in display.iterrows():
            values = [str(row[col]).replace("|", "\\|") for col in cols]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# Reference Entity Ablation Summary",
        "",
        "This experiment is read-only over `data/entities/*`. Reference matches are candidates only.",
        "",
        "## Metrics",
        "",
        _md_table(metrics),
        "",
        "## Candidate Counts by Source",
        "",
    ]
    if candidates.empty:
        lines.append("No reference candidates found.")
    else:
        by_source = (
            candidates.groupby(["variant", "reference_source", "reference_confidence"])
            .size()
            .reset_index(name="n")
            .sort_values(["variant", "n"], ascending=[True, False])
        )
        lines.append(_md_table(by_source))
    lines.extend([
        "",
        "## High-Risk Review",
        "",
        f"High-risk candidate rows: {len(high_risk):,}",
    ])
    (REPORT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    canonical, alias_map, store = _load()
    target = canonical[canonical["er_status"].isin(TARGET_STATUSES)].copy()

    metrics_rows = [_baseline_metrics(canonical, alias_map)]
    candidate_frames = []

    print(f"Loaded {len(canonical):,} canonical entities and {len(alias_map):,} aliases.")
    print(f"Reference cache loaded: {not store.is_empty}")
    print(f"Target entities for reference ablation: {len(target):,}")

    for variant in VARIANTS:
        print(f"Running variant: {variant.name} ({'+'.join(variant.sources)})")
        vstore = _variant_store(store, variant.sources)
        enriched = enrich_from_reference(target, vstore)
        candidates = enriched[enriched["reference_source"].notna()].copy()
        candidates["variant"] = variant.name
        candidate_frames.append(candidates)
        metrics_rows.append(_candidate_metrics(variant, enriched, target))
        print(f"  candidates: {len(candidates):,}")

    all_candidates = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames else pd.DataFrame()
    )
    metrics = pd.DataFrame(metrics_rows)
    probes = _probe_rows(canonical, all_candidates)
    high_risk = _high_risk(all_candidates)

    all_candidates.to_parquet(ABLATION_DIR / "reference_entity_candidates.parquet", index=False)
    metrics.to_csv(REPORT_DIR / "reference_ablation_metrics.csv", index=False)
    probes.to_csv(REPORT_DIR / "probe_entities.csv", index=False)
    high_risk.to_csv(REPORT_DIR / "high_risk_reference_candidates.csv", index=False)
    _write_summary(metrics, all_candidates, high_risk)

    print()
    print(f"Written: {ABLATION_DIR / 'reference_entity_candidates.parquet'}")
    print(f"Written: {REPORT_DIR / 'reference_ablation_metrics.csv'}")
    print(f"Written: {REPORT_DIR / 'probe_entities.csv'}")
    print(f"Written: {REPORT_DIR / 'high_risk_reference_candidates.csv'}")
    print(f"Written: {REPORT_DIR / 'summary.md'}")


if __name__ == "__main__":
    from procurement_graph.experiments.reference_ablation import cli_main

    cli_main()
