"""
Pipeline step 03: Heuristic entity resolution (Phase 2) + fuzzy candidate report.

Reads:  data/entities/canonical_orgs.parquet (from step 02)
        data/entities/alias_map.parquet       (from step 02)
        configs/gov_lookup.json

Writes: data/entities/canonical_orgs.parquet  (updated in-place)
        data/entities/alias_map.parquet        (updated in-place)
        data/entities/er_audit.csv             (provenance log, one row per entity)
        data/entities/er_candidates.csv        (fuzzy name-similarity pairs, REVIEW ONLY)

Usage:
    python pipelines/03_er_phase2.py                  # full run
    python pipelines/03_er_phase2.py --no-candidates  # skip fuzzy report (much faster)
    python pipelines/03_er_phase2.py --limit 2000 --output-dir data/smoke/entities
    python pipelines/03_er_phase2.py --sample 2000 --output-dir data/smoke/entities

Notes:
  --no-candidates skips the pairwise Jaro-Winkler comparison which is O(n^2) within each
  3-char name-prefix block. For full data this may take several minutes. Skip it on test runs.

  --limit / --sample restrict the canonical_orgs loaded from Phase 1 output. Partial
  outputs require a separate --output-dir and never replace the full entity tables.

  Fuzzy candidate report (er_candidates.csv) is for human review only — no automatic merges.

Run from the project root (fyp_new/).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import yaml
from procurement_graph.common.pipeline_paths import resolve_output_target
from procurement_graph.er.candidates import generate_candidates, write_candidates, describe_limits
from procurement_graph.er.phase1 import load_canonical_orgs, load_alias_map
from procurement_graph.er.phase2 import resolve_phase2, write_phase2_outputs

ROOT = Path(__file__).parent.parent


def load_settings() -> dict:
    with open(ROOT / "configs" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ts(t0: float) -> str:
    return f"+{time.time() - t0:5.1f}s"


def print_report(canonical_orgs, alias_map) -> None:
    total = len(canonical_orgs)
    status_counts = canonical_orgs["er_status"].value_counts()

    print()
    print("─" * 60)
    print("ENTITY RESOLUTION PHASE 2 — REPORT")
    print("─" * 60)
    print(f"  Total canonical entities : {total:,}")
    print()
    print("  Resolution status breakdown:")
    status_order = [
        "deterministic", "gov_lookup", "name_region_merge",
        "name_only_merge", "singleton", "unresolved",
    ]
    for s in status_order:
        cnt = status_counts.get(s, 0)
        pct = cnt / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"    {s:<22} : {cnt:>7,}  ({pct:5.1f}%) {bar}")
    print()

    # Alias map summary
    alias_source_counts = alias_map["alias_source"].value_counts()
    print("  Alias map entries by source:")
    for src in ["phase1", "phase2", "unresolved"]:
        cnt = alias_source_counts.get(src, 0)
        print(f"    {src:<12} : {cnt:,}")
    print(f"    {'TOTAL':<12} : {len(alias_map):,}")
    print()

    # GB-FTS specifically stored as aliases (never canonical)
    fts_alias_count = alias_map[
        alias_map["raw_id"].str.startswith("GB-FTS-", na=False)
        & (alias_map["raw_id"] != alias_map["canonical_id"])
    ]["raw_id"].nunique()
    print(f"  GB-FTS IDs stored as aliases (not canonical) : {fts_alias_count:,}")
    print()

    # Alias distribution — cast n_aliases to int in case concat produced object dtype
    canonical_orgs = canonical_orgs.copy()
    canonical_orgs["n_aliases"] = pd.to_numeric(canonical_orgs["n_aliases"], errors="coerce").fillna(0).astype(int)
    n_multi = (canonical_orgs["n_aliases"] > 1).sum()
    n_single = (canonical_orgs["n_aliases"] == 1).sum()
    print(f"  Entities with >1 alias (merged)   : {n_multi:,}")
    print(f"  Entities with 1 alias (singleton)  : {n_single:,}")
    print()

    # High-alias examples
    top = canonical_orgs.nlargest(8, "n_aliases")[
        ["canonical_id", "canonical_name", "er_status", "n_aliases"]
    ]
    print("  Top 8 most-aliased entities:")
    for _, row in top.iterrows():
        print(
            f"    [{row['er_status'][:3].upper()}] "
            f"{row['canonical_id']:<35} "
            f"({row['n_aliases']:>3} aliases) "
            f"{row['canonical_name'][:40]}"
        )
    print("─" * 60)


def main(
    run_candidates: bool = True,
    limit: int | None = None,
    sample: int | None = None,
    fuzzy_max_pairs_total: int = 10_000,
    fuzzy_max_pairs_per_block: int = 500,
    fuzzy_max_block_size: int = 50,
    output_dir: Path | None = None,
) -> None:
    cfg = load_settings()
    source_entities_dir = ROOT / cfg["data"]["entities_dir"]
    gov_lookup_path = ROOT / cfg["entity_resolution"]["gov_lookup_path"]
    fuzzy_threshold = cfg["entity_resolution"]["fuzzy_threshold"]

    is_partial = limit is not None or sample is not None
    entities_dir = resolve_output_target(
        source_entities_dir,
        output_dir,
        project_root=ROOT,
        partial=is_partial,
        option_name="--output-dir",
    )
    mode_label = (
        f"LIMIT {limit:,}" if limit is not None
        else f"SAMPLE {sample:,} (seed=42)" if sample is not None
        else "FULL"
    )

    print("=" * 60)
    print(f"PIPELINE 03 — ENTITY RESOLUTION PHASE 2 (Heuristic) [{mode_label}]")
    print("=" * 60)
    print(f"  Input    : {source_entities_dir}/canonical_orgs.parquet")
    print(f"             {source_entities_dir}/alias_map.parquet")
    print(f"  Gov lookup: {gov_lookup_path}  ({_count_gov_entries(gov_lookup_path)} entries)")
    print(f"  Outputs  : {entities_dir}/canonical_orgs.parquet  (updated)")
    print(f"             {entities_dir}/alias_map.parquet        (updated)")
    print(f"             {entities_dir}/er_audit.csv")
    if run_candidates:
        print(f"             {entities_dir}/er_candidates.csv  (fuzzy pairs, REVIEW ONLY)")
    else:
        print(f"             er_candidates.csv  SKIPPED (--no-candidates)")
    if run_candidates:
        print(f"  Fuzzy limits : {describe_limits(fuzzy_threshold, fuzzy_max_pairs_total, fuzzy_max_pairs_per_block, fuzzy_max_block_size)}")
    if is_partial:
        print(f"  WARNING  : Partial run ({mode_label}) — outputs are not a full-data result.")
    print()

    t0 = time.time()

    # --- Load ---
    t_load = time.time()
    canonical_orgs = load_canonical_orgs(source_entities_dir)
    alias_map = load_alias_map(source_entities_dir)
    total_available = len(canonical_orgs)
    if limit is not None:
        canonical_orgs = canonical_orgs.head(limit)
    elif sample is not None:
        canonical_orgs = canonical_orgs.sample(n=min(sample, len(canonical_orgs)), random_state=42)
    # Trim alias_map to match the reduced entity set
    if is_partial:
        kept_ids = set(canonical_orgs["canonical_id"])
        alias_map = alias_map[alias_map["canonical_id"].isin(kept_ids)]
    print(f"  [{_ts(t0)}] Loaded {len(canonical_orgs):,} entities"
          f"{f' (of {total_available:,})' if is_partial else ''}, "
          f"{len(alias_map):,} alias entries  ({time.time() - t_load:.1f}s)")

    unresolved_count = (canonical_orgs["er_status"] == "unresolved").sum()
    print(f"  [{_ts(t0)}] Unresolved (GB-FTS-only) entering Phase 2: {unresolved_count:,}")

    # --- Phase 2 resolution ---
    t_resolve = time.time()
    updated_orgs, updated_alias, audit_log = resolve_phase2(
        canonical_orgs, alias_map, gov_lookup_path
    )
    print(f"  [{_ts(t0)}] Phase 2 resolution complete  ({time.time() - t_resolve:.1f}s)")

    # --- Write outputs ---
    t_write = time.time()
    write_phase2_outputs(updated_orgs, updated_alias, audit_log, entities_dir)
    print(f"  [{_ts(t0)}] Outputs written  ({time.time() - t_write:.1f}s)")

    print_report(updated_orgs, updated_alias)

    # --- Fuzzy candidates ---
    if run_candidates:
        t_fuzz = time.time()
        print(f"  [{_ts(t0)}] Generating fuzzy candidate report ...")
        print(f"             Blocking: 3-char name prefix. Limits: {describe_limits(fuzzy_threshold, fuzzy_max_pairs_total, fuzzy_max_pairs_per_block, fuzzy_max_block_size)}")
        candidates = generate_candidates(
            updated_orgs,
            threshold=fuzzy_threshold,
            max_pairs_total=fuzzy_max_pairs_total,
            max_pairs_per_block=fuzzy_max_pairs_per_block,
            max_block_size=fuzzy_max_block_size,
        )
        write_candidates(candidates, entities_dir / "er_candidates.csv")
        print(f"  [{_ts(t0)}] Fuzzy candidates done  ({time.time() - t_fuzz:.1f}s)")
        if len(candidates):
            print(f"  Top 5 fuzzy candidates (REVIEW ONLY — not auto-merged):")
            for _, row in candidates.head(5).iterrows():
                print(
                    f"    {row['similarity']:.3f}  "
                    f"{row['entity_a_name'][:35]:<35} ↔  {row['entity_b_name'][:35]}"
                )

    print()
    print(f"Total elapsed: {time.time() - t0:.1f}s")
    print("DONE — Phase 2 entity resolution complete.")


def _count_gov_entries(path: Path) -> int:
    import json
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return sum(1 for k in data if not k.startswith("_"))
    except Exception:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ER Phase 2: heuristic entity resolution + fuzzy candidate report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipelines/03_er_phase2.py                          # full run
  python pipelines/03_er_phase2.py --no-candidates          # skip fuzzy report (fast)
  python pipelines/03_er_phase2.py --limit 2000 --output-dir data/smoke/entities
  python pipelines/03_er_phase2.py --sample 2000 --no-candidates --output-dir data/smoke/entities
        """,
    )
    parser.add_argument(
        "--no-candidates", action="store_true",
        help="Skip fuzzy candidate report (skips pairwise Jaro-Winkler comparison)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N entities from Phase 1 output")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use a random sample of N entities (fixed seed=42)")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (required and non-canonical with --limit/--sample)",
    )
    parser.add_argument("--fuzzy-max-pairs", type=int, default=10_000,
                        help="Global cap on total fuzzy candidate pairs (default 10000)")
    parser.add_argument("--fuzzy-max-block-size", type=int, default=50,
                        help="Skip prefix blocks larger than this (default 50)")
    parser.add_argument("--fuzzy-max-pairs-per-block", type=int, default=500,
                        help="Cap on pairs from a single prefix block (default 500)")
    args = parser.parse_args()
    if args.limit is not None and args.sample is not None:
        parser.error("--limit and --sample are mutually exclusive")
    main(
        run_candidates=not args.no_candidates,
        limit=args.limit,
        sample=args.sample,
        fuzzy_max_pairs_total=args.fuzzy_max_pairs,
        fuzzy_max_pairs_per_block=args.fuzzy_max_pairs_per_block,
        fuzzy_max_block_size=args.fuzzy_max_block_size,
        output_dir=args.output_dir,
    )
