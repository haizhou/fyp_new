"""
Pipeline step 02: Deterministic entity resolution → data/entities/

Usage:
    python pipelines/02_er_phase1.py                # full run
    python pipelines/02_er_phase1.py --limit 5000 --output-dir data/smoke/entities
    python pipelines/02_er_phase1.py --sample 5000 --output-dir data/smoke/entities

Partial runs must use --output-dir so they cannot replace full-data entities.

Run from the project root (fyp_new/).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from procurement_graph.common.pipeline_paths import resolve_output_target
from procurement_graph.er.phase1 import resolve_phase1, write_entities
from procurement_graph.ingest.loader import load_interim

ROOT = Path(__file__).parent.parent


def load_settings() -> dict:
    with open(ROOT / "configs" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ts(t0: float) -> str:
    return f"+{time.time() - t0:5.1f}s"


def print_report(canonical_orgs, alias_map) -> None:
    total = len(canonical_orgs)
    det = (canonical_orgs["er_status"] == "deterministic").sum()
    unres = (canonical_orgs["er_status"] == "unresolved").sum()

    print()
    print("─" * 60)
    print("ENTITY RESOLUTION PHASE 1 — REPORT")
    print("─" * 60)
    print(f"  Total canonical entities     : {total:,}")
    print(f"  Deterministically resolved   : {det:,}  ({det/total*100:.1f}%)")
    print(f"  Unresolved (GB-FTS-only)     : {unres:,}  ({unres/total*100:.1f}%)")
    print()

    # Breakdown by official_scheme
    resolved = canonical_orgs[canonical_orgs["er_status"] == "deterministic"]
    scheme_counts = resolved["official_scheme"].value_counts()
    print("  Resolved by scheme:")
    for scheme, cnt in scheme_counts.items():
        print(f"    {scheme:<12} : {cnt:,}")
    print()

    # Alias coverage — split phase1 (official+cross-ref) vs unresolved (self-alias)
    alias_source_counts = alias_map["alias_source"].value_counts()
    phase1_aliases = alias_source_counts.get("phase1", 0)
    unresolved_aliases = alias_source_counts.get("unresolved", 0)
    fts_cross_ref = phase1_aliases - det  # phase1 entries beyond the canonical ones = FTS cross-refs
    print("  Alias map entries:")
    print(f"    phase1 (official + GB-FTS cross-ref) : {phase1_aliases:,}")
    print(f"      of which GB-FTS stored as alias    : {max(fts_cross_ref, 0):,}")
    print(f"    unresolved (self-alias, FTS-only)    : {unresolved_aliases:,}")
    print(f"    TOTAL                                : {len(alias_map):,}")
    print()

    # High-risk examples (entities with many aliases)
    top_multi = (
        canonical_orgs.nlargest(10, "n_aliases")[
            ["canonical_id", "canonical_name", "er_status", "n_aliases"]
        ]
    )
    print("  Top 10 entities by alias count (fragmentation risk):")
    for _, row in top_multi.iterrows():
        print(
            f"    [{row['er_status'][:3].upper()}] {row['canonical_id']:<35} "
            f"{row['canonical_name'][:40]:<40} ({row['n_aliases']} aliases)"
        )
    print()

    # Unresolved examples with >1 alias count (already-merged by name, will be handled in ph2)
    unresolved_multi = canonical_orgs[
        (canonical_orgs["er_status"] == "unresolved") & (canonical_orgs["n_aliases"] > 1)
    ].nlargest(5, "n_aliases")
    if len(unresolved_multi):
        print("  Unresolved entities with multiple aliases (Phase 2 targets):")
        for _, row in unresolved_multi.iterrows():
            print(f"    {row['canonical_id']:<35} {row['canonical_name'][:40]} ({row['n_aliases']} raw_ids)")
    print("─" * 60)


def main(
    limit: int | None = None,
    sample: int | None = None,
    output_dir: Path | None = None,
) -> None:
    cfg = load_settings()
    interim_path = ROOT / cfg["data"]["interim_dir"] / "releases.parquet"
    canonical_entities_dir = ROOT / cfg["data"]["entities_dir"]

    is_partial = limit is not None or sample is not None
    entities_dir = resolve_output_target(
        canonical_entities_dir,
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
    print(f"PIPELINE 02 — ENTITY RESOLUTION PHASE 1 (Deterministic) [{mode_label}]")
    print("=" * 60)
    print(f"  Input    : {interim_path}")
    print(f"  Output   : {entities_dir}/")
    print(f"    canonical_orgs.parquet")
    print(f"    alias_map.parquet")
    if is_partial:
        print(f"  WARNING  : Partial run ({mode_label}) — outputs are not a full-data result.")
    print()

    t0 = time.time()

    # --- Load ---
    t_load = time.time()
    releases = load_interim(interim_path)
    total_available = len(releases)
    if limit is not None:
        releases = releases.head(limit)
    elif sample is not None:
        releases = releases.sample(n=min(sample, len(releases)), random_state=42)
    print(f"  [{_ts(t0)}] Loaded {len(releases):,} releases"
          f"{f' (of {total_available:,} available)' if is_partial else ''}"
          f"  ({time.time() - t_load:.1f}s)")

    # --- Resolve ---
    t_resolve = time.time()
    canonical_orgs, alias_map = resolve_phase1(releases)
    print(f"  [{_ts(t0)}] Entity resolution complete  ({time.time() - t_resolve:.1f}s)")

    # --- Write ---
    t_write = time.time()
    write_entities(canonical_orgs, alias_map, entities_dir)
    print(f"  [{_ts(t0)}] Outputs written  ({time.time() - t_write:.1f}s)")

    print_report(canonical_orgs, alias_map)
    print(f"Total elapsed: {time.time() - t0:.1f}s")
    print("DONE — Phase 1 entity resolution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ER Phase 1: deterministic entity resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipelines/02_er_phase1.py                # full run (~166K releases)
  python pipelines/02_er_phase1.py --limit 5000 --output-dir data/smoke/entities
  python pipelines/02_er_phase1.py --sample 5000 --output-dir data/smoke/entities
        """,
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N releases from interim")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use a random sample of N releases (fixed seed=42)")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (required and non-canonical with --limit/--sample)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.sample is not None:
        parser.error("--limit and --sample are mutually exclusive")
    main(limit=args.limit, sample=args.sample, output_dir=args.output_dir)
