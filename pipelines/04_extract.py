"""
Pipeline step 04: Rich OCDS field extraction → data/extracted/

Reads:  data/raw/*.jsonl.gz     (raw records, for full field access)
        data/interim/releases.parquet  (OCID deduplication authority)

Writes: data/extracted/tender_core.parquet
        data/extracted/lots.parquet
        data/extracted/award_criteria.parquet
        data/extracted/awards.parquet
        data/extracted/bid_stats.parquet
        data/extracted/text_evidence.parquet
        data/extracted/documents.parquet

Usage:
    python pipelines/04_extract.py                  # full run
    python pipelines/04_extract.py --years 2025 --output-dir data/smoke/extracted
    python pipelines/04_extract.py --years 2024 2025 --output-dir data/smoke/extracted

IMPORTANT: Run 01_ingest.py first (data/interim/releases.parquet must exist).

Run from the project root (fyp_new/).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from procurement_graph.common.pipeline_paths import resolve_output_target
from procurement_graph.extract.tables import extract_all, write_extracted
from procurement_graph.ingest.loader import load_interim

ROOT = Path(__file__).parent.parent


def load_settings() -> dict:
    with open(ROOT / "configs" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ts(t0: float) -> str:
    return f"+{time.time() - t0:5.1f}s"


def print_report(tables: dict) -> None:
    print()
    print("─" * 60)
    print("EXTRACTION — TABLE SUMMARY")
    print("─" * 60)
    table_order = [
        "tender_core", "lots", "award_criteria",
        "awards", "bid_stats", "text_evidence", "documents",
    ]
    for name in table_order:
        df = tables.get(name)
        if df is None:
            continue
        print(f"  {name:<25} : {len(df):>8,} rows")

    # Quick spot-checks
    tc = tables.get("tender_core")
    if tc is not None and len(tc):
        framework_pct = tc["tender_has_framework"].fillna(False).sum() / len(tc) * 100
        dps_pct = tc["tender_has_dps"].fillna(False).sum() / len(tc) * 100
        recur_pct = tc["tender_has_recurrence"].fillna(False).sum() / len(tc) * 100
        print()
        print("  Tender flags (% of records with flag=True):")
        print(f"    has_framework    : {framework_pct:5.1f}%")
        print(f"    has_dps          : {dps_pct:5.1f}%")
        print(f"    has_recurrence   : {recur_pct:5.1f}%")

    bs = tables.get("bid_stats")
    if bs is not None and len(bs):
        print()
        print("  Bid stat measures:")
        for measure, cnt in bs["measure"].value_counts().head(8).items():
            print(f"    {measure:<35} : {cnt:,}")

    docs = tables.get("documents")
    if docs is not None and len(docs):
        print()
        print("  Document sources:")
        for src, cnt in docs["source"].value_counts().items():
            print(f"    {src:<30} : {cnt:,}")

    print("─" * 60)


def main(
    years: list[int] | None = None,
    output_dir: Path | None = None,
) -> None:
    cfg = load_settings()
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    interim_path = ROOT / cfg["data"]["interim_dir"] / "releases.parquet"
    canonical_extracted_dir = ROOT / "data" / "extracted"
    extracted_dir = resolve_output_target(
        canonical_extracted_dir,
        output_dir,
        project_root=ROOT,
        partial=bool(years),
        option_name="--output-dir",
    )

    year_label = f"years={years}" if years else "all years"
    print("=" * 60)
    print(f"PIPELINE 04 — RICH OCDS EXTRACTION [{year_label}]")
    print("=" * 60)
    print(f"  Raw dir      : {raw_dir}")
    print(f"  Interim      : {interim_path}")
    print(f"  Output dir   : {extracted_dir}/")
    print()

    t0 = time.time()

    # Build ocid → release_id map from the deduplication output (snapshot authority)
    t_load = time.time()
    releases = load_interim(interim_path)
    ocid_to_release_id = dict(zip(releases["ocid"], releases["release_id"]))
    print(
        f"  [{_ts(t0)}] Loaded {len(ocid_to_release_id):,} ocid→release_id mappings from interim"
        f"  ({time.time()-t_load:.1f}s)"
    )
    print(
        f"             Extraction is snapshot mode: one record per OCID, "
        f"aligned to releases.parquet by release_id."
    )

    # Extract
    t_ext = time.time()
    tables = extract_all(raw_dir, ocid_to_release_id, years=years, show_progress=True)
    print(f"  [{_ts(t0)}] Extraction complete  ({time.time()-t_ext:.1f}s)")

    # Write
    t_write = time.time()
    write_extracted(tables, extracted_dir)
    print(f"  [{_ts(t0)}] Outputs written  ({time.time()-t_write:.1f}s)")

    print_report(tables)
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")
    print("DONE — extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract rich OCDS fields from raw records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipelines/04_extract.py                   # all years (~166K records)
  python pipelines/04_extract.py --years 2025 --output-dir data/smoke/extracted
  python pipelines/04_extract.py --years 2024 2025 --output-dir data/smoke/extracted
        """,
    )
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Limit to specific year files (e.g. --years 2025)")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (required and non-canonical with --years)",
    )
    args = parser.parse_args()
    main(years=args.years, output_dir=args.output_dir)
