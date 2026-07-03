"""
Pipeline step 01: Ingest raw OCDS data → data/interim/releases.parquet

Usage:
    python pipelines/01_ingest.py [--years 2022 2023 2024 2025 2026]

Run from the project root (fyp_new/).
"""

import argparse
import sys
import time
from pathlib import Path

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from ingest import flatten_all_years, write_interim

ROOT = Path(__file__).parent.parent


def load_settings() -> dict:
    cfg_path = ROOT / "configs" / "settings.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(years: list[int] | None = None) -> None:
    cfg = load_settings()
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    interim_path = ROOT / cfg["data"]["interim_dir"] / "releases.parquet"

    print("=" * 60)
    print("PIPELINE 01 — INGEST")
    print("=" * 60)
    print(f"  Raw dir   : {raw_dir}")
    print(f"  Output    : {interim_path}")
    if years:
        print(f"  Years     : {years}")
    print()

    t0 = time.time()
    df = flatten_all_years(raw_dir, years=years, show_progress=True)
    write_interim(df, interim_path)
    elapsed = time.time() - t0

    print()
    print("Summary")
    print(f"  Rows (unique OCIDs)  : {len(df):,}")
    print(f"  Date range           : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Years in data        : {sorted(df['year'].dropna().unique().tolist())}")
    print(f"  Records with buyer   : {df['buyer_raw_id'].ne('').sum():,}")
    print(f"  Records with contracts: {df['contracts_json'].ne('[]').sum():,}")
    print(f"  Elapsed              : {elapsed:.1f}s")
    print()
    print("DONE — data/interim/releases.parquet written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest OCDS JSONL.gz files")
    parser.add_argument(
        "--years", nargs="*", type=int, default=None,
        help="Limit to specific years (e.g. --years 2022 2023)"
    )
    args = parser.parse_args()
    main(years=args.years)
