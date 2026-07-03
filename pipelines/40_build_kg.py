"""
Pipeline step 40: deterministic KG v0 build.

Reads:
    data/interim/releases.parquet
    data/extracted/awards.parquet
    data/extracted/text_evidence.parquet
    data/entities/canonical_orgs.parquet
    data/entities/alias_map.parquet

Writes:
    data/kg/nodes/org_nodes.parquet
    data/kg/nodes/contract_nodes.parquet
    data/kg/nodes/cpv_nodes.parquet
    data/kg/nodes/evidence_nodes.parquet
    data/kg/edges/buyer_of.parquet
    data/kg/edges/supplier_of.parquet
    data/kg/edges/categorized_by.parquet
    data/kg/edges/evidence_for.parquet
    reports/kg/kg_validation_summary.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_graph.kg import build_and_validate
from procurement_graph.kg.validate import FAIL, validation_summary


ROOT = Path(__file__).resolve().parents[1]


def _ts(t0: float) -> str:
    return f"+{time.time() - t0:5.1f}s"


def main(output_dir: Path | None = None) -> int:
    t0 = time.time()
    kg_dir = output_dir or (ROOT / "data" / "kg")

    print("=" * 60)
    print("PIPELINE 40 - DETERMINISTIC KG V0 BUILD")
    print("=" * 60)
    print(f"  Output dir: {kg_dir}")
    print()

    tables, checks, did_write = build_and_validate(ROOT, kg_dir=kg_dir)
    if did_write:
        print(f"  [{_ts(t0)}] KG tables written")
    else:
        print(f"  [{_ts(t0)}] KG tables built in memory; outputs not written because validation failed")
    for name in [
        "org_nodes",
        "contract_nodes",
        "cpv_nodes",
        "evidence_nodes",
        "buyer_of",
        "supplier_of",
        "categorized_by",
        "evidence_for",
    ]:
        print(f"    {name:<18} {len(tables[name]):>10,} rows")

    summary = validation_summary(checks)
    print()
    print("VALIDATION")
    print(f"  PASS: {summary['PASS']}")
    print(f"  WARN: {summary['WARN']}")
    print(f"  FAIL: {summary['FAIL']}")
    for check in checks:
        print(f"  [{check.status}] {check.area}.{check.name} - {check.detail}")

    print()
    print(f"Validation report: {ROOT / 'reports' / 'kg' / 'kg_validation_summary.json'}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")

    if summary[FAIL]:
        print("DONE - KG build completed with validation failures.")
        return 1
    print("DONE - KG build validated.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build deterministic KG v0 tables")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override KG output directory; defaults to data/kg",
    )
    args = parser.parse_args()
    raise SystemExit(main(output_dir=args.output_dir))
