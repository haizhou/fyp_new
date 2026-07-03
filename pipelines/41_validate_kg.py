"""
Pipeline step 41: validate existing KG v0 outputs without rebuilding them.

Reads:
    data/kg/nodes/*.parquet
    data/kg/edges/*.parquet
    data/extracted/awards.parquet
    data/entities/alias_map.parquet

Writes:
    reports/kg/kg_validation_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_graph.kg.build import load_kg_inputs, run_validation, write_validation_report
from procurement_graph.kg.validate import FAIL, validation_summary


ROOT = Path(__file__).resolve().parents[1]


def _load_tables(kg_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "org_nodes": pd.read_parquet(kg_dir / "nodes" / "org_nodes.parquet"),
        "contract_nodes": pd.read_parquet(kg_dir / "nodes" / "contract_nodes.parquet"),
        "cpv_nodes": pd.read_parquet(kg_dir / "nodes" / "cpv_nodes.parquet"),
        "evidence_nodes": pd.read_parquet(kg_dir / "nodes" / "evidence_nodes.parquet"),
        "buyer_of": pd.read_parquet(kg_dir / "edges" / "buyer_of.parquet"),
        "supplier_of": pd.read_parquet(kg_dir / "edges" / "supplier_of.parquet"),
        "categorized_by": pd.read_parquet(kg_dir / "edges" / "categorized_by.parquet"),
        "evidence_for": pd.read_parquet(kg_dir / "edges" / "evidence_for.parquet"),
    }


def main(kg_dir: Path | None = None) -> int:
    kg_dir = kg_dir or (ROOT / "data" / "kg")
    inputs = load_kg_inputs(ROOT)
    tables = _load_tables(kg_dir)
    checks = run_validation(tables, inputs)
    write_validation_report(checks, ROOT / "reports" / "kg" / "kg_validation_summary.json")

    summary = validation_summary(checks)
    print("=" * 60)
    print("PIPELINE 41 - KG V0 VALIDATION")
    print("=" * 60)
    print(f"  KG dir: {kg_dir}")
    print(f"  PASS: {summary['PASS']}")
    print(f"  WARN: {summary['WARN']}")
    print(f"  FAIL: {summary['FAIL']}")
    for check in checks:
        print(f"  [{check.status}] {check.area}.{check.name} - {check.detail}")
    print(f"Validation report: {ROOT / 'reports' / 'kg' / 'kg_validation_summary.json'}")

    return 1 if summary[FAIL] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate existing deterministic KG v0 tables")
    parser.add_argument(
        "--kg-dir",
        type=Path,
        default=None,
        help="KG directory to validate; defaults to data/kg",
    )
    args = parser.parse_args()
    raise SystemExit(main(kg_dir=args.kg_dir))
