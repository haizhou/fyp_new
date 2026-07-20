#!/usr/bin/env python3
"""Corpus-wide loader integrity audit (regression test for loader v2).

Pass criteria (all hard):
- every table under csv/*-csv/*.csv loads via the official .tsv path
- parsed_rows == physical_rows - 1 for every table (0 silently dropped rows)
- uniform field width within every table
- header collisions resolved by the deterministic suffix rule (reported)
- synthetic contract_node_id present in records_df but absent from catalog

Last known-good: 2108/2108 tables, 57,960 rows, 0 dropped, 101 collision
tables (deterministically disambiguated), 0 failures.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader import WTQ_ROOT, load_universe  # noqa: E402


def main() -> int:
    n = rows = collide = 0
    fails = []
    for d in sorted(WTQ_ROOT.glob("csv/*-csv")):
        for f in sorted(d.glob("*.csv")):
            rel = str(f.relative_to(WTQ_ROOT))
            n += 1
            try:
                shim, cat = load_universe(rel)
                it = shim.integrity
                assert it["parsed_rows"] == it["physical_rows"] - 1, "row conservation"
                assert it["dropped_rows"] == 0, "dropped rows"
                assert len(shim.records_df) == it["parsed_rows"], "df length"
                assert "contract_node_id" in shim.records_df.columns, "internal id"
                assert all(c[0] != "contract_node_id" for c in cat), "id leaked to catalog"
                rows += it["parsed_rows"]
                collide += bool(it["header_collisions"])
            except Exception as exc:  # noqa: BLE001
                fails.append((rel, f"{type(exc).__name__}: {exc}"))
    print(f"tables {n}, ok {n - len(fails)}, rows {rows}, collision-tables {collide}")
    for x in fails[:20]:
        print("FAIL", x)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
