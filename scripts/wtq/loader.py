"""WTQ table -> record-universe loader (v2: official TSV, dual view, integrity).

v2 changes (after loader-reliability review):
- Reads the official .tsv sibling (uniform field width across all 2,108 tables;
  embedded newlines escaped as literal \\n) instead of pandas CSV guessing.
  No on_bad_lines=skip: row conservation is ASSERTED, never silently violated.
- Dual view: `raw_df` keeps every cell as its display string (what the table
  shows, what a planner copies, what the answer should look like); `records_df`
  is the typed computation view (numeric columns coerced for gte/sum/argmax).
  WTQEvaluator computes on typed, projects answers from raw.
- The synthetic `contract_node_id` exists only inside the evaluator universe;
  it is NOT part of the field catalog and must not enter the model-visible
  schema enum (spurious row-number programs).
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

WTQ_ROOT = Path("/var/tmp/cicada/wtq/WikiTableQuestions")

_NUM_RE = re.compile(r"^[\s$£€]*[-+]?[\d,]+(?:\.\d+)?\s*%?\s*$")


def _norm_header(h: str, seen: set) -> tuple[str, bool]:
    base = re.sub(r"[^a-z0-9]+", "_", str(h).strip().casefold()).strip("_") or "col"
    name, i, collided = base, 2, False
    while name in seen:
        name, i, collided = f"{base}_{i}", i + 1, True
    seen.add(name)
    return name, collided


def _to_num(cell: str):
    s = str(cell).strip()
    if not _NUM_RE.match(s):
        return None
    s = re.sub(r"[,$£€%\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _unescape(cell: str) -> str:
    # official WTQ tsv escaping: backslash-escaped backslash and newline
    return cell.replace("\\\\", "\x00").replace("\\n", "\n").replace("\x00", "\\")


def _read_tsv(csv_rel: str):
    """Deterministic parse of the official .tsv. Returns (header, rows, integrity)."""
    path = (WTQ_ROOT / csv_rel).with_suffix(".tsv")
    lines = [l for l in path.read_text().split("\n") if l != ""]
    cells = [[_unescape(c) for c in l.split("\t")] for l in lines]
    widths = {len(r) for r in cells}
    if len(widths) != 1:
        raise ValueError(f"nonuniform_tsv_width:{csv_rel}:{sorted(widths)}")
    integrity = {
        "physical_rows": len(lines),
        "parsed_rows": len(cells) - 1,
        "width": len(cells[0]),
        "dropped_rows": 0,  # by construction: every physical line is parsed
    }
    return cells[0], cells[1:], integrity


def load_universe(csv_rel: str):
    """Returns (shim, field_catalog) for one WTQ table.

    shim.records_df -- typed computation view (numeric columns coerced)
    shim.raw_df     -- display view (every cell its original string)
    shim.integrity  -- row/width conservation report for the audit
    field_catalog   -- list of (field_name, dtype_label, RAW sample values)
    """
    header, data, integrity = _read_tsv(csv_rel)
    seen: set = set()
    names, collisions = [], 0
    for h in header:
        name, collided = _norm_header(h, seen)
        names.append(name)
        collisions += collided
    integrity["header_collisions"] = collisions

    raw_df = pd.DataFrame(data, columns=names, dtype=str)
    typed = {}
    catalog = []
    for col in names:
        series = raw_df[col].astype(str).str.strip()
        non_empty = series[series != ""]
        nums = non_empty.map(_to_num)
        if len(non_empty) > 0 and nums.notna().mean() >= 0.8:
            typed[col] = series.map(_to_num)
            dtype = "number"
        else:
            typed[col] = series
            dtype = "text"
        catalog.append((col, dtype, list(non_empty.head(3))))
    records_df = pd.DataFrame(typed)

    # synthetic row id: internal to the evaluator (dedup key); NOT in catalog.
    records_df["contract_node_id"] = [f"row{i}" for i in range(len(records_df))]
    shim = SimpleNamespace(records_df=records_df, raw_df=raw_df, integrity=integrity)
    return shim, catalog


def catalog_text(catalog, df=None, max_rows: int = 3) -> str:
    lines = ["Columns:"]
    for name, dtype, samples in catalog:
        sample_txt = ", ".join(str(s)[:30] for s in samples[:3])
        lines.append(f"- {name} ({dtype}; e.g. {sample_txt})")
    return "\n".join(lines)
