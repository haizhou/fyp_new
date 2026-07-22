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

    # v2: per-cell numeric view — for text columns where >=50% of non-empty
    # cells contain an extractable number, add <col>__num (first number in the
    # cell, commas stripped). Fixed rule, identical across splits; raw_df gets
    # the display string of the extracted number for projection.
    _first_num = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
    def _cell_num(s):
        m = _first_num.search(str(s))
        if not m:
            return None
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            return None
    for col, dtype, _ in list(catalog):
        if dtype == "number":
            continue
        series = raw_df[col].astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) == 0:
            continue
        nums = non_empty.map(_cell_num)
        if nums.notna().mean() >= 0.5:
            aux = f"{col}__num"
            records_df[aux] = series.map(_cell_num)
            raw_df[aux] = records_df[aux].map(
                lambda v: "" if pd.isna(v) else (f"{int(v)}" if float(v).is_integer() else f"{v:g}"))
            catalog.append((aux, "number",
                            [v for v in nums.dropna().head(3)]))

    # v3 typed views (attribution flag WTQ_VIEWS: "v3" default, "v2" disables).
    # Fixed cross-split rules, never question- or gold-SQL-conditioned:
    # __number_k = k-th valid numeric token IN ORIGINAL TEXT ORDER (unified
    #   token extractor; span/provenance = the raw cell, preserved alongside).
    # __min_year/__max_year = min/max of 4-DIGIT year tokens (1000-2099).
    #   Two-digit shorthand ("1998-99") is NOT expanded: yields [1998] only —
    #   fixed pre-declared convention. Parse failure -> null, never guessed.
    import os as _os
    _views_on = _os.environ.get("WTQ_VIEWS", "v4") in ("v3", "v4")
    _year_re = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
    for col, dtype, _ in (list(catalog) if _views_on else []):
        if dtype == "number" or col.endswith("__num"):
            continue
        series = raw_df[col].astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) == 0:
            continue
        years = non_empty.map(lambda s: [int(y) for y in _year_re.findall(s)])
        if years.map(bool).mean() >= 0.5:
            aux = f"{col}__min_year"
            records_df[aux] = series.map(lambda s: float(_year_re.findall(s)[0]) if _year_re.findall(s) else None)
            raw_df[aux] = records_df[aux].map(lambda v: "" if pd.isna(v) else str(int(v)))
            catalog.append((aux, "number", [y[0] for y in years if y][:3]))
            if years.map(lambda y: len(y) >= 2).mean() >= 0.2:
                aux2 = f"{col}__max_year"
                records_df[aux2] = series.map(lambda s: float(_year_re.findall(s)[-1]) if _year_re.findall(s) else None)
                raw_df[aux2] = records_df[aux2].map(lambda v: "" if pd.isna(v) else str(int(v)))
                catalog.append((aux2, "number", [y[-1] for y in years if y][:3]))
        # typed positional numbers: "5-14 (29)" -> number_1=5, number_2=14
        # parse failure -> null, never guessed; raw always preserved
        allnums = non_empty.map(lambda s: [float(m.replace(",", "")) for m in
                                           re.findall(r"-?\d[\d,]*(?:\.\d+)?", s)])
        if allnums.map(lambda x: len(x) >= 2).mean() >= 0.5:
            for idx in (0, 1):
                aux = f"{col}__number_{idx+1}"
                records_df[aux] = series.map(
                    lambda s, i=idx: (lambda ns: ns[i] if len(ns) > i else None)(
                        [float(m.replace(",", "")) for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", s)]))
                raw_df[aux] = records_df[aux].map(
                    lambda v: "" if pd.isna(v) else (f"{int(v)}" if float(v).is_integer() else f"{v:g}"))
                catalog.append((aux, "number", [n[idx] for n in allnums if len(n) > idx][:3]))

    # v4 views (WTQ_VIEWS=v4): row order as DATA plus typed composite parts.
    # row_index = 1-based source-table position, a semantic field of the table
    # (distinct from the opaque synthetic id below, which stays internal).
    # __part_k = k-th cell segment under a FIXED delimiter priority
    # (newline, then " - "/en-dash with spaces, then ", "), added when >=50%
    # of non-empty cells split; no semantic claim (part, not surname);
    # parse failure -> null; raw preserved.
    if _os.environ.get("WTQ_VIEWS", "v4") == "v4":
        n_rows = len(records_df)
        records_df["row_index"] = [float(i + 1) for i in range(n_rows)]
        raw_df["row_index"] = [str(i + 1) for i in range(n_rows)]
        catalog.append(("row_index", "number", [1, 2, 3]))
        _delim = re.compile(r"\n| [-\u2013\u2014] |, ")
        for col, dtype, _ in list(catalog):
            if dtype == "number" or "__" in col or col == "row_index":
                continue
            series = raw_df[col].astype(str).str.strip()
            non_empty = series[series != ""]
            if len(non_empty) == 0:
                continue
            parts = non_empty.map(lambda s: [x.strip() for x in _delim.split(s) if x.strip()])
            if parts.map(lambda x: len(x) >= 2).mean() >= 0.5:
                for k in (0, 1):
                    aux = f"{col}__part_{k+1}"
                    vals = series.map(lambda s, i=k: (lambda seg: seg[i] if len(seg) > i else "")(
                        [x.strip() for x in _delim.split(s) if x.strip()]))
                    records_df[aux] = vals
                    raw_df[aux] = vals
                    catalog.append((aux, "text", [p2[k] for p2 in parts if len(p2) > k][:3]))

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
