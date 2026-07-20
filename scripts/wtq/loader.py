"""WTQ table -> record-universe loader.

Adapts an arbitrary WTQ CSV into the flat typed DataFrame our evaluator
executes over, plus the per-table field catalog for prompt + guided schema.
The algebra, type checker, and evaluator are unchanged; only this front end
is new — that is the portability claim in code form.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

WTQ_ROOT = Path("/var/tmp/cicada/wtq/WikiTableQuestions")

_NUM_RE = re.compile(r"^[\s$£€]*[-+]?[\d,]+(?:\.\d+)?\s*%?\s*$")


def _norm_header(h: str, seen: set) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", str(h).strip().casefold()).strip("_") or "col"
    name, i = base, 2
    while name in seen:
        name, i = f"{base}_{i}", i + 1
    seen.add(name)
    return name


def _to_num(cell: str):
    s = str(cell).strip()
    if not _NUM_RE.match(s):
        return None
    s = re.sub(r"[,$£€%\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def load_universe(csv_rel: str):
    """Returns (backend_shim, field_catalog) for one WTQ table.
    field_catalog: list of (field_name, dtype_label, sample_values)."""
    df = pd.read_csv(WTQ_ROOT / csv_rel, dtype=str, keep_default_na=False,
                     on_bad_lines="skip")
    seen: set = set()
    df.columns = [_norm_header(h, seen) for h in df.columns]

    catalog = []
    for col in list(df.columns):
        series = df[col].astype(str).str.strip()
        non_empty = series[series != ""]
        nums = non_empty.map(_to_num)
        if len(non_empty) > 0 and nums.notna().mean() >= 0.8:
            df[col] = series.map(lambda s: _to_num(s))
            dtype = "number"
            samples = [v for v in nums.dropna().head(3)]
        else:
            df[col] = series
            dtype = "text"
            samples = list(non_empty.head(3))
        catalog.append((col, dtype, samples))

    # synthetic row id so the evaluator's dedup key exists (each row unique)
    df["contract_node_id"] = [f"row{i}" for i in range(len(df))]
    return SimpleNamespace(records_df=df), catalog


def catalog_text(catalog, df=None, max_rows: int = 3) -> str:
    lines = ["Columns:"]
    for name, dtype, samples in catalog:
        sample_txt = ", ".join(str(s)[:30] for s in samples[:3])
        lines.append(f"- {name} ({dtype}; e.g. {sample_txt})")
    return "\n".join(lines)
