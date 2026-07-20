"""Table value linker: column-aware cell candidates for the dynamic schema.

Transfer of the main system's entity-grounding component (org_resolver) to
tables. LOCKED RULES (pre-declared): retrieval only from the current table;
never uses gold answers/SQL/test annotations; identical logic for train/dev/
test; fixed top-k; synthetic row id excluded; typed normalization; candidates
are presented per-column, never as a bare value list.
"""
from __future__ import annotations

import re

TOP_COLS = 4        # max columns shown
TOP_CELLS = 5       # max candidate cells per column
_STOP = set("the a an of in on at for to and or is are was were what which who how many much "
            "does did do with by from as that this it its their there be been total number".split())


def _toks(s: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", str(s).casefold()) if t not in _STOP and len(t) > 1}


def link(question: str, raw_df, catalog):
    """Returns list of (column, [candidate display values]) ranked by overlap."""
    q = _toks(question)
    if not q:
        return []
    scored = []
    for col, dtype, _ in catalog:
        series = raw_df[col].dropna().astype(str)
        col_score = len(_toks(col) & q)
        cells = {}
        for v in series:
            s = v.strip()
            if not s:
                continue
            ov = len(_toks(s) & q)
            if ov > 0:
                cells.setdefault(s, ov)
        ranked = sorted(cells.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_CELLS]
        score = col_score * 2 + sum(o for _, o in ranked)
        if score > 0:
            scored.append((score, col, [c for c, _ in ranked]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(col, cells) for _, col, cells in scored[:TOP_COLS]]


def render(links) -> str:
    if not links:
        return ""
    lines = ["Relevant grounded candidates (copy values EXACTLY from here):"]
    for col, cells in links:
        if cells:
            lines.append(f"- column={col}: {cells!r}")
        else:
            lines.append(f"- column={col}")
    return "\n".join(lines)
