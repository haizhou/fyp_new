#!/usr/bin/env python3
"""Step-2.5 mechanical adapter: deterministic tree repairs for cloud planners.

Cloud APIs cannot enforce the algebra schema, so their trees fail on FORMAT
(pred-op vocabulary, missing answer-extraction wrapper, answer-shape, operand
order, hallucinated view suffixes) far more often than on logic. Each repair
below is a semantics-preserving or answer-shape variant; every variant is
re-verified by the full deterministic chain, so the oracle, not the repair
rule, decides what survives. No repair invents constraints.
"""
from __future__ import annotations

import copy
import json
from typing import Any

NUMERIC = (int, float)


def _walk(tree: Any):
    if isinstance(tree, dict):
        yield tree
        for v in tree.values():
            yield from _walk(v)
    elif isinstance(tree, list):
        for v in tree:
            yield from _walk(v)


def _fix_strict_ops(tree: dict) -> dict | None:
    """gt(v) -> gte(v+1), lt(v) -> lte(v-1) for integer literals."""
    t = copy.deepcopy(tree)
    changed = False
    for node in _walk(t):
        op = node.get("op")
        if op in ("gt", "lt") and "field" in node and isinstance(node.get("value"), NUMERIC):
            node["op"] = "gte" if op == "gt" else "lte"
            node["value"] = node["value"] + (1 if op == "gt" else -1)
            changed = True
    return t if changed else None


def _fix_column_names(tree: dict, fields: list[str]) -> dict | None:
    """Map hallucinated field names to the nearest real column (prefix match)."""
    t = copy.deepcopy(tree)
    fset = set(fields)
    changed = False
    for node in _walk(t):
        f = node.get("field")
        if isinstance(f, str) and f and f not in fset:
            base = f.split("__")[0]
            cands = [c for c in fields if c == base] or \
                    [c for c in fields if c.startswith(base + "__num")] or \
                    [c for c in fields if c.startswith(base)]
            if cands:
                node["field"] = cands[0]
                changed = True
    return t if changed else None


def _wrap_records_root(tree: dict, fields: list[str], name_hints: list[str]) -> list[dict]:
    """Bare RECORDS root -> select/values wrappers over likely answer columns."""
    out = []
    if tree.get("node") in ("filter", "extreme_rows"):
        cols = [c for c in name_hints if c in fields] or fields[:3]
        for c in cols[:3]:
            for shape in ("select", "values"):
                out.append({"node": shape, "of": copy.deepcopy(tree), "field": c})
    return out


def _debase_answer_column(tree: dict, fields: list[str]) -> dict | None:
    """Root select/values on a __part/__num/__date view -> try the base column."""
    if tree.get("node") in ("select", "values"):
        f = tree.get("field")
        if isinstance(f, str) and "__" in f:
            base = f.split("__")[0]
            if base in fields:
                t = copy.deepcopy(tree)
                t["field"] = base
                return t
    return None


def _swap_answer_shape(tree: dict) -> dict | None:
    """select <-> values at the root (unique-value vs list mismatch)."""
    if tree.get("node") in ("select", "values"):
        t = copy.deepcopy(tree)
        t["node"] = "values" if tree["node"] == "select" else "select"
        return t
    return None


def _swap_diff_operands(tree: dict) -> dict | None:
    """combine diff/ratio: negative-result questions usually want |a-b|."""
    if tree.get("node") == "combine" and tree.get("op") in ("diff", "ratio"):
        t = copy.deepcopy(tree)
        t["left"], t["right"] = t["right"], t["left"]
        return t
    return None


def _numeric_view_upgrade(tree: dict, fields: list[str]) -> dict | None:
    """extreme/extreme_rows/sum over a text column that has a __num twin."""
    t = copy.deepcopy(tree)
    changed = False
    for node in _walk(t):
        if node.get("node") in ("extreme", "extreme_rows", "sum", "groupby"):
            for key in ("field",):
                f = node.get(key)
                if isinstance(f, str) and f in fields and not f.endswith("__num") \
                        and f + "__num" in fields:
                    node[key] = f + "__num"
                    changed = True
    return t if changed else None


def repair_variants(tree: dict, fields: list[str], name_hints: list[str] | None = None,
                    max_variants: int = 12) -> list[dict]:
    """Ordered, deduplicated repair variants for one tree (original NOT included)."""
    name_hints = name_hints or []
    variants: list[dict] = []
    for fn in (lambda t: _fix_strict_ops(t),
               lambda t: _debase_answer_column(t, fields),
               lambda t: _fix_column_names(t, fields),
               lambda t: _numeric_view_upgrade(t, fields),
               lambda t: _swap_answer_shape(t),
               lambda t: _swap_diff_operands(t)):
        v = fn(tree)
        if v:
            variants.append(v)
            for fn2 in (lambda t: _numeric_view_upgrade(t, fields),
                        lambda t: _swap_answer_shape(t)):
                v2 = fn2(v)
                if v2:
                    variants.append(v2)
    variants.extend(_wrap_records_root(tree, fields, name_hints))
    seen, out = set(), []
    for v in variants:
        k = json.dumps(v, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out[:max_variants]
