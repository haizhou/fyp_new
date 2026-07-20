#!/usr/bin/env python3
"""Attach surface channels a (independent grammar) and c (training-renderer
idiom, diagnostic only) to the PACS intent pool. Channel b (naturalized) is a
separate GPU stage. Channel c deliberately reuses the training renderer's
idiom: "; "-joined scope clauses and its stem forms — that is its purpose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/pacs"))
from surface_independent import render as render_a  # noqa: E402

_CLAUSE = {"buyer": "published by {}", "b1": "published by {}", "b2": "published by {}",
           "k": "published by {}", "supplier": "awarded to {}", "year": "published in {}",
           "y1": "published in {}", "y2": "published in {}", "cpv": "under CPV {}",
           "cat": "in the {} category", "c1": "in the {} category", "c2": "in the {} category",
           "scope": "in the {} slice", "anchor": "for {}", "metric": None, "op": None,
           "years": None, "flag": None}


def _clauses(params: dict) -> str:
    parts = []
    for key, value in params.items():
        tpl = _CLAUSE.get(key)
        if tpl:
            parts.append(tpl.format(value))
        elif key == "years" and isinstance(value, list):
            parts.append(f"published in {value[0]} or {value[1]}")
    return "; ".join(parts)


def _deco_clauses(tree) -> list[str]:
    out = []

    def walk(n):
        if isinstance(n, dict):
            if n.get("op") == "not" and isinstance(n.get("pred"), dict):
                p = n["pred"]
                if p.get("field") == "tender_category":
                    out.append(f"not classified as {p['value']}")
                elif p.get("field") == "supplier_name":
                    out.append(f"not awarded to {p['value']}")
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(tree)
    return out


_ROOT_STEM = {
    "count": "How many contract notices were {c}?",
    "sum": "What is the total contract value, in GBP, of notices {c}; counting only additive contract values?",
    "exists": "Were there any contract notices {c}?",
    "values": "Which distinct {noun} appear on contract notices {c}?",
    "size": "How many distinct {noun} appear on contract notices {c}?",
    "select": "For contract notices {c}, what is the value?",
    "extreme": "Among contract notices {c}, which single notice has the extreme value?",
    "argext": "Considering contract notices {c}: which {key} accounts for the most of them?",
    "top": "Rank the top {k} {noun} by number of contract notices {c}. Give each with its count.",
    "combine": "Is the first quantity for notices {c} greater than the second?",
    "setop": "Which {noun} satisfy the set condition over notices {c}?",
    "keys_where": "Which {noun} have more notices {c} in the first scope than the second?",
}

_NOUN = {"buyer_name": "buyers", "supplier_name": "first-listed suppliers",
         "tender_cpv_id": "CPV codes", "release_year": "years"}


def _find(tree, key, default=""):
    if isinstance(tree, dict):
        if key in tree and isinstance(tree[key], str):
            return tree[key]
        for v in tree.values():
            found = _find(v, key)
            if found:
                return found
    if isinstance(tree, list):
        for v in tree:
            found = _find(v, key)
            if found:
                return found
    return default


def render_c(row: dict) -> str | None:
    tree = row.get("compose_tree")
    if not tree:
        return None
    root = tree.get("node")
    stem = _ROOT_STEM.get(root)
    if stem is None:
        return None
    clauses = _clauses(row.get("params", {}))
    deco = _deco_clauses(tree)
    c = "; ".join(x for x in [clauses] + deco if x)
    field = _find(tree, "field") or _find(tree, "key")
    return stem.format(c=c, noun=_NOUN.get(field, "entries"),
                       key=_NOUN.get(_find(tree, "key"), "group").rstrip("s"),
                       k=tree.get("k", 3))


def main() -> None:
    pool = ROOT / "data/qa/pacs_v1/intent_pool.jsonl"
    out = ROOT / "data/qa/pacs_v1/intent_pool_surfaced.jsonl"
    rows = [json.loads(l) for l in open(pool)]
    n_a = n_c = 0
    with out.open("w") as fh:
        for row in rows:
            if row.get("expected_status") == "requires_missing_operator":
                row["question_a"] = row.get("question_seed")
                row["question_c"] = row.get("question_seed")
            else:
                qa = render_a(row)
                qc = render_c(row)
                row["question_a"], row["question_c"] = qa, qc
                n_a += qa is not None
                n_c += qc is not None
            fh.write(json.dumps(row, default=str) + "\n")
    print(f"{len(rows)} rows; channel a rendered {n_a}, channel c rendered {n_c} -> {out}")
    for row in rows[:3]:
        if row.get("question_a") and row.get("question_c"):
            print("A:", row["question_a"][:110])
            print("C:", row["question_c"][:110])
            print("---")


if __name__ == "__main__":
    main()
