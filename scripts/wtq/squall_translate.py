#!/usr/bin/env python3
"""Squall gold SQL -> compose algebra: SQL expression NORMALIZATION and
LOWERING into the frozen v2 algebra (phase 1 — no algebra/schema/evaluator
changes; every construct lowers to existing v2 nodes or is censused).
Attribution flag WTQ_LOWERING: "v3" (default) enables the normalization pack
(in-list, between, scalar comparisons, unified scalar-expression lowering);
"v2" reproduces the pre-pack translator for the 4-way audit.

Two numbers this produces:
  1. Expressibility census: % of Squall gold programs the algebra can express
     (with skip reasons for the rest — the honest coverage boundary).
  2. Executor oracle accuracy: on the translated subset, execute the gold-derived
     tree with OUR evaluator on OUR loader and compare to the WTQ target.
     This is the ceiling a perfect planner could reach through our stack.

SQL dialect notes (Squall): table `w`, columns c1..cn positional, suffixed
views (c5_number, c5_year, ...); `id` is the 1-based row number. We translate
plain cN and cN_number only; other suffixes are counted as out-of-scope.

Usage: .venv/bin/python scripts/wtq/squall_translate.py [--limit N] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/wtq"))

from loader import load_universe  # noqa: E402

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402
from wtq_eval import WTQEvaluator  # noqa: E402

SQUALL = Path("/var/tmp/cicada/squall/squall-main/data/squall.json")


import os as _os
LOWERING_ON = _os.environ.get("WTQ_LOWERING", "v3") == "v3"


class Skip(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


_COL_RE = re.compile(r"^c(\d+)(?:_(\w+))?$")


def resolve_col(tok: str, colmap: dict) -> str:
    if tok == "id":
        raise Skip("row_id_navigation")
    m = _COL_RE.match(tok)
    if not m:
        raise Skip(f"unknown_column_token")
    idx, suffix = int(m.group(1)), m.group(2)
    if suffix not in (None, "number", "year", "minimum_year", "maximum_year"):
        raise Skip(f"column_transform:{suffix}")
    entry = colmap.get(idx)
    if entry is None:
        raise Skip("column_index_out_of_range")
    name, dtype = entry
    _SUFFIX_VIEWS = {"number": "__num", "year": "__min_year", "minimum_year": "__min_year",
                     "maximum_year": "__max_year"}  # first/second stay censused: ambiguous parse rejected
    if suffix in _SUFFIX_VIEWS and not (suffix == "number" and dtype == "number"):
        aux = f"{name}{_SUFFIX_VIEWS[suffix]}"
        if aux in colmap.get(0, set()):
            return aux  # loader-materialized derived view
        if suffix == "number" and dtype == "number":
            return name
        raise Skip(f"column_transform:{suffix}")
    return name


def literal(tok_type: str, tok_val: str):
    if tok_type == "Literal.Number" or re.fullmatch(r"-?\d+(?:\.\d+)?", tok_val):
        f = float(tok_val)
        return int(f) if f.is_integer() else f
    if tok_type == "Literal.String":
        return tok_val.strip("'\"")
    raise Skip("literal_unparsed")


_GROUND_CACHE: dict = {}


def ground(value, field, df):
    """Squall literals are lowercase-normalized; map them back to the raw cell
    value of THIS table (the value a planner would copy from the catalog)."""
    if field not in df.columns:
        return value
    series = df[field].dropna()
    if series.dtype != object:  # numeric column: align literal type
        if isinstance(value, str):
            from loader import _to_num
            n = _to_num(value)
            return n if n is not None else value
        return value
    key = (id(df), field)
    if key not in _GROUND_CACHE:
        m = {}
        for raw in series.astype(str):
            n1 = re.sub(r"\s+", " ", raw.casefold().strip())
            n2 = re.sub(r"[^a-z0-9]+", "", n1)
            m.setdefault(n1, raw)
            if n2:
                m.setdefault(n2, raw)
        _GROUND_CACHE[key] = m
    m = _GROUND_CACHE[key]
    s = re.sub(r"\s+", " ", str(value).casefold().strip())
    return m.get(s) or m.get(re.sub(r"[^a-z0-9]+", "", s)) or value


def split_top(toks, seps: set):
    """Split a token list on top-level separators (paren-aware)."""
    parts, cur, depth = [], [], 0
    for t in toks:
        if t[1] == "(":
            depth += 1
        elif t[1] == ")":
            depth -= 1
        if depth == 0 and t[1] in seps and cur:
            parts.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        parts.append(cur)
    return parts


def trans_conds(toks, colmap, df) -> list:
    """WHERE tokens -> list of preds (AND only; OR/subqueries are out of scope)."""
    if any(t[1] == "or" for t in toks):
        raise Skip("or_condition")
    if any(t[1] == "select" for t in toks):
        raise Skip("where_subquery")
    parts_raw = split_top(toks, {"and"})
    merged = []
    for part in parts_raw:
        if merged and "between" in [t[1] for t in merged[-1]] and \
                len([t for t in merged[-1] if t[1] == "between"]) > len([t for t in merged[-1] if t[1] == "and"]):
            merged[-1] = merged[-1] + [("Keyword", "and")] + part
        else:
            merged.append(part)
    preds = []
    for part in merged:
        vals = [t[1] for t in part]
        # col is [not] null
        if len(vals) >= 3 and vals[1] == "is":
            field = resolve_col(vals[0], colmap)
            pred = {"field": field, "op": "exists"}
            if "not" in vals[2:]:
                preds.append(pred)
            else:
                preds.append({"op": "not", "pred": pred})
            continue
        if LOWERING_ON and len(vals) >= 5 and vals[1] == "in" and vals[2] == "(":
            field = resolve_col(vals[0], colmap)
            lits = [ground(literal(part[i][0], vals[i]), field, df)
                    for i in range(3, len(vals) - 1) if vals[i] != ","]
            preds.append({"field": field, "op": "in", "value": lits})
            continue
        if LOWERING_ON and len(vals) == 5 and vals[1] == "between" and vals[3] == "and":
            field = resolve_col(vals[0], colmap)
            preds.append({"field": field, "op": "gte", "value": ground(literal(part[2][0], vals[2]), field, df)})
            preds.append({"field": field, "op": "lte", "value": ground(literal(part[4][0], vals[4]), field, df)})
            continue
        if len(vals) != 3:
            raise Skip("cond_shape")
        field = resolve_col(vals[0], colmap)
        cmp_tok = vals[1]
        value = ground(literal(part[2][0], vals[2]), field, df)
        if cmp_tok == "=":
            preds.append({"field": field, "op": "eq", "value": value})
        elif cmp_tok in ("!=", "<>"):
            preds.append({"op": "not", "pred": {"field": field, "op": "eq", "value": value}})
        elif cmp_tok == ">=":
            preds.append({"field": field, "op": "gte", "value": value})
        elif cmp_tok == "<=":
            preds.append({"field": field, "op": "lte", "value": value})
        elif cmp_tok == ">":
            preds.append({"op": "not", "pred": {"field": field, "op": "lte", "value": value}})
        elif cmp_tok == "<":
            preds.append({"op": "not", "pred": {"field": field, "op": "gte", "value": value}})
        else:
            raise Skip(f"cmp_op:{cmp_tok}")
    return preds


def trans_select(toks, colmap, df, want_number=False) -> dict:
    """One SELECT statement -> tree. toks include the leading 'select'."""
    vals = [t[1] for t in toks]
    if vals[0] != "select":
        raise Skip("not_select")
    if "from" not in vals:
        raise Skip("no_from")
    fi = vals.index("from")
    agg = vals[1:fi]
    rest = toks[fi + 2:]  # skip 'from' 'w'
    rvals = [t[1] for t in rest]

    # clause boundaries in the tail
    def clause(kw):
        return rvals.index(kw) if kw in rvals else None

    wi, gi, oi, li = clause("where"), clause("group"), clause("order"), clause("limit")
    ends = sorted(i for i in (gi, oi, li) if i is not None)
    preds = []
    if wi is not None:
        wend = ends[0] if ends else len(rest)
        preds = trans_conds(rest[wi + 1:wend], colmap, df)
    base = {"node": "filter", "where": preds}

    group_key = None
    if gi is not None:
        gtoks = rvals[gi:oi if oi is not None else li if li is not None else len(rvals)]
        if len(gtoks) != 3 or gtoks[1] != "by":
            raise Skip("group_shape")
        group_key = resolve_col(gtoks[2], colmap)

    order_toks, limit_n = None, None
    if oi is not None:
        order_toks = rvals[oi + 2:li if li is not None else len(rvals)]
    if li is not None:
        limit_n = int(rvals[li + 1])
        if oi is None:
            raise Skip("limit_without_order")

    # ---- grouped queries: select K group by K order by metric desc limit n
    if group_key is not None:
        if len(agg) != 1 or resolve_col(agg[0], colmap) != group_key:
            raise Skip("group_select_mismatch")
        if order_toks is None:
            raise Skip("group_without_order")
        direction = "desc" if "desc" in order_toks else "asc"
        metric_toks = [t for t in order_toks if t not in ("asc", "desc")]
        if metric_toks[:2] == ["count", "("]:
            g = {"node": "groupby", "of": base, "key": group_key, "metric": "count"}
        elif metric_toks[:2] == ["sum", "("]:
            g = {"node": "groupby", "of": base, "key": group_key, "metric": "sum",
                 "field": resolve_col(metric_toks[2], colmap)}
        else:
            raise Skip("group_order_metric")
        if limit_n == 1:
            return {"node": "argext", "of": g,
                    "op": "argmax" if direction == "desc" else "argmin"}
        if limit_n and limit_n > 1 and direction == "desc":
            return {"node": "top", "of": g, "k": limit_n}
        raise Skip("group_limit_shape")

    # ---- ungrouped order/limit: extremum-row projection
    if order_toks is not None:
        if limit_n != 1:
            raise Skip("limit_k_rows")
        direction = "desc" if "desc" in order_toks else "asc"
        okey = [t for t in order_toks if t not in ("asc", "desc")]
        if len(okey) != 1:
            raise Skip("order_key_shape")
        okcol = resolve_col(okey[0], colmap)
        rows = {"node": "extreme_rows", "of": base, "field": okcol,
                "op": "argmax" if direction == "desc" else "argmin"}
        if len(agg) == 1:
            if want_number:
                raise Skip("scalar_extremum_in_arith")
            return {"node": "select", "of": rows, "field": resolve_col(agg[0], colmap)}
        raise Skip("order_agg_shape")

    # ---- plain aggregates
    if agg[:2] == ["count", "("]:
        if agg[2] == "*":
            return {"node": "count", "of": base}
        if agg[2] == "distinct":
            return {"node": "size", "of": {"node": "values", "of": base,
                                           "field": resolve_col(agg[3], colmap)}}
        preds.append({"field": resolve_col(agg[2], colmap), "op": "exists"})
        return {"node": "count", "of": base}
    if agg[:2] == ["sum", "("]:
        return {"node": "sum", "of": base, "field": resolve_col(agg[2], colmap)}
    if agg[:2] in (["max", "("], ["min", "("]):
        if want_number:
            raise Skip("scalar_extremum_in_arith")
        field = resolve_col(agg[2], colmap)
        rows = {"node": "extreme_rows", "of": base, "field": field,
                "op": "argmax" if agg[0] == "max" else "argmin"}
        return {"node": "select", "of": rows, "field": field}
    if agg[:2] == ["avg", "("]:
        field = resolve_col(agg[2], colmap)
        return {"node": "combine", "op": "ratio",
                "left": {"node": "sum", "of": base, "field": field},
                "right": {"node": "count", "of": base}}
    if agg[0] in ("abs", "julianday"):
        raise Skip(f"unsupported_agg:{agg[0]}")
    if len(agg) == 1:
        field = resolve_col(agg[0], colmap)
        if want_number:
            # APPROXIMATION: SQL scalar subquery = one row's value; sum equals it
            # only when the filter matches exactly one row. Differential audit
            # classifies divergences as class B (translated, semantics differ).
            return {"node": "sum", "of": base, "field": field}
        return {"node": "values", "of": base, "field": field}
    if "distinct" in agg and len(agg) == 2:
        return {"node": "values", "of": base, "field": resolve_col(agg[1], colmap)}
    raise Skip("multi_select" if "," in agg else "agg_shape")


def translate(sql_tokens, colmap, df) -> dict:
    toks = [(t[0], t[1]) for t in sql_tokens]
    vals = [t[1] for t in toks]
    # scalar arithmetic: select ( Q1 ) OP ( Q2 )
    if vals[:2] == ["select", "("]:
        body = toks[1:]
        _ops = [("-", "diff"), ("+", "add"), ("/", "ratio")]
        if LOWERING_ON:
            _ops += [(">", "gt"), (">=", "ge"), ("<", "lt"), ("<=", "le"), ("=", "eq")]
        for op_tok, op in _ops:
            parts = split_top(body, {op_tok})
            if len(parts) == 2:
                sides = []
                for p in parts:
                    inner = [t for t in p]
                    while inner and inner[0][1] == "(":
                        inner = inner[1:]
                    while inner and inner[-1][1] == ")":
                        inner = inner[:-1]
                    if len(inner) == 1 and inner[0][1].replace(".", "").replace("-", "").isdigit():
                        sides.append({"node": "num", "value": float(inner[0][1])})
                    else:
                        sides.append(trans_select(inner, colmap, df, want_number=True))
                return {"node": "combine", "op": op, "left": sides[0], "right": sides[1]}
        raise Skip("scalar_arith_shape")
    if vals.count("select") > 1:
        raise Skip("nested_subquery")
    if "*" in vals and "count" not in vals:
        raise Skip("select_star")
    return trans_select(toks, colmap, df, want_number=False)


# ---------------------------------------------------------------- scoring
def norm(x) -> str:
    s = str(x).strip().strip('"').casefold()
    s = re.sub(r"\s+", " ", s)
    try:
        f = float(s.replace(",", ""))
        return f"{f:g}"
    except ValueError:
        return s


def num_of(s: str):
    m = re.search(r"-?[\d,]+(?:\.\d+)?", str(s))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def match(pred, targets: list) -> tuple:
    """(strict, tolerant). tolerant also accepts numeric-content equality."""
    if isinstance(pred, list) and pred and isinstance(pred[0], list):
        pred = [p[0] for p in pred]
    preds = pred if isinstance(pred, list) else [pred]
    strict = sorted(norm(x) for x in preds) == sorted(norm(x) for x in targets)
    tolerant = strict
    if not strict and len(preds) == 1 and len(targets) == 1:
        pn, tn = num_of(preds[0]), num_of(targets[0])
        tolerant = pn is not None and tn is not None and abs(pn - tn) < 1e-6
    return strict, tolerant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "data/qa/wtq/squall_oracle.jsonl"))
    args = ap.parse_args()

    data = json.loads(SQUALL.read_text())
    if args.limit:
        data = data[: args.limit]

    universes: dict = {}
    skip_census: Counter = Counter()
    exec_census: Counter = Counter()
    n_strict = n_tolerant = n_translated = n_exec_ok = 0
    out_rows = []

    for e in data:
        row = {"nt": e["nt"], "tbl": e["tbl"], "tgt": e["tgt"]}
        num, grp = e["tbl"].split("_", 1)
        csv_rel = f"csv/{num}-csv/{grp}.csv"
        if csv_rel not in universes:
            try:
                shim, catalog = load_universe(csv_rel)
                orig = [c for c in catalog if "__" not in c[0]]
                cm = {i + 1: (c[0], c[1]) for i, c in enumerate(orig)}
                cm[0] = {c[0] for c in catalog if "__" in c[0]}
                universes[csv_rel] = (shim, cm)
            except Exception as exc:
                universes[csv_rel] = None
        if universes[csv_rel] is None:
            skip_census["table_load_error"] += 1
            row["skip"] = "table_load_error"
            out_rows.append(row)
            continue
        shim, colmap = universes[csv_rel]
        try:
            tree = translate(e["sql"], colmap, shim.records_df)
            validate_tree(tree)
        except Skip as exc:
            skip_census[exc.reason.split(":")[0]] += 1
            row["skip"] = exc.reason
            out_rows.append(row)
            continue
        except AlgebraError as exc:
            skip_census["translator_invalid"] += 1
            row["skip"] = f"translator_invalid:{exc.reason}"
            out_rows.append(row)
            continue
        n_translated += 1
        row["tree"] = tree
        res = WTQEvaluator(shim).run(tree)
        if res.get("status") != "ok":
            exec_census[str(res.get("reason", "?")).split(":")[0]] += 1
            row["exec"] = f"failed:{res.get('reason')}"
            out_rows.append(row)
            continue
        n_exec_ok += 1
        strict, tolerant = match(res["answer"], e["tgt"].split("|"))
        n_strict += strict
        n_tolerant += tolerant
        row.update(exec="ok", answer=str(res["answer"])[:200],
                   strict=bool(strict), tolerant=bool(tolerant))
        out_rows.append(row)

    n = len(data)
    with open(args.out, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, default=str) + "\n")

    print(f"total gold programs: {n}")
    print(f"translated (expressible, first-cut): {n_translated} = {100*n_translated/n:.2f}%")
    print(f"  executed ok: {n_exec_ok} = {100*n_exec_ok/max(1,n_translated):.2f}% of translated")
    print(f"  oracle strict:   {n_strict}/{n_translated} = {100*n_strict/max(1,n_translated):.2f}%")
    print(f"  oracle tolerant: {n_tolerant}/{n_translated} = {100*n_tolerant/max(1,n_translated):.2f}%")
    print("\nskip census (not translated):")
    for k, v in skip_census.most_common():
        print(f"  {k:28s} {v}")
    print("\nexec-failure census (translated but failed):")
    for k, v in exec_census.most_common(15):
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
