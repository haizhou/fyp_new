#!/usr/bin/env python3
"""Independent evaluator (implementation #2) for the compose algebra.

Pure pandas over the RAW node/edge parquet tables. Deliberately imports NOTHING
from src/procurement_graph/ -- it rebuilds the flat first-party record universe
from raw artifacts with its own code, and evaluates algebra trees with its own
recursion. Shared with implementation #1 are only the *documented conventions*:

  C1  one row per contract-award; scalar buyer_name/supplier_name = first of the
      sorted distinct canonical names
  C2  counts deduplicate on contract_node_id
  C3  group keys that are empty/blank/null are excluded
  C4  money sums are exact decimal arithmetic over deduplicated rows
  C5  ranking ties break on (-value, key-string); argext ties likewise
  C6  values sets drop empty strings; multi-value cells expand

Usage:
  .venv/bin/python scripts/compose_independent_eval.py --trees trees.jsonl
  (each line: {"id": ..., "tree": {...}}) -> writes answers JSONL
Also importable: load_universe(), evaluate_tree(df, tree).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "contract_node_id"


# --------------------------------------------------------------- universe
def load_universe(kg_dir: Path | str = ROOT / "data/kg") -> pd.DataFrame:
    kg = Path(kg_dir)
    contracts = pd.read_parquet(kg / "nodes/contract_nodes.parquet")
    orgs = pd.read_parquet(kg / "nodes/org_nodes.parquet", columns=["canonical_id", "canonical_name"])
    name_of = dict(zip(orgs["canonical_id"], orgs["canonical_name"]))

    def side(edge_file: str, prefix: str) -> pd.DataFrame:
        edges = pd.read_parquet(kg / f"edges/{edge_file}", columns=[ID, "canonical_id"])
        edges["name"] = edges["canonical_id"].map(name_of)
        edges = edges[edges["name"].notna() & (edges["name"].astype(str) != "")]
        grouped = edges.groupby(ID)["name"].agg(lambda s: tuple(sorted(set(map(str, s)))))
        out = grouped.to_frame(f"{prefix}_names").reset_index()
        out[f"{prefix}_name"] = out[f"{prefix}_names"].map(lambda t: t[0] if t else "")
        return out

    df = contracts.merge(side("buyer_of.parquet", "buyer"), on=ID, how="left")
    df = df.merge(side("supplier_of.parquet", "supplier"), on=ID, how="left")
    for col in ("buyer_name", "supplier_name"):
        df[col] = df[col].fillna("")
    for col in ("buyer_names", "supplier_names"):
        df[col] = df[col].map(lambda v: v if isinstance(v, tuple) else tuple())
    return df


# ------------------------------------------------------------------ preds
def _expand(cell):
    if isinstance(cell, (list, tuple, set)):
        return list(cell)
    if cell is None:
        return []
    return [cell]


def _pred_mask(df: pd.DataFrame, pred: dict) -> pd.Series:
    op = pred.get("op")
    if op == "not":
        return ~_pred_mask(df, pred["pred"])
    if op == "any":
        out = pd.Series(False, index=df.index)
        for sub in pred["preds"]:
            out |= _pred_mask(df, sub)
        return out
    field = pred["field"]
    if field not in df.columns:
        raise EvalFail(f"unknown_field:{field}")
    col = df[field]
    if op == "in_expr":
        members = set(map(str, evaluate_tree(df, pred["expr"])))
        multi = col.map(lambda c: isinstance(c, (list, tuple, set))).any()
        if multi:
            mask = col.map(lambda c: any(str(x) in members for x in _expand(c)))
        else:
            mask = col.astype(str).isin(members)
        return ~mask if pred.get("negate") else mask
    value = pred.get("value")
    if op == "eq":
        if col.map(lambda c: isinstance(c, (list, tuple, set))).any():
            return col.map(lambda c: value in _expand(c))
        return col == value
    if op == "in":
        targets = set(value or [])
        return col.map(lambda c: any(x in targets for x in _expand(c)))
    if op == "contains":
        needle = str(value).casefold()
        return col.map(lambda c: any(needle in str(x).casefold() for x in _expand(c)))
    if op == "exists":
        return col.map(lambda c: any(x not in (None, "") for x in _expand(c)))
    if op in ("gte", "lte"):
        # evaluator v1.1: coerce numeric-string literals (shared convention)
        if isinstance(value, str):
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass
        nums = pd.to_numeric(col, errors="coerce")
        return nums >= value if op == "gte" else nums <= value
    raise EvalFail(f"unknown_pred_op:{op}")


class EvalFail(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


# ------------------------------------------------------------------- eval
def _nonblank(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def _grouped(df: pd.DataFrame, tree: dict) -> dict[str, float]:
    sub = evaluate_tree(df, tree["of"])
    key = tree["key"]
    if key not in sub.columns:
        raise EvalFail(f"unknown_field:{key}")
    sub = sub[_nonblank(sub[key])]
    if sub.empty:
        return {}
    if tree.get("metric", "count") == "count":
        series = sub.groupby(sub[key].astype(str))[ID].nunique()
    else:
        field = tree["field"]
        if field not in sub.columns:
            raise EvalFail(f"unknown_field:{field}")
        nums = pd.to_numeric(sub[field], errors="coerce").fillna(0)
        series = nums.groupby(sub[key].astype(str)).sum()
    return {str(k): float(v) for k, v in series.items()}


def evaluate_tree(df: pd.DataFrame, tree: dict):
    node = tree.get("node")
    if node == "filter":
        mask = pd.Series(True, index=df.index)
        for pred in tree.get("where", []):
            m = _pred_mask(df, pred)
            if m.dtype != bool:
                m = m.fillna(False).astype(bool)
            mask &= m
        return df[mask]
    if node == "values":
        sub = evaluate_tree(df, tree["of"])
        field = tree["field"]
        if field not in sub.columns:
            raise EvalFail(f"unknown_field:{field}")
        seen: set[str] = set()
        for cell in sub[field].dropna():
            for item in _expand(cell):
                text = str(item)
                if text and text != "nan":
                    seen.add(text)
        return sorted(seen)
    if node == "count":
        sub = evaluate_tree(df, tree["of"])
        return int(sub[ID].nunique())
    if node == "size":
        return len(evaluate_tree(df, tree["of"]))
    if node == "sum":
        sub = evaluate_tree(df, tree["of"]).drop_duplicates(ID)
        field = tree["field"]
        if field not in sub.columns:
            raise EvalFail(f"unknown_field:{field}")
        total = Decimal(0)
        for v in sub[field].dropna():
            if isinstance(v, float) and not math.isfinite(v):
                raise EvalFail("non_finite_in_sum")
            try:
                total += Decimal(str(v))
            except InvalidOperation:
                raise EvalFail(f"non_numeric_in_sum:{v!r}")
        return float(total)
    if node == "exists":
        return bool(len(evaluate_tree(df, tree["of"])))
    if node == "select":
        sub = evaluate_tree(df, tree["of"])
        field = tree["field"]
        if field not in sub.columns:
            raise EvalFail(f"unknown_field:{field}")
        uniq = {str(v) for v in sub[field].dropna() if str(v) and str(v) != "nan"}
        if not uniq:
            raise EvalFail("no_results")
        if len(uniq) > 1:
            raise EvalFail(f"multiple_answers:{len(uniq)}")
        return uniq.pop()
    if node == "extreme":
        sub = evaluate_tree(df, tree["of"]).drop_duplicates(ID)
        nums = pd.to_numeric(sub[tree["field"]], errors="coerce")
        sub, nums = sub[nums.notna()], nums.dropna()
        if sub.empty:
            raise EvalFail("no_results")
        pos = nums.values.argmax() if tree["op"] == "argmax" else nums.values.argmin()
        return str(sub.iloc[pos][ID])
    if node == "groupby":
        return _grouped(df, tree)
    if node == "argext":
        groups = evaluate_tree(df, tree["of"])
        if not groups:
            raise EvalFail("no_groups")
        items = sorted(groups.items(), key=lambda kv: ((-kv[1]) if tree["op"] == "argmax" else kv[1], kv[0]))
        return items[0][0]
    if node == "top":
        groups = evaluate_tree(df, tree["of"])
        if not groups:
            raise EvalFail("no_groups")
        ranked = sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))[: tree["k"]]
        return [[k, int(v) if v == int(v) else v] for k, v in ranked]
    if node == "num":
        v = tree["value"]
        return int(v) if float(v).is_integer() else float(v)
    if node == "combine":
        left, right = evaluate_tree(df, tree["left"]), evaluate_tree(df, tree["right"])
        op = tree["op"]
        table = {"gt": lambda: left > right, "lt": lambda: left < right,
                 "ge": lambda: left >= right, "le": lambda: left <= right,
                 "eq": lambda: left == right}
        if op in table:
            return bool(table[op]())
        if op == "diff":
            return float(Decimal(str(left)) - Decimal(str(right)))
        if op == "add":
            return float(Decimal(str(left)) + Decimal(str(right)))
        if right == 0:
            raise EvalFail("ratio_division_by_zero")
        return float(left) / float(right)
    if node == "vcompare":
        left = str(evaluate_tree(df, tree["of"]))
        right = str(tree["value"])
        if tree.get("normalize") == "date":
            left, right = left[:10], right[:10]
        return {"gt": left > right, "lt": left < right, "ge": left >= right,
                "le": left <= right, "eq": left == right}[tree["op"]]
    if node == "setop":
        left = set(evaluate_tree(df, tree["left"]))
        right = set(evaluate_tree(df, tree["right"]))
        return sorted({"union": left | right, "intersect": left & right,
                       "difference": left - right}[tree["op"]])
    if node == "gcombine":
        left, right = evaluate_tree(df, tree["left"]), evaluate_tree(df, tree["right"])
        out = {}
        for key in set(left) | set(right):
            lv, rv = left.get(key, 0.0), right.get(key, 0.0)
            if tree["op"] == "gt":
                out[key] = 1.0 if lv > rv else 0.0
            elif tree["op"] == "diff":
                out[key] = float(Decimal(str(lv)) - Decimal(str(rv)))
            elif rv != 0:
                out[key] = lv / rv
        return out
    if node == "keys_where":
        groups = evaluate_tree(df, tree["of"])
        value = float(tree["value"])
        keep = {"gt": lambda v: v > value, "ge": lambda v: v >= value,
                "lt": lambda v: v < value, "le": lambda v: v <= value,
                "eq": lambda v: v == value}[tree["op"]]
        return sorted(k for k, v in groups.items() if keep(v))
    raise EvalFail(f"unknown_node:{node}")


def run_tree(df: pd.DataFrame, tree: dict) -> dict:
    try:
        return {"status": "ok", "answer": evaluate_tree(df, tree)}
    except EvalFail as exc:
        return {"status": "failed", "reason": exc.reason}
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {"status": "failed", "reason": f"eval_error:{type(exc).__name__}:{exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", required=True, help="JSONL of {id, tree}")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    df = load_universe()
    out_path = Path(args.out) if args.out else Path(args.trees).with_suffix(".answers2.jsonl")
    with open(args.trees) as fh, open(out_path, "w") as out:
        for line in fh:
            row = json.loads(line)
            result = run_tree(df, row["tree"])
            out.write(json.dumps({"id": row["id"], **result}, default=str) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
