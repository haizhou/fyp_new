#!/usr/bin/env python3
"""Training-data engine for the compose algebra: random typed trees, plan-first.

Samples type-correct trees from a compositional recipe space (random scope
blocks x random operation wrappers, nestable), renders each node to an English
clause compositionally, evaluates with BOTH implementations, and keeps only
dual-agreeing, non-degenerate rows. No teacher, no LLM anywhere: supervision is
fully verified by construction.

Every row carries a SHAPE SIGNATURE (the tree skeleton: node labels + pred ops,
no leaf literals). The A/B split for the compositional-generalisation test cuts
on these signatures, entire shapes moving together.

Usage:
  .venv/bin/python scripts/build_compose_train.py --n 8000 --seed 20260710
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402
from procurement_graph.compose.eval_runtime import RuntimeAlgebraEvaluator  # noqa: E402
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend  # noqa: E402

_spec = importlib.util.spec_from_file_location("indep", ROOT / "scripts/compose_independent_eval.py")
indep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(indep)

ADD = {"field": "value_is_additive", "op": "eq", "value": True}


def eqp(field, value):
    return {"field": field, "op": "eq", "value": value}


def flt(*preds):
    return {"node": "filter", "where": list(preds)}


def signature(tree) -> str:
    """Tree skeleton: node labels + ops, leaf literals stripped."""
    if isinstance(tree, dict):
        if "node" in tree:
            parts = [tree["node"]]
            if "op" in tree:
                parts.append(str(tree["op"]))
            if "metric" in tree:
                parts.append(str(tree["metric"]))
            kids = [signature(v) for k, v in sorted(tree.items())
                    if k in ("of", "left", "right", "expr", "where")]
            return f"{'.'.join(parts)}({','.join(k for k in kids if k)})"
        if "op" in tree:  # pred
            inner = [signature(v) for k, v in sorted(tree.items()) if k in ("pred", "preds", "expr")]
            return f"p:{tree['op']}({','.join(k for k in inner if k)})"
        return ""
    if isinstance(tree, list):
        return ",".join(s for s in (signature(v) for v in tree) if s)
    return ""


class Sampler:
    """Random scope blocks + operation wrappers, with compositional English."""

    def __init__(self, df, rng: random.Random):
        self.rng = rng
        d = df[(df["buyer_name"] != "")]
        self.buyers = d["buyer_name"].value_counts()
        self.buyers = list(self.buyers[(self.buyers >= 8) & (self.buyers <= 3000)].index)
        s = df[df["supplier_name"] != ""]["supplier_name"].value_counts()
        self.suppliers = list(s[(s >= 5) & (s <= 1500)].index)
        c = df[df["tender_cpv_id"].astype(str).str.len() == 8]["tender_cpv_id"].value_counts()
        self.cpvs = list(c[c >= 8].index)
        self.years = sorted({int(y) for y in df["release_year"].dropna().unique() if 2022 <= int(y) <= 2026})
        self.cats = [x for x in df["tender_category"].astype(str).unique() if x]

    # ---------------------------------------------------------- scope blocks
    def scope(self, want_money: bool = False):
        """Returns (preds, clause). 1-3 random atomic preds, maybe OR/NOT."""
        rng = self.rng
        pool = []
        pool.append((lambda: (dict(field="buyer_name", op="eq", value=rng.choice(self.buyers)),)))
        pool.append((lambda: (dict(field="supplier_name", op="eq", value=rng.choice(self.suppliers)),)))
        pool.append((lambda: (dict(field="release_year", op="eq", value=rng.choice(self.years)),)))
        pool.append((lambda: (dict(field="tender_category", op="eq", value=rng.choice(self.cats)),)))
        pool.append((lambda: (dict(field="tender_cpv_id", op="eq", value=str(rng.choice(self.cpvs))),)))
        n = rng.choice([1, 1, 2, 2, 3])
        preds, clauses = [], []
        used = set()
        for maker in rng.sample(pool, min(n + 2, len(pool))):
            (p,) = maker()
            if p["field"] in used:
                continue
            used.add(p["field"])
            preds.append(p)
            clauses.append(self._pred_clause(p))
            if len(preds) >= n:
                break
        # random negation / disjunction decorations (25% each, disjoint fields)
        if rng.random() < 0.25:
            f = rng.choice([x for x in ("tender_category", "release_year") if x not in used] or ["tender_category"])
            if f == "tender_category":
                v = rng.choice(self.cats)
                preds.append({"op": "not", "pred": dict(field=f, op="eq", value=v)})
                clauses.append(f"not classified as {v}")
            else:
                v = rng.choice(self.years)
                preds.append({"op": "not", "pred": dict(field=f, op="eq", value=v)})
                clauses.append(f"not published in {v}")
        elif rng.random() < 0.25 and "release_year" not in used and len(self.years) >= 2:
            y1, y2 = rng.sample(self.years, 2)
            preds.append({"op": "any", "preds": [dict(field="release_year", op="eq", value=y1),
                                                 dict(field="release_year", op="eq", value=y2)]})
            clauses.append(f"published in {y1} or {y2}")
        if want_money:
            preds.append(dict(ADD))
            clauses.append("counting only additive contract values")
        # v2: clause-order randomisation — the check-2 audit measured 97.9->53.2
        # under meaning-preserving reorders, i.e. the v1 model anchored on the
        # fixed rendering order. Randomising here teaches order invariance.
        rng.shuffle(clauses)
        return preds, "; ".join(clauses)

    @staticmethod
    def _pred_clause(p) -> str:
        f, v = p["field"], p.get("value")
        return {"buyer_name": f"published by {v}", "supplier_name": f"awarded to {v}",
                "release_year": f"published in {v}", "tender_category": f"in the {v} category",
                "tender_cpv_id": f"under CPV {v}"}[f]

    def filt(self, want_money=False):
        preds, clause = self.scope(want_money)
        return {"node": "filter", "where": preds}, clause

    # --------------------------------------------------------- tree recipes
    def sample(self):
        """Returns (tree, question, answer_type)."""
        rng = self.rng
        recipes = [self.r_count, self.r_sum, self.r_exists, self.r_values, self.r_size,
                   self.r_groupby_argext, self.r_top, self.r_combine_counts,
                   self.r_combine_sums, self.r_vs_ratio, self.r_setop, self.r_keys_where,
                   self.r_bind, self.r_extreme, self.r_nested_bind_agg, self.r_universal]
        return rng.choice(recipes)()

    def r_count(self):
        f, c = self.filt()
        return {"node": "count", "of": f}, f"How many contract notices were {c}?", "number"

    def r_sum(self):
        f, c = self.filt(want_money=True)
        return ({"node": "sum", "of": f, "field": "value_amount"},
                f"What is the total contract value, in GBP, of notices {c}?", "number")

    def r_exists(self):
        f, c = self.filt()
        return {"node": "exists", "of": f}, f"Were there any contract notices {c}?", "boolean"

    def r_values(self):
        f, c = self.filt()
        field = self.rng.choice(["buyer_name", "supplier_name", "tender_cpv_id"])
        noun = {"buyer_name": "buyers", "supplier_name": "first-listed suppliers",
                "tender_cpv_id": "CPV codes"}[field]
        return ({"node": "values", "of": f, "field": field},
                f"Which distinct {noun} appear on contract notices {c}?", "value_set")

    def r_size(self):
        tree, q, _ = self.r_values()
        return ({"node": "size", "of": tree},
                q.replace("Which distinct", "How many distinct").rstrip("?") + "?", "number")

    def r_groupby_argext(self):
        f, c = self.filt()
        key = self.rng.choice(["release_year", "buyer_name", "supplier_name"])
        op = self.rng.choice(["argmax", "argmin"])
        noun = {"release_year": "year", "buyer_name": "buyer", "supplier_name": "first-listed supplier"}[key]
        side = "most" if op == "argmax" else "fewest"
        tree = {"node": "argext", "op": op,
                "of": {"node": "groupby", "of": f, "key": key, "metric": "count"}}
        return tree, f"Considering contract notices {c}: which {noun} accounts for the {side} of them?", "value"

    def r_top(self):
        f, c = self.filt()
        key = self.rng.choice(["buyer_name", "supplier_name", "release_year"])
        k = self.rng.choice([3, 5])
        noun = {"buyer_name": "buyers", "supplier_name": "first-listed suppliers", "release_year": "years"}[key]
        tree = {"node": "top", "k": k, "of": {"node": "groupby", "of": f, "key": key, "metric": "count"}}
        return tree, f"Rank the top {k} {noun} by number of contract notices {c}. Give each with its count.", "ranking"

    def r_combine_counts(self):
        f1, c1 = self.filt()
        f2, c2 = self.filt()
        op = self.rng.choice(["gt", "diff"])
        left = {"node": "count", "of": f1}
        right = {"node": "count", "of": f2}
        tree = {"node": "combine", "op": op, "left": left, "right": right}
        if op == "gt":
            q = f"Are there more contract notices {c1} than notices {c2}?"
            return tree, q, "boolean"
        return tree, (f"How many more contract notices were {c1} than {c2}? "
                      f"Answer with the difference (negative if fewer)."), "number"

    def r_combine_sums(self):
        f1, c1 = self.filt(want_money=True)
        f2, c2 = self.filt(want_money=True)
        op = self.rng.choice(["gt", "diff"])
        tree = {"node": "combine", "op": op,
                "left": {"node": "sum", "of": f1, "field": "value_amount"},
                "right": {"node": "sum", "of": f2, "field": "value_amount"}}
        if op == "gt":
            return tree, f"Is the total contract value of notices {c1} higher than that of notices {c2}?", "boolean"
        return tree, f"By how much, in GBP, does the total value of notices {c1} exceed that of notices {c2}?", "number"

    def r_vs_ratio(self):
        f1, c1 = self.filt()
        preds2 = [p for p in f1["where"] if not (p.get("field") and self.rng.random() < 0.0)]
        extra = {"field": "tender_category", "op": "eq", "value": self.rng.choice(self.cats)}
        narrow = {"node": "filter", "where": list(f1["where"]) + [extra]}
        tree = {"node": "combine", "op": "ratio",
                "left": {"node": "count", "of": narrow}, "right": {"node": "count", "of": f1}}
        return tree, (f"Of the contract notices {c1}, what fraction are in the {extra['value']} category? "
                      f"Answer as a number between 0 and 1."), "number"

    def r_setop(self):
        if self.rng.random() < 0.3:
            # cross-role union (the C4 pattern v2 regressed on): organisations
            # appearing as a buyer OR as a first-listed supplier in one scope
            f, c = self.filt()
            tree = {"node": "setop", "op": "union",
                    "left": {"node": "values", "of": f, "field": "buyer_name"},
                    "right": {"node": "values", "of": f, "field": "supplier_name"}}
            q = (f"organisations appear as a buyer or as a first-listed supplier "
                 f"on contract notices {c}")
            if self.rng.random() < 0.5:
                return ({"node": "size", "of": tree}, f"How many distinct {q}?", "number")
            return tree, f"Which distinct {q}?", "value_set"
        f1, c1 = self.filt()
        f2, c2 = self.filt()
        field = self.rng.choice(["buyer_name", "supplier_name"])
        noun = {"buyer_name": "buyers", "supplier_name": "first-listed suppliers"}[field]
        op = self.rng.choice(["difference", "intersect", "union"])
        tree = {"node": "setop", "op": op,
                "left": {"node": "values", "of": f1, "field": field},
                "right": {"node": "values", "of": f2, "field": field}}
        phr = {"difference": f"appear on notices {c1} but on no notice {c2}",
               "intersect": f"appear both on notices {c1} and on notices {c2}",
               "union": f"appear on notices {c1}, on notices {c2}, or both"}[op]
        if op == "union" and self.rng.random() < 0.5:
            return ({"node": "size", "of": tree}, f"How many distinct {noun} {phr}?", "number")
        return tree, f"Which {noun} {phr}?", "value_set"

    def r_keys_where(self):
        key = self.rng.choice(["buyer_name", "supplier_name"])
        noun = {"buyer_name": "buyers", "supplier_name": "first-listed suppliers"}[key]
        if self.rng.random() < 0.5 and len(self.years) >= 2:
            # same-anchor / different-year ELLIPTICAL variant — the exact content
            # pattern the C5 probe exposed (model read the year as a threshold)
            anchor_field = self.rng.choice(["tender_cpv_id", "tender_category"])
            anchor = str(self.rng.choice(self.cpvs)) if anchor_field == "tender_cpv_id" \
                else self.rng.choice(self.cats)
            y1, y2 = self.rng.sample(self.years, 2)
            f1 = flt(eqp(anchor_field, anchor), eqp("release_year", y1))
            f2 = flt(eqp(anchor_field, anchor), eqp("release_year", y2))
            label = f"CPV {anchor}" if anchor_field == "tender_cpv_id" else anchor
            q = (f"Which {noun} published more {label} contract notices in {y1} than in {y2}? "
                 f"Count one even if it published none in {y2}.")
        else:
            f1, c1 = self.filt()
            f2, c2 = self.filt()
            q = (f"Which {noun} have more contract notices {c1} than notices {c2}? "
                 f"Include those with none in the second group.")
        g1 = {"node": "groupby", "of": f1, "key": key, "metric": "count"}
        g2 = {"node": "groupby", "of": f2, "key": key, "metric": "count"}
        tree = {"node": "keys_where", "op": "eq", "value": 1,
                "of": {"node": "gcombine", "op": "gt", "left": g1, "right": g2}}
        return tree, q, "value_set"

    def r_universal(self):
        # relational division (C3): entities ALL of whose in-scope notices carry
        # a per-notice flag -> everyone MINUS those with a counterexample
        f, c = self.filt()
        key = self.rng.choice(["buyer_name", "supplier_name"])
        noun = {"buyer_name": "buyers", "supplier_name": "first-listed suppliers"}[key]
        flag, phrase = self.rng.choice([
            ("has_award_signed_date", "a recorded award signing date"),
            ("has_contract_period", "a recorded contract period"),
            ("value_is_additive", "an additive (award- or contract-sourced) value"),
        ])
        everyone = {"node": "values", "of": f, "field": key}
        offenders = {"node": "values",
                     "of": {"node": "filter",
                            "where": list(f["where"]) + [{"op": "not", "pred": eqp(flag, True)}]},
                     "field": key}
        tree = {"node": "setop", "op": "difference", "left": everyone, "right": offenders}
        return tree, (f"Among {noun} that appear on contract notices {c}, which had {phrase} on "
                      f"every single one of those notices? One notice without it disqualifies."), "value_set"

    def r_bind(self):
        f_inner, c_inner = self.filt()
        f_outer, c_outer = self.filt()
        field = self.rng.choice(["buyer_name", "supplier_name"])
        inner = {"node": "values", "of": f_inner, "field": field}
        negate = self.rng.random() < 0.4
        pred = {"field": field, "op": "in_expr", "expr": inner}
        if negate:
            pred["negate"] = True
        outer = {"node": "filter", "where": list(f_outer["where"]) + [pred]}
        role = {"buyer_name": "buyer", "supplier_name": "first-listed supplier"}[field]
        cond = f"whose {role} does {'NOT ' if negate else ''}appear on any notice {c_inner}"
        agg = self.rng.choice(["count", "exists"])
        if agg == "count":
            return {"node": "count", "of": outer}, f"How many contract notices {c_outer} have a {role} {cond.split(' ', 2)[2]}?", "number"
        return {"node": "exists", "of": outer}, f"Is there any contract notice {c_outer} {cond}?", "boolean"

    def r_extreme(self):
        f, c = self.filt(want_money=True)
        op = self.rng.choice(["argmax", "argmin"])
        side = "highest" if op == "argmax" else "lowest"
        tree = {"node": "extreme", "op": op, "of": f, "field": "value_amount"}
        return tree, (f"Among contract notices {c}, which single notice has the {side} value? "
                      f"Answer with its contract_node_id."), "value"

    def r_nested_bind_agg(self):
        f_inner, c_inner = self.filt()
        field = "supplier_name"
        inner = {"node": "values", "of": f_inner, "field": field}
        f_outer, c_outer = self.filt(want_money=True)
        outer = {"node": "filter", "where": list(f_outer["where"]) + [
            {"field": field, "op": "in_expr", "expr": inner}]}
        tree = {"node": "sum", "of": outer, "field": "value_amount"}
        return tree, (f"What is the total contract value, in GBP, of notices {c_outer} whose "
                      f"first-listed supplier also appears on notices {c_inner}?"), "number"


def sensible(answer, answer_type) -> bool:
    if answer_type == "number":
        if isinstance(answer, bool) or answer is None:
            return False
        return answer != 0 and abs(float(answer)) < 1e13
    if answer_type == "boolean":
        return isinstance(answer, bool)
    if answer_type == "value_set":
        return isinstance(answer, list) and 1 <= len(answer) <= 30
    if answer_type == "ranking":
        return isinstance(answer, list) and len(answer) >= 2
    if answer_type == "value":
        return bool(str(answer))
    return False


def agree(a, b) -> bool:
    if type(a) is bool or type(b) is bool:
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 0.01
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--out", default="data/qa/compose_train_v1/pool.jsonl")
    ap.add_argument("--max-attempts-mult", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    backend = ParquetKGQueryBackend.from_directory(ROOT / "data/kg", include_evidence=False)
    ev1 = RuntimeAlgebraEvaluator(backend)
    df2 = indep.load_universe()
    sampler = Sampler(backend.records_df, rng)

    rows, seen_q, rejects = [], set(), {}
    attempts = 0
    while len(rows) < args.n and attempts < args.n * args.max_attempts_mult:
        attempts += 1
        try:
            tree, question, answer_type = sampler.sample()
        except (IndexError, ValueError):
            continue
        if question in seen_q:
            continue
        try:
            validate_tree(tree)
        except AlgebraError as e:
            rejects[f"invalid:{e.reason[:30]}"] = rejects.get(f"invalid:{e.reason[:30]}", 0) + 1
            continue
        r1 = ev1.run(tree)
        if r1.get("status") != "ok" or not sensible(r1["answer"], answer_type):
            rejects["not_sensible_or_failed"] = rejects.get("not_sensible_or_failed", 0) + 1
            continue
        r2 = indep.run_tree(df2, tree)
        if r2.get("status") != "ok" or not agree(r1["answer"], r2["answer"]):
            rejects["dual_disagree"] = rejects.get("dual_disagree", 0) + 1
            continue
        seen_q.add(question)
        rows.append({"id": f"CT::{len(rows):05d}", "question": question,
                     "tree": tree, "answer": r1["answer"], "answer_type": answer_type,
                     "shape_signature": signature(tree)})
        if len(rows) % 500 == 0:
            print(f"{len(rows)}/{args.n} (attempts {attempts})", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

    shapes = {}
    for r in rows:
        shapes[r["shape_signature"]] = shapes.get(r["shape_signature"], 0) + 1
    print(f"\nwrote {len(rows)} rows, {len(shapes)} distinct shape signatures -> {out}")
    print("top shapes:", sorted(shapes.values(), reverse=True)[:10])
    print("rejects:", dict(sorted(rejects.items(), key=lambda kv: -kv[1])[:6]))

    sample = random.Random(args.seed).sample(rows, min(10, len(rows)))
    print("\n=== RAW SAMPLE (10, human-scan) ===")
    for r in sample:
        print(json.dumps({"q": r["question"][:150], "a": str(r["answer"])[:80]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
