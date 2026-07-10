#!/usr/bin/env python3
"""Generate the compose probe: novel-composition questions with dual-verified oracles.

Plan-first: each instance is born as an algebra tree, evaluated by BOTH
implementations (runtime evaluator #1 and the independent raw-parquet evaluator
#2); any disagreement discards the row. Anchor mining may look at the KG to find
non-degenerate parameters, but every oracle answer comes only from dual tree
evaluation.

Ladder (compositional distance from the training distribution):
  near   N1 temporal_argmax          argext over groupby(year)
         N2 filtered_sum_compare     per-side filtered sums, gt
  mid    C1a count_ratio             ratio of two counts
         C1b sum_diff                difference of two additive sums
         C5 grouped_compare          keys whose y1 count exceeds y2 count
  far    C2 supplier_difference      set difference of supplier sets
         C3 universal_buyers         relational division via set difference
         C4 role_union_count         size of buyer∪supplier value union
  ctrl   P1 median_value             OUT OF GRAMMAR (expect abstain)
         P2 monotonic_trend          OUT OF GRAMMAR (expect abstain)

Usage: .venv/bin/python scripts/build_compose_probe.py [--per-family 40] [--seed 20260710]
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

from procurement_graph.compose.eval_runtime import RuntimeAlgebraEvaluator  # noqa: E402
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend  # noqa: E402

_spec = importlib.util.spec_from_file_location("indep", ROOT / "scripts/compose_independent_eval.py")
indep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(indep)

ADD = {"field": "value_is_additive", "op": "eq", "value": True}


def flt(*preds):
    return {"node": "filter", "where": list(preds)}


def eqp(field, value):
    return {"field": field, "op": "eq", "value": value}


def _agree(a, b) -> bool:
    if type(a) is bool or type(b) is bool:
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 0.01
    if isinstance(a, list) and isinstance(b, list):
        return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    return str(a) == str(b)


class Builder:
    def __init__(self, per_family: int, seed: int):
        self.per_family = per_family
        self.rng = random.Random(seed)
        backend = ParquetKGQueryBackend.from_directory(ROOT / "data/kg", include_evidence=False)
        self.df = backend.records_df
        self.ev1 = RuntimeAlgebraEvaluator(backend)
        self.df2 = indep.load_universe()
        self.rows: list[dict] = []
        self.rejects: dict[str, int] = {}

    # ------------------------------------------------------------ emitters
    def emit(self, family: str, band: str, question: str, tree: dict,
             answer_type: str, sensible) -> bool:
        r1 = self.ev1.run(tree)
        if r1.get("status") != "ok":
            self.rejects[f"{family}:eval1:{r1.get('reason', r1.get('status'))}"] = \
                self.rejects.get(f"{family}:eval1:{r1.get('reason', r1.get('status'))}", 0) + 1
            return False
        r2 = indep.run_tree(self.df2, tree)
        if r2.get("status") != "ok" or not _agree(r1["answer"], r2["answer"]):
            key = f"{family}:dual_disagree"
            self.rejects[key] = self.rejects.get(key, 0) + 1
            return False
        if not sensible(r1["answer"]):
            key = f"{family}:not_sensible"
            self.rejects[key] = self.rejects.get(key, 0) + 1
            return False
        n = sum(1 for r in self.rows if r["template_family"] == family)
        self.rows.append({
            "id": f"CP::{family}_{n:04d}",
            "template_family": family,
            "distance_band": band,
            "question": question,
            "expected_status": "answerable",
            "answer_type": answer_type,
            "oracle_answer": r1["answer"],
            "compose_tree": tree,
            "benchmark": "compose_probe_v1",
        })
        return True

    def done(self, family: str) -> bool:
        return sum(1 for r in self.rows if r["template_family"] == family) >= self.per_family

    # ------------------------------------------------------- anchor mining
    def buyers_with_years(self, min_years=3, min_rows=30):
        d = self.df[(self.df["buyer_name"] != "")]
        g = d.groupby("buyer_name").agg(years=("release_year", "nunique"), n=("contract_node_id", "nunique"))
        picks = g[(g["years"] >= min_years) & (g["n"] >= min_rows)].index.tolist()
        self.rng.shuffle(picks)
        return picks

    def category_cpv_years(self, min_rows=8):
        d = self.df[(self.df["tender_cpv_id"].astype(str) != "") & self.df["value_is_additive"].map(bool)]
        g = d.groupby(["tender_cpv_id", "release_year"])["contract_node_id"].nunique()
        pairs = [(cpv, int(y)) for (cpv, y), n in g.items() if n >= min_rows]
        self.rng.shuffle(pairs)
        return pairs

    def buyer_pairs(self, min_sup=5, max_sup=400):
        d = self.df[(self.df["buyer_name"] != "") & (self.df["supplier_name"] != "")]
        g = d.groupby("buyer_name")["supplier_name"].nunique()
        names = g[(g >= min_sup) & (g <= max_sup)].index.tolist()
        self.rng.shuffle(names)
        return list(zip(names[::2], names[1::2]))

    # ----------------------------------------------------------- templates
    def n1_temporal_argmax(self):
        for buyer in self.buyers_with_years():
            if self.done("N1_temporal_argmax"):
                return
            tree = {"node": "argext", "op": "argmax",
                    "of": {"node": "groupby", "of": flt(eqp("buyer_name", buyer)),
                           "key": "release_year", "metric": "count"}}
            # strict unique max only: recompute the group table to check
            groups = self.ev1.run(tree["of"])
            if groups.get("status") != "ok" or len(groups["answer"]) < 3:
                continue
            top = sorted(groups["answer"].values(), reverse=True)
            if len(top) >= 2 and top[0] == top[1]:
                continue  # tied max -> ambiguous surface, skip
            q = f"In which year did {buyer} publish the most contract notices?"
            self.emit("N1_temporal_argmax", "near", q, tree, "value", lambda a: bool(str(a)))

    def n2_filtered_sum_compare(self):
        for cpv, year in self.category_cpv_years():
            if self.done("N2_filtered_sum_compare"):
                return
            y2 = year - 1
            left = {"node": "sum", "of": flt(eqp("tender_cpv_id", cpv), eqp("release_year", year), ADD),
                    "field": "value_amount"}
            right = {"node": "sum", "of": flt(eqp("tender_cpv_id", cpv), eqp("release_year", y2), ADD),
                     "field": "value_amount"}
            s1, s2 = self.ev1.run(left), self.ev1.run(right)
            if s1.get("status") != "ok" or s2.get("status") != "ok":
                continue
            a, b = float(s1["answer"]), float(s2["answer"])
            if a <= 0 or b <= 0 or abs(a - b) < 0.05 * max(a, b):
                continue  # both sides real and not knife-edge
            tree = {"node": "combine", "op": "gt", "left": left, "right": right}
            q = (f"Considering only additive contract values, was the total value of CPV {cpv} "
                 f"notices higher in {year} than in {y2}?")
            self.emit("N2_filtered_sum_compare", "near", q, tree, "boolean", lambda a: isinstance(a, bool))

    def c1a_count_ratio(self):
        for buyer in self.buyers_with_years(min_years=2, min_rows=40):
            if self.done("C1a_count_ratio"):
                return
            cats = self.df[(self.df["buyer_name"] == buyer)]["tender_category"].astype(str)
            cats = cats[cats != ""].value_counts()
            if len(cats) < 2:
                continue
            category = cats.index[0]
            num = {"node": "count", "of": flt(eqp("buyer_name", buyer), eqp("tender_category", category))}
            den = {"node": "count", "of": flt(eqp("buyer_name", buyer))}
            tree = {"node": "combine", "op": "ratio", "left": num, "right": den}
            q = f"What fraction of all contract notices published by {buyer} are {category} notices?"
            self.emit("C1a_count_ratio", "mid", q, tree, "number",
                      lambda a: isinstance(a, float) and 0.0 < a < 1.0)

    def c1b_sum_diff(self):
        for cpv, year in self.category_cpv_years(min_rows=10):
            if self.done("C1b_sum_diff"):
                return
            y2 = year - 1
            left = {"node": "sum", "of": flt(eqp("tender_cpv_id", cpv), eqp("release_year", year), ADD),
                    "field": "value_amount"}
            right = {"node": "sum", "of": flt(eqp("tender_cpv_id", cpv), eqp("release_year", y2), ADD),
                     "field": "value_amount"}
            s1, s2 = self.ev1.run(left), self.ev1.run(right)
            if s1.get("status") != "ok" or s2.get("status") != "ok":
                continue
            if float(s1["answer"]) <= 0 or float(s2["answer"]) <= 0:
                continue
            tree = {"node": "combine", "op": "diff", "left": left, "right": right}
            q = (f"Considering only additive contract values, how much higher (or lower, as a negative "
                 f"number) was the total value of CPV {cpv} notices in {year} compared with {y2}, in GBP?")
            self.emit("C1b_sum_diff", "mid", q, tree, "number", lambda a: isinstance(a, float) and a != 0)

    def c5_grouped_compare(self):
        cats = [c for c in self.df["tender_category"].astype(str).unique() if c]
        years = sorted({int(y) for y in self.df["release_year"].dropna().unique() if 2022 <= int(y) <= 2025})
        combos = [(c, y) for c in cats for y in years if y - 1 >= 2022]
        cpvs = self.category_cpv_years(min_rows=25)
        anchors = [("tender_category", c, y) for c, y in combos] + \
                  [("tender_cpv_id", c, y) for c, y in cpvs if y - 1 >= 2022]
        self.rng.shuffle(anchors)
        for field, val, year in anchors:
            if self.done("C5_grouped_compare"):
                return
            y2 = year - 1
            g1 = {"node": "groupby", "of": flt(eqp(field, val), eqp("release_year", year)),
                  "key": "buyer_name", "metric": "count"}
            g2 = {"node": "groupby", "of": flt(eqp(field, val), eqp("release_year", y2)),
                  "key": "buyer_name", "metric": "count"}
            tree = {"node": "keys_where", "op": "eq", "value": 1,
                    "of": {"node": "gcombine", "op": "gt", "left": g1, "right": g2}}
            label = f"CPV {val}" if field == "tender_cpv_id" else f"{val}"
            q = (f"Which buyers published more {label} contract notices in {year} than in {y2}? "
                 f"Count a buyer even if it published none in {y2}.")
            self.emit("C5_grouped_compare", "mid", q, tree, "value_set",
                      lambda a: isinstance(a, list) and 1 <= len(a) <= 25)

    def c2_supplier_difference(self):
        for a, b in self.buyer_pairs():
            if self.done("C2_supplier_difference"):
                return
            left = {"node": "values", "of": flt(eqp("buyer_name", a)), "field": "supplier_name"}
            right = {"node": "values", "of": flt(eqp("buyer_name", b)), "field": "supplier_name"}
            tree = {"node": "setop", "op": "difference", "left": left, "right": right}
            q = (f"Which suppliers have been awarded contract notices by {a} "
                 f"but never by {b}?")
            self.emit("C2_supplier_difference", "far", q, tree, "value_set",
                      lambda ans: isinstance(ans, list) and 1 <= len(ans) <= 20)

    def c3_universal_buyers(self):
        # relational division (∀): buyers ALL of whose in-scope notices carry a
        # signed award date. tender_category is a function of CPV here, so the
        # universal property must be a per-notice flag that genuinely varies:
        # has_award_signed_date does (True 156k / False 8k, mixed in 195/224
        # division-year slices).
        d = self.df[(self.df["tender_cpv_id"].astype(str).str.len() == 8) & (self.df["buyer_name"] != "")]
        div = d["tender_cpv_id"].astype(str).str[:2]
        g = d.assign(_div=div).groupby(["_div", "release_year"]).agg(
            n=("contract_node_id", "nunique"),
            mix=("has_award_signed_date", lambda s: s.astype(bool).nunique()))
        anchors = [(str(dv), int(y)) for (dv, y), r in g.iterrows()
                   if r["n"] >= 12 and r["mix"] >= 2]
        self.rng.shuffle(anchors)
        for dv, year in anchors:
            if self.done("C3_universal_buyers"):
                return
            lo, hi = int(dv) * 1_000_000, int(dv) * 1_000_000 + 999_999
            base = [{"field": "tender_cpv_id", "op": "gte", "value": lo},
                    {"field": "tender_cpv_id", "op": "lte", "value": hi},
                    eqp("release_year", year)]
            everyone = {"node": "values", "of": flt(*base), "field": "buyer_name"}
            offenders = {"node": "values",
                         "of": flt(*base, {"op": "not", "pred": eqp("has_award_signed_date", True)}),
                         "field": "buyer_name"}
            tree = {"node": "setop", "op": "difference", "left": everyone, "right": offenders}
            q = (f"Among buyers that published notices under CPV division {dv} (codes {dv}000000 to "
                 f"{dv}999999) in {year}, which buyers had a recorded award signing date on every "
                 f"single one of those notices? One notice without a signing date disqualifies the buyer.")
            self.emit("C3_universal_buyers", "far", q, tree, "value_set",
                      lambda ans: isinstance(ans, list) and 1 <= len(ans) <= 25)

    def c4_role_union_count(self):
        cpvs = self.category_cpv_years(min_rows=15)
        for cpv, year in cpvs:
            if self.done("C4_role_union_count"):
                return
            base = flt(eqp("tender_cpv_id", cpv), eqp("release_year", year))
            buyers = {"node": "values", "of": base, "field": "buyer_name"}
            sups = {"node": "values", "of": base, "field": "supplier_name"}
            tree = {"node": "size", "of": {"node": "setop", "op": "union", "left": buyers, "right": sups}}
            q = (f"How many distinct organisations appear as a buyer or as a first-listed supplier "
                 f"on CPV {cpv} notices published in {year}?")
            self.emit("C4_role_union_count", "far", q, tree, "number",
                      lambda a: isinstance(a, int) and a >= 3)

    # out-of-grammar negative controls: no tree exists; correct behaviour = abstain
    def p_controls(self, n=12):
        cpvs = self.category_cpv_years(min_rows=20)[: n]
        for i, (cpv, year) in enumerate(cpvs):
            fam = "P1_median_value" if i % 2 == 0 else "P2_monotonic_trend"
            if fam == "P1_median_value":
                q = f"What is the median additive contract value of CPV {cpv} notices published in {year}?"
            else:
                q = (f"Did the yearly count of CPV {cpv} contract notices increase every single year "
                     f"from 2022 through {year}?")
            self.rows.append({
                "id": f"CP::{fam}_{i // 2:04d}", "template_family": fam,
                "distance_band": "out_of_grammar", "question": q,
                "expected_status": "unanswerable_out_of_grammar",
                "answer_type": "abstain", "oracle_answer": None,
                "compose_tree": None, "benchmark": "compose_probe_v1",
            })

    def build(self):
        self.n1_temporal_argmax()
        self.n2_filtered_sum_compare()
        self.c1a_count_ratio()
        self.c1b_sum_diff()
        self.c5_grouped_compare()
        self.c2_supplier_difference()
        self.c3_universal_buyers()
        self.c4_role_union_count()
        self.p_controls()
        return self.rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--out", default="data/qa/compose_probe_v1/probe.jsonl")
    args = ap.parse_args()

    b = Builder(args.per_family, args.seed)
    rows = b.build()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

    from collections import Counter
    fams = Counter(r["template_family"] for r in rows)
    print(f"wrote {len(rows)} rows -> {out}")
    for fam, n in sorted(fams.items()):
        print(f"  {fam:28s} {n}")
    if b.rejects:
        print("rejects:", dict(sorted(b.rejects.items())))

    rng = random.Random(args.seed)
    sample = rng.sample([r for r in rows if r["compose_tree"]], min(10, len(rows)))
    print("\n=== RAW SAMPLE (10, deterministic, human-scan) ===")
    for r in sample:
        print(json.dumps({"id": r["id"], "q": r["question"], "answer": r["oracle_answer"]},
                         default=str)[:300])


if __name__ == "__main__":
    main()
