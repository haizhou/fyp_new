"""PACS family x depth tree templates (spec v2.2, frozen).

21 cells: F1-F7 x L1/L2/L3. Each template returns (tree, params, template_id).
Intent sampling is KG-grounded; trees are executed by BOTH evaluators by the
generator driver (not here). Depth definitions per spec Axis 3:
  L1 atomic: one filter + one reduction
  L2 simple: 2-3 operators
  L3 nested: semijoin/anti-join, per-side filtered comparison, set operations,
             group-wise combination.
Decoration predicates (NOT / OR) are available to every scope so the driver can
steer a cell's unseen-shape quota by shifting shape signatures.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ADD = {"field": "value_is_additive", "op": "eq", "value": True}


def eqp(field, value):
    return {"field": field, "op": "eq", "value": value}


def flt(*preds):
    return {"node": "filter", "where": list(preds)}


def _sum(records):
    return {"node": "sum", "of": records, "field": "value_amount"}


def _count(records):
    return {"node": "count", "of": records}


def _values(records, field):
    return {"node": "values", "of": records, "field": field}


class Anchors:
    """KG-grounded anchor mining shared by all templates."""

    def __init__(self, records_df, rng: random.Random):
        self.rng = rng
        d = records_df
        b = d[d["buyer_name"] != ""]["buyer_name"].value_counts()
        self.buyers = list(b[(b >= 10) & (b <= 3000)].index)
        s = d[d["supplier_name"] != ""]["supplier_name"].value_counts()
        self.suppliers = list(s[(s >= 5) & (s <= 1500)].index)
        c = d[d["tender_cpv_id"].astype(str).str.len() == 8]["tender_cpv_id"].value_counts()
        self.cpvs = list(c[c >= 10].index)
        self.years = sorted({int(y) for y in d["release_year"].dropna().unique() if 2022 <= int(y) <= 2026})
        self.cats = [x for x in d["tender_category"].astype(str).unique() if x]
        # buyers with activity in >= 3 years (for temporal templates)
        g = d[d["buyer_name"] != ""].groupby("buyer_name")["release_year"].nunique()
        self.multi_year_buyers = list(g[g >= 3].index)
        # buyer pairs sharing a category (for comparison templates)
        self.buyer_pairs = None  # built lazily by driver if needed

    def buyer(self):
        return self.rng.choice(self.buyers)

    def supplier(self):
        return self.rng.choice(self.suppliers)

    def cpv(self):
        return str(self.rng.choice(self.cpvs))

    def year(self):
        return self.rng.choice(self.years)

    def two_years(self):
        return self.rng.sample(self.years, 2)

    def cat(self):
        return self.rng.choice(self.cats)

    def decoration(self, used_fields):
        """Optional NOT/OR decoration to shift shape signature (unseen steering)."""
        rng = self.rng
        if "tender_category" not in used_fields and rng.random() < 0.5:
            return {"op": "not", "pred": eqp("tender_category", self.cat())}
        y1, y2 = self.two_years()
        return {"op": "any", "preds": [eqp("release_year", y1), eqp("release_year", y2)]}


# --------------------------------------------------------------------------
# F1 spending & volume
def f1_l1(a: Anchors):
    if a.rng.random() < 0.5:
        b, y = a.buyer(), a.year()
        return (_count(flt(eqp("buyer_name", b), eqp("release_year", y))),
                {"buyer": b, "year": y}, "f1_count_buyer_year")
    c, y = a.cpv(), a.year()
    return (_sum(flt(eqp("tender_cpv_id", c), eqp("release_year", y), dict(ADD))),
            {"cpv": c, "year": y}, "f1_sum_cpv_year")


def f1_l2(a: Anchors):
    b, y, cat = a.buyer(), a.year(), a.cat()
    if a.rng.random() < 0.5:
        return (_sum(flt(eqp("buyer_name", b), eqp("tender_category", cat),
                         eqp("release_year", y), dict(ADD))),
                {"buyer": b, "cat": cat, "year": y}, "f1_sum_buyer_cat_year")
    y1, y2 = a.two_years()
    return (_count(flt(eqp("buyer_name", b),
                       {"op": "any", "preds": [eqp("release_year", y1), eqp("release_year", y2)]})),
            {"buyer": b, "years": [y1, y2]}, "f1_count_buyer_or_years")


def f1_l3(a: Anchors):
    # spend on notices whose first-listed supplier also served category C
    b, cat = a.buyer(), a.cat()
    members = _values(flt(eqp("tender_category", cat)), "supplier_name")
    outer = {"node": "filter", "where": [eqp("buyer_name", b), dict(ADD),
             {"field": "supplier_name", "op": "in_expr", "expr": members}]}
    return (_sum(outer), {"buyer": b, "cat": cat}, "f1_sum_bind_supplier_cat")


# F2 temporal change
def f2_l1(a: Anchors):
    b, y = a.rng.choice(a.multi_year_buyers), a.year()
    return ({"node": "exists", "of": flt(eqp("buyer_name", b), eqp("release_year", y))},
            {"buyer": b, "year": y}, "f2_exists_buyer_year")


def f2_l2(a: Anchors):
    anchor_field, anchor = (("tender_cpv_id", a.cpv()) if a.rng.random() < 0.5
                            else ("tender_category", a.cat()))
    y1, y2 = a.two_years()
    left = _sum(flt(eqp(anchor_field, anchor), eqp("release_year", y1), dict(ADD)))
    right = _sum(flt(eqp(anchor_field, anchor), eqp("release_year", y2), dict(ADD)))
    op = a.rng.choice(["gt", "diff"])
    return ({"node": "combine", "op": op, "left": left, "right": right},
            {"anchor": anchor, "y1": y1, "y2": y2, "op": op}, f"f2_year_{op}")


def f2_l3(a: Anchors):
    if a.rng.random() < 0.5:
        b = a.rng.choice(a.multi_year_buyers)
        grouped = {"node": "groupby", "of": flt(eqp("buyer_name", b)),
                   "key": "release_year", "metric": "count"}
        return ({"node": "argext", "op": "argmax", "of": grouped},
                {"buyer": b}, "f2_argmax_year")
    # per-side different categories across two years
    y1, y2 = a.two_years()
    c1, c2 = a.rng.sample(a.cats, 2)
    left = _count(flt(eqp("tender_category", c1), eqp("release_year", y1)))
    right = _count(flt(eqp("tender_category", c2), eqp("release_year", y2)))
    return ({"node": "combine", "op": "gt", "left": left, "right": right},
            {"c1": c1, "y1": y1, "c2": c2, "y2": y2}, "f2_perside_cat_year")


# F3 supplier concentration
def f3_l1(a: Anchors):
    s = a.supplier()
    return (_count(flt(eqp("supplier_name", s))), {"supplier": s}, "f3_count_supplier")


def f3_l2(a: Anchors):
    scope_field, scope = (("tender_cpv_id", a.cpv()) if a.rng.random() < 0.5
                          else ("tender_category", a.cat()))
    metric = a.rng.choice(["count", "sum"])
    preds = [eqp(scope_field, scope)] + ([dict(ADD)] if metric == "sum" else [])
    grouped = {"node": "groupby", "of": flt(*preds), "key": "supplier_name", "metric": metric}
    if metric == "sum":
        grouped["field"] = "value_amount"
    if a.rng.random() < 0.5:
        return ({"node": "top", "of": grouped, "k": a.rng.choice([3, 5])},
                {"scope": scope, "metric": metric}, "f3_topk_supplier")
    return ({"node": "argext", "op": "argmax", "of": grouped},
            {"scope": scope, "metric": metric}, "f3_argmax_supplier")


def f3_l3(a: Anchors):
    # concentration inside a computed scope: suppliers of buyers active in category C
    cat, y = a.cat(), a.year()
    buyer_set = _values(flt(eqp("tender_category", cat)), "buyer_name")
    outer = {"node": "filter", "where": [eqp("release_year", y),
             {"field": "buyer_name", "op": "in_expr", "expr": buyer_set}]}
    grouped = {"node": "groupby", "of": outer, "key": "supplier_name", "metric": "count"}
    return ({"node": "top", "of": grouped, "k": 3},
            {"cat": cat, "year": y}, "f3_topk_in_bound_scope")


# F4 cross-buyer comparison
def f4_l1(a: Anchors):
    b, cat = a.buyer(), a.cat()
    return (_sum(flt(eqp("buyer_name", b), eqp("tender_category", cat), dict(ADD))),
            {"buyer": b, "cat": cat}, "f4_sum_one_side")


def f4_l2(a: Anchors):
    b1, b2 = a.rng.sample(a.buyers, 2)
    cat = a.cat()
    metric = a.rng.choice(["count", "sum"])
    def side(b):
        preds = [eqp("buyer_name", b), eqp("tender_category", cat)]
        if metric == "sum":
            preds.append(dict(ADD))
            return _sum(flt(*preds))
        return _count(flt(*preds))
    op = a.rng.choice(["gt", "diff"])
    return ({"node": "combine", "op": op, "left": side(b1), "right": side(b2)},
            {"b1": b1, "b2": b2, "cat": cat, "metric": metric}, f"f4_two_buyers_{op}")


def f4_l3(a: Anchors):
    # per-side independent filters: A on cat1/y1 vs B on cat2/y2
    b1, b2 = a.rng.sample(a.buyers, 2)
    c1, c2 = a.rng.sample(a.cats, 2)
    y1, y2 = a.two_years()
    left = _sum(flt(eqp("buyer_name", b1), eqp("tender_category", c1),
                    eqp("release_year", y1), dict(ADD)))
    right = _sum(flt(eqp("buyer_name", b2), eqp("tender_category", c2),
                     eqp("release_year", y2), dict(ADD)))
    return ({"node": "combine", "op": "gt", "left": left, "right": right},
            {"b1": b1, "c1": c1, "y1": y1, "b2": b2, "c2": c2, "y2": y2},
            "f4_perside_full")


# F5 overlap & exclusion
def f5_l1(a: Anchors):
    b = a.buyer()
    return ({"node": "size", "of": _values(flt(eqp("buyer_name", b)), "supplier_name")},
            {"buyer": b}, "f5_size_suppliers")


def f5_l2(a: Anchors):
    b1, b2 = a.rng.sample(a.buyers, 2)
    op = a.rng.choice(["intersect", "union"])
    tree = {"node": "setop", "op": op,
            "left": _values(flt(eqp("buyer_name", b1)), "supplier_name"),
            "right": _values(flt(eqp("buyer_name", b2)), "supplier_name")}
    if op == "union" or a.rng.random() < 0.5:
        tree = {"node": "size", "of": tree}
    return (tree, {"b1": b1, "b2": b2, "op": op}, f"f5_{op}")


def f5_l3(a: Anchors):
    b1, b2 = a.rng.sample(a.buyers, 2)
    if a.rng.random() < 0.5:
        tree = {"node": "setop", "op": "difference",
                "left": _values(flt(eqp("buyer_name", b1)), "supplier_name"),
                "right": _values(flt(eqp("buyer_name", b2)), "supplier_name")}
        return (tree, {"b1": b1, "b2": b2}, "f5_only_a_never_b")
    # anti-join formulation: notices of b1 whose supplier never served b2
    members = _values(flt(eqp("buyer_name", b2)), "supplier_name")
    outer = {"node": "filter", "where": [eqp("buyer_name", b1),
             {"field": "supplier_name", "op": "in_expr", "expr": members, "negate": True}]}
    return (_count(outer), {"b1": b1, "b2": b2}, "f5_antijoin_count")


# F6 relational composition
def f6_l1(a: Anchors):
    s = a.supplier()
    return (_values(flt(eqp("supplier_name", s)), "buyer_name"),
            {"supplier": s}, "f6_buyers_of_supplier")


def f6_l2(a: Anchors):
    s, y = a.supplier(), a.year()
    return ({"node": "size", "of": _values(flt(eqp("supplier_name", s),
                                               eqp("release_year", y)), "buyer_name")},
            {"supplier": s, "year": y}, "f6_size_buyers_year")


def f6_l3(a: Anchors):
    # which OTHER buyers do suppliers of buyer K serve
    k = a.buyer()
    suppliers_of_k = _values(flt(eqp("buyer_name", k)), "supplier_name")
    reached = _values({"node": "filter", "where": [
        {"field": "supplier_name", "op": "in_expr", "expr": suppliers_of_k}]}, "buyer_name")
    k_only = _values(flt(eqp("buyer_name", k)), "buyer_name")
    tree = {"node": "setop", "op": "difference", "left": reached, "right": k_only}
    return (tree, {"k": k}, "f6_other_buyers_via_suppliers")


# F7 disclosure & compliance
_FLAGS = [("has_award_signed_date", "award signing date"),
          ("has_contract_period", "contract period")]


def f7_l1(a: Anchors):
    flag, _ = a.rng.choice(_FLAGS)
    y = a.year()
    return (_count(flt(eqp("release_year", y), {"op": "not", "pred": eqp(flag, True)})),
            {"year": y, "flag": flag}, "f7_count_missing_flag")


def f7_l2(a: Anchors):
    flag, _ = a.rng.choice(_FLAGS)
    scope_field, scope = (("tender_category", a.cat()) if a.rng.random() < 0.5
                          else ("tender_cpv_id", a.cpv()))
    tree = _values(flt(eqp(scope_field, scope), {"op": "not", "pred": eqp(flag, True)}),
                   "buyer_name")
    return (tree, {"scope": scope, "flag": flag}, "f7_buyers_with_missing")


def f7_l3(a: Anchors):
    # universal: buyers ALL of whose in-scope notices carry the flag
    flag, _ = a.rng.choice(_FLAGS)
    cat, y = a.cat(), a.year()
    base = [eqp("tender_category", cat), eqp("release_year", y)]
    everyone = _values({"node": "filter", "where": list(base)}, "buyer_name")
    offenders = _values({"node": "filter",
                         "where": list(base) + [{"op": "not", "pred": eqp(flag, True)}]},
                        "buyer_name")
    return ({"node": "setop", "op": "difference", "left": everyone, "right": offenders},
            {"cat": cat, "year": y, "flag": flag}, "f7_universal_flag")


TEMPLATES = {
    ("F1", "L1"): f1_l1, ("F1", "L2"): f1_l2, ("F1", "L3"): f1_l3,
    ("F2", "L1"): f2_l1, ("F2", "L2"): f2_l2, ("F2", "L3"): f2_l3,
    ("F3", "L1"): f3_l1, ("F3", "L2"): f3_l2, ("F3", "L3"): f3_l3,
    ("F4", "L1"): f4_l1, ("F4", "L2"): f4_l2, ("F4", "L3"): f4_l3,
    ("F5", "L1"): f5_l1, ("F5", "L2"): f5_l2, ("F5", "L3"): f5_l3,
    ("F6", "L1"): f6_l1, ("F6", "L2"): f6_l2, ("F6", "L3"): f6_l3,
    ("F7", "L1"): f7_l1, ("F7", "L2"): f7_l2, ("F7", "L3"): f7_l3,
}
