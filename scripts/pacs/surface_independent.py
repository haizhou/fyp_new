"""PACS channel-a renderer: the INDEPENDENT surface grammar (spec v2.2).

Second authored voice. Shares NO stems, connectors, or ordering policy with the
training renderer (build_compose_train.py): measures lead, scopes follow as
prepositional chains ("for X, during Y, under Z"), decorations become
"excluding"/"whether-or" phrases. Renders from params PLUS tree inspection so
injected decoration predicates always reach the surface (faithfulness).
"""
from __future__ import annotations


def _money(scope: str) -> str:
    return f"In pounds, what was the total additive contract value {scope}?"


def _decorations(tree, params) -> list[str]:
    """Verbalise decoration predicates present in the tree but absent from the
    template's own parameters (unseen-steering injections)."""
    known = {str(v) for v in _flatten(params)}
    phrases = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("op") == "not" and isinstance(node.get("pred"), dict):
                p = node["pred"]
                v = str(p.get("value"))
                if p.get("field") == "tender_category" and v not in known:
                    phrases.append(f"excluding {v}-classified notices")
                elif p.get("field") == "release_year" and v not in known:
                    phrases.append(f"leaving out year {v}")
            if node.get("op") == "any":
                ys = [str(p.get("value")) for p in node.get("preds", [])
                      if isinstance(p, dict) and p.get("field") == "release_year"]
                if ys and not set(ys) <= known:
                    phrases.append(f"taking {ys[0]} and {ys[1]} together" if len(ys) == 2 else "")
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(tree)
    return [p for p in phrases if p]


def _flatten(params):
    for v in params.values():
        if isinstance(v, list):
            yield from v
        else:
            yield v


_R = {}


def r(tid):
    def deco(fn):
        _R[tid] = fn
        return fn
    return deco


# F1 -----------------------------------------------------------------------
@r("f1_count_buyer_year")
def _(p): return f"How many contract notices did {p['buyer']} put out during {p['year']}?"
@r("f1_sum_cpv_year")
def _(p): return _money(f"for CPV code {p['cpv']} during {p['year']}")
@r("f1_sum_buyer_cat_year")
def _(p): return _money(f"for {p['buyer']}, in the {p['cat']} segment, during {p['year']}")
@r("f1_count_buyer_or_years")
def _(p): return (f"Counting {p['years'][0]} and {p['years'][1]} together, how many contract "
                  f"notices did {p['buyer']} put out?")
@r("f1_sum_bind_supplier_cat")
def _(p): return _money(f"for {p['buyer']}, restricted to notices whose lead supplier also "
                        f"holds work in the {p['cat']} segment")

# F2 -----------------------------------------------------------------------
@r("f2_exists_buyer_year")
def _(p): return f"During {p['year']}, did {p['buyer']} put out any contract notice at all?"
@r("f2_year_gt")
def _(p): return (f"For {p['anchor']}, was total additive spend larger during {p['y1']} "
                  f"than during {p['y2']}?")
@r("f2_year_diff")
def _(p): return (f"For {p['anchor']}, by what amount (in pounds, negative when lower) did "
                  f"total additive spend during {p['y1']} differ from {p['y2']}?")
@r("f2_argmax_year")
def _(p): return f"Across all years on record, when did {p['buyer']} put out the largest number of contract notices?"
@r("f2_perside_cat_year")
def _(p): return (f"Compare two slices: {p['c1']} notices during {p['y1']} on one side, "
                  f"{p['c2']} notices during {p['y2']}, on the other. Does the first side hold more notices?")

# F3 -----------------------------------------------------------------------
@r("f3_count_supplier")
def _(p): return f"How many contract notices name {p['supplier']} as the lead supplier?"
@r("f3_topk_supplier")
def _(p): return (f"Within the {p['scope']} slice, list the leading suppliers by "
                  f"{'notice volume' if p['metric']=='count' else 'total additive value'}, best first, with figures.")
@r("f3_argmax_supplier")
def _(p): return (f"Within the {p['scope']} slice, which single supplier leads on "
                  f"{'notice volume' if p['metric']=='count' else 'total additive value'}?")
@r("f3_topk_in_bound_scope")
def _(p): return (f"Take the buyers active in the {p['cat']} segment; during {p['year']}, "
                  f"which three lead suppliers appear most often on those buyers' notices? Give counts.")

# F4 -----------------------------------------------------------------------
@r("f4_sum_one_side")
def _(p): return _money(f"for {p['buyer']} within the {p['cat']} segment")
@r("f4_two_buyers_gt")
def _(p): return (f"Within the {p['cat']} segment, does {p['b1']} outweigh {p['b2']} on "
                  f"{'notice volume' if p['metric']=='count' else 'total additive spend'}?")
@r("f4_two_buyers_diff")
def _(p): return (f"Within the {p['cat']} segment, by how much (negative when behind) does "
                  f"{p['b1']} lead {p['b2']} on "
                  f"{'notice volume' if p['metric']=='count' else 'total additive spend'}?")
@r("f4_perside_full")
def _(p): return (f"Two slices to weigh: {p['b1']}'s {p['c1']} spend during {p['y1']}, against "
                  f"{p['b2']}'s {p['c2']} spend during {p['y2']} (additive values throughout). "
                  f"Is the first the larger?")

# F5 -----------------------------------------------------------------------
@r("f5_size_suppliers")
def _(p): return f"Across all its notices, how many distinct lead suppliers has {p['buyer']} used?"
@r("f5_intersect")
def _(p): return f"Which lead suppliers show up on the books of {p['b1']} and of {p['b2']} alike?"
@r("f5_union")
def _(p): return (f"Pooling the books of {p['b1']} and {p['b2']}, how many distinct lead "
                  f"suppliers appear in total?")
@r("f5_only_a_never_b")
def _(p): return (f"Which lead suppliers appear on the books of {p['b1']} while staying entirely "
                  f"absent from those of {p['b2']}?")
@r("f5_antijoin_count")
def _(p): return (f"Of {p['b1']}'s contract notices, how many carry a lead supplier that has "
                  f"never once appeared on a notice of {p['b2']}?")

# F6 -----------------------------------------------------------------------
@r("f6_buyers_of_supplier")
def _(p): return f"Name every buyer on whose notices {p['supplier']} appears as lead supplier."
@r("f6_size_buyers_year")
def _(p): return (f"During {p['year']}, how many distinct buyers placed {p['supplier']} as "
                  f"lead supplier on a notice?")
@r("f6_other_buyers_via_suppliers")
def _(p): return (f"Follow {p['k']}'s lead suppliers outward: setting {p['k']} itself aside, "
                  f"which buyers do those same suppliers also work for?")

# F7 -----------------------------------------------------------------------
_FLAG_PHRASE = {"has_award_signed_date": "a recorded signing date",
                "has_contract_period": "a recorded contract period"}
@r("f7_count_missing_flag")
def _(p): return (f"During {p['year']}, how many contract notices went out without "
                  f"{_FLAG_PHRASE[p['flag']]}?")
@r("f7_buyers_with_missing")
def _(p): return (f"Within the {p['scope']} slice, name the buyers that put out at least one "
                  f"notice lacking {_FLAG_PHRASE[p['flag']]}.")
@r("f7_universal_flag")
def _(p): return (f"Looking at {p['cat']} notices from {p['year']}: which buyers kept a perfect "
                  f"record, {_FLAG_PHRASE[p['flag']]} present on every single notice? Missing it "
                  f"once disqualifies.")


def render(row: dict) -> str | None:
    """Channel-a surface for one intent row (answerable or empty_result variant)."""
    tid = str(row.get("template_id", "")).replace("#empty", "")
    fn = _R.get(tid)
    if fn is None:
        return None
    text = fn(row["params"])
    extras = _decorations(row.get("compose_tree"), row.get("params", {}))
    if extras:
        text = text.rstrip("?") + ", " + " and ".join(extras) + "?"
    return text


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    pool = Path(sys.argv[1] if len(sys.argv) > 1 else
                "data/qa/pacs_v1/intent_pool.jsonl")
    rows = [json.loads(l) for l in open(pool)]
    missing, shown = set(), 0
    for row in rows:
        if row.get("expected_status") == "requires_missing_operator":
            continue
        text = render(row)
        if text is None:
            missing.add(row.get("template_id"))
        elif shown < 12:
            print(f"[{row['intent_id']}] {text}")
            shown += 1
    print(f"\nrendered {len(rows)-len(missing)} ok; missing template ids: {sorted(missing) or 'NONE'}")
