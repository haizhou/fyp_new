"""Deterministic translation: frozen flat gold plans -> compose algebra trees.

Used by the regression gate only: if the algebra + evaluator are correct, every
translated old question must reproduce its frozen oracle answer. Families whose
parameters live only in the question surface (the documented 'surface-parse
assistance' comparison families) are declared out of scope and reported as
skipped, mirroring the dual-oracle audit's own coverage statement.
"""
from __future__ import annotations

from typing import Any

# resolver kind -> (anchor field for the inner filter, projected field)
_RESOLVERS: dict[str, tuple[tuple[str, ...], str]] = {
    "suppliers_of_buyer": (("buyer_name", "buyer"), "supplier_name"),
    "buyers_of_supplier": (("supplier_name", "supplier"), "buyer_name"),
    "suppliers_for_cpv": (("tender_cpv_id", "cpv"), "supplier_name"),
    "cpvs_of_buyer": (("buyer_name", "buyer"), "tender_cpv_id"),
    "buyers_in_category": (("tender_category", "category"), "buyer_name"),
}


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}


def _parse_surface_date(surface: str) -> str:
    """Deterministic '1 May 2025' -> '2025-05-01'; ISO strings pass through."""
    text = surface.strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    parts = text.replace(",", " ").split()
    if len(parts) == 3:
        day_s, month_s, year_s = parts
        if day_s.isalpha():  # 'May 1 2025'
            day_s, month_s = month_s, day_s
        month = _MONTHS.get(month_s.lower())
        if month and day_s.isdigit() and year_s.isdigit():
            return f"{int(year_s):04d}-{month:02d}-{int(day_s):02d}"
    raise Unsupported(f"unparseable_pivot_date:{surface}")


class Unsupported(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


_K_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10}


def _parse_surface_k(question: str) -> int | None:
    import re

    text = question.casefold()
    m = re.search(r"top\s+(\d{1,2})\b", text)
    if m:
        return int(m.group(1))
    m = re.search(r"top\s+(" + "|".join(_K_WORDS) + r")\b", text)
    if m:
        return _K_WORDS[m.group(1)]
    m = re.search(r"\b(?:which|what)\s+(" + "|".join(_K_WORDS) + r")\b", text)
    if m:
        return _K_WORDS[m.group(1)]
    m = re.search(r"\b(" + "|".join(_K_WORDS) + r"|\d{1,2})\s+(?:buyers|suppliers|organisations|organizations)\b", text)
    if m:
        word = m.group(1)
        return _K_WORDS.get(word) or int(word)
    return None


def _subquery_pred(field: str, payload: dict) -> dict:
    kind = str(payload.get("resolve") or "")
    if kind == "suppliers_in_year_category":
        where = []
        if payload.get("year") is not None:
            where.append({"field": "release_year", "op": "eq", "value": payload["year"]})
        if payload.get("category"):
            where.append({"field": "tender_category", "op": "eq", "value": payload["category"]})
        if not where:
            raise Unsupported("subquery_missing_params")
        inner = {"node": "values", "of": {"node": "filter", "where": where}, "field": "supplier_name"}
        return {"field": field, "op": "in_expr", "expr": inner}
    if kind not in _RESOLVERS:
        raise Unsupported(f"unknown_resolver:{kind}")
    (anchor_field, payload_key), projected = _RESOLVERS[kind]
    anchor_value = payload.get(payload_key)
    if anchor_value in (None, ""):
        raise Unsupported(f"subquery_missing_anchor:{kind}")
    inner = {
        "node": "values",
        "of": {"node": "filter", "where": [{"field": anchor_field, "op": "eq", "value": anchor_value}]},
        "field": projected,
    }
    return {"field": field, "op": "in_expr", "expr": inner}


def _preds(constraints: list[dict]) -> list[dict]:
    preds: list[dict] = []
    for c in constraints:
        field, op, value = c.get("field", ""), c.get("op", ""), c.get("value")
        if op == "in_subquery":
            preds.append(_subquery_pred(field, value or {}))
        elif op == "between":
            lo, hi = (value or [None, None])[0], (value or [None, None])[1]
            preds.append({"field": field, "op": "gte", "value": lo})
            preds.append({"field": field, "op": "lte", "value": hi})
        elif op in {"eq", "in", "contains", "exists", "gte", "lte"}:
            pred = {"field": field, "op": op}
            if op != "exists":
                pred["value"] = value
            preds.append(pred)
        else:
            raise Unsupported(f"constraint_op:{op}")
    return preds


def gold_plan_to_tree(row: dict) -> dict:
    """Translate one benchmark row's gold plan into an algebra tree."""
    gp: dict = row.get("gold_plan") or {}
    op = str(gp.get("answer_operation") or row.get("answer_operation") or "")
    md: dict = gp.get("metadata") or {}
    constraints: list[dict] = list(gp.get("constraints") or row.get("constraints") or [])
    answer_field = str(gp.get("answer_field") or row.get("answer_field") or "")
    records: dict[str, Any] = {"node": "filter", "where": _preds(constraints)}

    if op == "count":
        return {"node": "count", "of": records}
    if op == "sum":
        return {"node": "sum", "of": records, "field": answer_field or "value_amount"}
    if op == "exists":
        return {"node": "exists", "of": records}
    if op == "select_unique":
        if not answer_field:
            raise Unsupported("select_missing_answer_field")
        return {"node": "select", "of": records, "field": answer_field}
    if op == "distinct_set":
        if not answer_field:
            raise Unsupported("distinct_missing_answer_field")
        return {"node": "values", "of": records, "field": answer_field}
    if op in {"argmax", "argmin"}:
        field = str(gp.get("sort_field") or md.get("metric_field") or "value_amount")
        family = str(md.get("template_family") or row.get("template_family") or "")
        if family == "min_max":
            # family convention (builder filtered _val > 0): extremum over non-zero values
            records = dict(records)
            records["where"] = list(records["where"]) + [
                {"op": "not", "pred": {"field": field, "op": "lte", "value": 0}}
            ]
        return {"node": "extreme", "of": records, "op": op, "field": field}
    if op == "rank_top_k":
        group_by = str(md.get("group_by") or "buyer_name")
        metric = str(md.get("metric") or "count")
        # Declared surface assistance for the historically under-specified top-k
        # family: the question's own count word wins over a conflicting metadata k
        # (production planners read k from the surface too).
        k_surface = _parse_surface_k(str(row.get("question") or ""))
        k = int(k_surface or md.get("k") or 3)
        grouped: dict[str, Any] = {"node": "groupby", "of": records, "key": group_by, "metric": metric}
        if metric == "sum":
            grouped["field"] = str(md.get("metric_field") or "value_amount")
        return {"node": "top", "of": grouped, "k": k}
    if op == "compare":
        family = str(md.get("template_family") or row.get("template_family") or "")
        params: dict = md.get("compare_params") or {}
        if family == "numeric_threshold" and params.get("threshold") is not None:
            total = {"node": "sum", "of": records, "field": "value_amount"}
            return {"node": "combine", "op": "gt", "left": total,
                    "right": {"node": "num", "value": float(params["threshold"])}}
        if family == "date_relation" and params.get("pivot_date_surface"):
            pivot = _parse_surface_date(str(params["pivot_date_surface"]))
            direction = "gt" if str(params.get("direction") or "after") == "after" else "lt"
            picked = {"node": "select", "of": records, "field": "award_date_signed"}
            return {"node": "vcompare", "op": direction, "of": picked, "value": pivot, "normalize": "date"}
        if family == "comparison" and isinstance(params.get("sides"), list) and len(params["sides"]) == 2:
            shared = [c for c in constraints if c.get("field") != "buyer_name"]
            if params.get("year") is not None and not any(c.get("field") == "release_year" for c in shared):
                shared.append({"field": "release_year", "op": "eq", "value": params["year"]})
            sides = []
            for name in params["sides"]:
                where = _preds(shared + [{"field": "buyer_name", "op": "eq", "value": name}])
                sides.append({"node": "count", "of": {"node": "filter", "where": where}})
            return {"node": "combine", "op": "gt", "left": sides[0], "right": sides[1]}
        raise Unsupported(f"compare_family_surface_borne:{family or 'unknown'}")
    raise Unsupported(f"answer_operation:{op or 'missing'}")
