"""Runtime evaluator (implementation #1) for the compose algebra.

Evaluates a validated tree over the flat first-party record universe
(ParquetKGQueryBackend.records_df). Shares the KG loading path and constraint
matching semantics with the production system on purpose: this is the
system-side implementation. The independent implementation
(scripts/compose_independent_eval.py) shares no code with this module.

Frozen conventions enforced here (identical to the production executor):
- counts deduplicate on contract_node_id
- group keys that are empty strings or null are excluded from grouping
- money sums use exact Decimal arithmetic; non-finite values are barred
- ties break deterministically: (-value, key-string) for max-style ranking
"""
from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd

from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend, _series_matches
from procurement_graph.qa.benchmark.models import Constraint

from .algebra import AlgebraError, validate_tree

_ID = "contract_node_id"


class EvalError(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class RuntimeAlgebraEvaluator:
    def __init__(self, backend: ParquetKGQueryBackend):
        self.df = backend.records_df

    # ------------------------------------------------------------- public
    def run(self, tree: dict) -> dict:
        """Validate + evaluate. Returns an answer envelope, never raises."""
        try:
            rtype = validate_tree(tree)
        except AlgebraError as exc:
            return {"status": "invalid", "reason": exc.reason, "path": exc.path}
        try:
            answer = self._eval(tree)
        except EvalError as exc:
            return {"status": "failed", "reason": exc.reason}
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return {"status": "failed", "reason": f"eval_error:{type(exc).__name__}:{exc}"}
        return {"status": "ok", "type": rtype, "answer": answer}

    # ------------------------------------------------------------ helpers
    def _field(self, name: str) -> pd.Series:
        if name not in self.df.columns:
            raise EvalError(f"unknown_field:{name}")
        return self.df[name]

    def _pred_mask(self, pred: dict) -> pd.Series:
        op = pred.get("op")
        if op == "not":
            return ~self._pred_mask(pred["pred"])
        if op == "any":
            mask = pd.Series(False, index=self.df.index)
            for sub in pred["preds"]:
                mask |= self._pred_mask(sub)
            return mask
        if op == "in_expr":
            members = self._eval(pred["expr"])  # VALUES -> sorted list
            series = self._field(pred["field"])
            mask = _series_matches(series, Constraint(field=pred["field"], op="in", value=list(members)))
            if pred.get("negate"):
                mask = ~mask
            return mask.fillna(False) if mask.dtype == object else mask
        series = self._field(pred["field"])
        value = pred.get("value")
        # evaluator v1.1: numeric range predicates coerce numeric-string literals
        # (CPV ids are string-typed; "24000000" in a gte/lte is a numeric intent)
        if op in ("gte", "lte") and isinstance(value, str):
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass
        constraint = Constraint(field=pred["field"], op=op, value=value)
        mask = _series_matches(series, constraint)
        if getattr(mask, "dtype", None) == object:
            mask = mask.fillna(False).astype(bool)
        return mask.fillna(False) if hasattr(mask, "fillna") else mask

    def _records(self, tree: dict) -> pd.DataFrame:
        mask = pd.Series(True, index=self.df.index)
        for pred in tree.get("where", []):
            mask &= self._pred_mask(pred)
        return self.df.loc[mask]

    @staticmethod
    def _clean_keys(series: pd.Series) -> pd.Series:
        return series[series.notna() & (series.astype(str).str.strip() != "")]

    def _groups(self, tree: dict) -> dict[str, float]:
        rows = self._eval(tree["of"])
        key, metric = tree["key"], tree.get("metric", "count")
        if key not in rows.columns:
            raise EvalError(f"unknown_field:{key}")
        rows = rows[rows[key].notna() & (rows[key].astype(str).str.strip() != "")]
        if rows.empty:
            return {}
        if metric == "count":
            grouped = rows.drop_duplicates([key, _ID]).groupby(key, dropna=True)[_ID].count()
            return {str(k): float(v) for k, v in grouped.items()}
        field = tree["field"]
        if field not in rows.columns:
            raise EvalError(f"unknown_field:{field}")
        vals = pd.to_numeric(rows[field], errors="coerce").fillna(0)
        grouped = vals.groupby(rows[key]).sum()
        return {str(k): float(v) for k, v in grouped.items()}

    # --------------------------------------------------------------- eval
    def _eval(self, tree: dict) -> Any:
        node = tree["node"]

        if node == "filter":
            return self._records(tree)

        if node == "values":
            rows = self._eval(tree["of"])
            field = tree["field"]
            if field not in rows.columns:
                raise EvalError(f"unknown_field:{field}")
            out: set[str] = set()
            for value in rows[field].dropna():
                items = value if isinstance(value, (list, tuple, set)) else (value,)
                for item in items:
                    text = str(item)
                    if text and text != "nan":
                        out.add(text)
            return sorted(out)

        if node == "count":
            rows = self._eval(tree["of"])
            return int(rows[_ID].nunique()) if _ID in rows.columns else int(len(rows))

        if node == "size":
            return len(self._eval(tree["of"]))

        if node == "sum":
            rows = self._eval(tree["of"])
            field = tree["field"]
            if field not in rows.columns:
                raise EvalError(f"unknown_field:{field}")
            rows = rows.drop_duplicates(_ID) if _ID in rows.columns else rows
            total = Decimal("0")
            for value in rows[field].dropna():
                if isinstance(value, float) and not math.isfinite(value):
                    raise EvalError("non_finite_value_in_sum")
                try:
                    total += Decimal(str(value))
                except InvalidOperation as exc:
                    raise EvalError(f"non_numeric_value_in_sum:{value!r}") from exc
            return float(total)

        if node == "exists":
            return bool(len(self._eval(tree["of"])) > 0)

        if node == "select":
            rows = self._eval(tree["of"])
            field = tree["field"]
            if field not in rows.columns:
                raise EvalError(f"unknown_field:{field}")
            distinct = {str(v) for v in rows[field].dropna() if str(v) and str(v) != "nan"}
            if not distinct:
                raise EvalError("no_results")
            if len(distinct) > 1:
                raise EvalError(f"multiple_answers:{len(distinct)}")
            return next(iter(distinct))

        if node == "extreme":
            rows = self._eval(tree["of"])
            field = tree["field"]
            if field not in rows.columns:
                raise EvalError(f"unknown_field:{field}")
            rows = rows.drop_duplicates(_ID) if _ID in rows.columns else rows
            numeric = pd.to_numeric(rows[field], errors="coerce")
            rows = rows[numeric.notna()]
            numeric = numeric.dropna()
            if rows.empty:
                raise EvalError("no_results")
            idx = numeric.idxmax() if tree["op"] == "argmax" else numeric.idxmin()
            return str(rows.loc[idx, _ID])

        if node == "groupby":
            return self._groups(tree)

        if node == "argext":
            groups = self._eval(tree["of"])
            if not groups:
                raise EvalError("no_groups")
            if tree["op"] == "argmax":
                return min(groups.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            return min(groups.items(), key=lambda kv: (kv[1], kv[0]))[0]

        if node == "top":
            groups = self._eval(tree["of"])
            if not groups:
                raise EvalError("no_groups")
            ranked = sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))[: tree["k"]]
            return [[k, int(v) if float(v).is_integer() else float(v)] for k, v in ranked]

        if node == "num":
            return float(tree["value"]) if not float(tree["value"]).is_integer() else int(tree["value"])

        if node == "vcompare":
            left = str(self._eval(tree["of"]))
            right = str(tree["value"])
            if tree.get("normalize") == "date":
                left, right = left[:10], right[:10]
            op = tree["op"]
            if op == "gt":
                return bool(left > right)
            if op == "lt":
                return bool(left < right)
            if op == "ge":
                return bool(left >= right)
            if op == "le":
                return bool(left <= right)
            return bool(left == right)

        if node == "combine":
            left, right = self._eval(tree["left"]), self._eval(tree["right"])
            op = tree["op"]
            if op == "gt":
                return bool(left > right)
            if op == "lt":
                return bool(left < right)
            if op == "ge":
                return bool(left >= right)
            if op == "le":
                return bool(left <= right)
            if op == "eq":
                return bool(left == right)
            if op == "diff":
                return float(Decimal(str(left)) - Decimal(str(right)))
            if op == "add":
                return float(Decimal(str(left)) + Decimal(str(right)))
            if op == "ratio":
                if right == 0:
                    raise EvalError("ratio_division_by_zero")
                return float(left) / float(right)

        if node == "setop":
            left, right = set(self._eval(tree["left"])), set(self._eval(tree["right"]))
            op = tree["op"]
            if op == "union":
                return sorted(left | right)
            if op == "intersect":
                return sorted(left & right)
            return sorted(left - right)

        if node == "gcombine":
            left, right = self._eval(tree["left"]), self._eval(tree["right"])
            keys = set(left) | set(right)
            op = tree["op"]
            out: dict[str, float] = {}
            for key in keys:
                lv, rv = left.get(key, 0.0), right.get(key, 0.0)
                if op == "gt":
                    out[key] = 1.0 if lv > rv else 0.0
                elif op == "diff":
                    out[key] = float(Decimal(str(lv)) - Decimal(str(rv)))
                elif op == "ratio":
                    if rv == 0:
                        continue  # undefined ratio: key dropped, deterministically
                    out[key] = lv / rv
            return out

        if node == "keys_where":
            groups = self._eval(tree["of"])
            op, value = tree["op"], float(tree["value"])
            cmp = {
                "gt": lambda v: v > value,
                "ge": lambda v: v >= value,
                "lt": lambda v: v < value,
                "le": lambda v: v <= value,
                "eq": lambda v: v == value,
            }[op]
            return sorted(k for k, v in groups.items() if cmp(v))

        raise EvalError(f"unknown_node:{node}")
