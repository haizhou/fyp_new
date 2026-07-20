"""WTQ evaluator adapter: dual-view projection over the core evaluator.

Subclasses the (unchanged) core RuntimeAlgebraEvaluator so that:
- all COMPUTATION (predicates, sum, extremum, counting) runs on the typed view;
- all PROJECTION (values, select, groupby keys) returns the raw display string
  the table actually shows — "2010" not 2010.0, "1,234 (est.)" not None.
- numeric-string literals a planner copied from raw samples ("1,234") are
  coerced before hitting a typed numeric column.

This file is adapter code: it lives in scripts/wtq/ and modifies nothing in
src/procurement_graph/compose/. Migration-cost accounting counts it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader import _to_num  # noqa: E402

from procurement_graph.compose.eval_runtime import EvalError, RuntimeAlgebraEvaluator  # noqa: E402

_ID = "contract_node_id"


class WTQEvaluator(RuntimeAlgebraEvaluator):
    def __init__(self, shim):
        super().__init__(shim)
        self.raw_df = shim.raw_df

    # numeric-string literal coercion for typed numeric columns
    def _pred_mask(self, pred: dict):
        op = pred.get("op")
        if op in ("eq", "gte", "lte", "in") and pred.get("field") in self.df.columns:
            field = pred["field"]
            if self.df[field].dtype != object:
                value = pred.get("value")
                def coerce(v):
                    n = _to_num(v) if isinstance(v, str) else None
                    return n if n is not None else v
                if isinstance(value, list):
                    pred = {**pred, "value": [coerce(v) for v in value]}
                else:
                    pred = {**pred, "value": coerce(value)}
        return super()._pred_mask(pred)

    def _raw(self, rows: pd.DataFrame, field: str) -> pd.Series:
        if field not in self.raw_df.columns:
            raise EvalError(f"unknown_field:{field}")
        return self.raw_df.loc[rows.index.intersection(self.raw_df.index), field]

    def _eval(self, tree: dict):
        node = tree.get("node")

        if node == "values":
            rows = super()._eval(tree["of"])
            raw = self._raw(rows, tree["field"])
            out = {s.strip() for s in raw.astype(str) if s.strip()}
            return sorted(out)

        if node == "select":
            rows = super()._eval(tree["of"])
            raw = self._raw(rows, tree["field"])
            distinct = {s.strip() for s in raw.astype(str) if s.strip()}
            if not distinct:
                raise EvalError("no_results")
            if len(distinct) > 1:
                raise EvalError(f"multiple_answers:{len(distinct)}")
            return next(iter(distinct))

        return super()._eval(tree)

    # groupby keys / argext / top come through _groups: group on raw key strings
    def _groups(self, tree: dict):
        rows = super()._eval(tree["of"])
        key, metric = tree["key"], tree.get("metric", "count")
        raw_keys = self._raw(rows, key).astype(str).str.strip()
        keep = raw_keys != ""
        rows, raw_keys = rows[keep.reindex(rows.index, fill_value=False)], raw_keys[keep]
        if rows.empty:
            return {}
        if metric == "count":
            sub = rows.assign(__k=raw_keys)
            grouped = sub.drop_duplicates(["__k", _ID]).groupby("__k")[_ID].count()
            return {str(k): float(v) for k, v in grouped.items()}
        field = tree["field"]
        if field not in rows.columns:
            raise EvalError(f"unknown_field:{field}")
        vals = pd.to_numeric(rows[field], errors="coerce").fillna(0)
        grouped = vals.groupby(raw_keys).sum()
        return {str(k): float(v) for k, v in grouped.items()}
