"""Typed validation for the compose plan algebra.

Types
-----
RECORDS  -- a subset of the flat first-party record universe
VALUES   -- a distinct set of scalar values (e.g. org names, years, CPV codes)
GROUPS   -- a mapping group-key -> number (result of a grouped aggregation)
NUMBER   -- a single number
VALUE    -- a single scalar (string/year/id)
BOOL     -- a single boolean
RANKING  -- an ordered list of [key, number] pairs

Node inventory (closed grammar; open composition)
-------------------------------------------------
filter     {"node":"filter","where":[PRED...]}                          -> RECORDS
values     {"node":"values","of":RECORDS,"field":F}                     -> VALUES
count      {"node":"count","of":RECORDS}                                -> NUMBER
size       {"node":"size","of":VALUES}                                  -> NUMBER
sum        {"node":"sum","of":RECORDS,"field":F}                        -> NUMBER
exists     {"node":"exists","of":RECORDS}                               -> BOOL
select     {"node":"select","of":RECORDS,"field":F}                     -> VALUE
extreme    {"node":"extreme","of":RECORDS,"op":"argmax|argmin","field":F} -> VALUE (record id)
groupby    {"node":"groupby","of":RECORDS,"key":F,"metric":"count|sum"[,"field":F]} -> GROUPS
argext     {"node":"argext","of":GROUPS,"op":"argmax|argmin"}           -> VALUE (group key)
top        {"node":"top","of":GROUPS,"k":int}                           -> RANKING
num        {"node":"num","value":number}                                -> NUMBER (literal)
combine    {"node":"combine","op":OP,"left":NUMBER,"right":NUMBER}      -> BOOL (gt/lt/ge/le/eq)
                                                                        -> NUMBER (diff/ratio/add)
vcompare   {"node":"vcompare","op":"gt|lt|ge|le|eq","of":VALUE,"value":literal[,"normalize":"date"]} -> BOOL
setop      {"node":"setop","op":"union|intersect|difference","left":VALUES,"right":VALUES} -> VALUES
gcombine   {"node":"gcombine","op":"gt|diff|ratio","left":GROUPS,"right":GROUPS} -> GROUPS
keys_where {"node":"keys_where","of":GROUPS,"op":"gt|ge|lt|le|eq","value":num}   -> VALUES

PRED forms (inside filter.where)
--------------------------------
{"field":F,"op":"eq|in|contains|exists|gte|lte","value":V}   -- base constraint (runtime semantics)
{"op":"not","pred":PRED}                                      -- negation
{"op":"any","preds":[PRED...]}                                -- disjunction (OR)
{"field":F,"op":"in_expr","expr":<VALUES expr>[,"negate":true]} -- semijoin / antijoin on a subtree

Every check failure raises AlgebraError with a structured reason so a planner
can be given typed feedback, mirroring the verifier-diagnostic discipline of
the main pipeline.
"""
from __future__ import annotations

from typing import Any

MAX_DEPTH = 16
MAX_NODES = 64

_BOOL_COMBINE = {"gt", "lt", "ge", "le", "eq"}
_NUM_COMBINE = {"diff", "ratio", "add"}
_SET_OPS = {"union", "intersect", "difference"}
_GCOMBINE_OPS = {"gt", "diff", "ratio"}
_CMP_OPS = {"gt", "ge", "lt", "le", "eq"}
_BASE_PRED_OPS = {"eq", "in", "contains", "exists", "gte", "lte"}


class AlgebraError(ValueError):
    def __init__(self, reason: str, path: str = "$"):
        self.reason = reason
        self.path = path
        super().__init__(f"{reason} at {path}")


def _require(cond: bool, reason: str, path: str) -> None:
    if not cond:
        raise AlgebraError(reason, path)


def validate_pred(pred: Any, path: str, state: dict) -> None:
    _require(isinstance(pred, dict), "pred_not_object", path)
    op = pred.get("op")
    if op == "not":
        validate_pred(pred.get("pred"), path + ".pred", state)
        return
    if op == "any":
        preds = pred.get("preds")
        _require(isinstance(preds, list) and len(preds) >= 2, "any_needs_two_preds", path)
        for i, p in enumerate(preds):
            validate_pred(p, f"{path}.preds[{i}]", state)
        return
    if op == "in_expr":
        _require(isinstance(pred.get("field"), str) and pred["field"], "in_expr_needs_field", path)
        sub_type = validate_expr(pred.get("expr"), path + ".expr", state)
        _require(sub_type == "VALUES", f"in_expr_expects_VALUES_got_{sub_type}", path)
        return
    _require(op in _BASE_PRED_OPS, f"unknown_pred_op:{op}", path)
    _require(isinstance(pred.get("field"), str) and pred["field"], "pred_needs_field", path)
    if op != "exists":
        _require("value" in pred, "pred_needs_value", path)


def validate_expr(tree: Any, path: str = "$", state: dict | None = None) -> str:
    """Type-check a tree; returns the result type or raises AlgebraError."""
    if state is None:
        state = {"nodes": 0, "depth": 0}
    state["nodes"] += 1
    state["depth"] = path.count(".")
    _require(state["nodes"] <= MAX_NODES, "tree_too_large", path)
    _require(state["depth"] <= MAX_DEPTH, "tree_too_deep", path)
    _require(isinstance(tree, dict), "node_not_object", path)
    node = tree.get("node")

    if node == "filter":
        where = tree.get("where", [])
        _require(isinstance(where, list), "filter_where_not_list", path)
        for i, pred in enumerate(where):
            validate_pred(pred, f"{path}.where[{i}]", state)
        return "RECORDS"
    if node == "values":
        _require(isinstance(tree.get("field"), str) and tree["field"], "values_needs_field", path)
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "RECORDS", f"values_expects_RECORDS_got_{sub}", path)
        return "VALUES"
    if node == "count":
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "RECORDS", f"count_expects_RECORDS_got_{sub}", path)
        return "NUMBER"
    if node == "size":
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "VALUES", f"size_expects_VALUES_got_{sub}", path)
        return "NUMBER"
    if node == "sum":
        _require(isinstance(tree.get("field"), str) and tree["field"], "sum_needs_field", path)
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "RECORDS", f"sum_expects_RECORDS_got_{sub}", path)
        return "NUMBER"
    if node == "exists":
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "RECORDS", f"exists_expects_RECORDS_got_{sub}", path)
        return "BOOL"
    if node == "select":
        _require(isinstance(tree.get("field"), str) and tree["field"], "select_needs_field", path)
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "RECORDS", f"select_expects_RECORDS_got_{sub}", path)
        return "VALUE"
    if node == "extreme_rows":
        _require(tree.get("op") in {"argmax", "argmin"}, "extreme_rows_op_invalid", path)
        _require(isinstance(tree.get("field"), str) and tree["field"], "extreme_rows_needs_field", path)
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "RECORDS", f"extreme_rows_expects_RECORDS_got_{sub}", path)
        return "RECORDS"
    if node == "extreme":
        _require(tree.get("op") in {"argmax", "argmin"}, "extreme_op_invalid", path)
        _require(isinstance(tree.get("field"), str) and tree["field"], "extreme_needs_field", path)
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "RECORDS", f"extreme_expects_RECORDS_got_{sub}", path)
        return "VALUE"
    if node == "groupby":
        _require(isinstance(tree.get("key"), str) and tree["key"], "groupby_needs_key", path)
        metric = tree.get("metric", "count")
        _require(metric in {"count", "sum"}, f"groupby_metric_invalid:{metric}", path)
        if metric == "sum":
            _require(isinstance(tree.get("field"), str) and tree["field"], "groupby_sum_needs_field", path)
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "RECORDS", f"groupby_expects_RECORDS_got_{sub}", path)
        return "GROUPS"
    if node == "argext":
        _require(tree.get("op") in {"argmax", "argmin"}, "argext_op_invalid", path)
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "GROUPS", f"argext_expects_GROUPS_got_{sub}", path)
        return "VALUE"
    if node == "top":
        k = tree.get("k")
        _require(isinstance(k, int) and 1 <= k <= 50, "top_k_out_of_range", path)
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "GROUPS", f"top_expects_GROUPS_got_{sub}", path)
        return "RANKING"
    if node == "num":
        _require(isinstance(tree.get("value"), (int, float)) and not isinstance(tree.get("value"), bool),
                 "num_needs_numeric_value", path)
        return "NUMBER"
    if node == "vcompare":
        _require(tree.get("op") in _CMP_OPS, "vcompare_op_invalid", path)
        _require("value" in tree, "vcompare_needs_value", path)
        norm = tree.get("normalize")
        _require(norm in (None, "date"), f"vcompare_normalize_invalid:{norm}", path)
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "VALUE", f"vcompare_expects_VALUE_got_{sub}", path)
        return "BOOL"
    if node == "combine":
        op = tree.get("op")
        _require(op in _BOOL_COMBINE | _NUM_COMBINE, f"combine_op_invalid:{op}", path)
        lt = validate_expr(tree.get("left"), path + ".left", state)
        rt = validate_expr(tree.get("right"), path + ".right", state)
        _require(lt == "NUMBER", f"combine_left_expects_NUMBER_got_{lt}", path)
        _require(rt == "NUMBER", f"combine_right_expects_NUMBER_got_{rt}", path)
        return "BOOL" if op in _BOOL_COMBINE else "NUMBER"
    if node == "setop":
        op = tree.get("op")
        _require(op in _SET_OPS, f"setop_op_invalid:{op}", path)
        lt = validate_expr(tree.get("left"), path + ".left", state)
        rt = validate_expr(tree.get("right"), path + ".right", state)
        _require(lt == "VALUES", f"setop_left_expects_VALUES_got_{lt}", path)
        _require(rt == "VALUES", f"setop_right_expects_VALUES_got_{rt}", path)
        return "VALUES"
    if node == "gcombine":
        op = tree.get("op")
        _require(op in _GCOMBINE_OPS, f"gcombine_op_invalid:{op}", path)
        lt = validate_expr(tree.get("left"), path + ".left", state)
        rt = validate_expr(tree.get("right"), path + ".right", state)
        _require(lt == "GROUPS", f"gcombine_left_expects_GROUPS_got_{lt}", path)
        _require(rt == "GROUPS", f"gcombine_right_expects_GROUPS_got_{rt}", path)
        return "GROUPS"
    if node == "keys_where":
        _require(tree.get("op") in _CMP_OPS, "keys_where_op_invalid", path)
        _require("value" in tree, "keys_where_needs_value", path)
        sub = validate_expr(tree.get("of"), path + ".of", state)
        _require(sub == "GROUPS", f"keys_where_expects_GROUPS_got_{sub}", path)
        return "VALUES"
    raise AlgebraError(f"unknown_node:{node}", path)


def validate_tree(tree: Any) -> str:
    """Full-tree validation entry point. Returns the root result type."""
    return validate_expr(tree)
