"""Recursive JSON schema for the compose algebra (guided decoding).

Mirrors algebra.py's validator: same node inventory, same required fields.
Guided decoding enforces SYNTAX (shape + enums); algebra.validate_tree still
runs afterwards for TYPE correctness (e.g. argext requires GROUPS), which a
context-free schema cannot express. Depth is bounded by the schema's own
recursion via vLLM/xgrammar; MAX_DEPTH/MAX_NODES remain enforced post-hoc.
"""
from __future__ import annotations

_FIELDS = [
    "contract_node_id", "buyer_name", "supplier_name", "release_year",
    "tender_category", "tender_cpv_id", "tender_title", "value_amount",
    "value_is_additive", "award_date_signed", "has_award_signed_date",
]

_CMP = ["gt", "lt", "ge", "le", "eq"]


def algebra_json_schema(strict_fields: bool = True, fields: list | None = None) -> dict:
    """JSON schema for {"tree": EXPR} | {"abstain": true, "reason": str}.
    `fields` overrides the enum — per-table dynamic catalogs (e.g. WTQ)."""
    field_schema = {"type": "string"}
    if strict_fields:
        field_schema = {"type": "string", "enum": list(fields) if fields else _FIELDS}

    defs = {
        "pred": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "field": field_schema,
                        "op": {"type": "string", "enum": ["eq", "in", "contains", "gte", "lte", "exists"]},
                        "value": {},
                    },
                    "required": ["field", "op"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"op": {"const": "not"}, "pred": {"$ref": "#/$defs/pred"}},
                    "required": ["op", "pred"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"op": {"const": "any"},
                                   "preds": {"type": "array", "items": {"$ref": "#/$defs/pred"}, "minItems": 2}},
                    "required": ["op", "preds"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"field": field_schema, "op": {"const": "in_expr"},
                                   "expr": {"$ref": "#/$defs/expr"}, "negate": {"type": "boolean"}},
                    "required": ["field", "op", "expr"],
                    "additionalProperties": False,
                },
            ]
        },
        "expr": {
            "anyOf": [
                {"type": "object",
                 "properties": {"node": {"const": "filter"},
                                "where": {"type": "array", "items": {"$ref": "#/$defs/pred"}}},
                 "required": ["node", "where"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "values"}, "of": {"$ref": "#/$defs/expr"},
                                "field": field_schema},
                 "required": ["node", "of", "field"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "count"}, "of": {"$ref": "#/$defs/expr"}},
                 "required": ["node", "of"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "size"}, "of": {"$ref": "#/$defs/expr"}},
                 "required": ["node", "of"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "sum"}, "of": {"$ref": "#/$defs/expr"},
                                "field": field_schema},
                 "required": ["node", "of", "field"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "exists"}, "of": {"$ref": "#/$defs/expr"}},
                 "required": ["node", "of"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "select"}, "of": {"$ref": "#/$defs/expr"},
                                "field": field_schema},
                 "required": ["node", "of", "field"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "extreme_rows"}, "of": {"$ref": "#/$defs/expr"},
                                "op": {"type": "string", "enum": ["argmax", "argmin"]},
                                "field": field_schema},
                 "required": ["node", "of", "op", "field"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "extreme"}, "of": {"$ref": "#/$defs/expr"},
                                "op": {"type": "string", "enum": ["argmax", "argmin"]},
                                "field": field_schema},
                 "required": ["node", "of", "op", "field"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "groupby"}, "of": {"$ref": "#/$defs/expr"},
                                "key": field_schema,
                                "metric": {"type": "string", "enum": ["count", "sum"]},
                                "field": field_schema},
                 "required": ["node", "of", "key", "metric"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "argext"}, "of": {"$ref": "#/$defs/expr"},
                                "op": {"type": "string", "enum": ["argmax", "argmin"]}},
                 "required": ["node", "of", "op"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "top"}, "of": {"$ref": "#/$defs/expr"},
                                "k": {"type": "integer", "minimum": 1, "maximum": 50}},
                 "required": ["node", "of", "k"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "num"}, "value": {"type": "number"}},
                 "required": ["node", "value"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "combine"},
                                "op": {"type": "string",
                                       "enum": _CMP + ["diff", "ratio", "add"]},
                                "left": {"$ref": "#/$defs/expr"}, "right": {"$ref": "#/$defs/expr"}},
                 "required": ["node", "op", "left", "right"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "vcompare"},
                                "op": {"type": "string", "enum": _CMP},
                                "of": {"$ref": "#/$defs/expr"}, "value": {},
                                "normalize": {"type": "string", "enum": ["date"]}},
                 "required": ["node", "op", "of", "value"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "setop"},
                                "op": {"type": "string", "enum": ["union", "intersect", "difference"]},
                                "left": {"$ref": "#/$defs/expr"}, "right": {"$ref": "#/$defs/expr"}},
                 "required": ["node", "op", "left", "right"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "gcombine"},
                                "op": {"type": "string", "enum": ["gt", "diff", "ratio"]},
                                "left": {"$ref": "#/$defs/expr"}, "right": {"$ref": "#/$defs/expr"}},
                 "required": ["node", "op", "left", "right"], "additionalProperties": False},
                {"type": "object",
                 "properties": {"node": {"const": "keys_where"}, "of": {"$ref": "#/$defs/expr"},
                                "op": {"type": "string", "enum": _CMP},
                                "value": {"type": "number"}},
                 "required": ["node", "of", "op", "value"], "additionalProperties": False},
            ]
        },
    }

    return {
        "$defs": defs,
        "anyOf": [
            {"type": "object", "properties": {"tree": {"$ref": "#/$defs/expr"}},
             "required": ["tree"], "additionalProperties": False},
            {"type": "object",
             "properties": {"abstain": {"const": True}, "reason": {"type": "string", "maxLength": 200}},
             "required": ["abstain", "reason"], "additionalProperties": False},
        ],
    }
