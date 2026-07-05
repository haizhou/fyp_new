"""Typed planning: LLMs build a reviewable understanding graph, then executable slots.

Architecture shift (2026-07-03): the rule recognisers stop being the primary understander. The
supported question types become a reasoning DSL: the LLM (1) classifies the question into one of
the types below and (2) fills that type's slots. Two deterministic layers then take over:

  Plan Consistency Check (this module): is the typed plan FAITHFUL TO THE QUESTION TEXT?
    years/CPVs/thresholds in the plan must appear in the question; none may be invented;
    comparison direction words must match; buyer/supplier role phrasing must match the role the
    plan constrains; the operation must belong to the declared type. This catches the class the
    execution probe provably cannot: plans that run "successfully" with wrong semantics.

  Compile + grounding: slots become a RuntimeQuerySpec / DecompositionPlan (org surfaces resolved
  against the KG; structured errors like multiple_entity_candidates are surfaced, not guessed).
  The shared executor remains the sole answer authority.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any

from .decomposition import Binding, DecompositionPlan, SubQuery
from .entity_resolution import resolve_confident_org
from .graph_planning import _normalise_category, compile_graph_plan, normalise_graph_plan, variables_map
from .grounding import CANONICAL_FIELDS
from .models import CandidatePlan, QueryConstraint, RuntimeQuerySpec
from .schema_grounding import canonical_slot_from_text, ground_field_text
from .schema_retrieval import retrieve_schema_context

QUESTION_TYPES: dict[str, dict[str, Any]] = {
    "factoid":      {"operations": ("select_unique",), "slots": ("answer_field", "constraints")},
    "count":        {"operations": ("count",), "slots": ("constraints",)},
    "sum":          {"operations": ("sum",), "slots": ("constraints",)},
    "boolean":      {"operations": ("exists", "predicate"), "slots": ("constraints", "comparison")},
    "comparison":   {"operations": ("predicate",), "slots": ("constraints", "comparison")},
    "min_max":      {"operations": ("argmax", "argmin"), "slots": ("constraints",)},
    "top_k":        {"operations": ("top_k",), "slots": ("constraints", "group_by", "k", "metric")},
    "set":          {"operations": ("distinct_set",), "slots": ("answer_field", "constraints")},
    "compare_two":  {"operations": ("compare",), "slots": ("left", "right", "constraints")},
    "bridge_join":  {"operations": ("count", "sum"), "slots": ("steps",)},
    "unanswerable": {"operations": (), "slots": ("reason",)},
}
REPAIR_ACTIONS = (
    "fix_question_type",
    "repair_constraints",
    "swap_buyer_supplier",
    "change_operator",
    "change_operation",
    "abstain",
)
TYPE_SHELLS: dict[str, dict[str, Any]] = {
    "factoid": {
        "question_type": "factoid",
        "operation": "select_unique",
        "answer_field": "buyer|supplier|category|cpv|title|date|value",
        "constraints": [{"slot": "buyer|supplier|year|years|year_range|cpv|category|title|date",
                         "surface": "verbatim text", "value": "normalised value", "operator": "eq|in|between"}],
    },
    "count": {
        "question_type": "count",
        "operation": "count",
        "constraints": [{"slot": "buyer|supplier|year|years|year_range|cpv|category|title|date",
                         "surface": "verbatim text", "value": "normalised value", "operator": "eq|in|between"}],
    },
    "sum": {
        "question_type": "sum",
        "operation": "sum",
        "constraints": [{"slot": "buyer|supplier|year|years|year_range|cpv|category|title|date",
                         "surface": "verbatim text", "value": "normalised value", "operator": "eq|in|between"}],
        "note": "Do not add value_is_additive; deterministic grounding adds and logs that guard.",
    },
    "boolean": {
        "question_type": "boolean",
        "operation": "exists|predicate",
        "answer_field": "value|date|buyer|supplier|category|cpv when predicate compares a field",
        "constraints": [{"slot": "buyer|supplier|year|years|year_range|cpv|category|title|date",
                         "surface": "verbatim text", "value": "normalised value", "operator": "eq|in|between"}],
        "comparison": {"operator": ">|>=|<|<=|=|after|before", "threshold": "number or ISO date"},
    },
    "comparison": {
        "question_type": "comparison",
        "operation": "predicate",
        "answer_field": "value|date|buyer|supplier|category|cpv",
        "constraints": [{"slot": "buyer|supplier|year|years|year_range|cpv|category|title|date",
                         "surface": "verbatim text", "value": "normalised value", "operator": "eq|in|between"}],
        "comparison": {"operator": ">|>=|<|<=|=|after|before", "threshold": "number or ISO date"},
    },
    "min_max": {
        "question_type": "min_max",
        "operation": "argmax|argmin",
        "constraints": [{"slot": "buyer|supplier|year|years|year_range|cpv|category|title|date",
                         "surface": "verbatim text", "value": "normalised value", "operator": "eq|in|between"}],
    },
    "top_k": {
        "question_type": "top_k",
        "operation": "top_k",
        "group_by": "buyer|supplier|category|cpv",
        "metric": "count|sum",
        "k": "integer explicitly requested, otherwise 3",
        "constraints": [{"slot": "year|years|year_range|cpv|category|title|date",
                         "surface": "verbatim text", "value": "normalised value", "operator": "eq|in|between"}],
    },
    "set": {
        "question_type": "set",
        "operation": "distinct_set",
        "answer_field": "buyer|supplier|category|cpv|title|date",
        "constraints": [{"slot": "buyer|supplier|year|years|year_range|cpv|category|title|date",
                         "surface": "verbatim text", "value": "normalised value", "operator": "eq|in|between"}],
    },
    "compare_two": {
        "question_type": "compare_two",
        "operation": "compare",
        "left": "left organisation surface",
        "right": "right organisation surface",
        "constraints": [{"slot": "year|years|year_range|cpv|category|title|date",
                         "surface": "verbatim text", "value": "normalised value", "operator": "eq|in|between"}],
    },
    "bridge_join": {
        "question_type": "bridge_join",
        "operation": "count|sum",
        "steps": [
            {"step": "anchor_set", "action": "find intermediate entities",
             "constraints": [{"slot": "buyer|supplier|year|cpv|category|title",
                              "surface": "verbatim text", "value": "normalised value"}],
             "bind_slot": "buyer|supplier"},
            {"step": "answer", "action": "aggregate over contracts for bound entities",
             "from_step": "anchor_set",
             "constraints": [{"slot": "year|cpv|category|title",
                              "surface": "verbatim text", "value": "normalised value"}]},
        ],
    },
    "unanswerable": {
        "question_type": "unanswerable",
        "operation": "",
        "reason": "unsupported concept or insufficient information explicitly present in the question",
    },
}
_SLOT_FIELDS = {"buyer": "buyer_name", "supplier": "supplier_name", "year": "release_year",
                "year_range": "release_year", "years": "release_year",
                "cpv": "tender_cpv_id", "category": "tender_category", "title": "tender_title",
                "date": "award_date_signed", "has_signed_date": "has_award_signed_date"}

# Strict json_schema for Step-2 (provider-enforced shape). variables is an ARRAY (strict mode
# forbids arbitrary-key maps); slot/kind/operator/operation/role are ENUMS (kills placeholder and
# alternation echo at the source); understanding_network and the recursive goal_tree are dropped
# (Step-1 already carries the reading; recursion is unsupported by strict schema). The output is a
# graph_plan the deterministic executor runs.
_SLOT_ENUM = ["buyer", "supplier", "year", "year_range", "years", "cpv", "category", "title", "date",
              "has_signed_date"]
_OP_ENUM = ["count", "sum", "select", "exists", "argmax", "argmin", "top_k", "distinct_set", "compare", "abstain"]
_RETURN_FIELD_ENUM = ["buyer", "supplier", "category", "cpv", "title", "date", "value", "contract", "none"]
_INTENT_OP_ENUM = [
    "filter_records", "distinct", "bind_filter", "count", "sum", "select", "exists",
    "compare", "argmin", "argmax", "top_k", "select_distinct", "abstain",
]
_VALUE_TYPE_ENUM = ["number", "money", "boolean", "entity", "date", "set", "unknown"]
_PROGRAM_RETURN_ENUM = [
    "record", "record_set", "entity_set", "number", "money", "boolean",
    "entity", "date", "text", "set", "unknown",
]
_TEMPLATE_ENUM = [
    "simple_filter_aggregate",
    "bridge_join",
    "comparison",
    "distinct_set",
    "min_max_top_k",
    "abstain",
]


def intent_program_schema() -> dict[str, Any]:
    filter_obj = {
        "type": "object", "additionalProperties": False,
        "required": ["field_text", "operator", "value", "value_type"],
        "properties": {
            "field_text": {"type": "string"},
            "operator": {"type": "string", "enum": ["eq", "in", "between", "gt", "gte", "lt", "lte", "none"]},
            "value": {"type": "string"},
            "value_type": {"type": "string",
                           "enum": ["year", "number", "money", "date", "entity", "category",
                                    "cpv", "code", "text", "string", "boolean", "unknown"]},
        },
    }
    args_obj = {
        "type": "object", "additionalProperties": False,
        "required": ["filters", "field_text", "bind_field_text", "metric_text", "comparator", "threshold", "k"],
        "properties": {
            "filters": {"type": "array", "maxItems": 8, "items": filter_obj},
            "field_text": {"type": "string"},
            "bind_field_text": {"type": "string"},
            "metric_text": {"type": "string"},
            "comparator": {"type": "string"},
            "threshold": {"type": "string"},
            "k": {"type": "integer"},
        },
    }
    step_obj = {
        "type": "object", "additionalProperties": False,
        "required": ["id", "op", "inputs", "args", "returns"],
        "properties": {
            "id": {"type": "string"},
            "op": {"type": "string", "enum": _INTENT_OP_ENUM},
            "inputs": {"type": "array", "items": {"type": "string"}},
            "args": args_obj,
            "returns": {"type": "string", "enum": _PROGRAM_RETURN_ENUM},
        },
    }
    return {
        "name": "procurement_intent_program",
        "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "required": ["answer_signature", "program", "answer_step", "unsupported_or_ambiguous"],
            "properties": {
                "answer_signature": {
                    "type": "object", "additionalProperties": False,
                    "required": ["operation", "value_type", "answer_field_text"],
                    "properties": {
                        "operation": {"type": "string",
                                      "enum": ["count", "sum", "select", "exists", "distinct_set",
                                               "compare", "argmin", "argmax", "top_k", "abstain"]},
                        "value_type": {"type": "string", "enum": _VALUE_TYPE_ENUM},
                        "answer_field_text": {"type": "string",
                                              "description": "natural-language answer field, or empty for count/exists/abstain"},
                    },
                },
                "program": {"type": "array", "maxItems": 8, "items": step_obj},
                "answer_step": {"type": "string"},
                "unsupported_or_ambiguous": {"type": "array", "items": {"type": "string"}},
            },
        },
    }


def graph_plan_schema(variant: str = "filler") -> dict[str, Any]:
    filter_obj = {
        "type": "object", "additionalProperties": False,
        "required": ["slot", "value", "operator"],
        "properties": {
            "slot": {"type": "string", "enum": _SLOT_ENUM},
            "value": {"type": "string", "description": "verbatim/normalised value from the question"},
            "operator": {"type": "string", "enum": ["eq", "in", "between"]},
        },
    }
    variable_obj = {
        "type": "object", "additionalProperties": False,
        "required": ["var_id", "kind", "role", "filters", "depends_on"],
        "properties": {
            "var_id": {"type": "string",
                       "description": "Flat tree id: a1/a2 feed answer; b1a1 feeds a1; c1b1 feeds b1."},
            "kind": {"type": "string", "enum": ["entity", "entity_set", "record_set", "scalar"]},
            "role": {"type": "string",
                     "enum": ["buyer", "supplier", "publisher", "contract_records", "value", "date",
                              "cpv", "category", "none"]},
            "filters": {"type": "array", "maxItems": 8, "items": filter_obj},
            "depends_on": {"type": "array", "items": {"type": "string"},
                           "description": "prior var_ids; use ONLY for an entity-set join (bridge)"},
        },
    }
    relations_obj = {
        "type": "array", "maxItems": 6,
        "items": {
            "type": "object", "additionalProperties": False,
            "required": ["from", "to", "bind_field"],
            "properties": {
                "from": {"type": "string"}, "to": {"type": "string"},
                "bind_field": {"type": "string",
                               "enum": ["buyer", "supplier", "cpv", "category", "contract", "none"]},
            },
        },
    }
    comparison_obj = {
        "type": "object", "additionalProperties": False,
        "required": ["operator", "threshold"],
        "properties": {
            "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "=", "after", "before", "none"]},
            "threshold": {"type": "string"},
        },
    }
    return_obj = {
        "type": "object", "additionalProperties": False,
        "required": ["operation", "input"] if variant == "optional" else [
            "operation", "input", "field", "guards", "k", "group_by", "metric",
            "comparator", "left", "right",
        ],
        "properties": {
            "operation": {"type": "string", "enum": _OP_ENUM},
            "input": {"type": "string", "description": "var_id to aggregate/return"},
            "field": {"type": "string", "enum": _RETURN_FIELD_ENUM},
            "guards": {"type": "array", "items": {"type": "string"}},
            "k": {"type": "integer"},
            "group_by": {"type": "string", "enum": ["buyer", "supplier", "category", "cpv", "none"]},
            "metric": {"type": "string", "enum": ["count", "sum", "none"]},
            "comparator": {"type": "string", "enum": ["gt", "gte", "lt", "lte", "eq", "none"]},
            "left": {"type": "string"}, "right": {"type": "string"},
        },
    }
    if variant == "optional":
        # v6-era shape: unused semantics are OMITTED, never filled. Endpoints that enforce the
        # official all-required strict subset (gpt-5.4-nano) reject this; grok's accepts it, and
        # forcing filler on grok measurably injected junk (year=0 padding, empty-slot echoes).
        return {
            "name": "procurement_graph_plan", "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "required": ["question_type", "operation", "variables", "return"],
                "properties": {
                    "question_type": {"type": "string", "enum": list(QUESTION_TYPES)},
                    "operation": {"type": "string", "enum": _OP_ENUM},
                    "answer_field": {"type": "string", "enum": _RETURN_FIELD_ENUM},
                    "variables": {"type": "array", "maxItems": 6, "items": variable_obj},
                    "relations": relations_obj,
                    "comparison": comparison_obj,
                    "return": return_obj,
                },
            },
        }
    return {
        "name": "procurement_graph_plan", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "template": {"type": "string", "enum": _TEMPLATE_ENUM},
                "question_type": {"type": "string", "enum": list(QUESTION_TYPES)},
                "operation": {"type": "string", "enum": _OP_ENUM},
                "answer_field": {"type": "string", "enum": _RETURN_FIELD_ENUM},
                "variables": {"type": "array", "maxItems": 6, "items": variable_obj},
                "relations": relations_obj,
                "comparison": comparison_obj,
                "return": return_obj,
            },
            "required": ["template", "question_type", "operation", "answer_field", "variables",
                         "relations", "comparison", "return"],
        },
    }


def repair_graph_plan_schema(variant: str = "filler") -> dict[str, Any]:
    """Strict schema for the reflector: repair metadata + repaired graph_plan."""
    return {
        "name": "procurement_graph_repair", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "required": [
                "diagnosis", "repair_action", "repaired_understanding_briefing",
                "repaired_graph_plan", "changed_fields", "unchanged_fields",
            ],
            "properties": {
                "diagnosis": {"type": "string"},
                "repair_action": {"type": "string", "enum": list(REPAIR_ACTIONS)},
                "repaired_understanding_briefing": {"type": "string"},
                "repaired_graph_plan": graph_plan_schema(variant)["schema"],
                "changed_fields": {"type": "array", "items": {"type": "string"}},
                "unchanged_fields": {"type": "array", "items": {"type": "string"}},
            },
        },
    }
_ROLE_CUES = {"buyer_name": ("awarded by", "published by", "buyer", "contracting authority", "by "),
              "supplier_name": ("awarded to", "supplier", "won by", "contract with", "vendor")}
_GT_WORDS = ("more than", "above", "over", "greater", "exceed", "at least")
_LT_WORDS = ("less than", "below", "under", "fewer", "at most", "no more than")
# "more/fewer ... than" with a noun phrase in between ("more contract notices than") is still a
# direction cue; literal substring matching missed it and falsely rejected valid comparisons.
_GT_CUE_RE = re.compile(r"(?<!no )\bmore\b[^.?;!]{0,60}?\bthan\b|\babove\b|\bover\b|\bgreater\b"
                        r"|\bexceed|\bat least\b|\bafter\b")
_LT_CUE_RE = re.compile(r"\b(?:less|fewer)\b[^.?;!]{0,60}?\bthan\b|\bbelow\b|\bunder\b"
                        r"|\bat most\b|\bno more than\b|\bbefore\b")
# Strong bridge phrasings: a flat single-hop type over one of these means the intermediate
# entity set was dropped (the classic bridge-flattening failure).
_BRIDGE_CUES = ("went on to award", "has used before", "have used before", "had used before",
                "also worked with", "also supplied", "awarded contract went to",
                "contract went to", "previously used by", "used before by")
_NUM = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])")
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_SCALED_NUMBER = re.compile(
    "(?:\\u00a3|gbp|eur|\\u20ac|usd|\\$)?\\s*(\\d+(?:\\.\\d+)?)\\s*"
    "(m|mn|million|bn|billion|k|thousand)\\b",
    re.I,
)
_CPV_CODE = re.compile(r"\bcpv\s*(?:code\s*)?(\d{5,8})\b", re.I)
_YEAR_TEXT = re.compile(r"\b(20\d{2})\b")
_CATEGORY_TEXT = re.compile(
    r"\b(goods|services|works)\b(?=\s+(?:contract\s+)?notices?\b|\s+contracts?\b)"
    r"|categorized\s+as\s+(goods|services|works)\b"
    r"|listed\s+under\s+(goods|services|works)\b",
    re.I,
)
_UNSUPPORTED_CUES = (
    "evaluation score", "marked as reliable", "reliable", "fairness", "payment terms",
    "social value", "delivery performance", "bidder count", "bidder counts", "bidders",
)
_MONTH_NAMES = {
    "01": ("january", "jan"),
    "02": ("february", "feb"),
    "03": ("march", "mar"),
    "04": ("april", "apr"),
    "05": ("may",),
    "06": ("june", "jun"),
    "07": ("july", "jul"),
    "08": ("august", "aug"),
    "09": ("september", "sept", "sep"),
    "10": ("october", "oct"),
    "11": ("november", "nov"),
    "12": ("december", "dec"),
}


def question_understanding_messages(question: str, schema_context: dict[str, Any] | None = None) -> tuple[str, str]:
    """Step 1 prompt: a DENSE reasoning scaffold (not an essay) for the Step-2 graph planner.

    Every line must be a fact the planner cannot trivially re-derive. Prose that Step 2 ignores
    (target meaning/origin/dependency sentences, a reverse tree that restates the procedure,
    per-question KG-mapping speculation) is cut structurally via hard line budgets, because the
    model does not obey soft word limits.
    """
    system = (
        "You are the semantic reading layer for a UK public-procurement KGQA system. "
        "Output a DENSE scaffold for a downstream planner, not an essay. Every line must state a "
        "fact the planner cannot trivially re-derive: no explanations, no restating the question, "
        "no speculation about how the knowledge graph stores things. Do not answer the question."
    )
    user = f"""Question:
{question}

Task:
Describe the solving procedure as a compact scaffold. Do NOT calculate or guess the answer.
Output exactly these 8 sections and OBEY the line budgets. Terse fragments, not sentences.

1. Answer Type: one word (count | sum | boolean | comparison | factoid | entity_list | min_max | top_k | ambiguous | unsupported).
2. Query Template: one word (simple_filter_aggregate | bridge_join | comparison | distinct_set | min_max_top_k | abstain).
3. Explicit Info: one bullet per literal atom as "key = value" (year, cpv, category, amount, threshold, org name, role phrase). Copy values verbatim. Do NOT echo the whole question. Max 8 bullets.
4. Reverse Tree: the answer's dependency chain in at most 3 lines (answer <- B <- A). If the answer is one filtered set, write "answer <- filtered set". Do not re-list filters already in Explicit Info.
5. Procedure: at most 6 lines, each "Step N: <verb> ..." verb-first (resolve, find, filter, count, sum, max, min, compare). No hedging and no dedup/edge-case notes.
5. Targets (A, B, C): ONLY real intermediate sets needed for a join or comparison. One line each: "A = <short label> | role=buyer|supplier|cpv|record_set|scalar | from=explicit|derived(X)". If the answer is a single filtered set, write exactly "Targets: none". MANDATORY: if the answer depends on an intermediate ENTITY SET (buyers/suppliers/CPV codes derived from other records, e.g. "suppliers who worked with X", "CPV codes Y has used"), Targets MUST name that set explicitly - NEVER collapse a bridge into "filtered set".
7. Roles & Direction: one line stating which role each org fills, with the exact quote (e.g. supplier via "awarded to"). If no org role, write "none".
8. Ambiguities: write "none" by default. Add a line ONLY if the question requires a concept absent from procurement records (name it). NEVER speculate about KG field names, which date field to use, or classification internals.

Query Template rules:
- bridge_join: the answer depends on an intermediate entity set, e.g. suppliers/buyers/CPV codes/categories derived from other records.
- comparison: compare two sets or compare one aggregate against a threshold.
- distinct_set: return a list/set of buyers, suppliers, CPV codes, categories, or titles.
- min_max_top_k: highest/lowest/top-k/ranking/extreme-value questions.
- abstain: unsupported or ambiguous concept required.
- simple_filter_aggregate: one directly filtered record set with count/sum/select/exists.

Rules:
- ZERO hallucination: preserve every literal number, date, CPV, amount, and name exactly.
- Explicit Info contains ONLY text from the question.
- Plain ASCII only. No JSON. No database/field names."""
    user = f"""Question:
{question}

Task:
Describe the solving procedure as a compact scaffold. Do NOT calculate or guess the answer.
Output exactly these 8 sections and OBEY the line budgets. Terse fragments, not sentences.

1. Answer Type: one word (count | sum | boolean | comparison | factoid | entity_list | min_max | top_k | ambiguous | unsupported).
2. Query Template: one word (simple_filter_aggregate | bridge_join | comparison | distinct_set | min_max_top_k | abstain).
3. Explicit Info: one bullet per literal atom as "key = value" (year, cpv, category, amount, threshold, org name, role phrase). Copy values verbatim. Do NOT echo the whole question. Max 8 bullets.
4. Reverse Tree: the answer's dependency chain in at most 3 lines (answer <- B <- A). If the answer is one filtered set, write "answer <- filtered set". Do not re-list filters already in Explicit Info.
5. Procedure: at most 6 lines, each "Step N: <verb> ..." verb-first (resolve, find, filter, count, sum, max, min, compare). No hedging and no dedup/edge-case notes.
6. Targets (A, B, C): ONLY real intermediate sets needed for a join or comparison. One line each: "A = <short label> | role=buyer|supplier|cpv|record_set|scalar | from=explicit|derived(X)". If the answer is a single filtered set, write exactly "Targets: none". MANDATORY: if the answer depends on an intermediate ENTITY SET (buyers/suppliers/CPV codes/categories derived from other records, e.g. "suppliers who worked with X", "CPV codes Y has used"), Targets MUST name that set explicitly - NEVER collapse a bridge into "filtered set".
7. Roles & Direction: one line stating which role each org fills, with the exact quote (e.g. supplier via "awarded to"). If no org role, write "none".
8. Ambiguities: write "none" by default. Add a line ONLY if the question requires a concept absent from procurement records (name it). NEVER speculate about KG field names, which date field to use, or classification internals.

Query Template rules:
- bridge_join: the answer depends on an intermediate entity set, e.g. suppliers/buyers/CPV codes/categories derived from other records.
- comparison: compare two sets or compare one aggregate against a threshold.
- distinct_set: return a list/set of buyers, suppliers, CPV codes, categories, or titles.
- min_max_top_k: highest/lowest/top-k/ranking/extreme-value questions.
- abstain: unsupported or ambiguous concept required.
- simple_filter_aggregate: one directly filtered record set with count/sum/select/exists.

Rules:
- ZERO hallucination: preserve every literal number, date, CPV, amount, and name exactly.
- Explicit Info contains ONLY text from the question.
- Plain ASCII only. No JSON. No database/field names."""
    return system, user


def question_intent_program_messages(question: str, schema_context: dict[str, Any] | None = None) -> tuple[str, str]:
    """Structured Step-1 prompt: nano emits the computation program, not a prose trace."""
    system = (
        "You are the typed intent programmer for a UK public-procurement KGQA system. "
        "Convert the question into a small computation program. Do not answer the question. "
        "Use only information explicitly present in the question. The output shape is enforced by "
        "a strict JSON schema."
    )
    schema_context = schema_context or retrieve_schema_context(question)
    user = json.dumps({
        "question": question,
        "task": [
            "Define answer_signature: operation, value_type, answer_field_text.",
            "Write program steps as executable computation units.",
            "Set answer_step to the step id that produces the final answer.",
            "Use unsupported_or_ambiguous when the question itself is underspecified or unsupported.",
        ],
        "allowed_slots": {
            "year/year_range/years": "publication or release year constraints from the question",
            "cpv_code": "numeric CPV code only, e.g. 85149000",
            "cpv_label": "text inside CPV parentheses, e.g. Pharmacy services; keep separate from procurement_category",
            "procurement_category": "ONLY goods, services, or works",
            "buyer": "contracting authority / buyer organisation",
            "supplier": "awarded supplier / winner organisation",
            "title": "exact quoted title only",
            "date": "explicit date constraints",
            "amount": "explicit money amount or threshold",
        },
        "operation_guidance": {
            "simple_count": "A filter_records with literal filters, then count(A)",
            "sum": "A filter_records, then sum(A) over value",
            "select": "A filter_records, then select(A) field=buyer|supplier|category|cpv|title|date|value",
            "distinct_set": "A filter_records, then distinct(A) field=buyer|supplier|cpv|procurement_category|title",
            "bridge": "source filter_records -> distinct entity_set -> bind_filter target records -> aggregate/select",
            "comparison": "produce left/right scalar steps, then compare(left,right)",
            "ambiguous": "If the question asks for a unique contract/entity but gives only an organisation and no unique anchor, use operation=abstain.",
        },
        "hard_rules": [
            "Root search steps must use inputs=[]; do NOT invent inputs such as all_records, contracts, contract_notices, or awards.",
            "Later steps may only reference previous step ids, e.g. inputs=[\"A\"].",
            "Filters use field_text, not system slot names. Good: field_text='published year', 'CPV code', 'high-level procurement category', 'buyer organisation'.",
            "Do not put CPV labels in procurement_category. Procurement category is only goods/services/works.",
            "Use CPV labels only as field_text='CPV label' audit filters; they are not executable category filters.",
            "For count/exists/abstain, answer_field_text must be empty.",
            "No-results years such as future years are still executable programs; do not predict database emptiness.",
            "Ambiguous means the question lacks enough constraints to identify the requested thing, independent of KG contents.",
            "If unsupported_or_ambiguous is non-empty, set operation=abstain, program=[], and answer_step=''. Candidate programs are not allowed in the official program.",
            "Do not invent years, CPV codes, amounts, dates, categories, or organisation names.",
            "Use ASCII strings only.",
        ],
        "retrieved_schema_context": schema_context,
    }, ensure_ascii=True)
    return system, user


def repair_understanding_messages(question: str, feedback: dict[str, Any]) -> tuple[str, str]:
    """Repair Step-1 prompt: nano sees the failed Stage-2 plan and structured failure feedback."""
    system = (
        "You are the repair reading layer for a verifier-guided UK procurement KGQA system. "
        "Revise the question-understanding scaffold using the failed Stage-2 graph plan and failure "
        "feedback. Do not answer the question and do not infer hidden reference answers."
    )
    safe_feedback = _feedback_for_understanding(feedback)
    user = f"""Question:
{question}

Repair task:
The previous Stage-2 graph plan failed. Re-read the question and produce a corrected compact
understanding scaffold for the repair planner.

You may use:
- the original question
- the previous Step-1 understanding, if present
- the failed Stage-2 graph plan
- grounding/schema/executor/verifier feedback

You must NOT use or infer any hidden reference/oracle answer.

Structured feedback:
{json.dumps(safe_feedback, ensure_ascii=False, default=str)}

Output exactly these 8 sections:
1. Answer Type:
2. Query Template:
3. Explicit Info:
4. Reverse Tree:
5. Procedure:
6. Targets (A, B, C):
7. Roles & Direction:
8. Ambiguities:

Rules:
- Mention what the failed Stage-2 plan got wrong when it affects the corrected scaffold.
- Explicit Info may contain ONLY text that appears in the question.
- Plain ASCII only. No JSON. No database/field names.
- ZERO hallucination: preserve every literal number, date, CPV, amount, and name exactly."""
    return system, user


def _feedback_for_understanding(feedback: dict[str, Any]) -> dict[str, Any]:
    """Drop hidden/offline labels before the repair-understanding LLM sees feedback."""
    banned = {"oracle_answer", "reference_answer", "gold_answer", "expected_answer"}
    allowed = {
        "previous_understanding", "failed_plan", "selected_plan", "failure_stage", "failure_reason",
        "execution_status", "submitted_answer", "grounding_issues", "schema_errors",
        "compiler_issues", "executor_verifier_result", "graph_failure_kind", "failed_variable",
        "failed_operation_unit", "allowed_repair_actions", "notes", "repair_hints",
    }
    out: dict[str, Any] = {}
    for key, value in feedback.items():
        if key in banned or "oracle" in key or "reference_answer" in key:
            continue
        if key in allowed:
            out[key] = value
    return out


def _query_template_from_understanding(question: str, understanding: dict[str, Any] | None) -> str:
    """Stage-1 owns intent classification; this fallback keeps older cached briefings usable."""
    understanding = understanding or {}
    for key in ("query_template", "template", "stage1_query_template"):
        value = str(understanding.get(key, "")).strip().casefold()
        value = value.replace("-", "_").replace(" ", "_")
        if value in _TEMPLATE_ENUM:
            return value
    answer_type = str(
        understanding.get("answer_type")
        or understanding.get("question_type")
        or understanding.get("needs_to_return")
        or ""
    ).casefold()
    text = " ".join(
        str(understanding.get(key, ""))
        for key in ("briefing", "explicit_info", "reverse_tree", "procedure", "targets",
                    "roles_direction", "ambiguities", "known_information", "reasoning_chain",
                    "notes_for_planner")
    ).casefold()
    q = question.casefold()
    combined = f"{q} {answer_type} {text}"
    if "unsupported" in answer_type or "ambiguous" in answer_type:
        return "abstain"
    if any(cue in combined for cue in _UNSUPPORTED_CUES):
        return "abstain"
    bridge_cues = (
        "worked with", "also worked", "also won", "also awarded", "used before",
        "previously used", "went on to", "suppliers who", "buyers who", "cpv codes used",
        "categories used", "derived(", "entity set", "entity_set",
    )
    if any(cue in combined for cue in bridge_cues):
        return "bridge_join"
    if "top_k" in answer_type or "top " in combined or "highest" in combined or "lowest" in combined:
        return "min_max_top_k"
    if "min_max" in answer_type or "maximum" in combined or "minimum" in combined:
        return "min_max_top_k"
    if "comparison" in answer_type or "compare" in combined or _GT_CUE_RE.search(combined) or _LT_CUE_RE.search(combined):
        return "comparison"
    if any(cue in answer_type for cue in ("entity_list", "set", "list")):
        return "distinct_set"
    if re.search(r"\bwhich\s+(buyers?|suppliers?|cpv codes?|categories|titles?)\b", q):
        return "distinct_set"
    return "simple_filter_aggregate"


def _template_specific_rules(template: str) -> list[str]:
    common = [
        "Copy the stage1_query_template into graph_plan.template.",
        "Do not switch templates unless Stage1 marked abstain/ambiguous and the question is plainly executable.",
    ]
    by_template = {
        "simple_filter_aggregate": [
            "Use exactly one primary record_set a1 with all literal filters on that set.",
            "Do not split year/category/CPV/buyer/supplier filters into dependency variables.",
        ],
        "bridge_join": [
            "Mandatory dataflow: source records -> distinct entity_set -> target records -> aggregate.",
            "The source record_set produces the entity_set; the target record_set depends on that entity_set.",
            "Never put two different CPV eq filters on one record_set to represent a bridge.",
        ],
        "comparison": [
            "Two-set comparison: a1 left record_set, a2 right record_set, return compare(left=a1,right=a2,metric=count|sum).",
            "Threshold comparison: a1 record_set, return compare(left=a1,right=literal threshold,metric=count|sum).",
        ],
        "distinct_set": [
            "Use a record_set input and return distinct_set over buyer/supplier/cpv/category/title.",
            "If the returned set itself is derived through another set, this is bridge_join, not distinct_set.",
        ],
        "min_max_top_k": [
            "Use argmin/argmax/top_k over a record_set with literal filters.",
            "For top_k, set k, group_by, and metric explicitly.",
        ],
        "abstain": [
            "Use return.operation='abstain' when the question needs unsupported concepts or is under-specified.",
            "Do not force unsupported concepts into date/title/category filters.",
        ],
    }
    return common + by_template.get(template, [])


def resolve_planner_variants(model: str) -> tuple[str, str]:
    """Measured per-model Step-2 config (worklog 2026-07-04 four-cell A/B): grok plans best with
    the lean briefing-only prompt + omit-unused schema (35->40/50); gpt-5.4-nano needs the
    capability card and the all-required schema its endpoint enforces.

    Local 8B students (zero-shot AND fine-tuned Qwen3/Llama via vLLM) take the grok profile:
    SFT data is rendered lean+optional, and the serving layer enforces the schema via
    guided-json uniformly across every ladder rung — including the zero-shot floor — so
    rung-to-rung deltas measure planning ability, not format compliance.
    Returns (prompt_variant, schema_variant)."""
    m = str(model).casefold()
    if "grok" in m:
        return "lean", "optional"
    if "nano" in m or "gpt" in m:
        return "card", "filler"
    return "lean", "optional"


def typed_plan_messages(question: str, understanding: dict[str, Any] | None = None,
                        schema_context: dict[str, Any] | None = None,
                        variant: str = "card") -> tuple[str, str]:
    """Lean Step-2 prompt. A strict json_schema (graph_plan_schema) enforces the OUTPUT shape, so
    the prompt is task guidance only: no giant nested schema dump, no understanding_network echo
    (Step-1 already carries the reading).

    variant="card" adds the executor capability card + template-shell machinery (introduced for
    nano Stage-2); variant="lean" is the v6-era briefing-only prompt — measured better for grok
    (the shell machinery anchored its bridge planning). Keep both for A/B until settled.
    """
    system = (
        "You are the structured graph planner for a UK public-procurement KGQA system. Turn the "
        "question and the Step-1 briefing into an executable graph_plan (variables + return). A "
        "strict schema fixes the output shape; fill the fields faithfully and never answer the "
        "question."
    )
    schema_context = schema_context or retrieve_schema_context(question)
    template = _query_template_from_understanding(question, understanding)
    payload: dict[str, Any] = {
        "question": question,
        "step1_understanding_briefing": understanding,
        "retrieved_schema_context": schema_context,
    }
    if variant == "card":
        capability_card = _executor_capability_card()
        payload.update({
            "stage1_query_template": template,
            "executor_capability_card": capability_card,
            "selected_template_shell": capability_card["template_shells"].get(template, ""),
            "template_specific_rules": _template_specific_rules(template),
            "stage2_process": [
                "Stage2A read stage1_query_template from Step1.",
                "Stage2B fill only the graph shape allowed by that template and selected_template_shell.",
                "Stage2C preserve all literal filters and leave unsupported/ambiguous questions as abstain.",
            ],
        })
    payload["instructions"] = ([
        "Copy stage1_query_template into graph_plan.template, then fill the graph_plan for that template.",
    ] if variant == "card" else []) + [
        "Use FLAT tree-coded var_id values: a1/a2 feed the final answer directly; b1a1/b2a1 "
            "feed a1; c1b1 feeds b1. The parent token is encoded in the id. Keep variables as a "
            "flat array, not nested objects.",
            "variables[] are the record/entity sets needed. A CONJUNCTIVE filter (e.g. cpv AND "
            "year AND category) is ONE record_set variable with multiple filters - do NOT split it "
            "into a dependency chain.",
            "depends_on may be empty because the executor reconstructs dependencies from var_id. "
            "Only add explicit depends_on as a fallback when the id cannot express the join.",
            "Bridge questions are not one fixed template, but they must preserve executor semantics: "
            "derive a distinct entity_set (buyer/supplier/cpv/category) from source records, then "
            "bind target records through that entity_set. Do not add the source record_set itself as "
            "a target dependency unless the question asks for the same exact records.",
            "For template=bridge_join, source records -> entity_set -> target records -> aggregate is "
            "mandatory. The exact variable ids may vary, but this dataflow must be present.",
            "For template=comparison, set return.left, return.right, return.metric=count|sum, and "
            "return.comparator=gt|lt|eq. For threshold comparisons, right is the literal threshold.",
            "Filters are ONLY literal constraints from the question (year, CPV, goods/services/works, "
            "quoted title, named buyer/supplier). Derived phrases like 'CPV codes used by X', "
            "'matching procurement records', 'services contracts', or 'works notices' are NOT literal "
            "filter values; model them as variables/relations or normal categories.",
            "return.input is the var_id to aggregate/return; return.operation is the final operation.",
            "The schema requires some fields even when unused. For unused optional semantics, use "
            "empty strings, empty arrays, k=0, and enum value 'none'. Do NOT invent meaning just to "
            "fill a required field.",
            "Copy years, CPV codes, categories (goods|services|works), titles, thresholds VERBATIM "
            "from the question. A CPV parenthetical label is NOT a category filter.",
            "operator='between' for from/between year ranges; operator='in' for alternatives like "
            "'2022 or 2023'.",
            "For 'has a signed award date' / 'award has been signed', use slot 'has_signed_date' "
            "(existence), NOT the 'date' slot.",
            "For a money sum, add guard 'value_is_additive'.",
            "If the question needs a concept absent from procurement records, or is ambiguous, set "
            "question_type accordingly and return.operation='abstain'.",
    ]
    return system, json.dumps(payload, ensure_ascii=False)


def _executor_capability_card() -> dict[str, Any]:
    return {
        "variable_kinds": {
            "record_set": "procurement contract/notice records filtered by literal constraints",
            "entity_set": "distinct buyer/supplier/cpv/category values derived from a record_set",
            "scalar": "count, sum, selected date/value, or comparison participant derived from records",
            "entity": "one explicit buyer or supplier entity from the question",
        },
        "filter_slots": {
            "buyer": "buyer organisation name only",
            "supplier": "supplier/winner organisation name only",
            "year": "publication/release year like 2024",
            "cpv": "CPV code only, e.g. 72000000",
            "category": "one of goods, services, works only",
            "title": "only exact quoted title text from the question",
            "has_signed_date": "existence of signed award date",
        },
        "bind_fields": ["buyer", "supplier", "cpv", "category"],
        "query_semantics": [
            "record_set(filters) executes a table filter over procurement records.",
            "entity_set(role=R, filters/dependencies) executes distinct values of R from its records.",
            "A dependency from entity_set B to record_set A adds A.<bind_field> IN B.",
            "count/sum/select/distinct_set/argmin/argmax run on the return.input record/entity set.",
            "compare runs both left and right sides, using metric=count or metric=sum.",
        ],
        "bridge_rules": [
            "The source side of a bridge exists to produce a distinct entity_set.",
            "The target side carries the answer's literal filters and depends on the entity_set.",
            "Use bind_field=supplier for 'suppliers who won/worked with'; buyer for buyers; cpv for CPV codes; category for goods/services/works categories.",
            "Do not filter supplier=ORG when the phrase means 'suppliers who worked with ORG'; instead ORG is usually a buyer filter on the source records.",
            "Do not put both CPV A and CPV B on the same record_set unless the question explicitly requires one record to have both.",
        ],
        "return_operations": ["count", "sum", "select", "exists", "argmax", "argmin",
                              "top_k", "distinct_set", "compare", "abstain"],
        "template_shells": {
            "simple_filter_aggregate": (
                "a1=record_set(literal filters); return count|sum|select|exists over a1"
            ),
            "bridge_join": (
                "source records produce b1=entity_set(role=buyer|supplier|cpv|category); "
                "target a1=record_set(target literal filters, depends_on=[b1]); "
                "relation b1 -> a1 with bind_field matching b1.role; return count|sum|distinct_set over a1"
            ),
            "comparison": (
                "two-set: a1=left record_set, a2=right record_set, return compare(left=a1,right=a2,metric=count|sum); "
                "threshold: a1=record_set, return compare(left=a1,right=literal threshold,metric=sum|count)"
            ),
            "distinct_set": "a1=record_set(literal filters); return distinct_set(a1, field=buyer|supplier|cpv|category)",
            "min_max_top_k": "a1=record_set(literal filters); return argmin|argmax|top_k over a1",
            "abstain": "return.operation=abstain and no executable filters if unsupported or ambiguous",
        },
        "negative_examples": [
            "WRONG bridge: one record_set with cpv=A AND cpv=B when the question says suppliers also won under CPV B.",
            "WRONG bridge: target depends_on=[source_record_set, entity_set]; source records should only produce the entity_set.",
            "WRONG role: supplier=ORG for 'suppliers who worked with ORG'; ORG is usually buyer on the source records.",
            "WRONG filter: category='matching procurement records' or category='services contracts'. Use services/works/goods or no filter.",
            "WRONG unsupported: answering invoice/payment/rationale/document-content questions from award_date_signed.",
        ],
        "pattern_examples_not_templates": [
            {
                "name": "simple_filtered_count",
                "shape": "a1=record_set(filters=[year/category/cpv/buyer/supplier]); return count(a1)",
            },
            {
                "name": "supplier_bridge_example",
                "question_shape": "target records sent to suppliers who also won/worked with source records",
                "query": "b1=entity_set(role=supplier, source filters); "
                         "a1=record_set(target filters, depends_on=[b1]); "
                         "relation b1 -> a1 bind_field=supplier; return count/sum(a1)",
            },
            {
                "name": "cpv_or_category_bridge_example",
                "question_shape": "records under CPV codes/categories previously used by X",
                "query": "b1=entity_set(role=cpv|category, filters=[records for X]); "
                         "a1=record_set(target filters, depends_on=[b1]); "
                         "relation b1 -> a1 bind_field=cpv|category",
            },
            {
                "name": "compare_two_counts",
                "shape": "a1=record_set(filters for left); a2=record_set(filters for right); "
                        "return compare(left=a1,right=a2,metric=count,comparator=gt|lt|eq)",
            },
            {
                "name": "distinct_supplier_list",
                "shape": "a1=record_set(filters=[buyer/category/year]); return distinct_set(a1, field=supplier)",
            },
            {
                "name": "lowest_nonzero_contract",
                "shape": "a1=record_set(filters=[cpv/category]); return argmin(a1, field=contract, guards=[exclude_zero])",
            },
        ],
    }


# ERASER-style deterministic reason -> prescription table: the failure code names WHAT broke,
# the guideline says WHAT TO DO. Matched by substring against failure_reason, first hit wins.
_REPAIR_GUIDELINES: tuple[tuple[str, str], ...] = (
    ("invalid_graph_structure:cycle",
     "Dependency edges conflict. Keep ONE direction only: a derived set depends on its source "
     "records (b1a1 depends_on a1 when b1a1 is extracted FROM a1). Delete reverse edges and "
     "redundant duplicate variables, then re-emit the plan."),
    ("unknown dependency variables",
     "A depends_on names a variable that does not exist. Either add the missing variable or fix "
     "the reference to an existing var_id."),
    ("ungroundable_variable",
     "A variable carries a literal the KG cannot ground (e.g. a non-date in a date slot, or a "
     "concept outside procurement records). Drop that variable if decorative; if the question "
     "NEEDS that concept, the correct action is abstain."),
    ("unconstrained_bind_source",
     "A consumed entity_set has no filters and no dependencies — that is the whole dimension, "
     "not a derived set. Define it with a literal filter or derive it from source records."),
    ("singular_question_set_return",
     "The question asks for ONE entity but the plan returns a set. Use operation=select over the "
     "filtered records (uniqueness is checked at run time); if several entities genuinely match, "
     "the system must clarify, not enumerate."),
    ("role_flipped",
     "Buyer/supplier direction contradicts the question. Swap the organisation into the correct "
     "role slot; on 'suppliers who worked with ORG' bridges the ORG is usually the BUYER of the "
     "source records."),
    ("surface_not_in_question",
     "A filter value is a derived phrase, not question text. Model it as a derived entity_set "
     "bridge (source records -> entity_set -> target records), never as a literal filter."),
    ("no_results_with_invented_literals",
     "The empty result comes from a literal the question never stated. Remove or correct the "
     "invented value; keep every literal the question DOES state."),
    ("multiple_answers",
     "Several entities match a unique-answer question. Add a discriminating literal from the "
     "question if one exists; otherwise abstain with a clarification."),
)


def _repair_guideline(failure_reason: str) -> str:
    for needle, guideline in _REPAIR_GUIDELINES:
        if needle in failure_reason:
            return guideline
    return ""


def typed_replan_messages(question: str, feedback: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are the repair reflector inside a verifier-guided KGQA graph planner. Diagnose the "
        "failed graph plan from the structured failure feedback, choose one allowed repair action, "
        "and return a repaired understanding network plus repaired graph plan using the SAME schema "
        "as the planning step. Never answer the question and never use hidden reference answers. "
        "Return strict JSON."
    )
    # Graph-native: mirror the planning step's contract (understanding_network + graph_plan +
    # retrieved schema) instead of the legacy type-shell vocabulary, so the reflector repairs the
    # graph the executor actually runs rather than being nudged back to the old slot format.
    schema_context = feedback.get("retrieved_schema_context") or retrieve_schema_context(question)
    failure_reason = str(feedback.get("failure_reason") or "")
    # surface the FAILED GRAPH at top level: on compile failures the compiled plan is None and the
    # raw graph sits three levels deep in raw_response — the repairer must see what it is fixing.
    failed_plan = feedback.get("failed_plan") if isinstance(feedback.get("failed_plan"), dict) else {}
    failed_graph = failed_plan.get("graph_plan")
    if not isinstance(failed_graph, dict):
        raw = failed_plan.get("raw_response") if isinstance(failed_plan.get("raw_response"), dict) else {}
        typed = raw.get("typed_plan") if isinstance(raw.get("typed_plan"), dict) else {}
        failed_graph = typed.get("graph_plan") if isinstance(typed.get("graph_plan"), dict) else None
    payload = {
        "task": "Feedback replan: diagnose the failure and produce one corrected graph plan.",
        "question": question,
        "failed_graph_plan": failed_graph,
        "repair_guideline": _repair_guideline(failure_reason),
        "failure_feedback": feedback,
        "retrieved_schema_context": schema_context,
        "allowed_repair_actions": list(REPAIR_ACTIONS),
        "allowed_operation_units": schema_context.get("allowed_operation_units"),
        "output_schema": {
            "diagnosis": "short cause analysis, based only on feedback and question text",
            "repair_action": "one of allowed_repair_actions",
            "repaired_understanding_briefing": "short corrected Step-1 briefing, no hidden answer",
            "repaired_graph_plan": "same schema as planning step graph_plan (variables/relations/operation_units/return)",
            "changed_fields": ["fields changed relative to the failed graph plan"],
            "unchanged_fields": ["fields deliberately kept"],
        },
        "notes": [
            "Use failure feedback as diagnosis, not as new evidence.",
            "Do not include or infer a reference/oracle answer.",
            "failed_operation_unit and graph_failure_kind name exactly which variable/unit failed.",
            "Use the same flat tree-coded var_id convention as planning: a1/a2 feed answer; b1a1 "
            "feeds a1; c1b1 feeds b1. Keep variables flat.",
            "A conjunctive filter (e.g. cpv AND year) is ONE record_set variable with multiple "
            "filters; do not split it into parent/child variables unless it is a genuine join.",
            "If repair_action is fix_question_type, the repaired graph may change return.operation and variable/return structure.",
            "All entities, years, CPVs, dates, amounts, and titles must still be supported by the original question text.",
            "If the question is genuinely unsupported, use repair_action='abstain' and repaired_graph_plan.return.operation='abstain'.",
        ],
    }
    # default=str: the failure feedback may still carry a compiled artifact from an upstream trace;
    # never let prompt assembly crash the repair loop.
    return system, json.dumps(payload, ensure_ascii=False, default=str)


def _question_type_from_context(context: dict[str, Any] | None) -> str:
    if not isinstance(context, dict):
        return ""
    direct = str(context.get("question_type") or context.get("QUESTION_TYPE") or "").strip()
    if direct in QUESTION_TYPES:
        return direct
    for key in ("repair_understanding", "step1_understanding", "understanding"):
        nested = context.get(key)
        if isinstance(nested, dict):
            found = _question_type_from_context(nested)
            if found:
                return found
    failed_plan = context.get("failed_plan")
    if isinstance(failed_plan, dict):
        raw = failed_plan.get("raw_response")
        if isinstance(raw, dict):
            typed = raw.get("typed_plan")
            if isinstance(typed, dict):
                found = _question_type_from_context(typed)
                if found:
                    return found
        spec = failed_plan.get("query_spec")
        if isinstance(spec, dict):
            intent = str(spec.get("intent") or "").strip()
            if intent in QUESTION_TYPES:
                return intent
    previous = context.get("previous_failure") or context.get("failure_feedback")
    if isinstance(previous, dict):
        return _question_type_from_context(previous)
    return ""


def _shell_for_type(question_type: str) -> dict[str, Any]:
    shell = TYPE_SHELLS.get(question_type)
    return dict(shell) if shell is not None else {"question_type": "one of question_types"}


@dataclass(frozen=True)
class ConsistencyVerdict:
    ok: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntentIssue:
    stage: str
    error_type: str
    reason: str
    step_id: str = ""
    operation: str = ""
    answer_operation: str = ""
    answer_step: str = ""
    answer_step_op: str = ""
    answer_step_returns: str = ""
    field_text: str = ""
    value: Any = ""
    value_type: str = ""
    candidate: str = ""
    top_candidates: tuple[dict[str, Any], ...] = ()
    suggested_actions: tuple[str, ...] = ()

    def code(self) -> str:
        if self.error_type == "schema_unknown_input":
            match = re.search(r"\binput\s+([^\s]+)\s+", self.reason)
            if match:
                return f"unknown_inputs:{match.group(1)}"
        if self.error_type == "answer_step_operation_mismatch" and self.answer_operation and self.answer_step_op:
            return f"operation_contract:{self.answer_operation}_requires_final_{self.answer_operation}_got_{self.answer_step_op}"
        if self.error_type == "answer_return_type_mismatch" and self.answer_step_returns:
            match = re.search(r"value_type\s+([^\s]+)", self.reason)
            expected = match.group(1) if match else ""
            suffix = f"_not_{expected}" if expected else ""
            return f"type_contract:answer_returns_{self.answer_step_returns}{suffix}"
        prefix = self.error_type
        if self.stage == "intent_program_validation":
            prefix = f"{self._contract_family()}:{self.error_type}"
        elif self.stage == "schema_grounding":
            prefix = self.error_type
        parts = [prefix]
        if self.step_id:
            parts.append(self.step_id)
        if self.reason:
            parts.append(self.reason)
        return ":".join(str(p) for p in parts if str(p))

    def _contract_family(self) -> str:
        if self.error_type.startswith("type_"):
            return "type_contract"
        if self.error_type.startswith("schema_"):
            return "schema"
        return "operation_contract"

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "error_type": self.error_type,
            "reason": self.reason,
            "step_id": self.step_id,
            "operation": self.operation,
            "answer_operation": self.answer_operation,
            "answer_step": self.answer_step,
            "answer_step_op": self.answer_step_op,
            "answer_step_returns": self.answer_step_returns,
            "field_text": self.field_text,
            "value": self.value,
            "value_type": self.value_type,
            "candidate": self.candidate,
            "top_candidates": list(self.top_candidates),
            "suggested_actions": list(self.suggested_actions),
        }


# nano echoes its filled plan NESTED under a drifting key: the prompt's `selected_type_shell`,
# `typed_plan`/`plan`, or the TYPE NAME itself ({"question_type": "count", "count": {...}}).
# Reading only the top level then compiles ZERO constraints, which executed as a whole-KG count
# in the L2 run. Unwrap any dict child that carries constraints/steps; outer explicit keys win.
# Dict-valued slots that legitimately never hold the plan are excluded.
_NON_PLAN_DICT_KEYS = frozenset({"comparison", "failure_feedback", "feedback",
                                 "step1_understanding", "understanding", "output_schema"})


def _unwrap_typed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for key, nested in payload.items():
        if key in _NON_PLAN_DICT_KEYS:
            continue
        if isinstance(nested, dict) and ("constraints" in nested or "steps" in nested):
            outer = {k: v for k, v in payload.items() if k != key and v not in (None, "", [], {})}
            return _unwrap_typed_payload({**nested, **outer})
    return payload


def _is_intent_program(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("answer_signature"), dict) \
        and isinstance(payload.get("program"), list)


def _intent_program_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if _is_intent_program(payload):
        return payload
    intent = payload.get("intent_program")
    if _is_intent_program(intent):
        return intent
    return None


_OP_RETURN_TYPES: dict[str, frozenset[str]] = {
    "filter_records": frozenset({"record_set"}),
    "bind_filter": frozenset({"record_set"}),
    "distinct": frozenset({"entity_set", "set"}),
    "select_distinct": frozenset({"entity_set", "set"}),
    "count": frozenset({"number"}),
    "sum": frozenset({"money", "number"}),
    # entity_set is nano shorthand on select steps; select IS the unique-value reduction and
    # runtime uniqueness enforces singularity (_SIGNATURE_VALUE_COMPAT already accepts it).
    "select": frozenset({"entity", "entity_set", "date", "text", "money", "number", "set", "unknown"}),
    "exists": frozenset({"boolean"}),
    "compare": frozenset({"boolean"}),
    "argmin": frozenset({"entity", "record", "record_set", "unknown"}),
    "argmax": frozenset({"entity", "record", "record_set", "unknown"}),
    "top_k": frozenset({"set", "entity_set"}),
}
_SIGNATURE_VALUE_COMPAT: dict[str, frozenset[str]] = {
    "number": frozenset({"number"}),
    "money": frozenset({"money", "number"}),
    "boolean": frozenset({"boolean"}),
    "entity": frozenset({"entity", "entity_set", "set", "record", "record_set", "unknown"}),
    "date": frozenset({"date", "unknown"}),
    "set": frozenset({"set", "entity_set"}),
    "unknown": frozenset({"unknown", "entity", "date", "text", "money", "number", "boolean", "set",
                          "entity_set", "record", "record_set"}),
}
_OP_FINAL_STEPS: dict[str, frozenset[str]] = {
    "count": frozenset({"count"}),
    "sum": frozenset({"sum"}),
    "select": frozenset({"select"}),
    "exists": frozenset({"exists"}),
    "distinct_set": frozenset({"distinct", "select_distinct"}),
    "compare": frozenset({"compare"}),
    "argmin": frozenset({"argmin"}),
    "argmax": frozenset({"argmax"}),
    "top_k": frozenset({"top_k"}),
}
# A signature reduction may run DIRECTLY over the program's final set step — the reduction is
# fully specified by answer_signature and lives in the graph return (reduction steps never become
# variables), so a ceremonial final step adds nothing. Mirrors _STEP_INPUT_TYPES.
_SET_REDUCTION_INPUTS: dict[str, frozenset[str]] = {
    "count": frozenset({"record_set", "entity_set", "set"}),
    "sum": frozenset({"record_set"}),
    "select": frozenset({"record_set", "entity_set", "set"}),
    "exists": frozenset({"record_set", "entity_set", "set"}),
    "distinct_set": frozenset({"record_set"}),
    "argmin": frozenset({"record_set"}),
    "argmax": frozenset({"record_set"}),
    "top_k": frozenset({"record_set"}),
}
_STEP_INPUT_TYPES: dict[str, tuple[frozenset[str], ...]] = {
    "filter_records": (),
    "bind_filter": (frozenset({"record_set"}), frozenset({"entity_set", "set"})),
    "distinct": (frozenset({"record_set"}),),
    "count": (frozenset({"record_set", "entity_set", "set"}),),
    "sum": (frozenset({"record_set"}),),
    "select": (frozenset({"record_set", "entity_set", "set"}),),
    "exists": (frozenset({"record_set", "entity_set", "set"}),),
    "compare": (frozenset({"number", "money", "date", "entity", "set", "boolean", "unknown"}),
                frozenset({"number", "money", "date", "entity", "set", "boolean", "unknown"})),
    "argmin": (frozenset({"record_set"}),),
    "argmax": (frozenset({"record_set"}),),
    "top_k": (frozenset({"record_set"}),),
}


def _verify_intent_program(intent: dict[str, Any]) -> list[IntentIssue]:
    issues: list[IntentIssue] = []
    signature = intent.get("answer_signature") if isinstance(intent.get("answer_signature"), dict) else {}
    operation = _normalise_intent_operation(signature.get("operation"))
    value_type = str(signature.get("value_type") or "unknown").casefold()
    program = [step for step in (intent.get("program") or []) if isinstance(step, dict)]
    reasons = [str(x) for x in (intent.get("unsupported_or_ambiguous") or ()) if str(x).strip()]
    answer_step_id = str(intent.get("answer_step") or "").strip()
    if operation == "abstain":
        if program:
            issues.append(_intent_issue(
                "abstain_program_must_be_empty",
                "abstain rows must not contain an executable official program",
                answer_operation=operation,
                suggested_actions=("move candidate steps to diagnostics, not program", "set program=[]"),
            ))
        if answer_step_id:
            issues.append(_intent_issue(
                "abstain_answer_step_must_be_empty",
                "abstain rows must leave answer_step empty",
                answer_operation=operation, answer_step=answer_step_id,
                suggested_actions=("set answer_step to empty string",),
            ))
        return issues
    if reasons and not program:
        # Doubts and nothing executable: the abstain contract applies.
        issues.append(_intent_issue(
            "unsupported_requires_abstain",
            f"unsupported_or_ambiguous requires operation=abstain, got {operation}",
            answer_operation=operation,
            suggested_actions=("set answer_signature.operation to abstain", "clear executable program"),
        ))
        return issues
    # reasons + an executable program is a HEDGE, not a contract violation: nano commits to an
    # answerable reading while noting wording caveats ("only matching records..."). The plan
    # proceeds — executor/verifier judge it — and the caveat is carried for disclosure. The v5/v6
    # live runs lost 21/50 answerable rows to treating these notes as abstain violations.
    if not operation:
        issues.append(_intent_issue("schema_missing_answer_operation", "answer_signature.operation is missing"))
    if operation not in _OP_FINAL_STEPS:
        issues.append(_intent_issue("unsupported_answer_operation", f"unsupported answer operation {operation}",
                                    answer_operation=operation))
    if not program:
        issues.append(_intent_issue("schema_empty_intent_program", "non-abstain program is empty",
                                    answer_operation=operation))
        return issues

    seen: dict[str, str] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for index, step in enumerate(program):
        sid = str(step.get("id") or "").strip()
        op = str(step.get("op") or "").strip().casefold()
        returns = str(step.get("returns") or "").strip().casefold()
        if not sid:
            issues.append(_intent_issue("schema_missing_step_id", f"step at index {index} has no id",
                                        operation=op))
            continue
        if sid in by_id:
            issues.append(_intent_issue("schema_duplicate_step_id", f"duplicate step id {sid}",
                                        step_id=sid, operation=op))
            continue
        by_id[sid] = step
        if op not in _OP_RETURN_TYPES:
            issues.append(_intent_issue("unsupported_op", f"unsupported step op {op}",
                                        step_id=sid, operation=op))
        elif returns not in _OP_RETURN_TYPES[op]:
            issues.append(_intent_issue("type_return_mismatch",
                                        f"{op} cannot return {returns}",
                                        step_id=sid, operation=op, answer_step_returns=returns,
                                        suggested_actions=(f"change returns to one of {sorted(_OP_RETURN_TYPES[op])}",)))
        for inp in [str(x) for x in (step.get("inputs") or []) if str(x).strip()]:
            if inp not in seen:
                issues.append(_intent_issue("schema_unknown_input",
                                            f"input {inp} is not a previous step id",
                                            step_id=sid, operation=op,
                                            suggested_actions=("root filter_records uses inputs=[]",
                                                       "later steps may only reference previous ids")))
        _verify_step_required_args(step, issues)
        seen[sid] = returns

    if not answer_step_id:
        issues.append(_intent_issue("missing_answer_step", "answer_step is required for non-abstain programs",
                                    answer_operation=operation))
        return issues
    answer_step = by_id.get(answer_step_id)
    if answer_step is None:
        issues.append(_intent_issue("unknown_answer_step", f"answer_step {answer_step_id} does not exist",
                                    answer_operation=operation, answer_step=answer_step_id))
        return issues
    final_op = str(answer_step.get("op") or "").strip().casefold()
    final_returns = str(answer_step.get("returns") or "").strip().casefold()
    allowed_final = _OP_FINAL_STEPS.get(operation, frozenset())
    # A program that ends at the SET being reduced is complete, not mismatched: the signature
    # names the reduction and _intent_return_spec compiles it over that step directly.
    reducible_final_set = final_op in {"filter_records", "bind_filter"} \
        and final_returns in _SET_REDUCTION_INPUTS.get(operation, frozenset())
    if final_op not in allowed_final and not reducible_final_set:
        issues.append(_intent_issue(
            "answer_step_operation_mismatch",
            f"{operation} requires final step {sorted(allowed_final)}, got {final_op}",
            answer_operation=operation, answer_step=answer_step_id,
            answer_step_op=final_op, answer_step_returns=final_returns,
            suggested_actions=(f"add a {next(iter(allowed_final), operation)} step and set answer_step to it",),
        ))
    if not reducible_final_set \
            and final_returns not in _SIGNATURE_VALUE_COMPAT.get(value_type, frozenset({value_type})):
        issues.append(_intent_issue(
            "answer_return_type_mismatch",
            f"answer_step returns {final_returns}, incompatible with answer value_type {value_type}",
            answer_operation=operation, answer_step=answer_step_id,
            answer_step_op=final_op, answer_step_returns=final_returns,
            suggested_actions=("set answer_step to a step whose returns matches answer_signature.value_type",),
        ))
    return issues


def _verify_step_required_args(step: dict[str, Any], issues: list[IntentIssue]) -> None:
    sid = str(step.get("id") or "").strip()
    op = str(step.get("op") or "").strip().casefold()
    args = step.get("args") if isinstance(step.get("args"), dict) else {}
    inputs = [str(x) for x in (step.get("inputs") or []) if str(x).strip()]
    filters = [f for f in (args.get("filters") or []) if isinstance(f, dict)]
    if op == "filter_records" and not filters:
        issues.append(_intent_issue("filter_records_requires_filters",
                                    "filter_records must have at least one filter",
                                    step_id=sid, operation=op))
    if op == "bind_filter" and (len(inputs) < 2 or not str(args.get("bind_field_text") or args.get("bind_field") or "").strip()):
        issues.append(_intent_issue("bind_filter_requires_record_and_entity_inputs",
                                    "bind_filter needs record/entity inputs and bind_field_text",
                                    step_id=sid, operation=op))
    if op in {"count", "sum", "select", "distinct", "exists", "argmin", "argmax", "top_k"} and not inputs:
        issues.append(_intent_issue(f"{op}_requires_input", f"{op} requires an input step",
                                    step_id=sid, operation=op))
    if op in {"sum", "argmin", "argmax", "top_k"} \
            and not str(args.get("metric_text") or args.get("metric") or args.get("field_text") or args.get("field") or "").strip():
        issues.append(_intent_issue(f"{op}_requires_metric_text", f"{op} requires metric_text or field_text",
                                    step_id=sid, operation=op,
                                    suggested_actions=("set metric_text to contract value/count/date as appropriate",)))
    if op in {"select", "distinct", "select_distinct"} and not str(args.get("field_text") or args.get("field") or "").strip():
        issues.append(_intent_issue(f"{op}_requires_field_text", f"{op} requires field_text",
                                    step_id=sid, operation=op))
    if op == "compare":
        if len(inputs) < 2 and not str(args.get("threshold") or "").strip():
            issues.append(_intent_issue("compare_requires_right_input_or_threshold",
                                        "compare requires a right input or threshold",
                                        step_id=sid, operation=op))
        if not str(args.get("comparator") or "").strip():
            issues.append(_intent_issue("compare_requires_comparator", "compare requires comparator",
                                        step_id=sid, operation=op))


def _intent_issue(error_type: str, reason: str, **kwargs: Any) -> IntentIssue:
    return IntentIssue(stage="intent_program_validation", error_type=error_type, reason=reason, **kwargs)


def _issue_codes(issues: list[IntentIssue]) -> list[str]:
    return [issue.code() for issue in issues]


def _pack_intent_failure(issues: list[IntentIssue]) -> str:
    return json.dumps({
        "codes": _issue_codes(issues),
        "issues": [issue.as_dict() for issue in issues],
    }, ensure_ascii=False)


def _unpack_intent_failure(reason: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(reason)
    except (TypeError, json.JSONDecodeError):
        return str(reason), []
    if not isinstance(payload, dict):
        return str(reason), []
    codes = [str(x) for x in payload.get("codes") or []]
    issues = [x for x in payload.get("issues") or [] if isinstance(x, dict)]
    return ";".join(codes), issues


def _intent_program_to_graph_payload(question: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Compile the Stage-1 typed intent program into the graph-plan contract.

    This is deliberately deterministic: malformed programs become structured feedback rather than
    another free-form interpretation layer.
    """
    intent = _intent_program_from_payload(payload)
    if intent is None:
        return None, "missing_intent_program"
    contract_issues = _verify_intent_program(intent)
    if contract_issues:
        return None, _pack_intent_failure(contract_issues[:8])
    signature = intent.get("answer_signature") or {}
    operation = _normalise_intent_operation(signature.get("operation"))
    reasons = [str(x) for x in (intent.get("unsupported_or_ambiguous") or ()) if str(x).strip()]
    # A hedge (reasons + executable program + non-abstain operation) proceeds; reasons force the
    # abstain graph only when the model committed to abstain or provided nothing executable.
    if operation == "abstain" or (reasons and not intent.get("program")):
        if operation == "abstain" and intent.get("program"):
            return None, "abstain_program_must_be_empty"
        graph = _empty_graph("abstain", "unanswerable", operation="abstain")
        return {
            "understanding_network": {"intent_program": intent, "unsupported_or_ambiguous": reasons},
            "graph_plan": graph,
        }, ""

    program = [step for step in (intent.get("program") or []) if isinstance(step, dict)]
    if not program:
        return None, "empty_intent_program"
    by_id = {str(step.get("id")): step for step in program if str(step.get("id") or "").strip()}
    answer_step_id = str(intent.get("answer_step") or "").strip()
    answer_step = by_id.get(answer_step_id) if answer_step_id else None
    variables: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    issues: list[IntentIssue] = []
    step_ops = [str(step.get("op") or "").strip().casefold() for step in program]
    bridge_capable = "bind_filter" in step_ops or step_ops.count("filter_records") >= 2
    for step in program:
        op = str(step.get("op") or "").strip()
        sid = str(step.get("id") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        inputs = [str(x) for x in (step.get("inputs") or []) if str(x).strip()]
        bad_inputs = [x for x in inputs if x not in by_id]
        if bad_inputs:
            issues.append(_intent_issue("schema_unknown_input",
                                        f"input {bad_inputs[0]} is not a known step id",
                                        step_id=sid, operation=op,
                                        suggested_actions=("root filter_records uses inputs=[]",)))
            continue
        if not sid:
            issues.append(_intent_issue("schema_missing_step_id", "step has no id", operation=op))
            continue
        if op in {"filter_records", "bind_filter"}:
            filters, f_issues = _intent_filters_to_graph(args.get("filters") or (),
                                                         allow_boolean_drop=bridge_capable)
            issues.extend(replace(issue, step_id=issue.step_id or sid, operation=issue.operation or op)
                          for issue in f_issues)
            var = {
                "var_id": sid,
                "kind": "record_set",
                "role": "contract_records",
                "filters": filters,
                "depends_on": inputs,
            }
            variables.append(var)
            for inp in inputs:
                relations.append({"from": inp, "to": sid,
                                  "bind_field": _normalise_bind_field(args.get("bind_field_text") or args.get("bind_field"))})
        elif op in {"distinct", "select_distinct"}:
            field = _normalise_bind_field(args.get("field_text") or args.get("field")
                                          or args.get("bind_field_text") or args.get("bind_field"))
            variables.append({
                "var_id": sid,
                "kind": "entity_set",
                "role": field if field in {"buyer", "supplier", "cpv", "category"} else "none",
                "filters": [],
                "depends_on": inputs,
            })
        elif op in {"count", "sum", "select", "exists", "compare", "argmin", "argmax", "top_k"}:
            # Producer steps are represented in the graph return, not duplicated as fake variables.
            continue
        elif op == "abstain":
            graph = _empty_graph("abstain", "unanswerable", operation="abstain")
            return {"understanding_network": {"intent_program": intent}, "graph_plan": graph}, ""
        else:
            issues.append(_intent_issue("unsupported_op", f"unsupported step op {op}",
                                        step_id=sid, operation=op))
    if issues:
        return None, _pack_intent_failure(issues[:6])

    return_spec = _intent_return_spec(signature, answer_step, operation)
    if not return_spec.get("input") and operation != "compare":
        fallback = _last_record_or_entity_variable(variables)
        if fallback:
            return_spec["input"] = fallback
    if not return_spec.get("input") and operation != "compare":
        return None, "intent_missing_return_input"
    qtype = _question_type_from_graph_return(str(return_spec.get("operation") or operation), return_spec)
    answer_field = _normalise_answer_field(signature.get("answer_field_text")
                                           or signature.get("answer_field")
                                           or return_spec.get("field"))
    graph = {
        "template": _template_from_intent_program(program, operation, reasons),
        "question_type": "unanswerable" if operation == "abstain" else qtype,
        "operation": return_spec["operation"],
        "answer_field": answer_field,
        "variables": variables or [{"var_id": "A", "kind": "record_set", "role": "contract_records",
                                    "filters": [], "depends_on": []}],
        "relations": relations,
        "comparison": {"operator": "none", "threshold": ""},
        "return": return_spec,
        "reasoning_order": [str(step.get("id")) for step in program if str(step.get("id") or "").strip()] + ["answer"],
    }
    network: dict[str, Any] = {"intent_program": intent}
    if reasons:
        network["caveats"] = reasons  # hedge notes ride along for answer-card disclosure
    return {"understanding_network": network, "graph_plan": graph}, ""


def _empty_graph(template: str, qtype: str, *, operation: str = "abstain") -> dict[str, Any]:
    return {
        "template": template,
        "question_type": qtype,
        "operation": operation,
        "answer_field": "none",
        "variables": [{"var_id": "A", "kind": "record_set", "role": "contract_records",
                       "filters": [], "depends_on": []}],
        "relations": [],
        "comparison": {"operator": "none", "threshold": ""},
        "return": _full_return("abstain", "A", "none"),
    }


def _normalise_intent_operation(value: Any) -> str:
    op = str(value or "").strip().casefold()
    return {"distinct": "distinct_set", "min": "argmin", "max": "argmax"}.get(op, op)


def _normalise_answer_field(value: Any) -> str:
    text = str(value or "").strip().casefold()
    grounded, _reason = canonical_slot_from_text(text, value_type="unknown") if text else ("", "")
    if grounded:
        text = grounded
    mapping = {
        "": "none", "none": "none", "contract_value": "value", "value_amount": "value",
        "money": "value", "procurement_category": "category", "cpv_code": "cpv",
        "cpv_label": "none", "organisation": "none", "organization": "none",
    }
    return mapping.get(text, text if text in _RETURN_FIELD_ENUM else "none")


def _normalise_bind_field(value: Any) -> str:
    text = str(value or "").strip().casefold()
    grounded, _reason = canonical_slot_from_text(text, value_type="unknown") if text else ("", "")
    if grounded:
        text = grounded
    mapping = {"cpv_code": "cpv", "procurement_category": "category", "contract_records": "contract"}
    return mapping.get(text, text if text in {"buyer", "supplier", "cpv", "category", "contract"} else "none")


def _intent_filters_to_graph(filters: Any, *,
                             allow_boolean_drop: bool = False) -> tuple[list[dict[str, Any]], list[IntentIssue]]:
    out: list[dict[str, Any]] = []
    issues: list[IntentIssue] = []
    for item in filters if isinstance(filters, list) else []:
        if not isinstance(item, dict):
            continue
        field_text = item.get("field_text") if item.get("field_text") is not None else item.get("slot")
        value_type = str(item.get("value_type") or "unknown")
        grounded = ground_field_text(field_text, value=item.get("value", ""), value_type=value_type)
        slot, ground_reason = grounded.slot, grounded.reason
        if not slot:
            # Backwards compatibility for v0 cached rows that already used canonical-ish slot names.
            legacy = str(item.get("slot") or "").strip().casefold()
            if legacy in {"year", "year_range", "years", "cpv_code", "cpv_label",
                          "procurement_category", "buyer", "supplier", "title", "date",
                          "amount", "has_signed_date", "unsupported"}:
                slot = legacy
            elif str(item.get("value") or "").strip().casefold() in {"true", "false", "yes", "no"} \
                    or value_type.casefold() == "boolean":
                # A boolean pseudo-filter is a RELATION description, not a filter ("supplier has
                # also won services contracts"=true — the bridge lives in the program steps, and
                # value_is_additive is re-added by grounding on every sum). Killing the whole
                # plan for it cost 5/50 answerable rows; genuinely unsupported concepts
                # ("reliable") are still caught by UNSUPPORTED_TERMS + the consistency cue.
                continue
            else:
                issues.append(_grounding_issue(grounded, item))
                continue
        value = str(item.get("value") or "").strip()
        op = str(item.get("operator") or "eq").strip().casefold()
        if not value or value.casefold() == "none":
            continue
        if slot == "cpv_label":
            # Useful audit atom but never an executable filter in the current KG runtime.
            continue
        if slot == "procurement_category":
            # strip decoration first: "services notice"/"goods contracts" ARE category values
            norm = _normalise_category(value) or value.casefold()
            if norm not in {"goods", "services", "works"}:
                issues.append(IntentIssue(
                    stage="schema_grounding",
                    error_type="type_gate_rejected",
                    reason="procurement_category only accepts goods/services/works",
                    field_text=str(field_text),
                    value=value,
                    value_type=value_type,
                    candidate="procurement_category",
                    suggested_actions=("treat as cpv_label if it came from CPV parentheses",
                                       "remove this filter if it is only a CPV label"),
                ))
                continue
            out.append({"slot": "category", "operator": _intent_operator(op), "value": norm})
            continue
        slot_map = {
            "cpv_code": "cpv", "year": "year", "years": "years", "year_range": "year_range",
            "buyer": "buyer", "supplier": "supplier", "title": "title", "date": "date",
            "has_signed_date": "has_signed_date",
        }
        graph_slot = slot_map.get(slot)
        if graph_slot:
            out.append({"slot": graph_slot, "operator": _intent_operator(op), "value": value})
        elif slot in {"amount", "unsupported"}:
            continue
        else:
            issues.append(IntentIssue(stage="schema_grounding",
                                      error_type="unsupported_filter_slot",
                                      reason=f"no executor mapping for grounded slot {slot}",
                                      field_text=str(field_text), value=value,
                                      value_type=value_type, candidate=slot))
    return out, issues


def _grounding_issue(grounded: Any, item: dict[str, Any]) -> IntentIssue:
    top = tuple(
        {"slot": c.slot, "score": round(c.score, 4), "method": c.method, "matched": c.matched}
        for c in getattr(grounded, "candidates", ())[:5]
    )
    candidate = top[0]["slot"] if top else ""
    reason = getattr(grounded, "reason", "") or "schema grounding failed"
    error_type = str(reason).split(":", 1)[0]
    nice_reason = _grounding_reason_text(error_type, candidate)
    return IntentIssue(
        stage="schema_grounding",
        error_type=error_type,
        reason=nice_reason,
        field_text=str(item.get("field_text") if item.get("field_text") is not None else item.get("slot") or ""),
        value=item.get("value", ""),
        value_type=str(item.get("value_type") or "unknown"),
        candidate=candidate,
        top_candidates=top,
        suggested_actions=_grounding_suggestions(error_type, candidate),
    )


def _grounding_reason_text(error_type: str, candidate: str) -> str:
    if error_type == "type_gate_rejected" and candidate == "procurement_category":
        return "procurement_category only accepts goods/services/works"
    if error_type == "ambiguous_schema_match":
        return "top schema candidates are too close to choose safely"
    if error_type == "no_confident_schema_match":
        return "no schema candidate exceeded the confidence threshold"
    return error_type


def _grounding_suggestions(error_type: str, candidate: str) -> tuple[str, ...]:
    if error_type == "type_gate_rejected" and candidate == "procurement_category":
        return ("use field_text='CPV label' for CPV parenthetical descriptions",
                "use procurement_category only for goods/services/works")
    if error_type == "ambiguous_schema_match":
        return ("rewrite field_text with a more specific role or value type",)
    if error_type == "no_confident_schema_match":
        return ("use a supported semantic field such as published year, CPV code, buyer, supplier, or contract value",)
    return ()


def _intent_operator(op: str) -> str:
    if op in {"in", "between"}:
        return op
    return "eq"


def _intent_return_spec(signature: dict[str, Any], answer_step: dict[str, Any] | None,
                        operation: str) -> dict[str, Any]:
    step_op = str((answer_step or {}).get("op") or "").strip().casefold()
    args = (answer_step or {}).get("args") if isinstance((answer_step or {}).get("args"), dict) else {}
    inputs = [str(x) for x in ((answer_step or {}).get("inputs") or []) if str(x).strip()]
    op = _normalise_intent_operation(step_op if step_op in _INTENT_OP_ENUM else operation)
    if op in {"distinct", "select_distinct"}:
        op = "distinct_set"
    if op in {"filter_records", "bind_filter"}:
        # The program ends at the set being reduced: the signature operation runs over THIS step,
        # not over its inputs (bind_filter's first input is the pre-bind set — never the answer).
        op = operation
        input_id = str((answer_step or {}).get("id") or "")
    elif op == "compare":
        input_id = str(inputs[0]) if inputs else ""
    else:
        input_id = str(inputs[0]) if inputs else str((answer_step or {}).get("id") or "")
    ret = _full_return(op, input_id, _normalise_answer_field(args.get("field_text") or args.get("field")
                                                            or signature.get("answer_field_text")
                                                            or signature.get("answer_field")))
    if op == "sum":
        ret["field"] = "value"
        ret["guards"] = ["value_is_additive"]
    if op == "distinct_set":
        ret["field"] = _normalise_answer_field(args.get("field_text") or args.get("field")
                                               or signature.get("answer_field_text")
                                               or signature.get("answer_field"))
    if op in {"argmin", "argmax"}:
        ret["field"] = _normalise_answer_field(args.get("field_text") or args.get("field")
                                               or signature.get("answer_field_text")
                                               or signature.get("answer_field") or "value")
    if op == "top_k":
        ret["k"] = int(args.get("k") or 3)
        ret["group_by"] = _normalise_bind_field(args.get("field_text") or args.get("field")
                                                or args.get("bind_field_text") or args.get("bind_field"))
        ret["metric"] = str(args.get("metric_text") or args.get("metric") or "count").casefold()
    if op == "compare":
        ret["left"] = str(inputs[0]) if inputs else ""
        ret["right"] = str(inputs[1]) if len(inputs) > 1 else str(args.get("threshold") or "")
        ret["metric"] = str(args.get("metric_text") or args.get("metric") or "count").casefold()
        ret["comparator"] = str(args.get("comparator") or "gt").casefold()
    return ret


def _full_return(operation: str, input_id: str, field: str) -> dict[str, Any]:
    return {
        "operation": operation,
        "input": input_id,
        "field": field or "none",
        "guards": [],
        "k": 0,
        "group_by": "none",
        "metric": "none",
        "comparator": "none",
        "left": "",
        "right": "",
    }


def _last_record_or_entity_variable(variables: list[dict[str, Any]]) -> str:
    for var in reversed(variables):
        if str(var.get("kind")) in {"record_set", "entity_set"}:
            return str(var.get("var_id") or "")
    return ""


def _template_from_intent_program(program: list[dict[str, Any]], operation: str,
                                  reasons: list[str]) -> str:
    if reasons or operation == "abstain":
        return "abstain"
    ops = {str(step.get("op") or "").casefold() for step in program}
    if "compare" in ops or operation == "compare":
        return "comparison"
    if {"distinct", "bind_filter"} & ops:
        if "bind_filter" in ops:
            return "bridge_join"
        if operation == "distinct_set":
            return "distinct_set"
    if operation in {"argmin", "argmax", "top_k"}:
        return "min_max_top_k"
    return "simple_filter_aggregate"


def _normalise_planner_payload(payload: dict[str, Any], question: str = "") -> dict[str, Any]:
    """Accept the new graph-plan contract and compile it into the legacy runtime DSL.

    LLMs no longer emit `executable_typed_plan`; the deterministic compiler derives it from
    `graph_plan`. Bare typed payloads remain supported for tests and old traces.
    """
    intent = _intent_program_from_payload(payload)
    if intent is not None:
        graph_payload, reason = _intent_program_to_graph_payload(question, payload)
        if graph_payload is None:
            return {"question_type": "unanswerable", "operation": "",
                    "reason": f"invalid_intent_program:{reason}", "_intent_program": intent}
        return _typed_payload_from_graph_plan(graph_payload, question)
    if isinstance(payload.get("graph_plan"), dict):
        return _typed_payload_from_graph_plan(payload, question)
    return _unwrap_typed_payload(payload)


def _typed_payload_from_graph_plan(payload: dict[str, Any], question: str = "") -> dict[str, Any]:
    graph = payload.get("graph_plan") if isinstance(payload.get("graph_plan"), dict) else payload
    graph = normalise_graph_plan(question or str(payload.get("question") or ""), graph)
    ret = graph.get("return") if isinstance(graph.get("return"), dict) else {}
    # Prefer the strict-schema top-level fields; fall back to deriving from return (legacy payloads).
    operation = str(graph.get("operation") or ret.get("operation") or "").strip()
    if operation == "abstain" or str(graph.get("question_type") or "") == "unanswerable":
        return {
            "question_type": "unanswerable",
            "operation": "",
            "reason": "graph planner abstained",
            "_understanding_network": payload.get("understanding_network"),
            "_graph_plan": graph,
        }
    qtype = str(graph.get("question_type") or "").strip() or _question_type_from_graph_return(operation, ret)
    if qtype not in QUESTION_TYPES:
        qtype = _question_type_from_graph_return(operation, ret)
    input_id = str(ret.get("input") or "")
    filters = _graph_filters(graph, input_id)
    variables = variables_map(graph)
    has_structure = any(
        (var.get("depends_on") or str(var.get("kind")) == "entity_set")
        for var in variables.values() if isinstance(var, dict)
    )
    typed: dict[str, Any] = {
        "question_type": qtype,
        "operation": _runtime_operation_from_graph(operation, qtype),
        "constraints": filters,
        "_understanding_network": payload.get("understanding_network"),
        "_graph_plan": graph,
        "_graph_has_structure": has_structure,
    }
    field = str(graph.get("answer_field") or ret.get("field") or "")
    if qtype in {"factoid", "set"}:
        typed["answer_field"] = _answer_field_from_semantic(field, graph, input_id)
    if qtype in {"boolean", "comparison"}:
        comparison = graph.get("comparison") if isinstance(graph.get("comparison"), dict) else \
            (ret.get("comparison") if isinstance(ret.get("comparison"), dict) else {})
        typed["comparison"] = comparison
        typed["answer_field"] = field or "value"
    if qtype == "top_k":
        typed["group_by"] = str(ret.get("group_by") or "buyer")
        typed["metric"] = "sum" if operation == "sum" or "sum" in field.casefold() else "count"
        typed["k"] = int(ret.get("k") or 3)
    if qtype == "min_max":
        typed["operation"] = "argmin" if operation in {"argmin", "min"} else "argmax"
    return typed


def _question_type_from_graph_return(operation: str, ret: dict[str, Any]) -> str:
    op = operation.casefold()
    if op in {"count"}:
        return "count"
    if op in {"sum", "total"}:
        return "sum"
    if op in {"exists", "boolean", "yes_no"}:
        return "boolean"
    if op in {"argmax", "argmin", "max", "min"}:
        return "min_max"
    if op in {"top_k", "rank"}:
        return "top_k"
    if op in {"set", "distinct_set"}:
        return "set"
    if op in {"compare", "comparison"}:
        return "compare_two" if ret.get("left") and ret.get("right") else "comparison"
    if op in {"select", "factoid"}:
        return "factoid"
    return "factoid"


def _runtime_operation_from_graph(operation: str, qtype: str) -> str:
    op = operation.casefold()
    if qtype == "set":
        return "distinct_set"
    if qtype == "factoid":
        return "select_unique"
    if qtype == "boolean":
        return "exists" if op in {"exists", "boolean", "yes_no"} else "predicate"
    if qtype == "comparison":
        return "predicate"
    if qtype == "compare_two":
        return "compare"
    if qtype == "min_max":
        return "argmin" if op in {"argmin", "min"} else "argmax"
    if qtype == "top_k":
        return "top_k"
    return op or (QUESTION_TYPES.get(qtype, {}).get("operations") or ("select_unique",))[0]


def _graph_filters(graph: dict[str, Any], input_id: str) -> list[dict[str, Any]]:
    variables = variables_map(graph)
    wanted = _dependency_closure(variables, input_id)
    # A variable other variables depend on is a bridge ANCHOR. Its org constraint's role follows
    # anchor semantics ("suppliers who worked with X" anchors on X as BUYER), which is the exact
    # inverse of flat-plan role phrasing; tag it so the role-flip check can exempt it.
    depended: set[str] = set()
    for var in variables.values():
        if isinstance(var, dict):
            depended.update(str(dep) for dep in (var.get("depends_on") or ()))
    filters: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for var_id in wanted:
        var = variables.get(var_id)
        if not isinstance(var, dict):
            continue
        is_anchor = var_id in depended
        for item in var.get("filters") or ():
            if isinstance(item, dict):
                _append_filter(filters, seen, item, anchor=is_anchor)
        inferred = _filter_from_grounded_variable(var)
        if inferred:
            _append_filter(filters, seen, inferred, anchor=is_anchor)
    return filters


def _dependency_closure(variables: dict[str, Any], root: str) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()

    def visit(var_id: str) -> None:
        if not var_id or var_id in seen:
            return
        seen.add(var_id)
        var = variables.get(var_id)
        if isinstance(var, dict):
            for dep in var.get("depends_on") or ():
                visit(str(dep))
        order.append(var_id)

    visit(root)
    return order


def _append_filter(filters: list[dict[str, Any]], seen: set[tuple[str, str, str]], item: dict[str, Any],
                   *, anchor: bool = False) -> None:
    slot = str(item.get("slot") or item.get("role") or "").strip()
    value = item.get("value", item.get("surface"))
    key = (slot, str(item.get("operator", item.get("op", "eq"))), str(value))
    if slot and value not in (None, "") and key not in seen:
        seen.add(key)
        # has_signed_date is an existence predicate: its value is a synthetic boolean, not a
        # question surface, so leave surface empty (otherwise the surface-fidelity check flags it).
        surface = "" if slot == "has_signed_date" else str(item.get("surface", value))
        entry = {
            "slot": slot,
            "surface": surface,
            "value": value,
            "operator": str(item.get("operator", item.get("op", "eq")) or "eq"),
        }
        if anchor:
            entry["anchor"] = True
        filters.append(entry)


def _filter_from_grounded_variable(var: dict[str, Any]) -> dict[str, Any] | None:
    grounding = var.get("grounding") if isinstance(var.get("grounding"), dict) else {}
    surface = grounding.get("surface") or var.get("known_surface")
    if not surface:
        return None
    slot = _slot_from_graph_variable(var)
    if slot not in {"buyer", "supplier"}:
        return None
    return {"slot": slot, "surface": surface, "value": surface, "operator": "eq"}


def _slot_from_graph_variable(var: dict[str, Any]) -> str:
    role = str(var.get("role") or var.get("description") or "").casefold()
    if "supplier" in role or "winner" in role:
        return "supplier"
    if "buyer" in role or "authority" in role or "publisher" in role:
        return "buyer"
    candidates = var.get("type_candidates") or var.get("node_type_candidates") or ()
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            typ = str(candidate.get("type") or "").casefold()
            if typ in {"buyer", "contracting_authority", "authority"}:
                return "buyer"
            if typ in {"supplier", "winner", "award_winner"}:
                return "supplier"
    return ""


def _answer_field_from_semantic(field: str, graph: dict[str, Any], input_id: str) -> str:
    low = field.casefold()
    if "buyer" in low or "authority" in low:
        return "buyer"
    if "supplier" in low or "winner" in low:
        return "supplier"
    if "date" in low:
        return "date"
    if "value" in low or "amount" in low:
        return "value"
    if "cpv" in low:
        return "cpv"
    if "categor" in low:
        return "category"
    variables = variables_map(graph)
    var = variables.get(input_id) if isinstance(variables.get(input_id), dict) else {}
    slot = _slot_from_graph_variable(var)
    return slot or "supplier"


def plan_consistency_check(question: str, payload: dict[str, Any]) -> ConsistencyVerdict:
    """Deterministic surface-fidelity check: question -> typed plan, BEFORE any KG access."""
    issues: list[str] = []
    payload = _normalise_planner_payload(payload, question)
    low = " ".join(question.casefold().split())
    qtype = str(payload.get("question_type", ""))
    spec = QUESTION_TYPES.get(qtype)
    if spec is None:
        return ConsistencyVerdict(False, (f"unknown_question_type:{qtype}",))
    # An ABSENT operation is not a semantic mismatch: compile_typed_plan defaults it to the
    # type's first operation, so the check must judge the same defaulted value. Only an
    # explicitly WRONG operation (e.g. count under a sum type) is a real inconsistency.
    operation = _resolve_operation(payload, qtype)
    if spec["operations"] and operation not in spec["operations"]:
        issues.append(f"operation_outside_type:{operation}!in:{qtype}")
    if qtype not in {"unanswerable", "ambiguous"}:
        issues.extend(_missing_explicit_atom_issues(question, payload))
        if any(cue in low for cue in _UNSUPPORTED_CUES):
            issues.append("unsupported_cue_requires_unanswerable")
    # STRUCTURAL bridge detection: fire the bridge cue only when the plan is genuinely FLAT.
    # A graph payload with a dependency edge or an entity_set variable IS bridge-shaped even
    # though its flattened question_type reads count/sum, the cue used to kill those plans.
    if qtype in {"count", "sum", "factoid", "min_max", "set"} and not payload.get("_graph_has_structure"):
        cue = next((c for c in _BRIDGE_CUES if c in low), "")
        if cue:
            issues.append(f"bridge_cue_requires_bridge_join:{cue}")
    # Singular-interrogative x set-return is handled DETERMINISTICALLY at normalise time (T10 in
    # graph_planning rewrites distinct_set -> select so run-time uniqueness adjudicates); a veto
    # here would also kill legitimately-unique cases the rewrite keeps alive.

    question_numbers = set(_NUM.findall(low.replace(",", "")))
    synthetic_surfaces = set(_SLOT_FIELDS) | set(_SLOT_FIELDS.values())
    for constraint in payload.get("constraints") or ():
        surface = str(constraint.get("surface", "") or "")
        value = constraint.get("value")
        # a surface that is a slot/field TOKEN (e.g. "award_date_signed") is a synthetic reference
        # the planner emitted, not question text; judge the value, not the token
        if surface and surface.casefold() not in low and surface.casefold() not in synthetic_surfaces:
            issues.append(f"surface_not_in_question:{surface[:40]}")
        for number in _NUM.findall(str(value)):
            if not _number_supported_by_question(number, question_numbers, low, raw_value=value):
                issues.append(f"invented_number:{number}")
        slot = str(constraint.get("slot", ""))
        field = _SLOT_FIELDS.get(slot, slot)
        # role check anchors on the surface, falling back to the VALUE (graph filters often carry
        # no surface; a silent role flip there produced confidently wrong bridge answers).
        # BRIDGE ANCHORS are exempt: "suppliers who worked with X" correctly constrains X as
        # BUYER even though the window reads supplier-ish; flat role phrasing does not apply.
        role_anchor = surface or str(value or "")
        if field in _ROLE_CUES and role_anchor and role_anchor.casefold() in low \
                and not constraint.get("anchor"):
            # mask interrogative ANSWER-role phrases ("which buyer is recorded where...") so the
            # asked-for role never counts as a cue against a legitimate constraint on the other role
            role_low = _INTERROG_ROLE_RE.sub(lambda m: " " * len(m.group(0)), low)
            span = role_low.find(role_anchor.casefold())
            window = role_low[max(0, span - 40): span]
            other = "supplier_name" if field == "buyer_name" else "buyer_name"
            if any(cue in window for cue in _ROLE_CUES[other]) \
                    and not any(cue in window for cue in _ROLE_CUES[field]):
                issues.append(f"role_flipped:{slot}:{role_anchor[:30]}")

    comparison = payload.get("comparison") or {}
    if comparison:
        op = str(comparison.get("operator", ""))
        if op in {">", "after", ">="} and not _GT_CUE_RE.search(low):
            issues.append("comparison_direction_gt_unsupported_by_text")
        if op in {"<", "before", "<="} and not _LT_CUE_RE.search(low):
            issues.append("comparison_direction_lt_unsupported_by_text")
        for number in _NUM.findall(str(comparison.get("threshold", "")).replace(",", "")):
            if not _number_supported_by_question(number, question_numbers, low,
                                                 raw_value=comparison.get("threshold")):
                issues.append(f"invented_threshold:{number}")
    return ConsistencyVerdict(not issues, tuple(issues))


def _missing_explicit_atom_issues(question: str, payload: dict[str, Any]) -> list[str]:
    low = question.casefold()
    present = _payload_text_blob(payload)
    issues: list[str] = []
    for year in sorted(set(_YEAR_TEXT.findall(question))):
        if year not in present:
            issues.append(f"missing_explicit_constraint:year:{year}")
    for cpv in sorted(set(_CPV_CODE.findall(question))):
        if cpv not in present:
            issues.append(f"missing_explicit_constraint:cpv:{cpv}")
    categories = set()
    for match in _CATEGORY_TEXT.findall(question):
        categories.update(part.casefold() for part in match if part)
    for category in sorted(categories):
        if category not in present:
            issues.append(f"missing_explicit_constraint:category:{category}")
    for title in re.findall(r'"([^"]{2,160})"', question):
        if title.casefold() not in present:
            issues.append(f"missing_explicit_constraint:title:{title[:40]}")
    if "award signed date" in low and not any(token in present for token in ("date", "award_date", "award signed")):
        issues.append("missing_answer_field:award_signed_date")
    return issues


def _payload_text_blob(value: Any) -> str:
    parts: list[str] = []
    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                parts.append(str(key))
                visit(val)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                visit(item)
        elif obj is not None:
            parts.append(str(obj))
    visit(value)
    return " ".join(parts).casefold()


def _number_supported_by_question(number: str, question_numbers: set[str], low_question: str, *,
                                  raw_value: Any) -> bool:
    if number in question_numbers or number.lstrip("0") in question_numbers:
        return True
    text = str(raw_value)
    date = _ISO_DATE.search(text)
    if date:
        year, month, day = date.groups()
        month_supported = any(name in low_question for name in _MONTH_NAMES.get(month, ()))
        day_supported = day in question_numbers or day.lstrip("0") in question_numbers
        year_supported = year in question_numbers
        if number in {month, month.lstrip("0")} and month_supported and day_supported and year_supported:
            return True
        if number in {day, day.lstrip("0")} and day_supported and (month_supported or month in question_numbers):
            return True
    return _scaled_number_supported(number, low_question)


def _scaled_number_supported(number: str, low_question: str) -> bool:
    target = _decimal_or_none(number.replace(",", ""))
    if target is None:
        return False
    multipliers = {
        "k": Decimal("1000"),
        "thousand": Decimal("1000"),
        "m": Decimal("1000000"),
        "mn": Decimal("1000000"),
        "million": Decimal("1000000"),
        "bn": Decimal("1000000000"),
        "billion": Decimal("1000000000"),
    }
    for raw, suffix in _SCALED_NUMBER.findall(low_question):
        amount = _decimal_or_none(raw)
        if amount is None:
            continue
        if amount * multipliers[suffix] == target:
            return True
    return False


def _decimal_or_none(value: str) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _resolve_operation(payload: dict[str, Any], qtype: str) -> str:
    """Resolve the payload operation the way compile does: default when absent, and collapse an
    echoed shell alternation ("exists|predicate") to its first option valid for the type."""
    allowed = QUESTION_TYPES.get(qtype, {}).get("operations") or ()
    raw = str(payload.get("operation", ""))
    if "|" in raw:
        for option in (part.strip() for part in raw.split("|")):
            if not allowed or option in allowed:
                return option
        raw = ""
    return raw or (allowed or ("",))[0]


# Parts of an echoed-placeholder constraint value ("NHS Blood and Transplant; 2023; CPV 33194220")
# that can be classified deterministically. Org role comes from the item's surface cues.
_ATOM_CPV_RE = re.compile(r"^(?:cpv\s*(?:code\s*)?)?(\d{8})$", re.I)
_ATOM_YEAR_RE = re.compile(r"^(?:published\s+in\s+|in\s+|year\s+)?(20[2-3]\d)$", re.I)
_SUPPLIER_SURFACE_CUES = ("awarded to", "supplier", "won by", "vendor")
_BUYER_SURFACE_CUES = ("published by", "buyer", "contracting authority", " by ")
_INTERROG_ROLE_RE = re.compile(r"\b(?:which|what|who(?:\s+is)?)\s+(?:the\s+)?(?:buyer|supplier)\b")


def _org_role_slot(part: str, surface: str, question_low: str) -> str:
    """Role for an org atom: from the item surface first, else from the question text just
    before the org mention. Empty string when neither side gives a cue."""
    for text in (surface, ""):
        if text:
            if any(cue in text for cue in _SUPPLIER_SURFACE_CUES):
                return "supplier"
            if any(cue in text for cue in _BUYER_SURFACE_CUES):
                return "buyer"
    span = question_low.find(part.casefold())
    if span >= 0:
        window = question_low[max(0, span - 30): span]
        if any(cue in window for cue in _SUPPLIER_SURFACE_CUES):
            return "supplier"
        if any(cue in window for cue in _BUYER_SURFACE_CUES):
            return "buyer"
    return ""


def _expand_placeholder_item(item: dict[str, Any], question_low: str) -> list[dict[str, Any]] | None:
    """Split one echoed-placeholder constraint into properly slotted items, or None if any part
    cannot be classified with certainty (the caller then fails structurally, never guesses)."""
    value = item.get("value", item.get("surface"))
    if not isinstance(value, str) or not value.strip():
        return None
    surface = str(item.get("surface", "")).casefold()
    expanded: list[dict[str, Any]] = []
    for part in (p.strip().rstrip(".") for p in value.split(";")):
        if not part:
            continue
        cpv = _ATOM_CPV_RE.match(part)
        year = _ATOM_YEAR_RE.match(part)
        category = _CATEGORY_TEXT.match(part.casefold()) or (
            part.casefold().replace("notices", "").replace("notice", "").strip() in ("goods", "services", "works"))
        if cpv:
            expanded.append({"slot": "cpv", "surface": part, "value": cpv.group(1), "operator": "eq"})
        elif year:
            expanded.append({"slot": "year", "surface": part, "value": int(year.group(1)), "operator": "eq"})
        elif category:
            cleaned = part.casefold().replace("notices", "").replace("notice", "").strip()
            expanded.append({"slot": "category", "surface": part, "value": cleaned, "operator": "eq"})
        else:
            role = _org_role_slot(part, surface, question_low)
            if not role:
                return None
            expanded.append({"slot": role, "surface": part, "value": part, "operator": "eq"})
    return expanded or None


def _quoted_spans(question: str) -> set[str]:
    return {span.casefold() for span in re.findall(r'"([^"]{2,160})"', question)}


# Payload-level misses that compile_typed_plan completes deterministically; when EVERY issue is
# one of these, the planner proceeds to compile instead of failing the plan outright.
_AUTOCOMPLETABLE_ISSUE_RE = re.compile(r"^missing_explicit_constraint:(?:year|cpv|category):")


def _only_autocompletable(issues: tuple[str, ...]) -> bool:
    return bool(issues) and all(_AUTOCOMPLETABLE_ISSUE_RE.match(issue) for issue in issues)


def _year_constraint(slot: str, op: str, value: Any) -> tuple[str, Any]:
    """Normalise typed year slots without silently narrowing multi-year questions."""
    years = [int(y) for y in _NUM.findall(str(value)) if len(y) == 4 and y.isdigit()]
    if isinstance(value, (list, tuple)):
        years = []
        for item in value:
            years.extend(int(y) for y in _NUM.findall(str(item)) if len(y) == 4 and y.isdigit())
    years = sorted(dict.fromkeys(years))
    if op == "between" or slot == "year_range":
        if len(years) >= 2:
            return "between", [years[0], years[-1]]
    if op == "in" or slot == "years" or len(years) > 1:
        return "in", years
    if years:
        return "eq", years[0]
    return op, value


def compile_typed_plan(question: str, payload: dict[str, Any], *,
                       org_resolver: Any = None) -> CandidatePlan:
    """Typed plan -> executable candidate. Entity surfaces are grounded via the resolver;
    unresolvable or ambiguous entities produce a structured non-planned candidate, never a guess."""
    graph_plan = None
    intent = _intent_program_from_payload(payload)
    if intent is not None:
        graph_payload, graph_reason = _intent_program_to_graph_payload(question, payload)
        if graph_payload is None:
            code_text, issue_payload = _unpack_intent_failure(graph_reason)
            candidate = _not_planned(question, "intent", f"invalid_intent_program:{code_text}")
            return replace(candidate, raw_response={"intent_program": intent,
                                                    "intent_issues": issue_payload,
                                                    "typed_plan": payload})
        payload = graph_payload
    if isinstance(payload.get("graph_plan"), dict):
        graph_plan, graph_reason = compile_graph_plan(question, payload, org_resolver=org_resolver)
        if graph_reason:
            return _not_planned(question, "graph", f"invalid_graph_plan:{graph_reason}")
    payload = _normalise_planner_payload(payload, question)
    qtype = str(payload.get("question_type", ""))
    operation = _resolve_operation(payload, qtype)
    spec_id = "typed_" + re.sub(r"\W+", "", qtype)[:24]
    if qtype == "unanswerable":
        spec = _spec(question, spec_id, "unsupported", (), intent="unsupported")
        return CandidatePlan(plan_id=f"{spec_id}:p0", query_spec=spec, status="unsupported",
                             rationale=str(payload.get("reason", "typed as unanswerable")),
                             planner_source="typed_llm", graph_plan=graph_plan)

    # Pre-pass: an echoed placeholder slot ("buyer|supplier|year|...") often still carries clean,
    # ';'-separated atoms in its value; split and re-slot them deterministically. Anything that
    # cannot be classified with certainty becomes a structured failure, never a guessed field.
    # A `title` filter whose value is NOT a quoted span of the question is L2 wrapper filler
    # ("matching procurement records only") that can only zero out the match set: drop it.
    question_low = " ".join(question.casefold().split())
    quoted = _quoted_spans(question)
    notes: list[str] = []
    items: list[dict[str, Any]] = []
    for item in payload.get("constraints") or ():
        slot = str(item.get("slot", ""))
        if slot in ("title",):
            value_text = str(item.get("value", item.get("surface") or ""))
            if value_text and value_text.casefold() not in quoted:
                notes.append(f"dropped phantom title filter {value_text[:40]!r}")
                continue
        if _SLOT_FIELDS.get(slot, slot) in CANONICAL_FIELDS:
            items.append(item)
            continue
        expanded = _expand_placeholder_item(item, question_low)
        if expanded is None:
            return _not_planned(question, spec_id, f"invalid_constraint_slot:{slot[:48]}")
        items.extend(expanded)

    constraints: list[QueryConstraint] = []
    for item in items:
        slot = str(item.get("slot", ""))
        field = _SLOT_FIELDS.get(slot, slot)
        value = item.get("value", item.get("surface"))
        op = str(item.get("operator", item.get("op", "eq")) or "eq")
        if field == "has_award_signed_date":
            constraints.append(QueryConstraint(field, "eq", True, source_text=str(item.get("surface", ""))))
            continue
        if field in ("buyer_name", "supplier_name") and org_resolver is not None:
            resolved = resolve_confident_org(org_resolver, str(value))
            if not resolved.ok:
                suffix = ""
                if resolved.candidates:
                    suffix = ":" + ";".join(h.linked_id for h in resolved.candidates[:4])
                return _not_planned(question, spec_id, f"{resolved.reason}:{value}{suffix}")
            if len(resolved.variants) > 1:
                constraints.append(QueryConstraint(field, "in", list(resolved.variants),
                                                   source_text=str(item.get("surface", ""))))
                continue
            value = resolved.hit.linked_id
        if field == "release_year":
            op, value = _year_constraint(str(item.get("slot", "")), op, value)
        constraints.append(QueryConstraint(field, op, value, source_text=str(item.get("surface", ""))))

    metadata: dict[str, Any] = {}
    answer_field, value_type = "", "string"
    comparison = payload.get("comparison") or {}
    if qtype in ("boolean", "comparison") and comparison and operation == "predicate":
        op_map = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte", "=": "eq",
                  "after": "after", "before": "before"}
        metadata = {"predicate_subject": "sum" if "value" in str(payload.get("answer_field", "")) or comparison.get("of") == "sum" else "field",
                    "comparator": op_map.get(str(comparison.get("operator")), "gt"),
                    "threshold": comparison.get("threshold"),
                    "threshold_type": "date" if str(comparison.get("operator")) in ("after", "before") else "number"}
        if metadata["predicate_subject"] == "sum":
            answer_field = "value_amount"
        else:
            metadata["predicate_field"] = _SLOT_FIELDS.get(str(payload.get("answer_field", "")),
                                                           str(payload.get("answer_field", "")) or "award_date_signed")
        value_type = "boolean"
    elif operation in ("select_unique", "distinct_set"):
        answer_field = _SLOT_FIELDS.get(str(payload.get("answer_field", "")),
                                        str(payload.get("answer_field", "")) or "supplier_name")
    elif operation in ("argmax", "argmin"):
        answer_field = "contract_node_id"
    elif operation == "top_k":
        metadata = {"group_by_field": _SLOT_FIELDS.get(str(payload.get("group_by", "")), "buyer_name"),
                    "metric": str(payload.get("metric", "count")), "k": int(payload.get("k", 3)),
                    "metric_field": "value_amount"}
    elif operation == "sum":
        answer_field, value_type = "value_amount", "currency"

    decomposition = None
    if qtype == "compare_two":
        left, right = str(payload.get("left", "")), str(payload.get("right", ""))
        year = tuple(c for c in constraints if c.field == "release_year")
        steps = tuple(SubQuery(sid, _spec(question, spec_id, "count",
                                          year + (QueryConstraint("buyer_name", "eq", name),)))
                      for sid, name in (("a", left), ("b", right)))
        decomposition = DecompositionPlan(spec_id, question, steps, combine="compare_gt",
                                          final_steps=("a", "b"))
        operation = "compare"
    elif qtype == "bridge_join" and payload.get("steps"):
        decomposition = _bridge_from_steps(question, spec_id, payload["steps"], constraints, operation)
        if decomposition is None:
            return _not_planned(question, spec_id, "bridge_steps_uncompilable")

    def build_spec(cons: list[QueryConstraint]) -> RuntimeQuerySpec:
        return _spec(question, spec_id, operation, tuple(cons), answer_field=answer_field,
                     value_type=value_type, intent=qtype, metadata=metadata,
                     sort_field="value_amount" if operation in ("argmax", "argmin") else "")

    spec = build_spec(constraints)
    # Post-compile fidelity net: the payload-level atom gate can pass on text ANYWHERE in the
    # payload while compile drops the constraint (nested-shell echo did exactly this and ran an
    # unconstrained whole-KG count). A single year / CPV / category atom is an unambiguous regex
    # extraction with a canonical field, so complete it deterministically (same philosophy as
    # grounding adding the additive guard); anything else missing still blocks execution.
    dropped = _compiled_atom_issues(question, spec, decomposition)
    if dropped and decomposition is None:
        addable = [d for d in dropped if d.partition(":")[0] in ("year", "cpv", "category")]
        if sum(1 for d in addable if d.startswith("year:")) > 1:
            # several years = range/union semantics; never guess how they distribute
            addable = [d for d in addable if not d.startswith("year:")]
        blocking = [d for d in dropped if d not in addable]
        if not blocking and addable:
            for entry in addable:
                kind, _, val = entry.partition(":")
                field, value = {"year": ("release_year", int(val) if val.isdigit() else val),
                                "cpv": ("tender_cpv_id", val),
                                "category": ("tender_category", val)}[kind]
                constraints.append(QueryConstraint(field, "eq", value,
                                                   source_text="deterministic atom completion"))
            notes.append("auto-completed atoms: " + ";".join(addable))
            spec = build_spec(constraints)
            dropped = []
    if dropped:
        return _not_planned(question, spec_id, "compiled_plan_dropped_atoms:" + ";".join(dropped[:4]))
    rationale = f"typed plan: {qtype}/{operation}" + (f" ({'; '.join(notes)})" if notes else "")
    return CandidatePlan(plan_id=f"{spec_id}:p0", query_spec=spec, status="planned", confidence=0.7,
                         rationale=rationale, warnings=tuple(notes), planner_source="typed_llm",
                         graph_plan=graph_plan, decomposition=decomposition)


def _compiled_atom_issues(question: str, spec: RuntimeQuerySpec,
                          decomposition: DecompositionPlan | None) -> list[str]:
    """Explicit question atoms (year/CPV/category/quoted title) missing from the compiled spec."""
    parts: list[str] = [spec.answer_field]
    def collect(constraints) -> None:
        for c in constraints:
            parts.extend((str(c.field), str(c.value), str(c.source_text)))
    collect(spec.constraints)
    parts.extend(str(v) for v in spec.metadata.values())
    if decomposition is not None:
        for step in decomposition.steps:
            collect(step.spec.constraints)
    blob = " ".join(parts).casefold()
    issues: list[str] = []
    for year in sorted(set(_YEAR_TEXT.findall(question))):
        if year not in blob:
            issues.append(f"year:{year}")
    for cpv in sorted(set(_CPV_CODE.findall(question))):
        if cpv not in blob:
            issues.append(f"cpv:{cpv}")
    categories = set()
    for match in _CATEGORY_TEXT.findall(question):
        categories.update(part.casefold() for part in match if part)
    for category in sorted(categories):
        if category not in blob:
            issues.append(f"category:{category}")
    for title in re.findall(r'"([^"]{2,160})"', question):
        if title.casefold() not in blob:
            issues.append(f"title:{title[:40]}")
    return issues


def _bridge_from_steps(question, spec_id, steps, extra, operation):
    """Compile a bridge skeleton (find_X -> find_Y(from_step) -> aggregate) into a DecompositionPlan."""
    try:
        entity_steps = [s for s in steps if not any(c.get("from_step") for c in s.get("constraints", ()))]
        hop2 = next(s for s in steps if any(c.get("from_step") for c in s.get("constraints", ())))
    except StopIteration:
        return None
    if not entity_steps:
        return None
    anchor = entity_steps[0]
    anchor_cons = tuple(QueryConstraint(_SLOT_FIELDS.get(str(c.get("slot", "")), str(c.get("slot", ""))),
                                        "eq", c.get("value", c.get("surface")))
                        for c in anchor.get("constraints", ()) if not c.get("from_step"))
    emit = _SLOT_FIELDS.get(str(hop2.get("bind_slot", "buyer")),
                            "buyer_name" if any(c.field == "supplier_name" for c in anchor_cons) else "supplier_name")
    h1 = SubQuery("h1", _spec(question, spec_id, "count", anchor_cons), kind="entity_set", emit_field=emit)
    hop2_cons = tuple(QueryConstraint(_SLOT_FIELDS.get(str(c.get("slot", "")), str(c.get("slot", ""))),
                                      "eq", c.get("value", c.get("surface")))
                      for c in hop2.get("constraints", ()) if not c.get("from_step")) + tuple(extra)
    op_spec = _spec(question, spec_id, operation, hop2_cons,
                    answer_field="value_amount" if operation == "sum" else "",
                    value_type="currency" if operation == "sum" else "string")
    ans = SubQuery("ans", op_spec, binds=(Binding("h1", emit),))
    return DecompositionPlan(spec_id, question, (h1, ans), final_steps=("ans",))


def _spec(question, spec_id, op, constraints, *, answer_field="", value_type="string",
          intent="typed", metadata=None, sort_field=""):
    return RuntimeQuerySpec(spec_id=spec_id, question=question, intent=intent,
                            constraints=tuple(constraints), answer_operation=op,
                            answer_field=answer_field, answer_value_type=value_type,
                            sort_field=sort_field, dedupe_key="contract_node_id",
                            requires_exhaustive_retrieval=True, metadata=metadata or {},
                            planner_version="typed_planner_v1")


def _not_planned(question, spec_id, reason):
    spec = _spec(question, spec_id, "count", (), intent="ambiguous")
    return CandidatePlan(plan_id=f"{spec_id}:p0", query_spec=spec, status="ambiguous",
                         rationale=reason, planner_source="typed_llm",
                         warnings=(reason,))


@dataclass
class TypedLLMPlanner:
    """LLM fills the DSL; deterministic consistency check + compile gate the result.

    A consistency failure or grounding ambiguity yields a structured non-planned candidate whose
    rationale carries the mismatch reason: the reflector's repair signal, never a silent guess.
    """

    client: Any
    model: str
    org_resolver: Any = None
    two_step: bool = True
    # Optional teacher split: a different model may run Step-1 understanding (semantic reading)
    # while `client`/`model` runs Step-2 structuring + repairs (strict JSON discipline). Label
    # authority stays with the executor/verifier either way, so the teacher mix never defines
    # correctness; record it as provenance.
    understanding_client: Any = None
    understanding_model: str = ""
    # Optional Step-1 injection: map normalised question -> parsed understanding dict. On a hit the
    # Step-1 LLM call is skipped and the cached briefing is used, so a Step-2 probe (or teacher run)
    # can reuse already-generated understandings and isolate Step-2 behaviour. Default None = live.
    understanding_cache: dict[str, dict[str, Any]] | None = None

    # Set False to force the legacy free-text json prompt even when the client supports schemas
    # (used to A/B strict json_schema vs prompt-only shaping).
    use_schema: bool = True
    # Step-2 prompt variant: "card" = capability card + template shells (added for nano);
    # "lean" = v6-era briefing-only prompt. Measured per planner model — see worklog 2026-07-04.
    plan_prompt_variant: str = "card"
    # json_schema variant: "filler" = all-required + none/empty conventions (the official strict
    # subset; gpt-5.4-nano's endpoint enforces it); "optional" = v6-era shape where unused
    # semantics are omitted (grok's endpoint accepts it and plans measurably cleaner).
    plan_schema_variant: str = "filler"
    # Structural resample budget: when sample 1 fails compile/consistency (a DETECTABLE defect),
    # ask for a different decomposition up to (plan_samples - 1) more times before falling back
    # to abstention/repair. Verifier stays the selector; wrong-but-executable plans are not
    # resampled (no oracle to catch them).
    plan_samples: int = 1
    # Optional semantic review: after the hard checks pass, the UNDERSTANDING model (nano) reads a
    # compact summary of the compiled plan and emits a diagnostic. It is deliberately NOT a hard
    # gate: a mismatch is fed to the pipeline's pre-execution reflector/replan hook, and if repair
    # fails the original plan still executes. Adds one cheap LLM call per planned question.
    plan_review: bool = False

    def _cached_understanding(self, question: str) -> dict[str, Any] | None:
        if not self.understanding_cache:
            return None
        return (self.understanding_cache.get(question)
                or self.understanding_cache.get(" ".join(str(question).split())))

    def _complete_graph(self, system: str, user: str) -> Any:
        """Prefer provider-enforced json_schema; fall back only for recoverable schema issues.

        Auth/resource errors (401/403/404, invalid subscription key, wrong endpoint) are
        infrastructure failures. Falling back to a plain JSON call only burns another request and
        hides the real cause, so those errors are re-raised for the caller to record explicitly.
        """
        if self.use_schema and hasattr(self.client, "complete_schema"):
            try:
                return self.client.complete_schema(model=self.model, system=system, user=user,
                                                   schema=graph_plan_schema(self.plan_schema_variant))
            except Exception as exc:  # noqa: BLE001 - live provider boundary
                if _is_non_fallback_schema_error(exc):
                    raise
        return self.client.complete_json(model=self.model, system=system, user=user)

    def _complete_intent_program(self, client: Any, model: str, system: str, user: str) -> Any:
        """Step-1 structured intent program; text fallback keeps older/simple clients usable."""
        if self.use_schema and hasattr(client, "complete_schema"):
            try:
                return client.complete_schema(model=model, system=system, user=user,
                                              schema=intent_program_schema())
            except Exception as exc:  # noqa: BLE001 - live provider boundary
                if _is_non_fallback_schema_error(exc):
                    raise
        if hasattr(client, "complete_text"):
            return client.complete_text(model=model, system=system, user=user)
        return client.complete_json(model=model, system=system, user=user)

    def _complete_repair_graph(self, system: str, user: str) -> Any:
        """Provider-enforced repair shape: diagnosis/action plus a repaired graph_plan."""
        if self.use_schema and hasattr(self.client, "complete_schema"):
            try:
                return self.client.complete_schema(model=self.model, system=system, user=user,
                                                   schema=repair_graph_plan_schema(self.plan_schema_variant))
            except Exception as exc:  # noqa: BLE001 - live provider boundary
                if _is_non_fallback_schema_error(exc):
                    raise
        return self.client.complete_json(model=self.model, system=system, user=user)

    def plan(self, question: str) -> tuple[CandidatePlan, ...]:
        understanding: dict[str, Any] | None = None
        understanding_result: Any = None
        understanding_source = "live"
        schema_context = retrieve_schema_context(question)
        if self.two_step:
            cached = self._cached_understanding(question)
            if cached is not None:
                understanding, understanding_source = cached, "cached"
            else:
                u_client = self.understanding_client if self.understanding_client is not None else self.client
                u_model = self.understanding_model or self.model
                u_system, u_user = question_intent_program_messages(question, schema_context=schema_context)
                try:
                    understanding_result = self._complete_intent_program(u_client, u_model, u_system, u_user)
                    understanding = _parse_understanding_result(understanding_result)
                except Exception as exc:  # pragma: no cover - live boundary
                    return (_not_planned(question, "typed_error", f"llm_understanding_error:{exc!r}"),)
                if not isinstance(understanding, dict):
                    return (_not_planned(question, "typed_error", "unparseable_question_understanding"),)
                if set(understanding.keys()) <= {"briefing"}:
                    # Step-1 lint: an unstructured briefing (labelled sections lost) starves
                    # Step-2; one cheap retry beats a downstream repair by an order of magnitude.
                    try:
                        understanding_result = self._complete_intent_program(
                            u_client, u_model,
                            u_system + " Your previous output lost the required labelled "
                                       "sections; emit EVERY section of the format.",
                            u_user)
                        retried = _parse_understanding_result(understanding_result)
                        if isinstance(retried, dict) and not (set(retried.keys()) <= {"briefing"}):
                            understanding = retried
                    except Exception:  # pragma: no cover - live boundary
                        pass  # keep the unstructured briefing; Step-2 may still cope

        if _is_intent_program(understanding):
            payload = {"intent_program": understanding}
            candidate = compile_typed_plan(question, payload, org_resolver=self.org_resolver)
            teacher = {"understanding": ("cached" if understanding_source == "cached"
                                         else (self.understanding_model or self.model)) if self.two_step else "",
                       "plan": "deterministic_intent_compiler",
                       "understanding_source": understanding_source}
            raw_response = {"understanding": understanding,
                            "typed_plan": payload,
                            "teacher": teacher,
                            "schema_context": schema_context,
                            "plan_review": {},
                            "understanding_raw": _chat_result_trace(understanding_result),
                            "typed_plan_raw": {}}
            raw_response.update(candidate.raw_response or {})
            return (replace(candidate, raw_response=raw_response),)

        system, user = typed_plan_messages(question, understanding=understanding,
                                           schema_context=schema_context,
                                           variant=self.plan_prompt_variant)
        # Structural resample: compile/consistency failures are DETECTABLE defects, so a fresh
        # decomposition may fix what feedback-free retries of the same text cannot. Verifier
        # remains the selector; executable-but-wrong plans are never resampled (no oracle).
        candidate = None
        payload: Any = None
        plan_result = None
        failure: tuple[str, str] | None = None
        for sample_index in range(max(1, int(self.plan_samples))):
            user_attempt = user if sample_index == 0 else user + (
                "\nRETRY NOTE: your previous graph plan failed structural validation "
                "(compile/consistency). Produce a DIFFERENT, simpler decomposition.")
            try:
                plan_result = self._complete_graph(system, user_attempt)
                payload = _wrap_schema_response(getattr(plan_result, "parsed", None))
            except Exception as exc:  # pragma: no cover - live boundary
                failure = ("typed_error", f"llm_error:{exc!r}")
                break
            if not isinstance(payload, dict):
                failure = ("typed_error", "unparseable_typed_plan")
                continue
            verdict = plan_consistency_check(question, payload)
            if not verdict.ok and not _only_autocompletable(verdict.issues):
                failure = ("typed_inconsistent",
                           "plan_semantic_mismatch:" + ";".join(verdict.issues[:4]))
                continue
            candidate = compile_typed_plan(question, payload, org_resolver=self.org_resolver)
            if candidate.status == "planned":
                failure = None
                break
            failure = (candidate.planner_source or "typed", candidate.rationale or "not_planned")
        if candidate is None or (failure is not None and candidate.status != "planned"):
            if candidate is None:
                return (_not_planned(question, failure[0] if failure else "typed_error",
                                     failure[1] if failure else "no_plan"),)
        review_note: dict[str, Any] = {}
        if self.plan_review and candidate.status == "planned":
            review_note = self._review_plan(question, candidate)
            if review_note.get("verdict") == "mismatch":
                candidate = replace(
                    candidate,
                    warnings=candidate.warnings + (
                        f"pre_execution_review_mismatch:{review_note.get('reason', '')[:120]}",
                    ),
                )
        teacher = {"understanding": ("cached" if understanding_source == "cached"
                                     else (self.understanding_model or self.model)) if self.two_step else "",
                   "plan": self.model, "understanding_source": understanding_source}
        return (replace(candidate, raw_response={"understanding": understanding, "typed_plan": payload,
                                                 "teacher": teacher,
                                                 "schema_context": schema_context,
                                                 "plan_review": review_note,
                                                 "understanding_raw": _chat_result_trace(understanding_result),
                                                 "typed_plan_raw": _chat_result_trace(plan_result)}),)

    def _review_plan(self, question: str, candidate: CandidatePlan) -> dict[str, Any]:
        """Semantic gut-check by the understanding model over the COMPILED plan (not raw JSON).

        The reviewer sees what will actually execute: operation, filters, bridge structure, and
        answers one thing: does this compute what the question asks? It never proposes a plan and
        never sees the oracle; a veto routes to the normal replan loop.
        """
        spec = candidate.query_spec
        summary = {
            "operation": spec.answer_operation,
            "answer_field": spec.answer_field,
            "filters": [{"field": c.field, "op": c.op, "value": c.value}
                        for c in spec.constraints if c.visible_to_user],
        }
        graph = getattr(candidate, "graph_plan", None)
        if graph is not None:
            summary["variables"] = [
                {"var_id": v.var_id, "kind": v.kind, "emit": v.emit_field,
                 "depends_on": list(v.depends_on),
                 "filters": [{"field": c.field, "value": c.value} for c in v.constraints]}
                for v in graph.variables]
            summary["return"] = {"operation": graph.return_spec.operation,
                                 "input": graph.return_spec.input_id}
        system = ("You review a compiled database plan for a procurement question. Judge ONLY "
                  "whether executing it answers the question (right aggregate, right filters, "
                  "right role direction, bridge present when the question needs one). Reply with "
                  "exactly two lines:\nVERDICT: ok|mismatch\nREASON: <one short line>")
        user = json.dumps({"question": question, "compiled_plan": summary}, ensure_ascii=False,
                          default=str)
        u_client = self.understanding_client if self.understanding_client is not None else self.client
        u_model = self.understanding_model or self.model
        try:
            if hasattr(u_client, "complete_text"):
                raw = u_client.complete_text(model=u_model, system=system, user=user).raw_text
            else:
                raw = str(getattr(u_client.complete_json(model=u_model, system=system, user=user),
                                  "raw_text", ""))
        except Exception as exc:  # pragma: no cover - live boundary; review must never break planning
            return {"verdict": "error", "reason": repr(exc)[:120]}
        verdict, reason = "ok", ""
        for line in str(raw).splitlines():
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                verdict = "mismatch" if "mismatch" in line.casefold() else "ok"
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        return {"verdict": verdict, "reason": reason, "raw": str(raw)[:400]}

    def replan_with_feedback(self, question: str, feedback: dict[str, Any]) -> tuple[CandidatePlan, ...]:
        repair_understanding: dict[str, Any] | None = None
        repair_understanding_result: Any = None
        understanding_source = "live"
        schema_context = retrieve_schema_context(question)
        if self.two_step:
            # Repair understanding is trace-aware: nano sees the failed Stage-2 graph plan and
            # structured failure feedback, not just the original question. A cached Step-1 briefing
            # is included as context, but repair still gets a chance to correct it.
            # MECHANICAL failures (grounding/schema rejections) don't need a semantic re-read of
            # the question; skip the extra LLM call and reuse the cached briefing (latency/cost).
            cached = self._cached_understanding(question)
            failure_stage = str(feedback.get("failure_stage") or "")
            # plan_compile failures (cycle, ungroundable variable, consistency rejection) are
            # structural, not semantic — the question reading did not fail, the graph shape did.
            mechanical = failure_stage in ("grounding", "schema", "plan_compile")
            if mechanical:
                repair_understanding = cached
                understanding_source = "cached" if cached is not None else "skipped_mechanical"
            else:
                feedback_for_repair = {**feedback}
                if cached is not None:
                    feedback_for_repair["previous_understanding"] = cached
                    understanding_source = "cached+repair"
                u_client = self.understanding_client if self.understanding_client is not None else self.client
                u_model = self.understanding_model or self.model
                u_system, u_user = repair_understanding_messages(question, feedback_for_repair)
                try:
                    if hasattr(u_client, "complete_text"):
                        repair_understanding_result = u_client.complete_text(model=u_model, system=u_system, user=u_user)
                        repair_understanding = _parse_understanding_result(repair_understanding_result)
                    else:
                        repair_understanding_result = u_client.complete_json(model=u_model, system=u_system, user=u_user)
                        repair_understanding = _parse_understanding_result(repair_understanding_result)
                except Exception as exc:  # pragma: no cover - live boundary
                    return (_not_planned(question, "typed_feedback_error",
                                         f"llm_feedback_understanding_error:{exc!r}"),)
                if not isinstance(repair_understanding, dict):
                    return (_not_planned(question, "typed_feedback_error",
                                         "unparseable_feedback_question_understanding"),)
            if repair_understanding is not None:
                feedback = {**feedback, "repair_understanding": repair_understanding,
                            "retrieved_schema_context": schema_context}
            else:
                feedback = {**feedback, "retrieved_schema_context": schema_context}

        repair_guideline = _repair_guideline(str(feedback.get("failure_reason") or ""))
        system, user = typed_replan_messages(question, feedback)
        try:
            feedback_result = self._complete_repair_graph(system, user)
            payload = getattr(feedback_result, "parsed", None)
        except Exception as exc:  # pragma: no cover - live boundary
            return (_not_planned(question, "typed_feedback_error", f"llm_feedback_error:{exc!r}"),)
        if not isinstance(payload, dict):
            return (_not_planned(question, "typed_feedback_error", "unparseable_feedback_typed_plan"),)
        plan_payload = _extract_repaired_plan(payload)
        if plan_payload is None:
            action = str(payload.get("repair_action", ""))
            if action == "abstain":
                plan_payload = {
                    "question_type": "unanswerable",
                    "operation": "",
                    "reason": str(payload.get("diagnosis", "reflector abstained")),
                }
            else:
                return (_not_planned(question, "typed_feedback_error", "missing_repaired_plan"),)
        verdict = plan_consistency_check(question, plan_payload)
        if not verdict.ok and not _only_autocompletable(verdict.issues):
            return (_not_planned(question, "typed_feedback_inconsistent",
                                 "feedback_plan_semantic_mismatch:" + ";".join(verdict.issues[:4])),)
        candidate = compile_typed_plan(question, plan_payload, org_resolver=self.org_resolver)
        action = str(payload.get("repair_action", "")) if payload is not plan_payload else ""
        rationale_suffix = f"feedback replan: {action}" if action else "feedback replan"
        teacher = {"understanding": ("cached" if understanding_source == "cached"
                                     else (self.understanding_model or self.model)) if self.two_step else "",
                   "plan": self.model, "understanding_source": understanding_source}
        return (replace(candidate, plan_id=f"{candidate.plan_id}:feedback",
                        rationale=f"{candidate.rationale} ({rationale_suffix})",
                        raw_response={"feedback": feedback, "reflector_output": payload,
                                      "repair_guideline": repair_guideline,
                                      "typed_plan": plan_payload,
                                      "teacher": teacher,
                                      "schema_context": schema_context,
                                      "repair_understanding": repair_understanding,
                                      "repair_understanding_raw": _chat_result_trace(repair_understanding_result),
                                      "reflector_raw": _chat_result_trace(feedback_result)}),)


def _chat_result_trace(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    return {
        "model": getattr(result, "model", ""),
        "raw_text": getattr(result, "raw_text", ""),
        "usage": getattr(result, "usage", {}) or {},
        "attempts": getattr(result, "attempts", None),
    }


def _is_non_fallback_schema_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return (
        "error code: 401" in text
        or "error code: 403" in text
        or "error code: 404" in text
        or "error code: 400" in text and (
            "invalid schema" in text
            or "response_format" in text
            or "json_schema" in text
            or "strict" in text
            or "unsupported" in text
        )
        or "status code: 401" in text
        or "status code: 403" in text
        or "status code: 404" in text
        or "status code: 400" in text and (
            "invalid schema" in text
            or "response_format" in text
            or "json_schema" in text
            or "strict" in text
            or "unsupported" in text
        )
        or "invalid subscription key" in text
        or "wrong api endpoint" in text
        or "access denied" in text
        or "resource not found" in text
        or "invalid schema" in text
    )


def _wrap_schema_response(payload: Any) -> Any:
    """A strict-schema response IS the graph_plan content (top-level variables/return). Wrap it as
    {"graph_plan": ...} so the compiler sees it; leave already-wrapped/legacy payloads untouched."""
    if not isinstance(payload, dict):
        return payload
    if "graph_plan" in payload:
        return payload
    if "answer_signature" in payload and "program" in payload:
        return payload
    if "variables" in payload or "return" in payload:
        return {"graph_plan": payload}
    return payload


def _extract_repaired_plan(payload: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(payload.get("repaired_graph_plan"), dict):
        return {
            "understanding_network": payload.get("repaired_understanding_network") or payload.get("understanding_network") or {},
            "graph_plan": payload["repaired_graph_plan"],
        }
    if isinstance(payload.get("graph_plan"), dict):
        return payload
    repaired = payload.get("repaired_plan")
    if isinstance(repaired, dict):
        return repaired
    if "question_type" in payload:
        return payload
    return None


def understanding_from_text(text: str) -> dict[str, Any]:
    """Parse a cached Step-1 briefing (raw text) into the understanding dict the Step-2 planner
    consumes. Falls back to `{"briefing": <sanitised text>}` when no labelled sections are found."""
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _normalise_understanding_keys(parsed)
    except json.JSONDecodeError:
        pass
    labelled = _parse_labelled_understanding(text)
    return labelled or {"briefing": _ascii_sanitise_text(text.strip())}


def _parse_understanding_result(result: Any) -> dict[str, Any] | None:
    parsed = getattr(result, "parsed", None)
    if isinstance(parsed, dict) and parsed:
        return _normalise_understanding_keys(parsed)
    raw = getattr(result, "raw_text", "")
    if isinstance(raw, str) and raw.strip():
        labelled = _parse_labelled_understanding(raw)
        if labelled:
            return labelled
        return {"briefing": _ascii_sanitise_text(raw.strip())}
    return parsed if isinstance(parsed, dict) else None


def _normalise_understanding_keys(payload: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "QUESTION_TYPE": "question_type",
        "QUERY_TEMPLATE": "query_template",
        "QUERY TEMPLATE": "query_template",
        "TEMPLATE": "query_template",
        "NEEDS_TO_RETURN": "needs_to_return",
        "KNOWN_INFORMATION": "known_information",
        "ROLE_DIRECTION": "role_direction",
        "REASONING_CHAIN": "reasoning_chain",
        "MISSING_OR_UNSUPPORTED_INFORMATION": "missing_or_unsupported_information",
        "NOTES_FOR_PLANNER": "notes_for_planner",
    }
    out = dict(payload)
    for old, new in mapping.items():
        if old in out and new not in out:
            out[new] = out[old]
    for key, value in list(out.items()):
        if isinstance(value, str):
            out[key] = _ascii_sanitise_text(value)
    return out


def _parse_labelled_understanding(text: str) -> dict[str, Any]:
    labels = {
        "QUESTION_TYPE": "question_type",
        "QUERY_TEMPLATE": "query_template",
        "QUERY TEMPLATE": "query_template",
        "TEMPLATE": "query_template",
        "NEEDS_TO_RETURN": "needs_to_return",
        "KNOWN_INFORMATION": "known_information",
        "ROLE_DIRECTION": "role_direction",
        "REASONING_CHAIN": "reasoning_chain",
        "MISSING_OR_UNSUPPORTED_INFORMATION": "missing_or_unsupported_information",
        "NOTES_FOR_PLANNER": "notes_for_planner",
        "ANSWER TYPE": "answer_type",
        "QUERY TEMPLATE": "query_template",
        "TEMPLATE": "query_template",
        "EXPLICIT INFO": "explicit_info",
        "REVERSE TREE": "reverse_tree",
        "PROCEDURE": "procedure",
        "TARGETS": "targets",
        "TARGETS (A, B, C)": "targets",
        "ROLES & DIRECTION": "roles_direction",
        "ROLES AND DIRECTION": "roles_direction",
        "AMBIGUITIES": "ambiguities",
    }
    current = ""
    values = {key: "" for key in labels.values()}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        matched = False
        for label, key in labels.items():
            prefix = f"{label}:"
            normalised = re.sub(r"^\d+\.\s*", "", line.upper())
            if normalised.startswith(prefix):
                current = key
                value = re.sub(r"^\d+\.\s*", "", line, count=1)
                values[key] = value[len(label) + 1:].strip()
                matched = True
                break
        if not matched and current and line:
            values[current] = (values[current] + " " + line).strip()
    return {key: _ascii_sanitise_text(value) for key, value in values.items() if value}


def _ascii_sanitise_text(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u2192": "->",
        "\u2194": "<->",
        "\u2208": " in ",
        "\u2229": " intersection ",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u00a3": "GBP ",
    }
    cleaned = "".join(replacements.get(ch, ch) for ch in text)
    return cleaned.encode("ascii", "ignore").decode("ascii")


__all__ = ["QUESTION_TYPES", "REPAIR_ACTIONS", "TYPE_SHELLS", "ConsistencyVerdict", "TypedLLMPlanner", "compile_typed_plan",
           "graph_plan_schema", "intent_program_schema", "repair_graph_plan_schema",
           "plan_consistency_check", "question_intent_program_messages",
           "question_understanding_messages", "repair_understanding_messages",
           "typed_plan_messages",
           "typed_replan_messages", "understanding_from_text"]

